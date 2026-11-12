use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use mdf4_rs::index::{IndexedChannel, IndexedChannelGroup, RecoveredMdfIndex, TimeRangeData};
use mdf4_rs::parsing::decoder::decode_channel_value_with_validity;
use mdf4_rs::{
    BufferedRangeReader, ByteRangeReader, Channel, DataType, FileRangeReader, MDF, MdfIndex,
};
use numpy::PyArray1;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rayon::prelude::*;

const DEFAULT_DENSE_LIMIT: usize = 10_000_000;
type MergedBases = HashMap<Vec<usize>, (Arc<Vec<f64>>, Vec<usize>)>;
type Series = Vec<(String, Arc<Vec<f64>>, Vec<f64>)>;
type IndexedParts = Vec<PyResult<Vec<(String, Part)>>>;

enum Output {
    Sparse(Series),
    Dense(Vec<(String, Vec<f64>)>),
}

#[pyclass]
struct ReplayData {
    #[pyo3(get)]
    timeline: Vec<f64>,
    #[pyo3(get)]
    channels: Py<PyDict>,
    #[pyo3(get)]
    source: String,
}

#[pyclass]
struct SparseReplayData {
    #[pyo3(get)]
    timeline: Py<PyAny>,
    #[pyo3(get)]
    series: Py<PyDict>,
    #[pyo3(get)]
    source: String,
}

/// An indexed MDF4 recording for inspecting groups and reading selected samples.
#[pyclass]
struct MdfFile {
    source: String,
    index: RecordingIndex,
}

enum RecordingIndex {
    Finalized(MdfIndex),
    Recovered(RecoveredMdfIndex),
}

#[pymethods]
impl MdfFile {
    /// Open an MDF4 file and index its structure.
    #[new]
    fn new(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<Self> {
        let source = path.str()?.to_str()?.to_owned();
        let index = py
            .detach(|| {
                let mut reader = FileRangeReader::new(&source)?;
                let unfinalized = reader.read_range(0, 8)?.starts_with(b"UnFinMF");
                if unfinalized {
                    Ok(RecordingIndex::Recovered(RecoveredMdfIndex::from_file(
                        &source,
                    )?))
                } else {
                    Ok(RecordingIndex::Finalized(MdfIndex::from_file_streaming(
                        &source,
                    )?))
                }
            })
            .map_err(mdf_error)?;
        Ok(Self { source, index })
    }

    #[getter]
    fn source(&self) -> &str {
        &self.source
    }

    /// Return groups with their indices, names, channel names, and record counts.
    fn structure<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let groups = PyList::empty(py);
        for (index, group) in self.groups().iter().enumerate() {
            let entry = PyDict::new(py);
            entry.set_item("index", index)?;
            entry.set_item("name", &group.name)?;
            entry.set_item(
                "channels",
                group
                    .channels
                    .iter()
                    .map(|channel| &channel.name)
                    .collect::<Vec<_>>(),
            )?;
            entry.set_item("record_count", group.record_count)?;
            groups.append(entry)?;
        }
        Ok(groups)
    }

    /// Return the group's timestamps in record order, or sample indices if no time channel exists.
    fn timeline(&self, py: Python<'_>, group: usize) -> PyResult<Vec<f64>> {
        let metadata = self.group(group)?;
        let time = group_time_index(metadata)?;
        Ok(self.read_group(py, group, &[], time, None, None)?.times)
    }

    /// Return the minimum and maximum non-NaN timestamps, or None for an empty timeline.
    fn time_bounds(&self, py: Python<'_>, group: usize) -> PyResult<Option<(f64, f64)>> {
        Ok(self
            .timeline(py, group)?
            .into_iter()
            .filter(|time| !time.is_nan())
            .fold(None, |bounds, time| {
                Some(match bounds {
                    Some((min, max)) => (f64::min(min, time), f64::max(max, time)),
                    None => (time, time),
                })
            }))
    }

    /// Return the number of records for a named channel in a group.
    fn channel_count(&self, group: usize, channel: &str) -> PyResult<u64> {
        let metadata = self.group(group)?;
        channel_index(metadata, channel)?;
        Ok(metadata.record_count)
    }

    /// Return (timestamps, values) for record indices [start, end).
    fn channel_slice(
        &self,
        py: Python<'_>,
        group: usize,
        channel: &str,
        start: u64,
        end: u64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let metadata = self.group(group)?;
        let selected = numeric_channel_index(metadata, channel)?;
        let time = group_time_index(metadata)?;
        if start > end {
            return Err(PyValueError::new_err("start must not exceed end"));
        }
        let start = start.min(metadata.record_count);
        let end = end.min(metadata.record_count);
        if start == end {
            return Ok((Vec::new(), Vec::new()));
        }
        if end > (1u64 << 53) {
            return Err(PyValueError::new_err(
                "Record indices above 2^53 are not supported",
            ));
        }
        let mut indices = vec![selected];
        if let Some(time_index) = time {
            indices.push(time_index);
        }
        let data = self.read_group(
            py,
            group,
            &indices,
            None,
            Some(start as f64),
            Some((end - 1) as f64),
        )?;
        let mut columns = data.values.into_iter();
        let values = numeric_values(columns.next().unwrap())?;
        let times = match time {
            Some(_) => numeric_values(columns.next().unwrap())?,
            None => data.times,
        };
        Ok((times, values))
    }

    /// Return (timestamps, values) for an inclusive time range in record order.
    fn channel_time_slice(
        &self,
        py: Python<'_>,
        group: usize,
        channel: &str,
        start: f64,
        end: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if start.is_nan() || end.is_nan() || start > end {
            return Err(PyValueError::new_err("Invalid time range"));
        }
        let metadata = self.group(group)?;
        let selected = numeric_channel_index(metadata, channel)?;
        let time = group_time_index(metadata)?;
        let data = self.read_group(py, group, &[selected], time, Some(start), Some(end))?;
        Ok((
            data.times,
            numeric_values(data.values.into_iter().next().unwrap())?,
        ))
    }
}

impl MdfFile {
    fn groups(&self) -> &[IndexedChannelGroup] {
        match &self.index {
            RecordingIndex::Finalized(index) => &index.channel_groups,
            RecordingIndex::Recovered(index) => index.channel_groups(),
        }
    }

    fn group(&self, index: usize) -> PyResult<&IndexedChannelGroup> {
        self.groups()
            .get(index)
            .ok_or_else(|| PyIndexError::new_err("Invalid group index"))
    }

    fn read_group(
        &self,
        py: Python<'_>,
        group: usize,
        channels: &[usize],
        time: Option<usize>,
        start: Option<f64>,
        end: Option<f64>,
    ) -> PyResult<TimeRangeData> {
        let metadata = self.group(group)?;
        if metadata.record_count == 0 {
            return Ok(TimeRangeData {
                times: Vec::new(),
                values: vec![Vec::new(); channels.len()],
            });
        }
        if matches!(&self.index, RecordingIndex::Finalized(_)) && metadata.data_blocks.is_empty() {
            return Err(PyRuntimeError::new_err(
                "MDF group has no indexed data blocks",
            ));
        }
        py.detach(|| {
            let mut reader = BufferedRangeReader::new(&self.source)?;
            match &self.index {
                RecordingIndex::Finalized(index) => {
                    index.read_group_time_range(group, channels, time, start, end, &mut reader)
                }
                RecordingIndex::Recovered(index) => {
                    index.read_group_time_range(group, channels, time, start, end, &mut reader)
                }
            }
        })
        .map_err(mdf_error)
    }
}

fn channel_index(group: &IndexedChannelGroup, name: &str) -> PyResult<usize> {
    let mut matching = group
        .channels
        .iter()
        .enumerate()
        .filter(|(_, channel)| channel.name.as_deref() == Some(name));
    let (index, _) = matching
        .next()
        .ok_or_else(|| PyKeyError::new_err(name.to_owned()))?;
    if matching.next().is_some() {
        return Err(PyValueError::new_err(format!(
            "Ambiguous channel name: {name}"
        )));
    }
    Ok(index)
}

fn is_indexed_scalar(channel: &IndexedChannel) -> bool {
    channel.component_addr == 0
        && channel.channel_type != 1
        && matches!(
            channel.data_type,
            DataType::UnsignedIntegerLE
                | DataType::UnsignedIntegerBE
                | DataType::SignedIntegerLE
                | DataType::SignedIntegerBE
                | DataType::FloatLE
                | DataType::FloatBE
        )
}

fn numeric_channel_index(group: &IndexedChannelGroup, name: &str) -> PyResult<usize> {
    let index = channel_index(group, name)?;
    if !is_indexed_scalar(&group.channels[index]) {
        return Err(PyValueError::new_err(format!(
            "Channel is not scalar numeric: {name}"
        )));
    }
    Ok(index)
}

fn group_time_index(group: &IndexedChannelGroup) -> PyResult<Option<usize>> {
    let index = group.channels.iter().position(|channel| {
        channel
            .name
            .as_deref()
            .is_some_and(|name| name.eq_ignore_ascii_case("time"))
    });
    if index.is_some_and(|index| !is_indexed_scalar(&group.channels[index])) {
        return Err(PyValueError::new_err(
            "MDF time channel is not scalar numeric",
        ));
    }
    Ok(index)
}

fn numeric_values(values: Vec<Option<f64>>) -> PyResult<Vec<f64>> {
    values
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| PyValueError::new_err("Channel conversion is not numeric"))
}

struct Part {
    group: usize,
    times: Arc<Vec<f64>>,
    values: Vec<f64>,
}

fn mdf_error(error: mdf4_rs::Error) -> PyErr {
    PyRuntimeError::new_err(format!("MDF error: {error}"))
}

fn is_scalar(channel: &Channel<'_>) -> bool {
    channel.block().component_addr == 0
        && matches!(
            channel.block().data_type,
            DataType::UnsignedIntegerLE
                | DataType::UnsignedIntegerBE
                | DataType::SignedIntegerLE
                | DataType::SignedIntegerBE
                | DataType::FloatLE
                | DataType::FloatBE
        )
        && channel.block().channel_type != 1
}

fn indexed_parts(
    py: Python<'_>,
    source: &str,
    keep: Option<&Bound<'_, PyAny>>,
    skipped: Option<&Bound<'_, PyList>>,
    start: Option<f64>,
    end: Option<f64>,
) -> PyResult<Option<IndexedParts>> {
    let index = match py.detach(|| {
        let mut reader = FileRangeReader::new(source)?;
        if reader.read_range(0, 8)?.starts_with(b"UnFinMF") {
            return Err(mdf4_rs::Error::BlockSerializationError(
                "Unfinalized files require the full parser".to_string(),
            ));
        }
        MdfIndex::from_file_streaming(source)
    }) {
        Ok(index) => index,
        Err(_) => return Ok(None),
    };
    if index.channel_groups.iter().any(|group| {
        group.record_id_size != 0 || (group.record_count > 0 && group.data_blocks.is_empty())
    }) {
        return Ok(None);
    }
    let mut selections = Vec::with_capacity(index.channel_groups.len());
    for group in &index.channel_groups {
        let time_name = group
            .channels
            .iter()
            .filter_map(|channel| channel.name.as_ref())
            .find(|name| name.eq_ignore_ascii_case("time"));
        let mut selected = Vec::new();
        let mut time_index = None;
        let mut seen = HashSet::new();
        for (channel_index, channel) in group.channels.iter().enumerate() {
            let Some(name) = channel.name.as_ref() else {
                continue;
            };
            let is_time = Some(name) == time_name;
            let wanted = is_time
                || match keep {
                    Some(predicate) => predicate.call1((name,))?.is_truthy()?,
                    None => true,
                };
            if !wanted {
                if let Some(skipped) = skipped {
                    skipped.append(name)?;
                }
                continue;
            }
            let scalar = channel.component_addr == 0
                && channel.channel_type != 1
                && matches!(
                    channel.data_type,
                    DataType::UnsignedIntegerLE
                        | DataType::UnsignedIntegerBE
                        | DataType::SignedIntegerLE
                        | DataType::SignedIntegerBE
                        | DataType::FloatLE
                        | DataType::FloatBE
                );
            if seen.insert(name.clone()) && scalar {
                if is_time {
                    time_index = Some(channel_index);
                } else {
                    selected.push((name.clone(), channel_index));
                }
            }
        }
        if !selected.is_empty() && time_name.is_some() && time_index.is_none() {
            return Err(PyValueError::new_err("MDF time channel is not scalar"));
        }
        selections.push((time_index, selected));
    }
    let parts = py.detach(|| {
        selections
            .into_par_iter()
            .enumerate()
            .map(|(group_index, (time_index, selected))| {
                if selected.is_empty() {
                    return Ok(Vec::new());
                }
                let mut reader = FileRangeReader::new(source).map_err(mdf_error)?;
                let indices = selected.iter().map(|(_, index)| *index).collect::<Vec<_>>();
                let data = index
                    .read_group_time_range(
                        group_index,
                        &indices,
                        time_index,
                        start,
                        end,
                        &mut reader,
                    )
                    .map_err(mdf_error)?;
                let mut order: Vec<usize> = (0..data.times.len()).collect();
                order.par_sort_by(|left, right| time_cmp(&data.times[*left], &data.times[*right]));
                let times = Arc::new(order.iter().map(|&index| data.times[index]).collect());
                let mut parts = Vec::new();
                for ((name, _), values) in selected.into_iter().zip(data.values) {
                    if values.is_empty() {
                        continue;
                    }
                    let Some(values) = values.into_iter().collect::<Option<Vec<_>>>() else {
                        continue;
                    };
                    parts.push((
                        name,
                        Part {
                            group: group_index,
                            times: Arc::clone(&times),
                            values: order.iter().map(|&index| values[index]).collect(),
                        },
                    ));
                }
                Ok(parts)
            })
            .collect()
    });
    Ok(Some(parts))
}

fn time_cmp(left: &f64, right: &f64) -> Ordering {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => left.partial_cmp(right).unwrap_or(Ordering::Equal),
    }
}

fn record_header(
    data: &[u8],
    id_size: usize,
    target_id: u64,
    record_sizes: &HashMap<u64, usize>,
) -> PyResult<Option<(u64, usize)>> {
    if data.len() < id_size {
        return Ok(None);
    }
    let id = match id_size {
        0 => target_id,
        1 | 2 | 4 | 8 => {
            let mut bytes = [0u8; 8];
            bytes[..id_size].copy_from_slice(&data[..id_size]);
            u64::from_le_bytes(bytes)
        }
        _ => return Err(PyValueError::new_err("Unsupported MDF record ID size")),
    };
    let size = *record_sizes
        .get(&id)
        .ok_or_else(|| PyValueError::new_err("Unknown MDF record ID"))?;
    if size == 0 || size < id_size {
        return Err(PyValueError::new_err("Invalid MDF record size"));
    }
    Ok(Some((id, size)))
}

/// Return channel names grouped by acquisition group.
#[pyfunction]
fn read_mdf_structure<'py>(py: Python<'py>, file_path: &str) -> PyResult<Bound<'py, PyDict>> {
    let mdf = MDF::from_file(file_path).map_err(mdf_error)?;
    let result = PyDict::new(py);
    for group in mdf.channel_groups() {
        let name = group
            .name()
            .ok()
            .flatten()
            .unwrap_or_else(|| "Unknown_Group".to_string());
        let channels = PyList::empty(py);
        for channel in group.channels() {
            if let Ok(Some(name)) = channel.name() {
                channels.append(name)?;
            }
        }
        result.set_item(name, channels)?;
    }
    Ok(result)
}

/// Load scalar channels from an MDF4 file into dense or sparse replay data.
///
/// Args:
///     path: MDF4 file path, as a string or pathlib.Path.
///     keep: Optional predicate called with each non-time channel name. Only
///         channels for which it returns a truthy value are selected.
///     skipped: Optional list to which names rejected by keep are appended.
///     start: Optional inclusive lower timestamp bound, in timeline units.
///     end: Optional inclusive upper timestamp bound, in timeline units. If
///         either bound is supplied, records with NaN timestamps are excluded.
///
/// Returns:
///     ReplayData or SparseReplayData, depending on pymdf4.DENSE_LIMIT.
#[pyfunction(signature = (path, keep=None, skipped=None, *, start=None, end=None))]
fn load_mf4(
    py: Python<'_>,
    path: &Bound<'_, PyAny>,
    keep: Option<&Bound<'_, PyAny>>,
    skipped: Option<&Bound<'_, PyList>>,
    start: Option<f64>,
    end: Option<f64>,
) -> PyResult<Py<PyAny>> {
    if start.is_some_and(f64::is_nan) || end.is_some_and(f64::is_nan) {
        return Err(PyValueError::new_err("Time bounds cannot be NaN"));
    }
    if matches!((start, end), (Some(start), Some(end)) if start > end) {
        return Err(PyValueError::new_err("start must not exceed end"));
    }
    let source = path.str()?.to_str()?.to_owned();
    let indexed = if start.is_some() || end.is_some() {
        indexed_parts(py, &source, keep, skipped, start, end)?
    } else {
        None
    };
    let group_parts = if let Some(parts) = indexed {
        parts
    } else {
        let mdf = py.detach(|| MDF::from_file(&source).map_err(mdf_error))?;
        let groups = mdf.channel_groups();
        let mut selections = Vec::with_capacity(groups.len());
        for group in &groups {
            let channels = group.channels();
            let names: Vec<Option<String>> = channels
                .iter()
                .map(|channel| channel.name())
                .collect::<mdf4_rs::Result<_>>()
                .map_err(mdf_error)?;
            let time_name = names
                .iter()
                .flatten()
                .find(|name| name.eq_ignore_ascii_case("time"));
            let mut selected = Vec::new();
            let mut seen = HashSet::new();
            for (index, name) in names.iter().enumerate() {
                let Some(name) = name else { continue };
                let is_time = Some(name) == time_name;
                let wanted = is_time
                    || match keep {
                        Some(predicate) => predicate.call1((name,))?.is_truthy()?,
                        None => true,
                    };
                if !wanted {
                    if let Some(skipped) = skipped {
                        skipped.append(name)?;
                    }
                    continue;
                }
                if seen.insert(name.clone()) && is_scalar(&channels[index]) {
                    selected.push((name.clone(), index, Vec::new()));
                }
            }
            selections.push((time_name.cloned(), selected));
        }

        py.detach(|| {
        groups
            .par_iter()
            .zip(selections.into_par_iter())
            .enumerate()
            .map(|(group_index, (group, (time_name, mut selected)))| -> PyResult<Vec<(String, Part)>> {
        if selected.iter().all(|(name, _, _)| Some(name) == time_name.as_ref()) {
            return Ok(Vec::new());
        }
        let channels = group.channels();
        let time_index = match time_name.as_ref() {
            Some(name) => Some(selected.iter().find(|(candidate, _, _)| candidate == name)
                .ok_or_else(|| PyValueError::new_err("MDF time channel is not scalar"))?.1),
            None => None,
        };
        let raw_group = group.raw_channel_group();
        let data_group = group.raw_data_group();
        let id_size = data_group.block.record_id_size as usize;
        let record_sizes: HashMap<u64, usize> = data_group
            .channel_groups
            .iter()
            .map(|candidate| {
                (
                    candidate.block.record_id,
                    id_size
                        + candidate.block.record_size as usize
                        + candidate.block.invalidation_size as usize,
                )
            })
            .collect();
        let target_id = raw_group.block.record_id;
        let mut record_count = 0usize;
        let mut fallback_times = Vec::new();
        let mut partial = Vec::new();
        let mut decode_record = |id, record: &[u8]| -> PyResult<()> {
            if id != target_id {
                return Ok(());
            }
            let fallback_time = record_count as f64;
            record_count += 1;
            if start.is_some() || end.is_some() {
                let time = match time_index {
                    Some(index) => {
                        let block = channels[index].block();
                        match decode_channel_value_with_validity(record, id_size, raw_group.block.record_size, block) {
                            Some(decoded) if decoded.is_valid => block
                                .apply_conversion_value(decoded.value, group.mmap())
                                .map_err(mdf_error)?
                                .as_f64()
                                .ok_or_else(|| PyValueError::new_err("MDF time channel is not numeric"))?,
                            _ => f64::NAN,
                        }
                    }
                    None => fallback_time,
                };
                if time.is_nan() || start.is_some_and(|bound| time < bound) || end.is_some_and(|bound| time > bound) {
                    return Ok(());
                }
            }
            if time_index.is_none() {
                fallback_times.push(fallback_time);
            }
            for (_, index, values) in &mut selected {
                let block = channels[*index].block();
                let value = match decode_channel_value_with_validity(
                    record,
                    id_size,
                    raw_group.block.record_size,
                    block,
                ) {
                    Some(decoded) if decoded.is_valid => block
                        .apply_conversion_value(decoded.value, group.mmap())
                        .map_err(mdf_error)?
                        .as_f64(),
                    _ => Some(f64::NAN),
                };
                values.push(value);
            }
            Ok(())
        };
        for block in data_group
            .resolved_data_blocks(group.mmap())
            .map_err(mdf_error)?
        {
            let mut data = block.data.as_slice();
            if !partial.is_empty() {
                let count = id_size.saturating_sub(partial.len()).min(data.len());
                partial.extend_from_slice(&data[..count]);
                data = &data[count..];
                let Some((id, size)) = record_header(&partial, id_size, target_id, &record_sizes)?
                else {
                    continue;
                };
                let count = (size - partial.len()).min(data.len());
                partial.extend_from_slice(&data[..count]);
                data = &data[count..];
                if partial.len() < size {
                    continue;
                }
                decode_record(id, &partial)?;
                partial.clear();
            }
            while !data.is_empty() {
                let Some((id, size)) = record_header(data, id_size, target_id, &record_sizes)?
                else {
                    partial.extend_from_slice(data);
                    break;
                };
                if data.len() < size {
                    partial.extend_from_slice(data);
                    break;
                }
                decode_record(id, &data[..size])?;
                data = &data[size..];
            }
        }
        if !partial.is_empty() {
            return Err(PyValueError::new_err("Truncated MDF record"));
        }

        let mut times = match time_name.as_ref() {
            Some(name) => {
                let time = selected
                    .iter()
                    .find(|(candidate, _, _)| candidate == name)
                    .ok_or_else(|| PyValueError::new_err("MDF time channel is not scalar"))?;
                time.2
                    .iter()
                    .copied()
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| PyValueError::new_err("MDF time channel is not numeric"))?
            }
            None => fallback_times,
        };
        let mut order: Vec<usize> = (0..times.len()).collect();
        order.par_sort_by(|left, right| time_cmp(&times[*left], &times[*right]));
        times = order.iter().map(|index| times[*index]).collect();
        let times = Arc::new(times);
        let mut result = Vec::new();
        for (name, _, values) in selected {
            if Some(&name) == time_name.as_ref() || values.is_empty() {
                continue;
            }
            let Some(values) = values.into_iter().collect::<Option<Vec<_>>>() else {
                continue;
            };
            let values = order.iter().map(|index| values[*index]).collect();
            result.push((name, Part {
                group: group_index,
                times: Arc::clone(&times),
                values,
            }));
        }
        Ok(result)
            })
            .collect::<Vec<_>>()
    })
    };
    let dense_limit: usize = py.import("pymdf4")?.getattr("DENSE_LIMIT")?.extract()?;
    let (timeline, output) = py.detach(|| -> PyResult<_> {
        let mut parts: Vec<(String, Vec<Part>)> = Vec::new();
        let mut part_indices = HashMap::new();
        for group in group_parts {
            for (name, part) in group? {
                let index = *part_indices.entry(name.clone()).or_insert_with(|| {
                    parts.push((name, Vec::new()));
                    parts.len() - 1
                });
                parts[index].1.push(part);
            }
        }

        let mut merged = MergedBases::new();
        let mut series = Vec::new();
        for (name, channel_parts) in parts {
            if channel_parts.len() == 1 {
                let part = channel_parts.into_iter().next().unwrap();
                series.push((name, part.times, part.values));
                continue;
            }
            let key = channel_parts.iter().map(|part| part.group).collect();
            let (times, order) = merged.entry(key).or_insert_with(|| {
                let all_times: Vec<f64> = channel_parts
                    .iter()
                    .flat_map(|part| part.times.iter().copied())
                    .collect();
                let mut order: Vec<usize> = (0..all_times.len()).collect();
                order.par_sort_by(|left, right| time_cmp(&all_times[*left], &all_times[*right]));
                (
                    Arc::new(order.iter().map(|index| all_times[*index]).collect()),
                    order,
                )
            });
            let values: Vec<f64> = channel_parts
                .into_iter()
                .flat_map(|part| part.values)
                .collect();
            series.push((
                name,
                Arc::clone(times),
                order.iter().map(|index| values[*index]).collect(),
            ));
        }

        let mut bases = HashSet::new();
        let mut timeline: Vec<f64> = series
            .iter()
            .filter(|(_, times, _)| bases.insert(Arc::as_ptr(times)))
            .flat_map(|(_, times, _)| times.iter().copied())
            .collect();
        timeline.par_sort_by(time_cmp);
        timeline.dedup_by(|left, right| time_cmp(left, right) == Ordering::Equal);
        let output = if timeline.len().saturating_mul(series.len()) > dense_limit {
            Output::Sparse(series)
        } else {
            Output::Dense(
                series
                    .into_par_iter()
                    .map(|(name, times, values)| {
                        let mut column = vec![f64::NAN; timeline.len()];
                        for (time, value) in times.iter().zip(values) {
                            let index = timeline.partition_point(|candidate| {
                                time_cmp(candidate, time) == Ordering::Less
                            });
                            column[index] = value;
                        }
                        (name, column)
                    })
                    .collect(),
            )
        };
        Ok((timeline, output))
    })?;
    if let Output::Sparse(series) = output {
        py.import("numpy")?;
        let sparse = PyDict::new(py);
        let mut shared_times: HashMap<*const Vec<f64>, Py<PyAny>> = HashMap::new();
        for (name, times, values) in series {
            let key = Arc::as_ptr(&times);
            if let std::collections::hash_map::Entry::Vacant(entry) = shared_times.entry(key) {
                entry.insert(
                    PyArray1::from_vec(py, times.as_ref().clone())
                        .into_any()
                        .unbind(),
                );
            }
            let time_array = shared_times.get(&key).unwrap().bind(py);
            sparse.set_item(name, (time_array, PyArray1::from_vec(py, values)))?;
        }
        return Ok(Py::new(
            py,
            SparseReplayData {
                timeline: PyArray1::from_vec(py, timeline).into_any().unbind(),
                series: sparse.unbind(),
                source,
            },
        )?
        .into_any());
    }
    let Output::Dense(columns) = output else {
        unreachable!()
    };
    let channels = PyDict::new(py);
    for (name, column) in columns {
        channels.set_item(name, column)?;
    }
    Ok(Py::new(
        py,
        ReplayData {
            timeline,
            channels: channels.unbind(),
            source,
        },
    )?
    .into_any())
}

#[pymodule]
fn pymdf4(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("DENSE_LIMIT", DEFAULT_DENSE_LIMIT)?;
    m.add_class::<ReplayData>()?;
    m.add_class::<SparseReplayData>()?;
    m.add_class::<MdfFile>()?;
    m.add_function(wrap_pyfunction!(read_mdf_structure, m)?)?;
    m.add_function(wrap_pyfunction!(load_mf4, m)?)?;
    Ok(())
}
