use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mdf4_rs::{BufferedRangeReader, ByteRangeReader};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyTimeoutError, PyValueError};
use pyo3::prelude::*;

use crate::RecordingIndex;

const BATCH_SIZE: u64 = 2048;
const BUFFER_SIZE: usize = 4096;

#[derive(Clone, Debug)]
pub(super) enum ChannelKey {
    Name(String),
    Group(usize, String),
}

#[derive(Clone)]
pub(super) struct StreamChannel {
    pub key: ChannelKey,
    pub group: usize,
    pub channel: usize,
    pub time: Option<usize>,
}

type Sample = (f64, f64);
type StreamSamples = Vec<(ChannelKey, Option<Sample>)>;

struct State {
    running: bool,
    terminal: bool,
    generation: u64,
    position: f64,
    channels: Vec<StreamChannel>,
    queues: Vec<VecDeque<Sample>>,
    done: Vec<bool>,
    error: Option<String>,
}

impl State {
    fn reset(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.queues = vec![VecDeque::new(); self.channels.len()];
        self.done = vec![false; self.channels.len()];
        self.error = None;
    }
}

struct Group {
    index: usize,
    time: Option<usize>,
    channels: Vec<usize>,
    outputs: Vec<usize>,
    next: u64,
    count: u64,
}

pub(super) struct Stream {
    shared: Arc<(Mutex<State>, Condvar)>,
    worker: Option<JoinHandle<()>>,
}

pub(super) struct StreamReader {
    shared: Arc<(Mutex<State>, Condvar)>,
}

pub(super) fn validate_position(position: f64) -> PyResult<()> {
    if position.is_nan() {
        return Err(PyValueError::new_err("Stream position cannot be NaN"));
    }
    Ok(())
}

impl Stream {
    pub fn start(
        index: Arc<RecordingIndex>,
        source: String,
        position: f64,
        channels: Vec<StreamChannel>,
    ) -> PyResult<Self> {
        let count = channels.len();
        let shared = Arc::new((
            Mutex::new(State {
                running: true,
                terminal: false,
                generation: 0,
                position,
                channels,
                queues: vec![VecDeque::new(); count],
                done: vec![false; count],
                error: None,
            }),
            Condvar::new(),
        ));
        let worker_shared = Arc::clone(&shared);
        let worker = thread::Builder::new()
            .name("pymdf4-stream".to_owned())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    worker(index, source, Arc::clone(&worker_shared));
                }));
                if result.is_err()
                    && let Ok(mut state) = worker_shared.0.lock()
                {
                    state.error = Some("Stream worker failed".to_owned());
                    state.terminal = true;
                    worker_shared.1.notify_all();
                }
            })
            .map_err(|error| PyRuntimeError::new_err(format!("Cannot start stream: {error}")))?;
        Ok(Self {
            shared,
            worker: Some(worker),
        })
    }

    pub fn reader(&self) -> StreamReader {
        StreamReader {
            shared: Arc::clone(&self.shared),
        }
    }

    pub fn set_position(&self, position: f64) -> PyResult<()> {
        let (lock, wake) = &*self.shared;
        let mut state = lock.lock().unwrap();
        if state.terminal || self.worker.as_ref().is_some_and(JoinHandle::is_finished) {
            return Err(PyRuntimeError::new_err(
                state
                    .error
                    .clone()
                    .unwrap_or_else(|| "Stream worker stopped".to_owned()),
            ));
        }
        state.position = position;
        state.reset();
        wake.notify_all();
        Ok(())
    }

    pub fn set_channels(&self, channels: Vec<StreamChannel>) -> PyResult<()> {
        let (lock, wake) = &*self.shared;
        let mut state = lock.lock().unwrap();
        if state.terminal || self.worker.as_ref().is_some_and(JoinHandle::is_finished) {
            return Err(PyRuntimeError::new_err(
                state
                    .error
                    .clone()
                    .unwrap_or_else(|| "Stream worker stopped".to_owned()),
            ));
        }
        state.channels = channels;
        state.reset();
        wake.notify_all();
        Ok(())
    }
}

impl StreamReader {
    pub fn read(&self, timeout: Option<f64>) -> PyResult<Option<StreamSamples>> {
        let timeout = timeout
            .map(|seconds| {
                Duration::try_from_secs_f64(seconds).map_err(|_| {
                    PyValueError::new_err("timeout must be a finite non-negative number of seconds")
                })
            })
            .transpose()?;
        let started = Instant::now();
        let (lock, wake) = &*self.shared;
        let mut state = lock.lock().unwrap();
        loop {
            if let Some(error) = &state.error {
                return Err(PyRuntimeError::new_err(error.clone()));
            }
            if !state.running {
                return Err(PyRuntimeError::new_err("Stream is not running"));
            }
            if !state
                .queues
                .iter()
                .zip(&state.done)
                .any(|(queue, done)| queue.is_empty() && !done)
            {
                break;
            }
            let Some(timeout) = timeout else {
                return Ok(None);
            };
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return Err(PyTimeoutError::new_err("Timed out waiting for stream data"));
            }
            state = wake.wait_timeout(state, remaining).unwrap().0;
        }
        if state.queues.iter().all(VecDeque::is_empty) {
            return Err(PyStopIteration::new_err("End of stream"));
        }
        let mut result = Vec::with_capacity(state.channels.len());
        let mut position = state.position;
        for index in 0..state.channels.len() {
            let sample = state.queues[index].pop_front();
            if let Some((time, _)) = sample {
                position = position.max(time);
            }
            result.push((state.channels[index].key.clone(), sample));
        }
        state.position = position;
        wake.notify_all();
        Ok(Some(result))
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let (lock, wake) = &*self.shared;
        lock.lock().unwrap().running = false;
        wake.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn groups(index: &RecordingIndex, channels: &[StreamChannel]) -> Vec<Group> {
    let metadata = match index {
        RecordingIndex::Finalized(index) => &index.channel_groups,
        RecordingIndex::Recovered(index) => index.channel_groups(),
    };
    let mut groups: Vec<Group> = Vec::new();
    for (output, channel) in channels.iter().enumerate() {
        let group = match groups.iter_mut().find(|group| group.index == channel.group) {
            Some(group) => group,
            None => {
                groups.push(Group {
                    index: channel.group,
                    time: channel.time,
                    channels: Vec::new(),
                    outputs: Vec::new(),
                    next: 0,
                    count: metadata[channel.group].record_count,
                });
                groups.last_mut().unwrap()
            }
        };
        group.channels.push(channel.channel);
        group.outputs.push(output);
    }
    groups
}

struct CachedReader {
    reader: BufferedRangeReader,
    headers: HashMap<u64, Vec<u8>>,
    compressed_offsets: HashSet<u64>,
}

impl ByteRangeReader for CachedReader {
    type Error = mdf4_rs::Error;

    fn read_range(&mut self, offset: u64, length: u64) -> Result<Vec<u8>, Self::Error> {
        if length == 48 && self.compressed_offsets.contains(&offset) {
            if let Some(header) = self.headers.get(&offset) {
                return Ok(header.clone());
            }
            let header = self.reader.read_range(offset, length)?;
            self.headers.insert(offset, header.clone());
            Ok(header)
        } else {
            self.reader.read_range(offset, length)
        }
    }
}

fn worker(index: Arc<RecordingIndex>, source: String, shared: Arc<(Mutex<State>, Condvar)>) {
    let (lock, wake) = &*shared;
    let compressed_offsets = match index.as_ref() {
        RecordingIndex::Finalized(index) => index
            .channel_groups
            .iter()
            .flat_map(|group| &group.data_blocks)
            .filter(|block| block.is_compressed)
            .map(|block| block.file_offset)
            .collect(),
        RecordingIndex::Recovered(_) => HashSet::new(),
    };
    let mut reader = match BufferedRangeReader::new(&source) {
        Ok(reader) => CachedReader {
            reader,
            headers: HashMap::new(),
            compressed_offsets,
        },
        Err(error) => {
            let mut state = lock.lock().unwrap();
            state.error = Some(format!("MDF error: {error}"));
            state.terminal = true;
            wake.notify_all();
            return;
        }
    };
    let mut generation = u64::MAX;
    let mut position = 0.0;
    let mut groups = Vec::new();
    loop {
        let mut state = lock.lock().unwrap();
        if !state.running {
            return;
        }
        if generation != state.generation {
            generation = state.generation;
            position = state.position;
            groups = self::groups(&index, &state.channels);
        }
        let next = groups.iter().position(|group| {
            group.next < group.count
                && group
                    .outputs
                    .iter()
                    .all(|&output| state.queues[output].len() <= BUFFER_SIZE - BATCH_SIZE as usize)
        });
        let Some(next) = next else {
            for group in &groups {
                if group.next >= group.count {
                    for &output in &group.outputs {
                        state.done[output] = true;
                    }
                }
            }
            wake.notify_all();
            drop(wake.wait(state).unwrap());
            continue;
        };
        let group = &mut groups[next];
        let first = group.next;
        let end = first.saturating_add(BATCH_SIZE).min(group.count);
        group.next = end;
        let mut channels = group.channels.clone();
        if let Some(time) = group.time {
            channels.push(time);
        }
        drop(state);
        let result = match index.as_ref() {
            RecordingIndex::Finalized(index) => index.read_group_time_range(
                group.index,
                &channels,
                None,
                Some(first as f64),
                Some((end - 1) as f64),
                &mut reader,
            ),
            RecordingIndex::Recovered(index) => index.read_group_time_range(
                group.index,
                &channels,
                None,
                Some(first as f64),
                Some((end - 1) as f64),
                &mut reader,
            ),
        };
        let mut state = lock.lock().unwrap();
        if !state.running || state.generation != generation {
            continue;
        }
        let mut data = match result {
            Ok(data) => data,
            Err(error) => {
                state.error = Some(format!("MDF error: {error}"));
                wake.notify_all();
                drop(wake.wait(state).unwrap());
                continue;
            }
        };
        let times = if group.time.is_some() {
            data.values.pop().unwrap()
        } else {
            data.times.into_iter().map(Some).collect()
        };
        for (row, timestamp) in times.into_iter().enumerate() {
            let Some(time) = timestamp else {
                state.error = Some("MDF time channel is not numeric".to_owned());
                break;
            };
            if time.is_nan() || time < position {
                continue;
            }
            for (column, &output) in group.outputs.iter().enumerate() {
                match data.values[column][row] {
                    Some(value) => state.queues[output].push_back((time, value)),
                    None => state.error = Some("Channel conversion is not numeric".to_owned()),
                }
            }
        }
        if group.next >= group.count {
            for &output in &group.outputs {
                state.done[output] = true;
            }
        }
        wake.notify_all();
        if state.error.is_some() {
            drop(wake.wait(state).unwrap());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reader() -> StreamReader {
        StreamReader {
            shared: Arc::new((
                Mutex::new(State {
                    running: true,
                    terminal: false,
                    generation: 0,
                    position: 0.0,
                    channels: vec![StreamChannel {
                        key: ChannelKey::Name("value".to_owned()),
                        group: 0,
                        channel: 0,
                        time: None,
                    }],
                    queues: vec![VecDeque::new()],
                    done: vec![false],
                    error: None,
                }),
                Condvar::new(),
            )),
        }
    }

    #[test]
    fn timed_reads_cover_wait_and_terminal_states() {
        Python::initialize();
        let reader = reader();
        assert!(reader.read(None).unwrap().is_none());
        Python::attach(|py| {
            assert!(
                reader
                    .read(Some(0.0))
                    .unwrap_err()
                    .is_instance_of::<PyTimeoutError>(py)
            );
            assert!(
                reader
                    .read(Some(0.01))
                    .unwrap_err()
                    .is_instance_of::<PyTimeoutError>(py)
            );
            for timeout in [-1.0, f64::NAN, f64::INFINITY] {
                assert!(
                    reader
                        .read(Some(timeout))
                        .unwrap_err()
                        .is_instance_of::<PyValueError>(py)
                );
            }
        });
        let shared = Arc::clone(&reader.shared);
        let producer = thread::spawn(move || {
            let (lock, wake) = &*shared;
            lock.lock().unwrap().queues[0].push_back((1.0, 2.0));
            wake.notify_all();
        });
        let samples = reader.read(Some(1.0)).unwrap().unwrap();
        assert_eq!(samples[0].1, Some((1.0, 2.0)));
        producer.join().unwrap();
        let (lock, wake) = &*reader.shared;
        lock.lock().unwrap().done[0] = true;
        wake.notify_all();
        Python::attach(|py| {
            assert!(
                reader
                    .read(Some(1.0))
                    .unwrap_err()
                    .is_instance_of::<PyStopIteration>(py)
            );
        });
        lock.lock().unwrap().running = false;
        wake.notify_all();
        Python::attach(|py| {
            assert!(
                reader
                    .read(Some(1.0))
                    .unwrap_err()
                    .is_instance_of::<PyRuntimeError>(py)
            );
        });
    }

    #[test]
    fn stopping_wakes_all_waiting_readers() {
        Python::initialize();
        let reader = reader();
        let (started, starts) = std::sync::mpsc::channel();
        let mut threads = Vec::new();
        for _ in 0..2 {
            let reader = StreamReader {
                shared: Arc::clone(&reader.shared),
            };
            let started = started.clone();
            threads.push(thread::spawn(move || {
                started.send(()).unwrap();
                reader.read(Some(2.0))
            }));
        }
        for _ in 0..2 {
            starts.recv_timeout(Duration::from_secs(1)).unwrap();
        }
        let (lock, wake) = &*reader.shared;
        lock.lock().unwrap().running = false;
        wake.notify_all();
        for thread in threads {
            let error = thread.join().unwrap().unwrap_err();
            Python::attach(|py| assert!(error.is_instance_of::<PyRuntimeError>(py)));
        }
    }
}
