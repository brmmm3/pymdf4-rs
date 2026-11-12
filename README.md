# pymdf4

`pymdf4.load_mf4(path, keep=None, skipped=None, *, start=None, end=None)` loads scalar numeric MDF 4
channels using `mdf4-rs`. `path` may be a string or `pathlib.Path`. The optional
`keep` predicate receives each non-time channel name; rejected names are
appended to `skipped` when supplied. Arrays, composite channels, text, and
unsupported channel types are omitted. Invalid samples become `NaN`.
The optional `start` and `end` bounds select timestamps inclusively, in the
same units as the recording's timeline. Omit either for an open-ended range.
When a bound is supplied, records with NaN timestamps are excluded. Bounds
cannot be NaN, and `start` must not exceed `end`.

The result has a `source` string and one of these forms:

- `ReplayData`: `timeline` is a list of sorted unique times and `channels` is
  a dictionary mapping each name to a NaN-padded list of values.
- `SparseReplayData`: `timeline` is a NumPy array and `series` maps each name
  to a pair of NumPy arrays `(times, values)`. Time arrays are shared between
  channels with the same time base.

The default `pymdf4.DENSE_LIMIT` is 10,000,000 cells. Set it before calling
`load_mf4` to select a different limit. NumPy is required for sparse results.
Groups without a channel named `time` use sample indices for their time base.
For bounded loads, the local `mdf4-rs` crate streams supported non-interleaved
groups in bounded chunks and decodes only matching samples. It can skip data
outside the range for groups using sample indices; explicit timestamps must be
scanned, and compressed blocks must be decompressed to inspect them. Unbounded
loads and unsupported indexed layouts fall back to reading the entire file.
The loader supports DT, DV, DZ, DL, and HL data blocks; other layouts, including
LD blocks, are not supported.

File loading and Rust data processing release the Python GIL, and independent
groups and output columns are processed in parallel. The `keep` predicate and
Python result creation still run with the GIL held. Sparse arrays use Rust-owned
memory and cannot be resized in place.

The vendored `mdf4-rs/` package is based on crate version 0.7.0 and retains
its original licenses.

For indexed inspection and selected-channel reads, open a recording
with `recording = pymdf4.MdfFile(path)`, where `path` is a string or `Path`:

- `recording.structure()` returns a list of groups, each with `index`, `name`,
  `channels`, and `record_count`. Group indices distinguish repeated names.
- `recording.timeline(group)` returns timestamps in record order, and
  `recording.time_bounds(group)` returns their minimum and maximum, or `None`
  when there are no non-NaN timestamps.
- `recording.channel_count(group, name)` returns the group's record count for
  the named channel.
- `recording.channel_slice(group, name, start, end)` returns `(times, values)`
  for record indices from `start` inclusive to `end` exclusive. Indices beyond
  the record count are clipped.
- `recording.channel_time_slice(group, name, start, end)` returns the same
  pair for an inclusive time range. Both slices preserve record order.

Finalized indexed reads support scalar numeric channels in non-interleaved groups;
duplicate channel names within a group are rejected. A group without a channel
named `time` uses sample indices as timestamps. Time slices exclude NaN
timestamps, and `time_bounds` ignores them. Explicit time ranges require scanning
the group's timestamps. Unsupported indexed layouts, including interleaved record
IDs and nested conversions, return errors instead of falling back to full loading.
Keep the recording unchanged while its `MdfFile` index is in use.

For unfinalized recordings, `MdfFile` recovers record counts and indexes
interleaved fixed-size records without modifying the file. This supports direct
DT data blocks and the standard finalization flags for channel group counts,
the final DT length, and variable-length data sizes. Variable-length records
are skipped during scalar reads. When the DT length needs recovery, only a
single header-only DT block extending to the end of the file is supported.
Other finalization flags, layouts, and reads
from variable-length channel groups are rejected. The index stores an offset
per record, and opening an unfinalized file scans its data.
This recovery applies to `MdfFile`; `load_mf4` has separate parser limitations.

On Windows, build the extension with `cargo build`, copy the resulting
`target/debug/pymdf4.dll` to `pymdf4.pyd`, and put it on Python's import path.
The integration tests require `numpy` and `asammdf` and can be run with
`python -m unittest discover -s tests -v` after building the extension.

Run `python examples/load_testdata.py` to load and summarize the bundled
`testdata/11-bit-obd2.MF4` recording.
Run `python examples/read_mdf_structure.py` to list its groups and channels.
Run `python examples/inspect_mf4.py [path]` for timeline and channel slice
operations; the path defaults to `testdata/11-bit-obd2.MF4`.