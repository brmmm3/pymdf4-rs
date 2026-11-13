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

For groups with explicit timestamps, the first `channel_time_slice` scans and
indexes that group's timestamps. Later slices reuse the index and read only the
span containing matching records, so narrow windows are faster when timestamps
are mostly in order. The index uses approximately 16 bytes per non-NaN sample
until the `MdfFile` is released. Unsorted timestamps remain supported but may
require reading a larger span. This cache does not apply to `load_mf4`.

For continuous reads, use the streaming methods on `MdfFile`:

```python
recording.stream_start(0.0, [(646, "MisfDet_stcEmiMisfCntrCyl_[1]")])
try:
  while True:
    try:
      samples = recording.stream_read(timeout=5.0)
    except StopIteration:
      break
    print(samples)
finally:
  recording.stream_stop()
```

`stream_start(start, channels)` starts a background reader. Channels may be
unique names or `(group_index, name)` pairs; use pairs for ambiguous names.
Only scalar numeric channels are supported, and the same channel cannot be
selected twice. Calling `stream_start` again replaces the existing stream.

`stream_read()` returns `None` immediately while any active channel needs data.
`stream_read(timeout=seconds)` instead waits up to the specified non-negative,
finite number of seconds and raises `TimeoutError` if data is still unavailable.
A timeout of zero checks immediately and raises `TimeoutError` if not ready.
When data is available, either form returns a dictionary keyed by the requested
channel identifiers with one `(timestamp, value)` per channel. A channel that finishes before the others has
the value `None`. When all channels finish, it raises `StopIteration`. Samples
retain each group's record order and are not synchronized across groups;
NaN timestamps are excluded. Background read errors raise `RuntimeError`.
Groups without a `time` channel use record indices as timestamps.

`stream_set_pos(time)` discards buffered data and starts again at the inclusive
time, and `stream_set_channels(channels)` replaces the selection from the
greatest timestamp already returned (or the start/seek position before any
read). Both operations restart reading in the background. With unsorted
timestamps, the latter can repeat or omit earlier records. `stream_stop()`
stops and joins the worker; it is safe to call more than once. Prefetching is
bounded per channel, but startup and seeking may require scanning records before
the requested time. The recording must remain unchanged while streaming.

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

Build wheels with maturin 1.9.4 or newer. For a manual extension build, set
`PYO3_BUILD_EXTENSION_MODULE=1` when running `cargo build`. On Windows, copy
the resulting `target/debug/pymdf4.dll` to `pymdf4.pyd` and put it on Python's
import path. Run Rust tests with `cargo test` without that environment variable.
The integration tests require `numpy` and `asammdf` and can be run with
`python -m unittest discover -s tests -v` after building the extension.
Wheels built with maturin include the Python type stubs and `py.typed` marker.

Run `python examples/load_testdata.py` to load and summarize the bundled
`testdata/11-bit-obd2.MF4` recording.
Run `python examples/read_mdf_structure.py` to list its groups and channels.
Run `python examples/inspect_mf4.py [path]` for timeline and channel slice
operations; the path defaults to `testdata/11-bit-obd2.MF4`.
Run `python examples/stream_mf4.py` for streaming reads, seeking, and channel
selection using `testdata/test.mf4`; use `--help` for other recordings.