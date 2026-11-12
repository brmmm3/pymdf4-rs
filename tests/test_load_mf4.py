import math
import shutil
import struct
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from asammdf import MDF, Signal

import pymdf4


def signal(name, values, times, **kwargs):
    return Signal(np.asarray(values), np.asarray(times, dtype=np.float64), name=name, **kwargs)


class LoadMf4Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "test.mf4"
        self.limit = pymdf4.DENSE_LIMIT
        self.addCleanup(setattr, pymdf4, "DENSE_LIMIT", self.limit)

    def make_recording(self, compression=0):
        mdf = MDF(version="4.10")
        first_times = [2, 0, 1, 1]
        matrix = np.zeros(4, dtype=[("matrix", "<i8", (2,))])
        matrix["matrix"] = [[1, 2], [3, 4], [5, 6], [7, 8]]
        mdf.append([
            signal("shared", [20, 0, 10, 11], first_times),
            signal("converted", [2, 0, 1, 3], first_times, conversion={"a": 2, "b": 1}),
            signal("matrix", matrix, first_times),
        ], common_timebase=True)
        mdf.append([
            signal("shared", [100, 300], [1, 3]),
            signal("other", [5, 6], [1, 3]),
        ], common_timebase=True)
        mdf.save(self.path, overwrite=True, compression=compression)
        mdf.close()

    def test_dense_and_compressed(self):
        for compression in (0, 1, 2):
            with self.subTest(compression=compression):
                self.make_recording(compression)
                result = pymdf4.load_mf4(self.path)
                self.assertIsInstance(result, pymdf4.ReplayData)
                self.assertEqual(result.source, str(self.path))
                self.assertEqual(result.timeline, [0, 1, 2, 3])
                self.assertEqual(result.channels["shared"], [0, 100, 20, 300])
                self.assertEqual(result.channels["converted"][:3], [1, 7, 5])
                self.assertTrue(math.isnan(result.channels["converted"][3]))
                self.assertTrue(math.isnan(result.channels["other"][0]))
                self.assertEqual(result.channels["other"][1], 5)
                self.assertNotIn("matrix", result.channels)

    def test_indexed_structure(self):
        self.make_recording()
        recording = pymdf4.MdfFile(self.path)
        groups = recording.structure()
        self.assertEqual([group["index"] for group in groups], [0, 1])
        self.assertEqual(groups[0]["record_count"], 4)
        self.assertIn("shared", groups[0]["channels"])
        self.assertIn("shared", groups[1]["channels"])
        with self.assertRaises(RuntimeError):
            pymdf4.MdfFile(self.path.with_name("missing.mf4"))

    def test_unfinalized_interleaved_recordings(self):
        testdata = Path(__file__).resolve().parents[1] / "testdata"
        for filename, group_indices in (("11-bit-obd2.MF4", [0]), ("29-bit-obd2.MF4", [0, 23, 24])):
            path = testdata / filename
            with self.subTest(filename=filename), MDF(path) as reference:
                recording = pymdf4.MdfFile(path)
                groups = recording.structure()
                for group in group_indices:
                    expected = reference.get("Timestamp", group=group).samples
                    self.assertEqual(groups[group]["record_count"], len(expected))
                    self.assertEqual(recording.channel_count(group, "Timestamp"), len(expected))
                    times, values = recording.channel_slice(group, "Timestamp", 0, len(expected))
                    np.testing.assert_array_equal(times, np.arange(len(expected)))
                    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)
                    begin, end = 1, min(4, len(expected))
                    if begin <= end:
                        times, values = recording.channel_time_slice(group, "Timestamp", begin, end)
                        np.testing.assert_array_equal(times, np.arange(begin, min(end + 1, len(expected))))
                        np.testing.assert_allclose(values, expected[begin:end + 1], rtol=0, atol=1e-12)

    def test_unfinalized_rejects_unsupported_flags_and_truncation(self):
        source = Path(__file__).resolve().parents[1] / "testdata" / "11-bit-obd2.MF4"
        shutil.copyfile(source, self.path)
        with self.path.open("r+b") as file:
            file.seek(60)
            file.write((0x27).to_bytes(2, "little"))
        with self.assertRaisesRegex(RuntimeError, "unsupported finalization flags"):
            pymdf4.MdfFile(self.path)
        shutil.copyfile(source, self.path)
        with self.path.open("r+b") as file:
            file.seek(88)
            dg = int.from_bytes(file.read(8), "little")
            file.seek(dg + 40)
            data_block = int.from_bytes(file.read(8), "little")
            file.seek(data_block + 8)
            file.write((25).to_bytes(8, "little"))
        with self.assertRaisesRegex(RuntimeError, "ambiguous unfinalized DT block length"):
            pymdf4.MdfFile(self.path)
        shutil.copyfile(source, self.path)
        with self.path.open("r+b") as file:
            file.truncate(file.seek(0, 2) - 1)
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            pymdf4.MdfFile(self.path)

    def test_unfinalized_recovers_converted_channels_and_time_ranges(self):
        self.make_recording()
        with self.path.open("r+b") as file:
            data = bytearray(file.read())
            data[:8] = b"UnFinMF "
            struct.pack_into("<H", data, 60, 1)
            dg = struct.unpack_from("<Q", data, 88)[0]
            while dg:
                cg = struct.unpack_from("<Q", data, dg + 32)[0]
                while cg:
                    struct.pack_into("<Q", data, cg + 80, 0)
                    cg = struct.unpack_from("<Q", data, cg + 24)[0]
                dg = struct.unpack_from("<Q", data, dg + 24)[0]
            file.seek(0)
            file.write(data)
        recording = pymdf4.MdfFile(self.path)
        self.assertEqual([group["record_count"] for group in recording.structure()], [4, 2])
        self.assertEqual(recording.timeline(0), [2, 0, 1, 1])
        self.assertEqual(recording.channel_slice(0, "converted", 1, 3), ([0, 1], [1, 3]))
        self.assertEqual(recording.channel_time_slice(0, "shared", 1, 2), ([2, 1, 1], [20, 10, 11]))

    def test_indexed_timeline_and_slices(self):
        for compression in (0, 1, 2):
            with self.subTest(compression=compression):
                self.make_recording(compression)
                recording = pymdf4.MdfFile(self.path)
                self.assertEqual(recording.source, str(self.path))
                self.assertEqual(recording.timeline(0), [2, 0, 1, 1])
                self.assertEqual(recording.time_bounds(0), (0, 2))
                self.assertEqual(recording.channel_count(0, "shared"), 4)
                self.assertEqual(recording.channel_count(1, "shared"), 2)
                self.assertEqual(recording.channel_slice(0, "converted", 1, 3), ([0, 1], [1, 3]))
                self.assertEqual(recording.channel_slice(0, "shared", 3, 10), ([1], [11]))
                self.assertEqual(recording.channel_slice(0, "shared", 10, 11), ([], []))
                self.assertEqual(recording.channel_time_slice(0, "shared", 1, 2), ([2, 1, 1], [20, 10, 11]))
                self.assertEqual(recording.channel_time_slice(1, "shared", 1, 1), ([1], [100]))
                with self.assertRaises(IndexError):
                    recording.timeline(99)
                with self.assertRaises(KeyError):
                    recording.channel_count(0, "missing")
                with self.assertRaises(ValueError):
                    recording.channel_slice(0, "shared", 3, 2)
                with self.assertRaises(ValueError):
                    recording.channel_time_slice(0, "shared", math.nan, 2)
                with self.assertRaises(ValueError):
                    recording.channel_slice(0, "matrix", 0, 1)

    def test_indexed_fallback_timeline(self):
        mdf = MDF(version="4.10")
        mdf.append([signal("first", [4, 5, 6], [10, 20, 30])], common_timebase=True)
        mdf.groups[0].channels[0].name = "clock"
        mdf.save(self.path, overwrite=True)
        mdf.close()
        recording = pymdf4.MdfFile(self.path)
        self.assertEqual(recording.timeline(0), [0, 1, 2])
        self.assertEqual(recording.time_bounds(0), (0, 2))
        self.assertEqual(recording.channel_slice(0, "first", 1, 3), ([1, 2], [5, 6]))
        self.assertEqual(recording.channel_time_slice(0, "first", 1, 2), ([1, 2], [5, 6]))

    def test_filter_and_sparse(self):
        self.make_recording()
        skipped = ["previous"]
        result = pymdf4.load_mf4(self.path, lambda name: name == "other", skipped)
        self.assertEqual(result.timeline, [1, 3])
        self.assertEqual(result.channels, {"other": [5, 6]})
        self.assertEqual(skipped, ["previous", "shared", "converted", "matrix", "shared"])

        pymdf4.DENSE_LIMIT = 1
        result = pymdf4.load_mf4(self.path)
        self.assertIsInstance(result, pymdf4.SparseReplayData)
        np.testing.assert_array_equal(result.timeline, [0, 1, 2, 3])
        np.testing.assert_array_equal(result.series["shared"][0], [0, 1, 1, 1, 2, 3])
        np.testing.assert_array_equal(result.series["shared"][1], [0, 10, 11, 100, 20, 300])
        np.testing.assert_array_equal(result.series["converted"][1], [1, 3, 7, 5])

    def test_filter_all_and_errors(self):
        self.make_recording()
        result = pymdf4.load_mf4(self.path, lambda name: False)
        self.assertEqual(result.timeline, [])
        self.assertEqual(result.channels, {})
        with self.assertRaises(RuntimeError):
            pymdf4.load_mf4(self.path.with_name("missing.mf4"))
        with self.assertRaisesRegex(ValueError, "predicate"):
            pymdf4.load_mf4(self.path, lambda name: (_ for _ in ()).throw(ValueError("predicate")))

    def test_fallback_time_and_shared_sparse_base(self):
        mdf = MDF(version="4.10")
        mdf.append([
            signal("first", [4, 5], [10, 20]),
            signal("second", [6, 7], [10, 20]),
        ], common_timebase=True)
        mdf.groups[0].channels[0].name = "clock"
        mdf.save(self.path, overwrite=True)
        mdf.close()

        result = pymdf4.load_mf4(self.path)
        self.assertEqual(result.timeline, [0, 1])
        self.assertEqual(result.channels["clock"], [10, 20])
        self.assertEqual(result.channels["first"], [4, 5])
        pymdf4.DENSE_LIMIT = 1
        result = pymdf4.load_mf4(self.path)
        self.assertIs(result.series["first"][0], result.series["second"][0])

    def test_threshold_is_strict(self):
        self.make_recording()
        pymdf4.DENSE_LIMIT = 4
        self.assertIsInstance(pymdf4.load_mf4(self.path, lambda name: name == "shared"), pymdf4.ReplayData)
        pymdf4.DENSE_LIMIT = 3
        self.assertIsInstance(pymdf4.load_mf4(self.path, lambda name: name == "shared"), pymdf4.SparseReplayData)

    def test_time_range(self):
        self.make_recording(compression=1)
        result = pymdf4.load_mf4(self.path, start=1, end=2)
        self.assertEqual(result.timeline, [1, 2])
        self.assertEqual(result.channels["shared"], [100, 20])
        self.assertEqual(result.channels["converted"], [7, 5])
        self.assertEqual(pymdf4.load_mf4(self.path, end=0).timeline, [0])
        self.assertEqual(pymdf4.load_mf4(self.path, start=3).timeline, [3])
        empty = pymdf4.load_mf4(self.path, start=4)
        self.assertEqual(empty.timeline, [])
        self.assertEqual(empty.channels, {})
        self.assertEqual(pymdf4.load_mf4(self.path, start=1, end=1).timeline, [1])
        skipped = []
        filtered = pymdf4.load_mf4(self.path, lambda name: name == "other", skipped, start=1, end=2)
        self.assertEqual(filtered.channels, {"other": [5]})
        self.assertEqual(skipped, ["shared", "converted", "matrix", "shared"])
        for bounds in ({"start": 2, "end": 1}, {"start": math.nan}, {"end": math.nan}):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                pymdf4.load_mf4(self.path, **bounds)

        pymdf4.DENSE_LIMIT = 1
        result = pymdf4.load_mf4(self.path, start=1, end=2)
        self.assertIsInstance(result, pymdf4.SparseReplayData)
        np.testing.assert_array_equal(result.timeline, [1, 2])
        np.testing.assert_array_equal(result.series["shared"][0], [1, 1, 1, 2])
        np.testing.assert_array_equal(result.series["shared"][1], [10, 11, 100, 20])

    def test_time_range_with_fallback_time(self):
        mdf = MDF(version="4.10")
        mdf.append([signal("first", [4, 5, 6], [10, 20, 30])], common_timebase=True)
        mdf.groups[0].channels[0].name = "clock"
        mdf.save(self.path, overwrite=True)
        mdf.close()
        result = pymdf4.load_mf4(self.path, start=1, end=2)
        self.assertEqual(result.timeline, [1, 2])
        self.assertEqual(result.channels["first"], [5, 6])
        self.assertEqual(result.channels["clock"], [20, 30])

    def test_time_range_excludes_nan_timestamps(self):
        mdf = MDF(version="4.10")
        mdf.append([signal("first", [4, 5, 6], [math.nan, 1, 2])], common_timebase=True)
        mdf.save(self.path, overwrite=True)
        mdf.close()
        unfiltered = pymdf4.load_mf4(self.path)
        self.assertTrue(math.isnan(unfiltered.timeline[-1]))
        result = pymdf4.load_mf4(self.path, start=1)
        self.assertEqual(result.timeline, [1, 2])
        self.assertEqual(result.channels["first"], [5, 6])
        recording = pymdf4.MdfFile(self.path)
        self.assertEqual(recording.time_bounds(0), (1, 2))
        self.assertEqual(recording.channel_time_slice(0, "first", 1, 2), ([1, 2], [5, 6]))
        times, values = recording.channel_slice(0, "first", 0, 1)
        self.assertTrue(math.isnan(times[0]))
        self.assertEqual(values, [4])

    def test_load_releases_gil(self):
        self.make_recording()
        ready = threading.Event()
        start = threading.Event()
        completed = threading.Event()

        def worker():
            ready.set()
            start.wait()
            completed.set()

        thread = threading.Thread(target=worker)
        interval = sys.getswitchinterval()
        try:
            sys.setswitchinterval(60)
            thread.start()
            self.assertTrue(ready.wait(5))
            start.set()
            pymdf4.load_mf4(self.path)
            self.assertTrue(completed.is_set())
        finally:
            sys.setswitchinterval(interval)
            start.set()
            thread.join()

    def test_concurrent_loads(self):
        self.make_recording(compression=1)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(pymdf4.load_mf4, [self.path, self.path]))
        for result in results:
            self.assertEqual(result.timeline, [0, 1, 2, 3])
            self.assertEqual(result.channels["shared"], [0, 100, 20, 300])

    def test_fragmented_testdata_records(self):
        path = Path(__file__).resolve().parents[1] / "testdata" / "11-bit-obd2.MF4"
        name = "ASAM.M.SCALAR.SLONG.IDENTICAL"
        with MDF(path) as reference:
            signal = reference.get(name, group=2)
            result = pymdf4.load_mf4(path, lambda candidate: candidate == name)
            np.testing.assert_array_equal(result.timeline, signal.timestamps)
            np.testing.assert_array_equal(result.channels[name], signal.samples)
            recording = pymdf4.MdfFile(path)
            self.assertEqual(recording.channel_count(2, name), len(signal.samples))
            times, values = recording.channel_slice(2, name, 1, 4)
            np.testing.assert_array_equal(times, signal.timestamps[1:4])
            np.testing.assert_array_equal(values, signal.samples[1:4])
            start, end = signal.timestamps[1], signal.timestamps[3]
            times, values = recording.channel_time_slice(2, name, start, end)
            in_range = (signal.timestamps >= start) & (signal.timestamps <= end)
            np.testing.assert_array_equal(times, signal.timestamps[in_range])
            np.testing.assert_array_equal(values, signal.samples[in_range])

    def test_load_seu690_recording(self):
        path = Path(__file__).resolve().parents[1] / "testdata" / "11-bit-obd2.MF4"
        start, end = 0.0, 10.0
        result = pymdf4.load_mf4(path, start=start, end=end)
        self.assertEqual(result.source, str(path))
        self.assertGreater(len(result.timeline), 0)
        self.assertTrue(np.all(np.asarray(result.timeline) >= start))
        self.assertTrue(np.all(np.asarray(result.timeline) <= end))
        name = "APP_cntDebMax_C"
        with MDF(path) as reference:
            channel = reference.get(name, group=1)
            in_range = (channel.timestamps >= start) & (channel.timestamps <= end)
            expected_times = channel.timestamps[in_range]
            expected_values = channel.samples[in_range]
        if isinstance(result, pymdf4.SparseReplayData):
            self.assertTrue(result.series)
            times, values = result.series[name]
            np.testing.assert_array_equal(times, expected_times)
            np.testing.assert_allclose(values, expected_values)
        else:
            self.assertIsInstance(result, pymdf4.ReplayData)
            self.assertTrue(result.channels)
            indices = np.searchsorted(result.timeline, expected_times)
            np.testing.assert_array_equal(np.asarray(result.timeline)[indices], expected_times)
            np.testing.assert_allclose(np.asarray(result.channels[name])[indices], expected_values)


if __name__ == "__main__":
    unittest.main()