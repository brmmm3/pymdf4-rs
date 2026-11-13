import argparse
from pathlib import Path

import pymdf4


def print_samples(recording, count):
    for _ in range(count):
        try:
            samples = recording.stream_read(timeout=5.0)
        except StopIteration:
            print("End of stream")
            return
        print(samples)


def main():
    parser = argparse.ArgumentParser(description="Stream selected MDF4 channels")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "testdata" / "test.mf4",
    )
    parser.add_argument("--group", type=int, default=646)
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["MisfDet_stcEmiMisfCntrCyl_[1]", "InjSys_cntDeltaPCylFast"],
    )
    parser.add_argument("--start", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    recording = pymdf4.MdfFile(args.path)
    channels = [(args.group, name) for name in args.channels]
    recording.stream_start(args.start, channels)
    try:
        print("Initial samples:")
        print_samples(recording, args.count)

        recording.stream_set_pos(args.start)
        recording.stream_set_channels(channels[1:] or channels)
        print("After seeking and changing channels:")
        print_samples(recording, args.count)
    finally:
        recording.stream_stop()


if __name__ == "__main__":
    main()