from itertools import islice
from pathlib import Path

import pymdf4


def main():
    path = Path(__file__).resolve().parents[1] / "testdata" / "29-bit-obd2.MF4"
    replay = pymdf4.load_mf4(path)

    print(f"Source: {replay.source}")
    print(f"Timeline: {len(replay.timeline)} timestamps")
    if isinstance(replay, pymdf4.SparseReplayData):
        print(f"Channels: {len(replay.series)} (sparse)")
        for name, (times, values) in islice(replay.series.items(), 5):
            print(f"  {name}: times={times[:5]}, values={values[:5]}")
    else:
        print(f"Channels: {len(replay.channels)} (dense)")
        for name, values in islice(replay.channels.items(), 5):
            print(f"  {name}: {values[:5]}")


if __name__ == "__main__":
    main()