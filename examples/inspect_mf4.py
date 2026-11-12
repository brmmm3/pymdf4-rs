import argparse
from pathlib import Path

import pymdf4


def main():
    parser = argparse.ArgumentParser(description="Inspect an MDF4 recording")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "testdata" / "11-bit-obd2.MF4",
    )
    recording = pymdf4.MdfFile(parser.parse_args().path)

    groups = recording.structure()
    for group in groups:
        print(f"Group {group['index']}: {group['name']}")
        for channel in group["channels"]:
            print(f"  {channel}")

    selected = None
    for group in groups:
        if not group["record_count"]:
            continue
        for channel in group["channels"]:
            try:
                recording.channel_slice(group["index"], channel, 0, 1)
            except ValueError as error:
                if not str(error).startswith("Channel is not scalar numeric:"):
                    raise
            else:
                selected = (group["index"], channel)
                break
        if selected is not None:
            break
    if selected is None:
        print("No scalar numeric channels with samples.")
        return

    group, channel = selected
    timeline = recording.timeline(group)
    bounds = recording.time_bounds(group)
    count = recording.channel_count(group, channel)
    print(f"Timeline: {len(timeline)} samples")
    print(f"Time begin and end: {bounds}")
    print(f"{channel}: {count} values")
    print(f"First values by index: {recording.channel_slice(group, channel, 0, min(5, count))}")
    if bounds is not None:
        start, end = bounds
        print(f"Values in first timeline unit: {recording.channel_time_slice(group, channel, start, min(start + 1, end))}")


if __name__ == "__main__":
    main()