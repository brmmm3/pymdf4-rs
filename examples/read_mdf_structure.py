from pathlib import Path

import pymdf4


def main():
    path = Path(__file__).resolve().parents[1] / "testdata" / "11-bit-obd2.MF4"
    groups = pymdf4.read_mdf_structure(str(path))

    for group_name, channel_names in groups.items():
        print(f"{group_name}:")
        for channel_name in channel_names:
            print(f"  {channel_name}")


if __name__ == "__main__":
    main()