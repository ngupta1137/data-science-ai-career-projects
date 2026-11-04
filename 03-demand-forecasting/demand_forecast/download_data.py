"""Command line: python -m demand_forecast.download_data [--write-checksum]"""

from __future__ import annotations

import argparse

from . import config, data


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the bike sharing data and verify it.")
    parser.add_argument(
        "--write-checksum",
        action="store_true",
        help="record the CSV checksum in data/CHECKSUMS.txt (do this once, then commit the file)",
    )
    args = parser.parse_args()

    path = data.download()
    frame = data.load(path)
    data.validate_frame(frame, expected_rows=config.EXPECTED_ROWS)
    print(f"wrote {path} with {len(frame)} rows")
    if args.write_checksum:
        print(f"recorded sha256 {data.write_checksum(path)}")
    else:
        print(f"sha256 {data.verify_checksum(path)}")


if __name__ == "__main__":
    main()
