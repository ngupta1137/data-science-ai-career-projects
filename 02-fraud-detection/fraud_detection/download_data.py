"""Command line: python -m fraud_detection.download_data [--write-checksum]"""

from __future__ import annotations

import argparse

from . import config, data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the credit card fraud data and verify it."
    )
    parser.add_argument(
        "--write-checksum",
        action="store_true",
        help="record the CSV checksum in data/CHECKSUMS.txt (do this once, then commit the file)",
    )
    args = parser.parse_args()

    path = data.download()
    frame = data.load(path)
    data.validate_frame(frame, config.EXPECTED_ROWS, config.EXPECTED_FRAUD)
    print(f"wrote {path} with {len(frame)} rows, {int(frame[config.TARGET].sum())} fraud")

    if args.write_checksum:
        print(f"recorded sha256 {data.write_checksum(path)}")
    else:
        print(f"sha256 {data.verify_checksum(path)}")


if __name__ == "__main__":
    main()
