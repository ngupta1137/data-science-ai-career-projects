"""Command line: python -m house_price.download_data [--write-checksum]"""

from __future__ import annotations

import argparse

from . import data


def main() -> None:
    parser = argparse.ArgumentParser(description="Download California Housing and verify it.")
    parser.add_argument(
        "--write-checksum",
        action="store_true",
        help="record the CSV checksum in data/CHECKSUMS.txt (do this once, then commit the file)",
    )
    args = parser.parse_args()

    path = data.download()
    frame = data.load(path)
    data.validate_frame(frame, expected_rows=None)
    print(f"wrote {path} with {len(frame)} rows")

    if args.write_checksum:
        digest = data.write_checksum(path)
        print(f"recorded sha256 {digest}")
    else:
        digest = data.verify_checksum(path)
        print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
