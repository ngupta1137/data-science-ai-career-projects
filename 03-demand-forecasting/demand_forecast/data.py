"""Data access and validation. Third-party data is downloaded, never committed."""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CHECKSUM_FILE, RAW_CSV, SOURCE_MEMBER, SOURCE_URL

REQUIRED = [
    "dteday",
    "season",
    "mnth",
    "holiday",
    "weekday",
    "workingday",
    "casual",
    "registered",
    "cnt",
]


class DataValidationError(ValueError):
    """Raised with every problem found, not just the first."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(raw_csv: Path = RAW_CSV, url: str = SOURCE_URL) -> Path:
    """Fetch the zip, extract the daily file, write one CSV."""
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        raw_csv.write_bytes(archive.read(SOURCE_MEMBER))
    return raw_csv


def write_checksum(raw_csv: Path = RAW_CSV, checksum_file: Path = CHECKSUM_FILE) -> str:
    digest = sha256_of(raw_csv)
    checksum_file.write_text(f"{digest}  {raw_csv.name}\n", encoding="utf-8")
    return digest


def verify_checksum(raw_csv: Path = RAW_CSV, checksum_file: Path = CHECKSUM_FILE) -> str:
    """Return the digest. Raise if a committed checksum exists and does not match."""
    digest = sha256_of(raw_csv)
    if checksum_file.exists():
        expected = checksum_file.read_text(encoding="utf-8").split()[0]
        if expected != digest:
            raise DataValidationError(
                f"checksum mismatch for {raw_csv.name}: expected {expected}, got {digest}"
            )
    return digest


def load(raw_csv: Path = RAW_CSV) -> pd.DataFrame:
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} not found. Run: python -m demand_forecast.download_data"
        )
    return pd.read_csv(raw_csv)


def validate_frame(df: pd.DataFrame, expected_rows: int | None = None) -> None:
    """Schema, range, and continuity checks. Collects all problems.

    Continuity matters: lag features assume one row per consecutive day.
    """
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")

    problems: list[str] = []
    if expected_rows is not None and len(df) != expected_rows:
        problems.append(f"expected {expected_rows} rows, found {len(df)}")

    nan_counts = df[REQUIRED].isna().sum()
    for col, count in nan_counts.items():
        if count:
            problems.append(f"{col}: {int(count)} missing values")

    if not nan_counts["dteday"]:
        days = pd.to_datetime(df["dteday"], errors="coerce")
        if days.isna().any():
            problems.append("dteday has unparseable dates")
        else:
            steps = days.diff().dropna()
            if not (steps == pd.Timedelta(days=1)).all():
                problems.append("dates are not one row per consecutive day in order")

    numeric = df[["casual", "registered", "cnt"]].apply(pd.to_numeric, errors="coerce")
    if (numeric < 0).any().any():
        problems.append("negative counts")
    if not np.array_equal(
        (numeric["casual"] + numeric["registered"]).to_numpy(), numeric["cnt"].to_numpy()
    ):
        problems.append("casual + registered != cnt (the leakage trap needs this identity)")
    if not df["weekday"].between(0, 6).all():
        problems.append("weekday outside 0..6")
    if not df["mnth"].between(1, 12).all():
        problems.append("mnth outside 1..12")

    if problems:
        raise DataValidationError("; ".join(problems))
