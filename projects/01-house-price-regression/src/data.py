"""Data access and validation. Third-party data is downloaded, never committed."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CACHE_DIR,
    CHECKSUM_FILE,
    FEATURES_RAW,
    RAW_CSV,
    TARGET,
    TARGET_CAP,
)


class DataValidationError(ValueError):
    """Raised with every problem found, not just the first."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(raw_csv: Path = RAW_CSV, cache_dir: Path = CACHE_DIR) -> Path:
    """Fetch California Housing through scikit-learn and write one CSV.

    scikit-learn verifies the downloaded archive against its own checksum.
    This function additionally records a checksum of the CSV (see write_checksum).
    """
    from sklearn.datasets import fetch_california_housing

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    bunch = fetch_california_housing(data_home=str(cache_dir), as_frame=True)
    bunch.frame.to_csv(raw_csv, index=False)
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
                f"checksum mismatch for {raw_csv.name}: expected {expected}, got {digest}. "
                "A different pandas or scikit-learn version can change CSV formatting. "
                "Check your versions against requirements.txt before trusting results."
            )
    return digest


def load(raw_csv: Path = RAW_CSV) -> pd.DataFrame:
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} not found. Run: python -m src.download_data"
        )
    return pd.read_csv(raw_csv)


def validate_frame(df: pd.DataFrame, expected_rows: int | None = None) -> None:
    """Schema, range, and missing-value checks. Collects all problems."""
    problems: list[str] = []
    required = FEATURES_RAW + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")

    if expected_rows is not None and len(df) != expected_rows:
        problems.append(f"expected {expected_rows} rows, found {len(df)}")

    nan_counts = df[required].isna().sum()
    for col, count in nan_counts.items():
        if count:
            problems.append(f"{col}: {int(count)} missing values")

    numeric = df[required].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().sum().sum() > nan_counts.sum():
        problems.append("non-numeric values in required columns")

    values = numeric.to_numpy(dtype=float)
    if np.isinf(values).any():
        problems.append("infinite values present")

    if (df["MedInc"] <= 0).any():
        problems.append("MedInc must be positive")
    if (df["AveRooms"] <= 0).any():
        problems.append("AveRooms must be positive")
    if (df["AveOccup"] <= 0).any():
        problems.append("AveOccup must be positive")
    if ((df["Latitude"] < 32) | (df["Latitude"] > 43)).any():
        problems.append("Latitude outside California range")
    if ((df["Longitude"] < -125) | (df["Longitude"] > -114)).any():
        problems.append("Longitude outside California range")
    if (df[TARGET] <= 0).any():
        problems.append(f"{TARGET} must be positive")
    if (df[TARGET] > TARGET_CAP + 1e-3).any():
        problems.append(f"{TARGET} exceeds the documented cap of {TARGET_CAP}")

    if problems:
        raise DataValidationError("; ".join(problems))
