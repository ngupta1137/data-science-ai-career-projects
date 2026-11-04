"""Data access and validation. Third-party data is downloaded, never committed."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CACHE_DIR,
    CHECKSUM_FILE,
    OPENML_DATA_ID,
    PCA_FEATURES,
    RAW_CSV,
    TARGET,
)

REQUIRED = ["Time", *PCA_FEATURES, "Amount", TARGET]


class DataValidationError(ValueError):
    """Raised with every problem found, not just the first."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(raw_csv: Path = RAW_CSV, cache_dir: Path = CACHE_DIR) -> Path:
    """Fetch the dataset from OpenML through scikit-learn and write one CSV."""
    from sklearn.datasets import fetch_openml

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, data_home=str(cache_dir))
    frame = bunch.frame.copy()
    frame[TARGET] = pd.to_numeric(frame[TARGET]).astype(int)
    frame.to_csv(raw_csv, index=False)
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
                "A different pandas version can change CSV formatting. "
                "Check your versions against requirements.txt before trusting results."
            )
    return digest


def load(raw_csv: Path = RAW_CSV) -> pd.DataFrame:
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} not found. Run: python -m fraud_detection.download_data"
        )
    return pd.read_csv(raw_csv)


def validate_frame(
    df: pd.DataFrame,
    expected_rows: int | None = None,
    expected_fraud: int | None = None,
) -> None:
    """Schema, range, and label checks. Collects all problems."""
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

    values = df[REQUIRED].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).sum() > int(nan_counts.sum()):
        problems.append("non-numeric values in required columns")
    if np.isinf(values).any():
        problems.append("infinite values present")

    labels = set(pd.unique(df[TARGET].dropna()).tolist())
    if not labels <= {0, 1}:
        problems.append(f"{TARGET} must be 0 or 1, found {sorted(labels)}")
    else:
        fraud = int((df[TARGET] == 1).sum())
        if fraud == 0:
            problems.append("no fraud rows")
        if expected_fraud is not None and fraud != expected_fraud:
            problems.append(f"expected {expected_fraud} fraud rows, found {fraud}")

    if (df["Amount"] < 0).any():
        problems.append("Amount must not be negative")
    if (df["Time"] < 0).any():
        problems.append("Time must not be negative")

    if problems:
        raise DataValidationError("; ".join(problems))
