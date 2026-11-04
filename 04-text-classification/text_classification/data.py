"""Data access and validation. Third-party data is downloaded, never committed.

The CSV holds two versions of each document: `text` with headers, footers, and quoted
replies removed, and `text_raw` with them kept. The raw version exists only for the
leakage trap.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .config import CACHE_DIR, CHECKSUM_FILE, METADATA_REMOVED, RAW_CSV

REQUIRED = ["text", "text_raw", "label", "split"]


class DataValidationError(ValueError):
    """Raised with every problem found, not just the first."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(raw_csv: Path = RAW_CSV, cache_dir: Path = CACHE_DIR) -> Path:
    """Fetch 20 Newsgroups twice (cleaned and raw) and write one CSV."""
    from sklearn.datasets import fetch_20newsgroups

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for split in ("train", "test"):
        kwargs = {"subset": split, "shuffle": False, "data_home": str(cache_dir)}
        clean = fetch_20newsgroups(remove=METADATA_REMOVED, **kwargs)
        raw = fetch_20newsgroups(remove=(), **kwargs)
        if len(clean.data) != len(raw.data) or list(clean.target) != list(raw.target):
            raise DataValidationError("cleaned and raw versions are not aligned")
        names = [clean.target_names[i] for i in clean.target]
        parts.append(
            pd.DataFrame({"text": clean.data, "text_raw": raw.data, "label": names, "split": split})
        )
    pd.concat(parts, ignore_index=True).to_csv(raw_csv, index=False)
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
            f"{raw_csv} not found. Run: python -m text_classification.download_data"
        )
    frame = pd.read_csv(raw_csv, keep_default_na=False)
    return frame


def validate_frame(
    df: pd.DataFrame,
    expected_rows: int | None = None,
    expected_classes: int | None = None,
) -> None:
    """Schema and label checks. Empty documents are allowed: removing metadata can empty a post."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")

    problems: list[str] = []
    if expected_rows is not None and len(df) != expected_rows:
        problems.append(f"expected {expected_rows} rows, found {len(df)}")
    if df["label"].isna().any() or (df["label"] == "").any():
        problems.append("label has missing values")
    if df["text"].isna().any() or df["text_raw"].isna().any():
        problems.append("text columns have missing values (empty text must be an empty string)")
    splits = set(df["split"].unique().tolist())
    if splits != {"train", "test"}:
        problems.append(f"split must be exactly train and test, found {sorted(splits)}")
    else:
        n_classes = df["label"].nunique()
        if expected_classes is not None and n_classes != expected_classes:
            problems.append(f"expected {expected_classes} classes, found {n_classes}")
        for split in ("train", "test"):
            seen = set(df.loc[df["split"] == split, "label"].unique().tolist())
            if seen != set(df["label"].unique().tolist()):
                problems.append(f"{split} does not contain every class")

    if problems:
        raise DataValidationError("; ".join(problems))
