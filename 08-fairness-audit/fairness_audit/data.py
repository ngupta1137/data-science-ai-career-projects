"""Simulated applicants with known mechanisms.

Group B is the smaller group. Its latent score has a lower mean, and one of its measurements
is noisier. A third feature is a proxy for group membership. Because the code sets all of
this, the audit can say which part of a gap comes from which mechanism. Real data never can.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .config import CHECKSUM_FILE


class DataValidationError(ValueError):
    """Raised on a schema problem or a fingerprint mismatch."""


def generate(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Draw `n` applicants. Column g is 0 for group A and 1 for group B."""
    g = (rng.random(n) < config.MINORITY_SHARE).astype(int)
    mean = np.where(g == 1, config.LATENT_MEAN["B"], config.LATENT_MEAN["A"])
    z = mean + rng.standard_normal(n)
    sd2 = np.where(g == 1, config.NOISE_SD["B"], config.NOISE_SD["A"])
    x1 = z + rng.standard_normal(n)
    x2 = z + sd2 * rng.standard_normal(n)
    x3 = config.PROXY_STRENGTH * g + rng.standard_normal(n)
    x4 = rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-config.SIGNAL * z))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"g": g, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})


def make_splits(seed: int = config.SEED) -> dict[str, pd.DataFrame]:
    """Independent fit, dev, and test sets. Dev picks thresholds and the headline policy."""
    rng = np.random.default_rng([seed, 8])
    return {
        "fit": generate(rng, config.N_FIT),
        "dev": generate(rng, config.N_DEV),
        "test": generate(rng, config.N_TEST),
    }


def validate(df: pd.DataFrame) -> None:
    problems = []
    for col in ("g", "x1", "x2", "x3", "x4", "y"):
        if col not in df.columns:
            problems.append(f"missing column {col}")
    if problems:
        raise DataValidationError("; ".join(problems))
    if df.isna().any().any():
        problems.append("missing values")
    if not set(df["g"].unique()) <= {0, 1} or df["g"].nunique() != 2:
        problems.append("g must contain both groups 0 and 1")
    if not set(df["y"].unique()) <= {0, 1}:
        problems.append("y must be 0 or 1")
    for gval in (0, 1):
        sub = df[df["g"] == gval]["y"]
        if sub.nunique() != 2:
            problems.append(f"group {gval} needs both outcomes present")
    if problems:
        raise DataValidationError("; ".join(problems))


def fingerprint(seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    df = generate(rng, 200)
    return hashlib.sha256(np.round(df.to_numpy(dtype=float), 6).tobytes()).hexdigest()


def write_fingerprint(path: Path = CHECKSUM_FILE) -> str:
    digest = fingerprint()
    path.write_text(f"{digest}  generator_probe(seed=0)\n", encoding="utf-8")
    return digest


def verify_fingerprint(path: Path = CHECKSUM_FILE) -> str:
    digest = fingerprint()
    if path.exists():
        expected = path.read_text(encoding="utf-8").split()[0]
        if expected != digest:
            raise DataValidationError(
                f"generator fingerprint mismatch: expected {expected}, got {digest}"
            )
    return digest
