"""Simulated experiments with a known true effect.

There is no external dataset. Every experiment is drawn from a model where the truth is
set by the code, so the project can measure how often each analysis is wrong. This is the
point of the project, not a shortcut: a real experiment never reveals the true effect.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .config import CHECKSUM_FILE


class DataValidationError(ValueError):
    """Raised when committed fingerprints do not match the generator."""


def _standardized(rng: np.random.Generator, shape: tuple, dist: str) -> np.ndarray:
    """Mean 0, variance 1. 'lognormal' is heavy tailed and skewed (sigma = 1)."""
    z = rng.standard_normal(shape)
    if dist == "normal":
        return z
    if dist == "lognormal":
        mean = np.exp(0.5)
        sd = np.sqrt((np.e - 1.0) * np.e)
        return (np.exp(z) - mean) / sd
    raise ValueError(f"unknown distribution: {dist}")


def draw(
    rng: np.random.Generator,
    reps: int,
    n: int,
    dist: str = "normal",
    rho: float = 0.6,
    effect: float = 0.0,
    contamination: float = 0.0,
) -> dict[str, np.ndarray]:
    """Draw `reps` independent experiments with `n` users per arm.

    Outcome y = rho * x + sqrt(1 - rho^2) * e + effect * treated, with x (the pre-period
    metric) and e both standardized, so the control outcome has variance 1 and the corr(x, y)
    is rho. `contamination` adds a shift to x in the treated arm only. A real pre-period
    metric cannot do that. A metric measured after launch can.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    out = {}
    for arm, shift_y, shift_x in (("t", effect, contamination), ("c", 0.0, 0.0)):
        x = _standardized(rng, (reps, n), dist)
        e = _standardized(rng, (reps, n), dist)
        y = rho * x + np.sqrt(1.0 - rho**2) * e + shift_y
        out[f"x_{arm}"] = x + shift_x
        out[f"y_{arm}"] = y
    return out


def draw_metrics(
    rng: np.random.Generator, reps: int, n: int, effects: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """`len(effects)` independent normal metrics per user. Shape (reps, metrics, n)."""
    m = len(effects)
    shifts = np.asarray(effects).reshape(1, m, 1)
    treated = rng.standard_normal((reps, m, n)) + shifts
    control = rng.standard_normal((reps, m, n))
    return treated, control


def fingerprint(seed: int = 0) -> str:
    """SHA-256 of a small fixed draw, rounded to 6 decimals. Detects generator drift."""
    rng = np.random.default_rng(seed)
    parts = []
    for dist in ("normal", "lognormal"):
        d = draw(rng, reps=3, n=20, dist=dist, rho=0.6, effect=0.1)
        for key in sorted(d):
            parts.append(np.round(d[key], 6).tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def write_fingerprint(path: Path = CHECKSUM_FILE) -> str:
    digest = fingerprint()
    path.write_text(f"{digest}  generator_probe(seed=0)\n", encoding="utf-8")
    return digest


def verify_fingerprint(path: Path = CHECKSUM_FILE) -> str:
    """Return the digest. Raise if a committed fingerprint exists and does not match."""
    digest = fingerprint()
    if path.exists():
        expected = path.read_text(encoding="utf-8").split()[0]
        if expected != digest:
            raise DataValidationError(
                f"generator fingerprint mismatch: expected {expected}, got {digest}"
            )
    return digest
