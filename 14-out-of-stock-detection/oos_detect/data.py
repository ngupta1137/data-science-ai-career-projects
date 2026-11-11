"""A simulated chain where some shelves go empty while the system shows stock.

Each shelf has a true daily sales rate. On any day an in-stock shelf can go empty and stay empty
for a random number of days. While empty it sells nothing. All random draws are made up front,
so every policy faces the same demand and the same empty-shelf episodes (common random numbers),
and differences between policies come from the policies alone.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .config import CHECKSUM_FILE


class DataValidationError(ValueError):
    """Raised on a schema problem or a fingerprint mismatch."""


@dataclass
class World:
    rate: np.ndarray  # (shelves,) true units per day
    store: np.ndarray  # (shelves,) store index
    demand: np.ndarray  # (shelves, days) units customers want
    start_u: np.ndarray  # (shelves, days) uniform draws that start an empty episode
    duration: np.ndarray  # (shelves, days) length of an episode if it starts that day

    @property
    def days(self) -> int:
        return self.demand.shape[1]

    def __len__(self) -> int:
        return len(self.rate)


def generate(rng: np.random.Generator, n_stores: int, n_items: int, days: int) -> World:
    n = n_stores * n_items
    rate = np.exp(rng.normal(config.RATE_LOG_MEAN, config.RATE_LOG_SD, n))
    store = np.repeat(np.arange(n_stores), n_items)
    demand = rng.poisson(rate[:, None], (n, days))
    start_u = rng.random((n, days))
    duration = rng.geometric(1 / config.OOS_MEAN_DAYS, (n, days))
    return World(rate, store, demand, start_u, duration)


def validate(w: World) -> None:
    problems = []
    if not (w.demand.shape == w.start_u.shape == w.duration.shape):
        problems.append("draw shapes differ")
    if (w.demand < 0).any() or (w.rate <= 0).any():
        problems.append("negative demand or non-positive rate")
    if (w.duration < 1).any():
        problems.append("episodes must last at least a day")
    if w.days < config.HISTORY_DAYS + 1:
        problems.append("need history and evaluation days")
    if problems:
        raise DataValidationError("; ".join(problems))


def fingerprint(seed: int = 0) -> str:
    w = generate(np.random.default_rng(seed), 3, 10, 20)
    blob = np.concatenate(
        [np.round(w.rate, 6), w.demand.ravel(), np.round(w.start_u, 6).ravel(), w.duration.ravel()]
    )
    return hashlib.sha256(blob.astype(float).tobytes()).hexdigest()


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
