"""A simulated customer base with a known individual effect of an offer.

Each customer has features x, an unmeasured driver u of buying, a baseline buying chance and a
buying chance if offered. The code records both, so the true effect of the offer on every customer
is known. Real data shows each customer under one condition only.
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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Customers:
    x: np.ndarray  # (n, features)
    t: np.ndarray  # offered, 0/1
    y: np.ndarray  # bought, 0/1
    p0: np.ndarray  # chance of buying with no offer
    p1: np.ndarray  # chance of buying with the offer
    propensity: np.ndarray  # chance the customer was offered, given what was seen when deciding

    def __len__(self) -> int:
        return len(self.y)

    @property
    def tau(self) -> np.ndarray:
        """True effect of the offer on the chance of buying, per customer."""
        return self.p1 - self.p0

    def subset(self, idx) -> Customers:
        return Customers(
            self.x[idx], self.t[idx], self.y[idx], self.p0[idx], self.p1[idx], self.propensity[idx]
        )


def baseline_logit(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return (
        -2.0
        + 0.8 * x[:, 3]
        + 0.6 * x[:, 4]
        + 0.3 * x[:, 5]
        + config.HIDDEN_CONFOUNDER_ON_BASELINE * u
    )


def effect_logit(x: np.ndarray) -> np.ndarray:
    """How the offer shifts the log odds. Positive for some customers and negative for others."""
    return 0.4 + 0.9 * x[:, 0] + 0.7 * (x[:, 1] > 0) * x[:, 2] - 0.5 * x[:, 0] * x[:, 1]


def generate(rng: np.random.Generator, n: int, randomised: bool = True) -> Customers:
    """n customers. If randomised, offers are a coin flip. Otherwise offers followed past buying signals."""
    x = rng.standard_normal((n, config.N_FEATURES))
    u = rng.standard_normal(n)
    base = baseline_logit(x, u)
    p0 = sigmoid(base)
    p1 = sigmoid(base + effect_logit(x))
    if randomised:
        e = np.full(n, config.TREATMENT_SHARE)
    else:
        a, b, c = config.OBS_PROPENSITY
        e = sigmoid(a * x[:, 3] + b * x[:, 4] + c * u)
    t = (rng.random(n) < e).astype(int)
    y = (rng.random(n) < np.where(t == 1, p1, p0)).astype(int)
    return Customers(x, t, y, p0, p1, e)


def make_splits(seed: int = config.SEED) -> dict[str, Customers]:
    rng = np.random.default_rng([seed, 12])
    return {
        "train": generate(rng, config.N_TRAIN),
        "dev": generate(rng, config.N_DEV),
        "test": generate(rng, config.N_TEST),
    }


def validate(c: Customers) -> None:
    if set(np.unique(c.t)) - {0, 1} or set(np.unique(c.y)) - {0, 1}:
        raise DataValidationError("t and y must be 0 or 1")
    if c.t.min() == c.t.max():
        raise DataValidationError("need both treated and control customers")
    if not ((c.p0 >= 0) & (c.p0 <= 1) & (c.p1 >= 0) & (c.p1 <= 1)).all():
        raise DataValidationError("probabilities out of range")


def fingerprint(seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    c = generate(rng, 300)
    o = generate(rng, 300, randomised=False)
    blob = np.concatenate(
        [np.round(a, 6).ravel() for a in (c.x, c.t, c.y, c.p0, c.p1, o.t, o.y, o.propensity)]
    )
    return hashlib.sha256(blob.tobytes()).hexdigest()


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
