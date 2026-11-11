"""Simulated loan applicants where the old approval policy saw something the data did not.

Each applicant has five recorded features and one unrecorded trait that raises the chance of
default. The old policy approved on a risk index that used the unrecorded trait, so among
approved applicants the trait is lower than among all applicants with the same recorded
features. Outcomes are seen only for approved loans, plus a small sample approved at random.
The test set gives every applicant's outcome, which no lender has.
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
class Applicants:
    x: np.ndarray  # (n, 5) recorded features
    pd: np.ndarray  # true default chance
    y: np.ndarray  # 1 if the applicant would default if approved
    old_approve: np.ndarray  # 1 if the old policy approved
    random_approve: np.ndarray  # 1 if approved by the random sample
    approve_prob: np.ndarray  # old policy's chance of approving, given everything it saw

    def __len__(self) -> int:
        return len(self.y)

    @property
    def approved(self) -> np.ndarray:
        return (self.old_approve == 1) | (self.random_approve == 1)


def true_logit(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return -2.0 + 0.9 * x[:, 0] + 0.6 * x[:, 1] - 0.5 * x[:, 2] + 0.4 * x[:, 0] * x[:, 1] + 0.5 * (x[:, 3] > 1) + config.HIDDEN_ON_DEFAULT * u


def generate(rng: np.random.Generator, n: int, random_share: float = config.RANDOM_APPROVE_SHARE) -> Applicants:
    x = rng.standard_normal((n, config.N_FEATURES))
    u = rng.standard_normal(n)
    pd = sigmoid(true_logit(x, u))
    y = (rng.random(n) < pd).astype(int)
    risk = 0.9 * x[:, 0] + 0.6 * x[:, 1] - 0.5 * x[:, 2] + config.OLD_POLICY_HIDDEN * u - 2.0
    p_app = sigmoid(config.OLD_POLICY_SHARPNESS * (config.OLD_POLICY_CUTOFF - risk))
    old = (rng.random(n) < p_app).astype(int)
    rnd = (rng.random(n) < random_share).astype(int)
    return Applicants(x, pd, y, old, rnd, p_app)


def validate(a: Applicants) -> None:
    problems = []
    if a.x.shape != (len(a.y), config.N_FEATURES):
        problems.append("feature shape")
    if set(np.unique(a.y)) - {0, 1}:
        problems.append("y must be 0 or 1")
    if not ((a.pd > 0) & (a.pd < 1)).all():
        problems.append("pd out of range")
    if a.approved.sum() == 0 or a.random_approve.sum() == 0:
        problems.append("need approved and randomly approved applicants")
    if problems:
        raise DataValidationError("; ".join(problems))


def fingerprint(seed: int = 0) -> str:
    a = generate(np.random.default_rng(seed), 300)
    blob = np.concatenate([np.round(a.x, 6).ravel(), np.round(a.pd, 6), a.y, a.old_approve, a.random_approve])
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
            raise DataValidationError(f"generator fingerprint mismatch: expected {expected}, got {digest}")
    return digest
