"""Simulated answer pairs, human annotators and judge calls with planted biases.

Each answer has a true quality q and a standardised length l. Humans prefer the higher q with
no slot or length bias. The judge sees q less sharply and is pushed by slot and by length. The
code knows the planted biases, so an audit can be checked against the answer.
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


@dataclass(frozen=True)
class JudgeParams:
    beta: float = config.J_BETA
    pos: float = config.POS_BIAS
    length: float = config.LEN_BIAS


@dataclass
class Pairs:
    """Answer pair (X, Y) per row. q is true quality, l is standardised length."""

    qx: np.ndarray
    qy: np.ndarray
    lx: np.ndarray
    ly: np.ndarray

    def __len__(self) -> int:
        return len(self.qx)

    @property
    def dq(self) -> np.ndarray:
        return self.qx - self.qy

    @property
    def dl(self) -> np.ndarray:
        return self.lx - self.ly

    @property
    def truth(self) -> np.ndarray:
        """True where X is really better."""
        return self.qx > self.qy


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def draw_answers(
    rng: np.random.Generator, n: int, mean: float, length_offset: float, rho: float
) -> tuple[np.ndarray, np.ndarray]:
    """Quality and length of n answers from one system. Length tracks quality with correlation rho."""
    eps = rng.standard_normal(n)
    length = length_offset + rho * eps + np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
    return mean + eps, length


def make_pairs(
    rng: np.random.Generator,
    n: int,
    x: tuple[float, float] = (0.0, 0.0),
    y: tuple[float, float] = (0.0, 0.0),
    rho: float = config.LENGTH_QUALITY_CORR,
) -> Pairs:
    """n pairs. x and y are (mean quality, length offset) of the two systems."""
    qx, lx = draw_answers(rng, n, x[0], x[1], rho)
    qy, ly = draw_answers(rng, n, y[0], y[1], rho)
    return Pairs(qx, qy, lx, ly)


def human_votes(rng: np.random.Generator, pairs: Pairs) -> np.ndarray:
    """Shape (n, annotators), True where the annotator prefers X. No slot or length bias."""
    p = sigmoid(config.HUMAN_BETA * pairs.dq)
    return rng.random((len(pairs), config.N_ANNOTATORS)) < p[:, None]


def consensus(votes: np.ndarray) -> np.ndarray:
    return votes.sum(axis=1) * 2 > votes.shape[1]


def judge_calls(
    rng: np.random.Generator, pairs: Pairs, params: JudgeParams
) -> tuple[np.ndarray, np.ndarray]:
    """Call the judge twice per pair, once with X first and once with Y first.

    Returns (v0, v1), each True where the judge says X is better. The two calls are independent
    draws, as with a sampled LLM at nonzero temperature.
    """
    z0 = params.beta * pairs.dq + params.pos + params.length * pairs.dl
    first0 = rng.random(len(pairs)) < sigmoid(z0)
    z1 = -params.beta * pairs.dq + params.pos - params.length * pairs.dl
    first1 = rng.random(len(pairs)) < sigmoid(z1)
    return first0, ~first1


def make_splits(seed: int = config.SEED) -> dict[str, Pairs]:
    rng = np.random.default_rng([seed, 9])
    return {"dev": make_pairs(rng, config.N_DEV), "test": make_pairs(rng, config.N_TEST)}


def validate(pairs: Pairs) -> None:
    arrays = (pairs.qx, pairs.qy, pairs.lx, pairs.ly)
    if len({len(a) for a in arrays}) != 1:
        raise DataValidationError("arrays differ in length")
    if any(not np.isfinite(a).all() for a in arrays):
        raise DataValidationError("non-finite values")


def fingerprint(seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    p = make_pairs(rng, 200)
    v = human_votes(rng, p)
    c0, c1 = judge_calls(rng, p, JudgeParams())
    blob = np.concatenate(
        [np.round(a, 6).ravel() for a in (p.qx, p.qy, p.lx, p.ly)]
        + [a.astype(float).ravel() for a in (v, c0, c1)]
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
