"""Simulated population, reference model, and drifting streams.

Everything is generated with a known truth: the code decides when and how the world drifts,
so a monitor can be scored on whether it noticed. A real system never tells you that.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import config
from .config import CHECKSUM_FILE

N_BINS = 10


class DataValidationError(ValueError):
    """Raised when committed fingerprints do not match the generator."""


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def draw_stream(
    rng: np.random.Generator,
    scenario: str,
    n_weeks: int = config.N_WEEKS,
    size: int = config.BATCH_SIZE,
    onset: int = config.ONSET_WEEK,
    magnitude=None,
) -> dict[str, np.ndarray]:
    """One batch of `size` rows per week. Drift starts at week `onset` (0 indexed)."""
    spec = config.SCENARIOS[scenario]
    kind = spec["kind"]
    mag = spec.get("magnitude") if magnitude is None else magnitude
    weeks = np.arange(n_weeks)
    ramp = np.clip((weeks - onset + 1) / spec.get("ramp_weeks", 1), 0.0, 1.0)
    ramp = np.where(weeks >= onset, ramp, 0.0)

    x = rng.standard_normal((n_weeks, size, config.N_NUM))
    probs = np.tile(np.asarray(config.BASE_CAT), (n_weeks, 1))
    coef = np.tile(np.asarray(config.COEF), (n_weeks, 1))
    if kind == "mean_shift":
        x[:, :, 0] += (mag * ramp)[:, None]
    elif kind == "variance_shift":
        x[:, :, 1] *= (1.0 + (mag - 1.0) * ramp)[:, None]
    elif kind == "category_shift":
        probs = (1 - ramp)[:, None] * probs + ramp[:, None] * np.asarray(mag)
    elif kind == "concept":
        coef[:, 0] = config.COEF[0] + (mag - config.COEF[0]) * ramp
    elif kind != "none":
        raise ValueError(f"unknown scenario kind: {kind}")

    u = rng.random((n_weeks, size))
    cum = np.cumsum(probs, axis=1)[:, :2]
    cat = (u[..., None] > cum[:, None, :]).sum(-1)
    logit = (x * coef[:, None, :]).sum(-1) + np.asarray(config.CAT_EFFECT)[cat]
    y = (rng.random((n_weeks, size)) < _sigmoid(logit)).astype(np.int8)
    return {"x": x, "cat": cat, "y": y}


def design_matrix(x: np.ndarray, cat: np.ndarray) -> np.ndarray:
    """Numeric features plus one-hot indicators for category levels 1 and 2."""
    return np.concatenate([x, (cat == 1)[..., None], (cat == 2)[..., None]], axis=-1).astype(float)


@dataclass
class Reference:
    coef: np.ndarray
    intercept: float
    edges: np.ndarray  # (N_NUM, N_BINS - 1) decile edges
    props: np.ndarray  # (N_NUM, N_BINS) reference bin shares
    sorted_ref: np.ndarray  # (N_NUM, REF_SIZE)
    cat_props: np.ndarray
    score_edges: np.ndarray
    score_props: np.ndarray
    ref_auc: float
    n_ref: int

    def score(self, x: np.ndarray, cat: np.ndarray) -> np.ndarray:
        return design_matrix(x, cat) @ self.coef + self.intercept


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    ranks = rankdata(score)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def make_reference(seed: int = config.SEED) -> Reference:
    """Train the monitored model on reference data and record the reference distributions."""
    rng = np.random.default_rng([seed, 1])
    train = draw_stream(rng, "null", n_weeks=1, size=config.REF_SIZE, onset=0)
    val = draw_stream(rng, "null", n_weeks=1, size=config.REF_VAL_SIZE, onset=0)
    x, cat, y = train["x"][0], train["cat"][0], train["y"][0]
    model = LogisticRegression(max_iter=1000).fit(design_matrix(x, cat), y)
    coef, intercept = model.coef_[0], float(model.intercept_[0])

    q = np.linspace(0, 1, N_BINS + 1)[1:-1]
    edges = np.quantile(x, q, axis=0).T
    props = np.stack(
        [
            np.bincount(np.searchsorted(edges[j], x[:, j]), minlength=N_BINS)
            for j in range(config.N_NUM)
        ]
    )
    props = props / len(x)
    cat_props = np.bincount(cat, minlength=3) / len(cat)
    score = design_matrix(x, cat) @ coef + intercept
    score_edges = np.quantile(score, q)
    score_props = np.bincount(np.searchsorted(score_edges, score), minlength=N_BINS) / len(score)
    val_score = design_matrix(val["x"][0], val["cat"][0]) @ coef + intercept
    return Reference(
        coef=coef,
        intercept=intercept,
        edges=edges,
        props=props,
        sorted_ref=np.sort(x, axis=0).T,
        cat_props=cat_props,
        score_edges=score_edges,
        score_props=score_props,
        ref_auc=_auc(val_score, val["y"][0]),
        n_ref=len(x),
    )


def fingerprint(seed: int = 0) -> str:
    """SHA-256 of small fixed draws from every scenario, rounded to 6 decimals."""
    rng = np.random.default_rng(seed)
    parts = []
    for name in sorted(config.SCENARIOS):
        s = draw_stream(rng, name, n_weeks=6, size=20, onset=2)
        parts.append(np.round(s["x"], 6).tobytes())
        parts.append(s["cat"].astype(np.int8).tobytes())
        parts.append(s["y"].tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


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
