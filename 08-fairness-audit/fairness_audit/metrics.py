"""Group metrics, disparities, and stratified bootstrap."""

from __future__ import annotations

import numpy as np
from scipy import stats

from . import config


def _safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def auc(score: np.ndarray, y: np.ndarray) -> float:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(score)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def group_report(y: np.ndarray, score: np.ndarray, approve: np.ndarray, g: np.ndarray) -> dict:
    """Per group: size, base rate, selection rate, TPR, FPR, PPV, AUC, calibration gap."""
    out = {}
    for name, gval in (("A", 0), ("B", 1)):
        m = g == gval
        yy, ss, aa = y[m], score[m], approve[m]
        out[name] = {
            "n": int(m.sum()),
            "positives": int(yy.sum()),
            "base_rate": float(yy.mean()),
            "selection_rate": float(aa.mean()),
            "tpr": _safe_mean(aa[yy == 1]),
            "fpr": _safe_mean(aa[yy == 0]),
            "ppv": _safe_mean(yy[aa]),
            "auc": auc(ss, yy),
            "mean_score_minus_base_rate": float(ss.mean() - yy.mean()),
        }
    return out


def disparities(rep: dict) -> dict:
    a, b = rep["A"], rep["B"]
    hi, lo = (
        max(a["selection_rate"], b["selection_rate"]),
        min(a["selection_rate"], b["selection_rate"]),
    )
    return {
        "tpr_gap": b["tpr"] - a["tpr"],
        "fpr_gap": b["fpr"] - a["fpr"],
        "ppv_gap": b["ppv"] - a["ppv"] if np.isfinite(a["ppv"]) and np.isfinite(b["ppv"]) else None,
        "selection_ratio_min_over_max": lo / hi if hi > 0 else None,
    }


def stratified_bootstrap(
    stat_fn, g: np.ndarray, n_boot: int = config.N_BOOT, seed: int = config.SEED
) -> np.ndarray:
    """Resample rows within each group. stat_fn takes an index array, returns a float or array."""
    rng = np.random.default_rng(seed)
    idx_a, idx_b = np.flatnonzero(g == 0), np.flatnonzero(g == 1)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(idx_a, size=len(idx_a)), rng.choice(idx_b, size=len(idx_b))]
        )
        draws.append(stat_fn(idx))
    return np.asarray(draws)


def ci95(draws: np.ndarray) -> tuple[float, float]:
    if np.isnan(draws).all():
        return float("nan"), float("nan")
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return float(low), float(high)
