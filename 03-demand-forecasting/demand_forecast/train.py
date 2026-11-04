"""Backtest, evaluate, and write every result file.

    python -m demand_forecast.train            # full run on the downloaded data
    python -m demand_forecast.train --fast     # one config per model, smoke tests only

Design:
  - Rolling-origin backtest. At each origin the model sees only earlier days and forecasts
    the next HORIZON days.
  - The first half of the origins tunes hyperparameters and picks the headline model and the
    best baseline. The second half is the test and is scored once.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold, cross_val_predict

from . import config, data
from .features import build_features, first_valid_row
from .metrics import (
    bias,
    bootstrap_blocks_ci,
    mae,
    mase,
    rmse,
    rolling_origins,
    seasonal_naive_scale,
    segment_table,
    smape,
)
from .models import BASELINES, TUNED, baseline_forecast, make_model, param_grid

H = config.HORIZON

CAVEATS = [
    "The data is two years of one city. It cannot show how a model behaves over more years or in other places.",
    "Weather is excluded because it is not known 7 days ahead. A real system would use weather forecasts, which this project does not.",
    "Forecast intervals are not produced. Point forecasts only.",
    "Only about half a year of test blocks is scored, so intervals are wide.",
    "The same-day and short-lag variants are deliberately wrong. Their scores are shown to measure the damage, not as results.",
]


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _backtest_model(X, y, origins, start_row, name, params, seed) -> np.ndarray:
    preds = np.empty((len(origins), H))
    for i, origin in enumerate(origins):
        rows = np.arange(start_row, origin)
        model = make_model(name, seed, params).fit(X.iloc[rows], y[rows])
        preds[i] = model.predict(X.iloc[origin : origin + H])
    return preds


def _backtest_baseline(X, y, origins, name) -> np.ndarray:
    return np.vstack([baseline_forecast(name, y[:o], X.iloc[o : o + H]) for o in origins])


def _actuals(y, origins) -> np.ndarray:
    return np.vstack([y[o : o + H] for o in origins])


def _save_plot(actual, forecast, results_dir: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(actual.ravel(), linewidth=1, label="actual")
    ax.plot(forecast.ravel(), linewidth=1, label=f"{name} (7 day blocks)")
    ax.set_xlabel("Test days in order")
    ax.set_ylabel("Daily count")
    ax.set_title("Test period: actual versus forecast")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "forecast_vs_actual.png", dpi=120)
    plt.close(fig)


def run(
    df: pd.DataFrame,
    results_dir: Path,
    fast: bool = False,
    seed: int = config.SEED,
    data_source: str = "bike_sharing_uci",
    data_sha256: str | None = None,
    expected_rows: int | None = config.EXPECTED_ROWS,
    initial_train: int = config.INITIAL_TRAIN,
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data.validate_frame(df, expected_rows=expected_rows)
    df = df.reset_index(drop=True)
    y = df[config.TARGET].to_numpy(dtype=float)
    X = build_features(df)
    start = first_valid_row(X)
    origins = rolling_origins(len(y), initial_train, config.STEP, H)
    if len(origins) < 4:
        raise ValueError("fewer than 4 forecast origins; need a longer series")
    n_tune = len(origins) // 2
    test_origins = origins[n_tune:]
    n_boot = 50 if fast else config.N_BOOT
    Y_all = _actuals(y, origins)

    # Predictions at every origin for every candidate. Tuning rows come first.
    candidates: dict[str, list[tuple[dict, np.ndarray]]] = {}
    for name in BASELINES:
        candidates[name] = [({}, _backtest_baseline(X, y, origins, name))]
    for name in TUNED:
        candidates[name] = [
            (params, _backtest_model(X, y, origins, start, name, params, seed))
            for params in param_grid(name, fast)
        ]

    def tune_mae(pred):
        return float(np.abs(Y_all[:n_tune] - pred[:n_tune]).mean())

    chosen: dict[str, tuple[dict, np.ndarray, float]] = {}
    for name, options in candidates.items():
        scored = [(tune_mae(p), params, p) for params, p in options]
        best = min(scored, key=lambda t: t[0])
        chosen[name] = (best[1], best[2], best[0])

    headline = min(TUNED, key=lambda n: chosen[n][2])
    best_baseline = min(BASELINES, key=lambda n: chosen[n][2])

    Y_test = Y_all[n_tune:]
    scale = seasonal_naive_scale(y[: test_origins[0]], H)
    model_results = {}
    for name, (params, pred, tuned) in chosen.items():
        P = pred[n_tune:]
        model_results[name] = {
            "tuning_mae": tuned,
            "test_mae": mae(Y_test, P),
            "test_rmse": rmse(Y_test, P),
            "test_smape_percent": smape(Y_test, P),
            "test_mase": mase(Y_test, P, scale),
            "test_bias": bias(Y_test, P),
            "best_params": _clean(params),
        }

    E = {name: np.abs(Y_test - chosen[name][1][n_tune:]) for name in chosen}
    n_blocks = len(test_origins)
    mae_ci = bootstrap_blocks_ci(lambda idx: E[headline][idx].mean(), n_blocks, n_boot, seed)
    gap_ci = bootstrap_blocks_ci(
        lambda idx: (E[best_baseline][idx] - E[headline][idx]).mean(), n_blocks, n_boot, seed
    )
    gate_passed = bool(gap_ci[0] > 0)
    fold_mae = E[headline].mean(axis=1)

    # Traps, measured on the same test blocks with the headline model's settings.
    head_params = chosen[headline][0]
    honest_mae = model_results[headline]["test_mae"]
    traps = {}
    for variant in ("same_day_components", "short_lag"):
        Xv = build_features(df, variant)
        pv = _backtest_model(Xv, y, test_origins, first_valid_row(Xv), headline, head_params, seed)
        traps[variant] = {
            "test_mae_with_leak": mae(Y_test, pv),
            "test_mae_honest": honest_mae,
            "difference": mae(Y_test, pv) - honest_mae,
        }
    traps["same_day_components"]["note"] = (
        "casual + registered equals the target on the same day, so the model reads the answer."
    )
    traps["short_lag"]["note"] = (
        "lag 1 and lag 2 are unknown 7 days ahead. Using them scores a forecast nobody could make."
    )
    rows = np.arange(start, len(y))
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    kpred = cross_val_predict(
        make_model(headline, seed, head_params), X.iloc[rows], y[rows], cv=kfold
    )
    traps["random_kfold"] = {
        "random_5fold_mae": mae(y[rows], kpred),
        "rolling_origin_test_mae": honest_mae,
        "difference": honest_mae - mae(y[rows], kpred),
        "note": "Shuffled folds train on days after the day being scored. Different rows, so this compares designs.",
    }

    # Error analysis for the headline model on the test blocks.
    test_rows = np.concatenate([np.arange(o, o + H) for o in test_origins])
    actual_flat = Y_test.ravel()
    forecast_flat = chosen[headline][1][n_tune:].ravel()
    sub = df.iloc[test_rows]
    level_edges = np.quantile(y[: test_origins[0]], [1 / 3, 2 / 3])
    level = np.array(["low", "mid", "high"])[np.digitize(actual_flat, level_edges)]
    error_analysis = {
        "model": headline,
        "by_weekday": segment_table(actual_flat, forecast_flat, sub["weekday"].to_numpy()),
        "by_workingday": segment_table(actual_flat, forecast_flat, sub["workingday"].to_numpy()),
        "by_holiday": segment_table(actual_flat, forecast_flat, sub["holiday"].to_numpy()),
        "by_month": segment_table(actual_flat, forecast_flat, sub["mnth"].to_numpy()),
        "by_demand_level": segment_table(actual_flat, forecast_flat, level),
        "demand_level_edges_from_history": [float(e) for e in level_edges],
        "fold_mae": {
            "min": float(fold_mae.min()),
            "median": float(np.median(fold_mae)),
            "max": float(fold_mae.max()),
        },
    }
    worst_weekday = max(error_analysis["by_weekday"].items(), key=lambda kv: kv[1]["mae"])
    error_analysis["worst_weekday"] = worst_weekday[0]

    metrics = {
        "project": config.PROJECT_NAME,
        "data_source": data_source,
        "fast_mode": bool(fast),
        "seed": seed,
        "versions": _versions(),
        "data": {"rows": len(df), "sha256": data_sha256},
        "backtest": {
            "horizon_days": H,
            "initial_train_days": initial_train,
            "step_days": config.STEP,
            "origins": len(origins),
            "tuning_origins": n_tune,
            "test_origins": n_blocks,
            "test_days": int(Y_test.size),
            "mase_scale_seasonal_naive_mae": scale,
        },
        "models": model_results,
        "headline": {
            "model": headline,
            "selected_by": "lowest tuning-fold MAE",
            "metric": "mae",
            "value": honest_mae,
            "ci95": [mae_ci[0], mae_ci[1]],
            "ci_method": f"block bootstrap over {n_blocks} forecast blocks, {n_boot} resamples",
        },
        "best_baseline": {
            "model": best_baseline,
            "selected_by": "lowest tuning-fold MAE",
            "test_mae": model_results[best_baseline]["test_mae"],
        },
        "gap_best_baseline_minus_headline_ci95": [gap_ci[0], gap_ci[1]],
        "gate": {"rule": config.GATE_RULE, "passed": gate_passed},
        "traps": traps,
        "caveats": CAVEATS,
    }

    (results_dir / "metrics.json").write_text(
        json.dumps(_clean(metrics), indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "error_analysis.json").write_text(
        json.dumps(_clean(error_analysis), indent=2) + "\n", encoding="utf-8"
    )
    _save_plot(Y_test, chosen[headline][1][n_tune:], results_dir, headline)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest and evaluate. Writes results/.")
    parser.add_argument("--fast", action="store_true", help="smoke-test mode, results are marked")
    args = parser.parse_args()

    frame = data.load(config.RAW_CSV)
    digest = data.verify_checksum(config.RAW_CSV)
    metrics = run(frame, config.RESULTS_DIR, fast=args.fast, data_sha256=digest)
    head = metrics["headline"]
    print(f"headline {head['model']}: test MAE {head['value']:.2f}, 95% interval {head['ci95']}")
    print(
        f"best baseline {metrics['best_baseline']['model']}: test MAE {metrics['best_baseline']['test_mae']:.2f}"
    )
    print(f"gate passed: {metrics['gate']['passed']}")
    print(f"wrote results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
