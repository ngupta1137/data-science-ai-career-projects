"""Train, evaluate, and write every result file.

    python -m house_price.train            # full run on the downloaded data
    python -m house_price.train --fast     # tiny search, for smoke tests only (marked in the output)

Model choice uses grouped cross-validation on the training rows only.
The test rows are scored once, after choices are made.
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
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score

from . import config, data
from .features import build_features
from .metrics import (
    bias,
    cluster_bootstrap_ci,
    mae,
    r2,
    rmse,
    segment_table,
    target_band,
)
from .models import BASELINES, TUNED, make_models, param_grid
from .splits import (
    add_geo_group,
    assert_no_overlap,
    random_split,
    spatial_split,
)

CAVEATS = [
    "Targets at or above the cap are censored in the source data. The model cannot learn true values above the cap.",
    "The interval resamples spatial cells. It does not capture uncertainty from the choice of split.",
    "Metrics are in units of $100,000 in 1990 dollars. They say nothing about current prices.",
    "Spatial cells are 0.5 degrees. Cells near the same city can still share information across the cell border.",
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
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _save_plots(y_true, y_pred, results_dir: Path, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3)
    lim = [0, max(float(np.max(y_true)), float(np.max(y_pred))) * 1.02]
    ax.plot(lim, lim, linewidth=1)
    ax.set_xlabel("Actual (units of $100,000)")
    ax.set_ylabel("Predicted (units of $100,000)")
    ax.set_title(f"{model_name}: predicted vs actual (test)")
    fig.tight_layout()
    fig.savefig(results_dir / "pred_vs_actual.png", dpi=120)
    plt.close(fig)

    residual = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residual, s=4, alpha=0.3)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Predicted (units of $100,000)")
    ax.set_ylabel("Residual (predicted minus actual)")
    ax.set_title(f"{model_name}: residuals (test)")
    fig.tight_layout()
    fig.savefig(results_dir / "residuals_vs_predicted.png", dpi=120)
    plt.close(fig)


def run(
    df: pd.DataFrame,
    results_dir: Path,
    fast: bool = False,
    seed: int = config.SEED,
    data_source: str = "california_housing",
    data_sha256: str | None = None,
    expected_rows: int | None = config.EXPECTED_ROWS,
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data.validate_frame(df, expected_rows=expected_rows)
    df = df.reset_index(drop=True)

    y = df[config.TARGET].to_numpy(dtype=float)
    X = build_features(df)
    groups = add_geo_group(df)

    train_idx, test_idx = spatial_split(groups, config.TEST_SIZE, seed)
    assert_no_overlap(groups, train_idx, test_idx)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train, g_test = groups[train_idx], groups[test_idx]

    n_boot = 50 if fast else config.N_BOOT
    cv = GroupKFold(n_splits=config.CV_FOLDS)

    models = make_models(seed)
    fitted = {}
    model_results: dict = {}
    for name, estimator in models.items():
        if name in TUNED:
            search = GridSearchCV(
                estimator,
                param_grid(name, fast),
                cv=cv,
                scoring="neg_root_mean_squared_error",
                refit=True,
            )
            search.fit(X_train, y_train, groups=g_train)
            fitted[name] = search.best_estimator_
            cv_rmse = float(-search.best_score_)
            best_params = _clean(search.best_params_)
        else:
            scores = cross_val_score(
                estimator,
                X_train,
                y_train,
                groups=g_train,
                cv=cv,
                scoring="neg_root_mean_squared_error",
            )
            cv_rmse = float(-np.mean(scores))
            fitted[name] = clone(estimator).fit(X_train, y_train)
            best_params = {}

        pred = fitted[name].predict(X_test)
        model_results[name] = {
            "cv_rmse": cv_rmse,
            "test_rmse": rmse(y_test, pred),
            "test_mae": mae(y_test, pred),
            "test_r2": r2(y_test, pred),
            "best_params": best_params,
        }

    # Headline model is chosen by cross-validation on train, never by test score.
    headline_name = min(TUNED, key=lambda n: model_results[n]["cv_rmse"])
    headline_pred = fitted[headline_name].predict(X_test)
    rmse_ci = cluster_bootstrap_ci(y_test, headline_pred, g_test, rmse, n_boot, seed)
    mae_ci = cluster_bootstrap_ci(y_test, headline_pred, g_test, mae, n_boot, seed)

    best_baseline = min(BASELINES, key=lambda n: model_results[n]["test_rmse"])
    baseline_rmse = model_results[best_baseline]["test_rmse"]
    gate_passed = bool(rmse_ci[1] < baseline_rmse)

    # The trap: same model, same settings, random split instead of spatial split.
    tr_r, te_r = random_split(len(df), config.TEST_SIZE, seed)
    trap_model = clone(models["hist_gradient_boosting"]).set_params(
        **model_results["hist_gradient_boosting"]["best_params"]
    )
    trap_model.fit(X.iloc[tr_r], y[tr_r])
    random_rmse = rmse(y[te_r], trap_model.predict(X.iloc[te_r]))
    spatial_rmse = model_results["hist_gradient_boosting"]["test_rmse"]
    leakage_trap = {
        "model": "hist_gradient_boosting",
        "same_hyperparameters": True,
        "random_split_test_rmse": random_rmse,
        "spatial_split_test_rmse": spatial_rmse,
        "difference": spatial_rmse - random_rmse,
        "note": (
            "The two test sets contain different rows, so this is a comparison of "
            "evaluation designs, not of identical rows."
        ),
    }

    # Error analysis on the headline model.
    bands = target_band(y_test)
    income_edges = np.quantile(X_train["MedInc"].to_numpy(), [0.2, 0.4, 0.6, 0.8])
    income_q = np.array(
        [f"income_q{i + 1}" for i in np.digitize(X_test["MedInc"].to_numpy(), income_edges)]
    )
    by_band = segment_table(y_test, headline_pred, bands)
    by_income = segment_table(y_test, headline_pred, income_q)
    by_cell = segment_table(y_test, headline_pred, g_test, min_n=20)
    worst_cells = dict(sorted(by_cell.items(), key=lambda kv: kv[1]["rmse"], reverse=True)[:5])

    ranked = {
        **{f"price_band:{k}": v for k, v in by_band.items() if v["n"] >= 30},
        **{f"income:{k}": v for k, v in by_income.items() if v["n"] >= 30},
    }
    worst_segment = max(ranked, key=lambda k: ranked[k]["rmse"]) if ranked else None

    error_analysis = {
        "model": headline_name,
        "overall": {
            "rmse": rmse(y_test, headline_pred),
            "mae": mae(y_test, headline_pred),
            "bias": bias(y_test, headline_pred),
        },
        "by_price_band": by_band,
        "by_income_quintile": by_income,
        "worst_geographic_cells_min_n_20": worst_cells,
        "worst_segment_min_n_30": worst_segment,
        "income_quintile_edges_from_train": [float(e) for e in income_edges],
    }

    censored = int((y >= config.TARGET_CAP).sum())
    metrics = {
        "project": config.PROJECT_NAME,
        "data_source": data_source,
        "fast_mode": bool(fast),
        "seed": seed,
        "versions": _versions(),
        "data": {
            "rows": len(df),
            "sha256": data_sha256,
            "censored_rows": censored,
            "censored_share": censored / len(df),
        },
        "split": {
            "strategy": f"spatial group split, {config.GROUP_CELL_DEG} degree cells",
            "test_size": config.TEST_SIZE,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_cells": len(set(g_train)),
            "test_cells": len(set(g_test)),
        },
        "cv": {"folds": config.CV_FOLDS, "grouping": "spatial cells, train rows only"},
        "models": model_results,
        "headline": {
            "model": headline_name,
            "selected_by": "lowest grouped-CV RMSE on train",
            "metric": "rmse",
            "value": model_results[headline_name]["test_rmse"],
            "ci95": [rmse_ci[0], rmse_ci[1]],
            "mae_ci95": [mae_ci[0], mae_ci[1]],
            "ci_method": f"cluster bootstrap over spatial cells, {n_boot} resamples",
        },
        "best_baseline": {"model": best_baseline, "test_rmse": baseline_rmse},
        "gate": {"rule": config.GATE_RULE, "passed": gate_passed},
        "leakage_trap": leakage_trap,
        "caveats": CAVEATS,
    }

    (results_dir / "metrics.json").write_text(
        json.dumps(_clean(metrics), indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "error_analysis.json").write_text(
        json.dumps(_clean(error_analysis), indent=2) + "\n", encoding="utf-8"
    )
    _save_plots(y_test, headline_pred, results_dir, headline_name)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate. Writes results/.")
    parser.add_argument("--fast", action="store_true", help="smoke-test mode, results are marked")
    args = parser.parse_args()

    frame = data.load(config.RAW_CSV)
    digest = data.verify_checksum(config.RAW_CSV)
    metrics = run(frame, config.RESULTS_DIR, fast=args.fast, data_sha256=digest)

    head = metrics["headline"]
    print(f"headline {head['model']}: test RMSE {head['value']:.4f}, 95% interval {head['ci95']}")
    print(
        f"best baseline {metrics['best_baseline']['model']}: test RMSE {metrics['best_baseline']['test_rmse']:.4f}"
    )
    print(f"gate passed: {metrics['gate']['passed']}")
    print(f"wrote results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
