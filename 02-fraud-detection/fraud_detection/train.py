"""Train, evaluate, and write every result file.

    python -m fraud_detection.train            # full run on the downloaded data
    python -m fraud_detection.train --fast     # tiny search, for smoke tests only (marked in the output)

Order of use of each split:
  fit   : models are fitted here
  val   : model choice, calibration, and threshold selection
  test  : scored once, after every choice is made
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
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import ParameterGrid

from . import config, data
from .features import build_features
from .metrics import (
    average_precision,
    best_global_threshold,
    bootstrap_ci,
    brier,
    calibration_ratio,
    expected_cost_flag,
    fit_isotonic,
    fit_platt,
    policy_cost,
    recall_at_top_fraction,
    reliability_table,
    roc_auc,
)
from .models import BASELINE, TUNED, make_models, param_grid
from .splits import assert_time_order, random_split, time_split

CAVEATS = [
    "The cost model is an assumption: a missed fraud costs its amount, a flag costs a fixed review cost. Real costs differ.",
    "The interval resamples rows as if independent. Transactions on one card are correlated, so the true uncertainty is wider.",
    "The test period is about one day of data. It cannot show how the model behaves over months of drift.",
    "The V1 to V28 features are PCA outputs computed on the whole dataset, including the test period, before this project saw them. That leak cannot be undone here.",
    "Duplicate rows are kept. The count is recorded in the data section of metrics.json.",
    "Hyperparameters, calibration, and the threshold all use the same validation split, so validation scores are optimistic. Test scores are not.",
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
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _save_plots(y_test, s_test, p_test, baseline_ap, results_dir: Path, model_name: str) -> None:
    precision, recall, _ = precision_recall_curve(y_test, s_test)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, linewidth=1.2, label=model_name)
    ax.axhline(baseline_ap, linestyle="--", linewidth=1, label="constant baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "precision_recall.png", dpi=120)
    plt.close(fig)

    table = reliability_table(y_test, p_test)
    pred = [r["mean_predicted"] for r in table]
    obs = [r["observed_rate"] for r in table]
    top = max(max(pred), max(obs), 1e-9) * 1.1
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, top], [0, top], linewidth=1)
    ax.plot(pred, obs, marker="o", linewidth=1)
    ax.set_xlabel("Mean predicted probability (equal-count bins)")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Calibration after sigmoid fit (test)")
    fig.tight_layout()
    fig.savefig(results_dir / "calibration.png", dpi=120)
    plt.close(fig)


def _segment(mask, y, flag, amount) -> dict:
    n = int(mask.sum())
    fraud = int(y[mask].sum())
    caught = int((y[mask] & flag[mask]).sum())
    return {
        "n": n,
        "fraud": fraud,
        "flagged": int(flag[mask].sum()),
        "recall": (caught / fraud) if fraud else None,
        "missed_fraud_amount": float(amount[mask & (y == 1) & ~flag].sum()),
    }


def run(
    df: pd.DataFrame,
    results_dir: Path,
    fast: bool = False,
    seed: int = config.SEED,
    data_source: str = "creditcard_openml",
    data_sha256: str | None = None,
    expected_rows: int | None = config.EXPECTED_ROWS,
    expected_fraud: int | None = config.EXPECTED_FRAUD,
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data.validate_frame(df, expected_rows, expected_fraud)
    duplicates = int(df.duplicated().sum())
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)

    fit_idx, val_idx, test_idx = time_split(df["Time"], config.TEST_FRAC, config.VAL_FRAC)
    assert_time_order(df["Time"], fit_idx, val_idx, test_idx)

    y = df[config.TARGET].to_numpy(dtype=int)
    amount = df["Amount"].to_numpy(dtype=float)
    X = build_features(df)
    for name, idx in (("fit", fit_idx), ("val", val_idx), ("test", test_idx)):
        if int(y[idx].sum()) < config.MIN_POSITIVES:
            raise ValueError(
                f"{name} split has {int(y[idx].sum())} fraud rows, fewer than {config.MIN_POSITIVES}"
            )

    X_fit, X_val, X_test = X.iloc[fit_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_fit, y_val, y_test = y[fit_idx], y[val_idx], y[test_idx]
    amt_val, amt_test = amount[val_idx], amount[test_idx]
    n_boot = 50 if fast else config.N_BOOT

    # Fit all models. Tuned models are chosen by validation PR-AUC.
    models = make_models(seed)
    fitted, scores, model_results = {}, {}, {}
    for name, estimator in models.items():
        if name in TUNED:
            best = None
            for params in ParameterGrid(param_grid(name, fast)):
                candidate = clone(estimator).set_params(**params).fit(X_fit, y_fit)
                val_ap = average_precision(y_val, candidate.predict_proba(X_val)[:, 1])
                if best is None or val_ap > best[0]:
                    best = (val_ap, params, candidate)
            val_ap, params, model = best
        else:
            model = clone(estimator).fit(X_fit, y_fit)
            params = {}
            val_ap = average_precision(y_val, model.predict_proba(X_val)[:, 1])
        fitted[name] = model
        s_val = model.predict_proba(X_val)[:, 1]
        s_test = model.predict_proba(X_test)[:, 1]
        scores[name] = (s_val, s_test)
        model_results[name] = {
            "val_pr_auc": val_ap,
            "test_pr_auc": average_precision(y_test, s_test),
            "test_roc_auc": roc_auc(y_test, s_test) if name != BASELINE else 0.5,
            "test_brier": brier(y_test, s_test),
            "best_params": _clean(params),
        }

    headline = max(TUNED, key=lambda n: model_results[n]["val_pr_auc"])
    s_val, s_test = scores[headline]
    baseline_ap = model_results[BASELINE]["test_pr_auc"]
    prevalence = float(y_test.mean())

    ap_ci = bootstrap_ci(
        lambda idx: average_precision(y_test[idx], s_test[idx]), len(y_test), n_boot, seed
    )

    # Calibration, fitted on validation only.
    platt = fit_platt(s_val, y_val)
    iso = fit_isotonic(s_val, y_val)
    p_platt, p_iso = platt(s_test), iso(s_test)
    calibration = {
        "calibrator_used_for_policy": "sigmoid (Platt), fixed in advance because validation holds few fraud rows",
        "val_fraud_rows": int(y_val.sum()),
        "test_brier_raw": brier(y_test, s_test),
        "test_brier_sigmoid": brier(y_test, p_platt),
        "test_brier_isotonic": brier(y_test, p_iso),
        "test_brier_constant_prevalence": brier(y_test, np.full(len(y_test), y_fit.mean())),
        "test_calibration_ratio_raw": calibration_ratio(y_test, s_test),
        "test_calibration_ratio_sigmoid": calibration_ratio(y_test, p_platt),
        "test_calibration_ratio_isotonic": calibration_ratio(y_test, p_iso),
        "reliability_sigmoid": reliability_table(y_test, p_platt),
    }

    # Cost policies at each review cost.
    cost_table = {}
    for c in config.REVIEW_COSTS:
        none_cost = policy_cost(y_test, amt_test, np.zeros(len(y_test), bool), c)
        all_cost = policy_cost(y_test, amt_test, np.ones(len(y_test), bool), c)
        t_val, _ = best_global_threshold(s_val, y_val, amt_val, c)
        flag_global = s_test >= t_val
        flag_platt = expected_cost_flag(p_platt, amt_test, c)
        flag_iso = expected_cost_flag(p_iso, amt_test, c)
        _, oracle_cost = best_global_threshold(s_test, y_test, amt_test, c)
        trivial = min(none_cost, all_cost)
        entry = {
            "flag_nothing": none_cost,
            "flag_everything": all_cost,
            "global_threshold_tuned_on_val": policy_cost(y_test, amt_test, flag_global, c),
            "per_transaction_sigmoid": policy_cost(y_test, amt_test, flag_platt, c),
            "per_transaction_isotonic": policy_cost(y_test, amt_test, flag_iso, c),
            "oracle_threshold_tuned_on_test": oracle_cost,
            "val_threshold": None if not np.isfinite(t_val) else float(t_val),
            "flagged_share_global": float(flag_global.mean()),
            "flagged_share_per_transaction": float(flag_platt.mean()),
        }
        entry["best_trivial"] = trivial
        entry["saving_vs_best_trivial_per_transaction"] = (
            (trivial - entry["per_transaction_sigmoid"]) / trivial if trivial > 0 else None
        )
        cost_table[str(c)] = entry

    base_key = str(config.REVIEW_COST_BASE)
    base = cost_table[base_key]
    c0 = config.REVIEW_COST_BASE
    t_val0, _ = best_global_threshold(s_val, y_val, amt_val, c0)

    def _cost_gap(idx):
        yy, aa = y_test[idx], amt_test[idx]
        g = policy_cost(yy, aa, s_test[idx] >= t_val0, c0)
        p = policy_cost(yy, aa, expected_cost_flag(p_platt[idx], aa, c0), c0)
        return g - p

    gap_ci = bootstrap_ci(_cost_gap, len(y_test), n_boot, seed)
    flag_policy = expected_cost_flag(p_platt, amt_test, c0)
    flag_global0 = s_test >= t_val0

    # Gate, fixed before results were seen.
    gate1 = bool(ap_ci[0] > baseline_ap)
    gate2 = bool(base["per_transaction_sigmoid"] < base["best_trivial"])
    gate_passed = gate1 and gate2

    # Traps, measured.
    accuracy_trap = {
        "constant_legit_accuracy": float(1.0 - prevalence),
        "model_accuracy_at_global_threshold": float(np.mean(flag_global0.astype(int) == y_test)),
        "model_recall_at_global_threshold": float(
            (flag_global0 & (y_test == 1)).sum() / max(int(y_test.sum()), 1)
        ),
        "note": "A model that flags nothing scores the constant accuracy and catches no fraud.",
    }
    optimism_trap = {
        "review_cost": c0,
        "cost_with_threshold_tuned_on_val": base["global_threshold_tuned_on_val"],
        "cost_with_threshold_tuned_on_test": base["oracle_threshold_tuned_on_test"],
        "optimism": base["global_threshold_tuned_on_val"] - base["oracle_threshold_tuned_on_test"],
        "note": "Tuning the threshold on test rows shows a cost you could not have achieved in advance.",
    }
    rs_train, rs_test = random_split(len(df), config.TEST_FRAC, seed)
    hgb_params = model_results["hist_gradient_boosting"]["best_params"]
    rs_model = (
        clone(models["hist_gradient_boosting"])
        .set_params(**hgb_params)
        .fit(X.iloc[rs_train], y[rs_train])
    )
    random_ap = average_precision(y[rs_test], rs_model.predict_proba(X.iloc[rs_test])[:, 1])
    split_trap = {
        "model": "hist_gradient_boosting",
        "same_hyperparameters": True,
        "random_split_test_pr_auc": random_ap,
        "time_split_test_pr_auc": model_results["hist_gradient_boosting"]["test_pr_auc"],
        "difference": model_results["hist_gradient_boosting"]["test_pr_auc"] - random_ap,
        "note": "The two test sets contain different rows, so this compares evaluation designs, not identical rows.",
    }

    # Error analysis for the per-transaction policy at the base review cost.
    amount_edges = np.quantile(amount[fit_idx], [0.5, 0.9])
    band = np.digitize(amt_test, amount_edges)
    band_names = np.array(
        ["amount_below_fit_median", "amount_median_to_p90", "amount_above_fit_p90"]
    )[band]
    by_amount = {
        name: _segment(band_names == name, y_test, flag_policy, amt_test)
        for name in sorted(set(band_names.tolist()))
    }
    by_time = {}
    for i, chunk in enumerate(np.array_split(np.arange(len(y_test)), 3), start=1):
        mask = np.zeros(len(y_test), dtype=bool)
        mask[chunk] = True
        seg = _segment(mask, y_test, flag_policy, amt_test)
        seg["pr_auc"] = (
            average_precision(y_test[chunk], s_test[chunk]) if y_test[chunk].sum() >= 5 else None
        )
        by_time[f"test_third_{i}"] = seg
    weakest = [
        (k, v["recall"])
        for k, v in by_amount.items()
        if v["recall"] is not None and v["fraud"] >= 10
    ]
    error_analysis = {
        "model": headline,
        "policy": f"per-transaction expected cost, sigmoid calibration, review cost {c0}",
        "overall": _segment(np.ones(len(y_test), bool), y_test, flag_policy, amt_test),
        "by_amount_band": by_amount,
        "by_time_third": by_time,
        "amount_band_edges_from_fit": [float(e) for e in amount_edges],
        "lowest_recall_amount_band_min_10_fraud": min(weakest, key=lambda kv: kv[1])[0]
        if weakest
        else None,
    }

    operating_points = {
        f"top_{frac:g}_share": {
            "recall": recall_at_top_fraction(y_test, s_test, frac),
            "note": "illustrative review budget, not a target",
        }
        for frac in (0.001, 0.005, 0.01)
    }

    metrics = {
        "project": config.PROJECT_NAME,
        "data_source": data_source,
        "fast_mode": bool(fast),
        "seed": seed,
        "versions": _versions(),
        "data": {
            "rows": len(df),
            "sha256": data_sha256,
            "fraud_rows": int(y.sum()),
            "fraud_rate": float(y.mean()),
            "duplicate_rows": duplicates,
        },
        "split": {
            "strategy": "time-ordered: fit, then validation, then test",
            "fit_rows": len(fit_idx),
            "val_rows": len(val_idx),
            "test_rows": len(test_idx),
            "fit_fraud": int(y_fit.sum()),
            "val_fraud": int(y_val.sum()),
            "test_fraud": int(y_test.sum()),
            "test_fraud_rate": prevalence,
        },
        "models": model_results,
        "headline": {
            "model": headline,
            "selected_by": "highest PR-AUC on validation",
            "metric": "pr_auc",
            "value": model_results[headline]["test_pr_auc"],
            "ci95": [ap_ci[0], ap_ci[1]],
            "ci_method": f"row bootstrap, {n_boot} resamples",
        },
        "baseline": {"model": BASELINE, "test_pr_auc": baseline_ap},
        "calibration": calibration,
        "assumptions": {
            "missed_fraud_cost": "transaction amount",
            "flag_cost": "fixed review cost per flagged transaction",
            "review_costs_tested": list(config.REVIEW_COSTS),
            "review_cost_base": c0,
        },
        "cost_by_review_cost": cost_table,
        "cost_gap_global_minus_per_transaction_ci95": [gap_ci[0], gap_ci[1]],
        "operating_points": operating_points,
        "gate": {
            "rule": config.GATE_RULE,
            "pr_auc_interval_above_baseline": gate1,
            "per_transaction_policy_beats_trivial": gate2,
            "passed": gate_passed,
        },
        "traps": {
            "accuracy": accuracy_trap,
            "threshold_on_test": optimism_trap,
            "random_split": split_trap,
        },
        "caveats": CAVEATS,
    }

    (results_dir / "metrics.json").write_text(
        json.dumps(_clean(metrics), indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "error_analysis.json").write_text(
        json.dumps(_clean(error_analysis), indent=2) + "\n", encoding="utf-8"
    )
    _save_plots(y_test, s_test, p_platt, baseline_ap, results_dir, headline)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate. Writes results/.")
    parser.add_argument("--fast", action="store_true", help="smoke-test mode, results are marked")
    args = parser.parse_args()

    frame = data.load(config.RAW_CSV)
    digest = data.verify_checksum(config.RAW_CSV)
    metrics = run(frame, config.RESULTS_DIR, fast=args.fast, data_sha256=digest)
    head = metrics["headline"]
    print(f"headline {head['model']}: test PR-AUC {head['value']:.4f}, 95% interval {head['ci95']}")
    print(f"constant baseline PR-AUC {metrics['baseline']['test_pr_auc']:.4f}")
    print(f"gate passed: {metrics['gate']['passed']}")
    print(f"wrote results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
