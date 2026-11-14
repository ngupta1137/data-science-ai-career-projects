"""Pipeline execution, model training, trap evaluation, and artifact generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import scipy
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config, data, metrics, models


def run_pipeline(results_dir: Path | None = None, fast_mode: bool = False) -> tuple[dict, dict]:
    """Execute complete marketing mix pipeline, returning (metrics_dict, error_analysis_dict)."""
    res_dir = results_dir or config.RESULTS_DIR
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify data generator fingerprint
    data.verify_fingerprint(config.CHECKSUM_FILE)

    n_train = config.N_WEEKS_TRAIN_FAST if fast_mode else config.N_WEEKS_TRAIN
    n_test = config.N_WEEKS_TEST_FAST if fast_mode else config.N_WEEKS_TEST
    n_boot = config.N_BOOT_FAST if fast_mode else config.N_BOOT

    rng_train = np.random.default_rng(config.SEED)
    train_data = data.generate(rng_train, n_train)
    data.validate(train_data)

    rng_test = np.random.default_rng(config.SEED + 1000)
    test_data = data.generate(rng_test, n_test)
    data.validate(test_data)

    # 2. Fit candidate models
    fitted_models = {
        "naive_linear": models.NaiveLinearModel().fit(train_data),
        "adstock_only": models.AdstockOnlyModel().fit(train_data),
        "saturation_only": models.SaturationOnlyModel().fit(train_data),
        "adstock_hill_mmm": models.AdstockHillMMM().fit(train_data),
        "oracle": models.OracleModel().fit(train_data),
    }

    # 3. Evaluate models on holdout test set
    avg_test_spend = {ch: float(np.mean(test_data.spend[ch])) for ch in config.CHANNELS}
    total_weekly_budget = float(sum(avg_test_spend.values()))

    model_metrics = {}
    preds = {}
    for name, model in fitted_models.items():
        pred = model.predict(test_data)
        preds[name] = pred
        r2 = metrics.r2_score(test_data.revenue, pred)
        err_rmse = metrics.rmse(test_data.revenue, pred)
        err_mape = metrics.mape(test_data.revenue, pred)
        mrois = model.marginal_roi(avg_test_spend)
        model_metrics[name] = {
            "r2": round(r2, 4),
            "rmse": round(err_rmse, 2),
            "mape": round(err_mape, 4),
            "marginal_rois": {ch: round(mrois[ch], 4) for ch in config.CHANNELS},
        }

    # 4. Bootstrap interval for headline model out-of-sample R^2
    r2_point, r2_ci_low, r2_ci_high = metrics.bootstrap_holdout_r2(
        test_data.revenue,
        preds["adstock_hill_mmm"],
        n_boot=n_boot,
        seed=config.SEED,
    )

    # 5. Budget Reallocation Decision
    headline_mmm = fitted_models["adstock_hill_mmm"]
    channel_params_mmm = {
        ch: {
            "half_sat": headline_mmm.Ks_[ch],
            "max_lift": headline_mmm.max_lifts_[ch],
        }
        for ch in config.CHANNELS
    }
    w_mmm = metrics.optimize_budget_mmm(channel_params_mmm, total_weekly_budget)

    naive_model = fitted_models["naive_linear"]
    w_naive = metrics.optimize_budget_linear(naive_model.channel_coefs_, total_weekly_budget)

    rev_mmm = metrics.evaluate_allocation_truth(w_mmm)
    rev_naive = metrics.evaluate_allocation_truth(w_naive)
    rel_gain = (rev_mmm - rev_naive) / rev_naive if rev_naive > 0 else 0.0

    # 6. Marginal ROI rank correlation
    true_mrois = [data.true_channel_marginal_roi(ch, avg_test_spend[ch]) for ch in config.CHANNELS]
    est_mrois = [model_metrics["adstock_hill_mmm"]["marginal_rois"][ch] for ch in config.CHANNELS]
    spearman_rho = metrics.spearman_rank_correlation(true_mrois, est_mrois)

    # 7. Decision Gate Evaluation
    check_r2 = bool(r2_ci_low >= config.MIN_HOLDOUT_R2)
    check_spearman = bool(spearman_rho >= config.MIN_ROI_SPEARMAN)
    check_allocation = bool(rel_gain >= config.MIN_ALLOCATION_GAIN)
    gate_passed = bool(check_r2 and check_spearman and check_allocation)

    gate = {
        "rule": config.GATE_RULE,
        "checks": {
            "holdout_r2_at_least_0.75": check_r2,
            "channel_roi_spearman_corr_at_least_0.80": check_spearman,
            "beats_naive_linear_allocation_revenue": check_allocation,
        },
        "passed": gate_passed,
    }

    # 8. Measure Traps
    # Trap 1: Ignoring adstock decay
    tv_true_alpha = config.TRUE_CHANNEL_PARAMS["tv"]["alpha"]
    tv_est_alpha_head = headline_mmm.alphas_["tv"]
    tv_half_life_true = np.log(0.5) / np.log(tv_true_alpha)
    trap_1 = {
        "tv_true_alpha": round(tv_true_alpha, 3),
        "tv_estimated_alpha_headline": round(tv_est_alpha_head, 3),
        "tv_estimated_alpha_naive": 0.0,
        "tv_true_half_life_weeks": round(float(tv_half_life_true), 2),
        "tv_naive_half_life_weeks": 0.0,
        "finding": (
            "Naive linear regression assumes zero carryover decay (half-life 0.0 weeks), "
            "missing TV's multi-week brand persistence (true half-life 1.36 weeks)."
        ),
    }

    # Trap 2: Ignoring saturation
    search_half_sat = config.TRUE_CHANNEL_PARAMS["search"]["half_sat"]
    search_max_lift = config.TRUE_CHANNEL_PARAMS["search"]["max_lift"]
    avg_search = avg_test_spend["search"]
    lift_1x = float(
        data.hill_saturation(np.array([avg_search]), search_half_sat, search_max_lift)[0]
    )
    lift_2x = float(
        data.hill_saturation(np.array([2.0 * avg_search]), search_half_sat, search_max_lift)[0]
    )
    true_search_diminish_ratio = lift_2x / lift_1x if lift_1x > 0 else 1.0

    trap_2 = {
        "search_spend_1x": round(avg_search, 1),
        "search_spend_2x": round(2.0 * avg_search, 1),
        "linear_projected_lift_ratio_at_2x": 2.0,
        "true_saturation_lift_ratio_at_2x": round(true_search_diminish_ratio, 3),
        "finding": (
            "Under linear assumptions, doubling Search spend projects a 2.00x revenue increase, "
            "whereas true Hill diminishing returns yield diminishing marginal return."
        ),
    }

    # Trap 3: Correlated spend inflates/distorts channel return
    corr_tv_search = float(np.corrcoef(train_data.spend["tv"], train_data.spend["search"])[0, 1])
    naive_search_slope = naive_model.channel_coefs_["search"]
    trap_3 = {
        "tv_search_spend_correlation": round(corr_tv_search, 4),
        "naive_search_slope": round(naive_search_slope, 4),
        "naive_search_recommended_spend": round(w_naive["search"], 1),
        "mmm_search_recommended_spend": round(w_mmm["search"], 1),
        "finding": (
            "TV and Search spend are synchronized; naive OLS suffers multicollinearity, "
            "inverting Search's slope and cutting Search spend to minimum compared to "
            "balanced MMM allocation."
        ),
    }

    traps = {
        "ignoring_adstock": trap_1,
        "ignoring_saturation": trap_2,
        "correlated_spend_bias": trap_3,
    }

    # 9. Format metrics.json
    metrics_out = {
        "project": config.PROJECT_NAME,
        "data_source": config.DATA_SOURCE,
        "fast_mode": fast_mode,
        "seed": config.SEED,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "data": {
            "train_weeks": n_train,
            "test_weeks": n_test,
            "sha256": data.fingerprint(),
        },
        "assumptions": {
            "base_sales": config.TRUE_BASE_SALES,
            "channels": list(config.CHANNELS),
            "total_weekly_budget": round(total_weekly_budget, 1),
            "min_holdout_r2": config.MIN_HOLDOUT_R2,
            "min_roi_spearman": config.MIN_ROI_SPEARMAN,
            "min_allocation_gain": config.MIN_ALLOCATION_GAIN,
        },
        "models": model_metrics,
        "headline": {
            "model": config.HEADLINE,
            "value": round(r2_point, 4),
            "ci_95": [round(r2_ci_low, 4), round(r2_ci_high, 4)],
            "ci_method": f"percentile bootstrap over weeks, {n_boot} resamples",
        },
        "budget_allocation": {
            "total_weekly_budget": round(total_weekly_budget, 1),
            "mmm_recommended_spend": {ch: round(w_mmm[ch], 1) for ch in config.CHANNELS},
            "naive_recommended_spend": {ch: round(w_naive[ch], 1) for ch in config.CHANNELS},
            "mmm_true_incremental_revenue": round(rev_mmm, 1),
            "naive_true_incremental_revenue": round(rev_naive, 1),
            "revenue_relative_gain": round(rel_gain, 4),
            "revenue_gain_pct": round(rel_gain * 100.0, 2),
        },
        "gate": gate,
        "traps": traps,
        "caveats": [
            "All data is generated by an econometric simulator with known adstock and Hill saturation.",
            "Real marketing channels exhibit competitive bidding dynamics and creative fatigue not modeled here.",
            "Observation noise is additive Gaussian without macroeconomic supply shocks or stock-outs.",
            "Bootstrap intervals resample holdout weeks; they do not cover structural shifts in channel effectiveness.",
        ],
    }

    # 10. Segment Error Analysis
    quarters = []
    for q_idx in range(4):
        mask = (test_data.weeks % 52.0 >= q_idx * 13) & (test_data.weeks % 52.0 < (q_idx + 1) * 13)
        if np.sum(mask) == 0:
            continue
        y_q = test_data.revenue[mask]
        p_q_head = preds["adstock_hill_mmm"][mask]
        p_q_naive = preds["naive_linear"][mask]
        quarters.append(
            {
                "segment": f"Q{q_idx + 1}",
                "weeks": int(np.sum(mask)),
                "actual_mean_revenue": round(float(np.mean(y_q)), 1),
                "headline_rmse": round(metrics.rmse(y_q, p_q_head), 2),
                "naive_rmse": round(metrics.rmse(y_q, p_q_naive), 2),
                "headline_r2": round(metrics.r2_score(y_q, p_q_head), 4),
            }
        )

    # Holiday vs Non-Holiday weeks
    holiday_mask = (test_data.weeks % 52.0 >= 44) & (test_data.weeks % 52.0 <= 51)
    non_holiday_mask = ~holiday_mask

    seasons = []
    if np.sum(holiday_mask) > 0:
        seasons.append(
            {
                "segment": "holiday_weeks_q4",
                "weeks": int(np.sum(holiday_mask)),
                "actual_mean_revenue": round(float(np.mean(test_data.revenue[holiday_mask])), 1),
                "headline_rmse": round(
                    metrics.rmse(
                        test_data.revenue[holiday_mask], preds["adstock_hill_mmm"][holiday_mask]
                    ),
                    2,
                ),
                "naive_rmse": round(
                    metrics.rmse(
                        test_data.revenue[holiday_mask], preds["naive_linear"][holiday_mask]
                    ),
                    2,
                ),
                "headline_r2": round(
                    metrics.r2_score(
                        test_data.revenue[holiday_mask], preds["adstock_hill_mmm"][holiday_mask]
                    ),
                    4,
                ),
            }
        )
    if np.sum(non_holiday_mask) > 0:
        seasons.append(
            {
                "segment": "non_holiday_weeks",
                "weeks": int(np.sum(non_holiday_mask)),
                "actual_mean_revenue": round(
                    float(np.mean(test_data.revenue[non_holiday_mask])), 1
                ),
                "headline_rmse": round(
                    metrics.rmse(
                        test_data.revenue[non_holiday_mask],
                        preds["adstock_hill_mmm"][non_holiday_mask],
                    ),
                    2,
                ),
                "naive_rmse": round(
                    metrics.rmse(
                        test_data.revenue[non_holiday_mask], preds["naive_linear"][non_holiday_mask]
                    ),
                    2,
                ),
                "headline_r2": round(
                    metrics.r2_score(
                        test_data.revenue[non_holiday_mask],
                        preds["adstock_hill_mmm"][non_holiday_mask],
                    ),
                    4,
                ),
            }
        )

    # Spend intensity (high spend weeks vs moderate spend weeks)
    total_spend_series = sum(test_data.spend[ch] for ch in config.CHANNELS)
    median_spend = float(np.median(total_spend_series))
    high_spend_mask = total_spend_series >= median_spend
    low_spend_mask = ~high_spend_mask

    intensity = [
        {
            "segment": "high_total_spend_weeks",
            "weeks": int(np.sum(high_spend_mask)),
            "actual_mean_revenue": round(float(np.mean(test_data.revenue[high_spend_mask])), 1),
            "headline_rmse": round(
                metrics.rmse(
                    test_data.revenue[high_spend_mask], preds["adstock_hill_mmm"][high_spend_mask]
                ),
                2,
            ),
            "naive_rmse": round(
                metrics.rmse(
                    test_data.revenue[high_spend_mask], preds["naive_linear"][high_spend_mask]
                ),
                2,
            ),
            "headline_r2": round(
                metrics.r2_score(
                    test_data.revenue[high_spend_mask], preds["adstock_hill_mmm"][high_spend_mask]
                ),
                4,
            ),
        },
        {
            "segment": "low_total_spend_weeks",
            "weeks": int(np.sum(low_spend_mask)),
            "actual_mean_revenue": round(float(np.mean(test_data.revenue[low_spend_mask])), 1),
            "headline_rmse": round(
                metrics.rmse(
                    test_data.revenue[low_spend_mask], preds["adstock_hill_mmm"][low_spend_mask]
                ),
                2,
            ),
            "naive_rmse": round(
                metrics.rmse(
                    test_data.revenue[low_spend_mask], preds["naive_linear"][low_spend_mask]
                ),
                2,
            ),
            "headline_r2": round(
                metrics.r2_score(
                    test_data.revenue[low_spend_mask], preds["adstock_hill_mmm"][low_spend_mask]
                ),
                4,
            ),
        },
    ]

    error_analysis_out = {
        "quarterly_segments": quarters,
        "seasonal_segments": seasons,
        "spend_intensity_segments": intensity,
    }

    # 11. Plot Response Curves & Allocations
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    spend_range = np.linspace(0, 80_000, 200)

    for i, ch in enumerate(config.CHANNELS):
        ax = axes[i]
        # True response
        p = config.TRUE_CHANNEL_PARAMS[ch]
        true_curve = data.hill_saturation(spend_range, p["half_sat"], p["max_lift"])
        ax.plot(spend_range / 1000.0, true_curve / 1000.0, "k-", lw=2, label="True Saturation")

        # MMM estimated response
        k = headline_mmm.Ks_[ch]
        m = headline_mmm.max_lifts_[ch]
        mmm_curve = m * (spend_range / (spend_range + k))
        ax.plot(spend_range / 1000.0, mmm_curve / 1000.0, "b--", lw=2, label="MMM Estimated")

        # Naive linear response
        slope = naive_model.channel_coefs_[ch]
        ax.plot(
            spend_range / 1000.0, (slope * spend_range) / 1000.0, "r:", lw=1.5, label="Naive Linear"
        )

        # Mark recommended spend
        ax.axvline(
            w_mmm[ch] / 1000.0, color="blue", ls="-.", alpha=0.7, label=f"MMM: ${w_mmm[ch]:,.0f}"
        )
        ax.axvline(
            w_naive[ch] / 1000.0,
            color="red",
            ls=":",
            alpha=0.7,
            label=f"Naive: ${w_naive[ch]:,.0f}",
        )

        ax.set_title(f"Channel: {ch.upper()}")
        ax.set_xlabel("Weekly Spend ($k)")
        if i == 0:
            ax.set_ylabel("Incremental Lift ($k)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Marketing Response Curves and Budget Reallocation", fontsize=12)
    plt.tight_layout()
    plt.savefig(res_dir / "response_curves.png", dpi=150)
    plt.close(fig)

    # 12. Write JSON outputs
    (res_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    (res_dir / "error_analysis.json").write_text(
        json.dumps(error_analysis_out, indent=2), encoding="utf-8"
    )

    return metrics_out, error_analysis_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train marketing mix models and evaluate decision gate."
    )
    parser.add_argument(
        "--write-checksum",
        action="store_true",
        help="Write simulator fingerprint to data/CHECKSUMS.txt",
    )
    parser.add_argument(
        "--fast", action="store_true", help="Run fast pipeline for quick validation testing"
    )
    args = parser.parse_args()

    if args.write_checksum:
        data.write_fingerprint(config.CHECKSUM_FILE)
        print(f"Wrote checksum: {data.fingerprint()}")
        return

    metrics_res, _ = run_pipeline(fast_mode=args.fast)
    verdict = "PASSED" if metrics_res["gate"]["passed"] else "FAILED"
    print(f"Project 17 Decision Gate {verdict}")
    for k, v in metrics_res["gate"]["checks"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
