"""Train pricing models, evaluate traps, and output metrics and error analysis."""

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

from . import config
from .data import generate, validate, verify_fingerprint, write_fingerprint
from .metrics import (
    bootstrap_gini_and_diff,
    decile_loss_ratios,
    gini_coefficient,
    loss_ratio,
    rmse,
)
from .models import (
    FlatBaseline,
    FreqSevTwoPartModel,
    NoExposureModel,
    OracleModel,
    SingleTweedieModel,
)


def run_pipeline(results_dir: Path | None = None, fast_mode: bool = False) -> tuple[dict, dict]:
    res_dir = results_dir or config.RESULTS_DIR
    # Check data fingerprint
    verify_fingerprint(config.CHECKSUM_FILE)

    n_train = 1000 if fast_mode else config.N_TRAIN
    n_test = 1000 if fast_mode else config.N_TEST
    n_boot = 50 if fast_mode else config.N_BOOT

    rng_train = np.random.default_rng(config.SEED)
    train_policies = generate(rng_train, n_train)
    validate(train_policies)

    rng_test = np.random.default_rng(config.SEED + 1000)
    test_policies = generate(rng_test, n_test)
    validate(test_policies)

    # Fit models
    models = {
        "flat_baseline": FlatBaseline().fit(train_policies),
        "no_exposure": NoExposureModel().fit(train_policies),
        "single_tweedie": SingleTweedieModel().fit(train_policies),
        "freq_sev_two_part": FreqSevTwoPartModel().fit(train_policies),
        "oracle": OracleModel().fit(train_policies),
    }

    # Evaluate pure premium predictions
    pred_rates = {name: model.predict_pure_premium(test_policies) for name, model in models.items()}

    # Charged technical premium: pure_premium * exposure / target_loss_ratio
    charged_premiums = {
        name: (pred_rates[name] * test_policies.exposure) / config.TARGET_LOSS_RATIO
        for name in models
    }

    # Model evaluation metrics
    policy_metrics = {}
    for name in config.POLICIES:
        rate = pred_rates[name]
        prem = charged_premiums[name]
        g = gini_coefficient(rate, test_policies.claim_amount, test_policies.exposure)
        lr = loss_ratio(test_policies.claim_amount, prem)
        err_rmse = rmse(rate, test_policies.true_pure_premium)
        rank_vec = pred_rates["freq_sev_two_part"] if name == "flat_baseline" else rate
        deciles = decile_loss_ratios(
            test_policies.claim_amount, prem, rank_vec, test_policies.exposure
        )
        policy_metrics[name] = {
            "gini": round(g, 4),
            "loss_ratio": round(lr, 4),
            "rmse_vs_truth": round(err_rmse, 2),
            "decile_loss_ratios": [d["loss_ratio"] for d in deciles],
            "top_decile_loss_ratio": deciles[-1]["loss_ratio"],
        }

    # Bootstrap intervals for headline vs single Tweedie
    ci_headline, ci_diff = bootstrap_gini_and_diff(
        pred_rates["freq_sev_two_part"],
        pred_rates["single_tweedie"],
        test_policies.claim_amount,
        test_policies.exposure,
        n_boot=n_boot,
        seed=config.SEED,
    )

    headline_gini = policy_metrics["freq_sev_two_part"]["gini"]
    headline_top_decile_lr = policy_metrics["freq_sev_two_part"]["top_decile_loss_ratio"]

    checks = {
        "gini_at_least_0.25": bool(ci_headline[0] >= config.MIN_GINI),
        "beats_single_tweedie": bool(ci_diff[0] > 0.0),
        "top_decile_loss_ratio_under_0.95": bool(headline_top_decile_lr < config.MAX_TOP_DECILE_LR),
    }
    gate_passed = all(checks.values())

    # Traps evaluation
    # Trap 1: Ignoring exposure time
    short_exp = test_policies.exposure < 0.30
    full_exp = test_policies.exposure > 0.80
    trap_exposure = {
        "short_exposure_count": int(np.sum(short_exp)),
        "full_exposure_count": int(np.sum(full_exp)),
        "no_exposure_lr_short": round(
            float(
                np.sum(test_policies.claim_amount[short_exp])
                / max(np.sum(charged_premiums["no_exposure"][short_exp]), 1e-6)
            ),
            4,
        ),
        "no_exposure_lr_full": round(
            float(
                np.sum(test_policies.claim_amount[full_exp])
                / max(np.sum(charged_premiums["no_exposure"][full_exp]), 1e-6)
            ),
            4,
        ),
        "two_part_lr_short": round(
            float(
                np.sum(test_policies.claim_amount[short_exp])
                / max(np.sum(charged_premiums["freq_sev_two_part"][short_exp]), 1e-6)
            ),
            4,
        ),
        "two_part_lr_full": round(
            float(
                np.sum(test_policies.claim_amount[full_exp])
                / max(np.sum(charged_premiums["freq_sev_two_part"][full_exp]), 1e-6)
            ),
            4,
        ),
        "note": (
            "The model that ignores exposure time misprices short-duration policies "
            "because it treats raw claim count as the annualized rate."
        ),
    }

    # Trap 2: Adverse selection by decile under flat pricing
    flat_deciles = decile_loss_ratios(
        test_policies.claim_amount,
        charged_premiums["flat_baseline"],
        pred_rates["freq_sev_two_part"],
        test_policies.exposure,
    )
    two_part_deciles = decile_loss_ratios(
        test_policies.claim_amount,
        charged_premiums["freq_sev_two_part"],
        pred_rates["freq_sev_two_part"],
        test_policies.exposure,
    )
    trap_deciles = {
        "flat_decile_1_loss_ratio": flat_deciles[0]["loss_ratio"],
        "flat_decile_10_loss_ratio": flat_deciles[-1]["loss_ratio"],
        "two_part_decile_1_loss_ratio": two_part_deciles[0]["loss_ratio"],
        "two_part_decile_10_loss_ratio": two_part_deciles[-1]["loss_ratio"],
        "note": (
            "Flat pricing heavily overcharges the safest segment and undercharges the "
            "riskiest segment, resulting in an unsustainable top-decile loss ratio."
        ),
    }

    # Trap 3: One model for frequency and size
    trap_one_model = {
        "two_part_rmse_vs_truth": policy_metrics["freq_sev_two_part"]["rmse_vs_truth"],
        "single_tweedie_rmse_vs_truth": policy_metrics["single_tweedie"]["rmse_vs_truth"],
        "two_part_gini": headline_gini,
        "single_tweedie_gini": policy_metrics["single_tweedie"]["gini"],
        "paired_gini_diff_95ci": [round(ci_diff[0], 6), round(ci_diff[1], 6)],
        "note": (
            "Fitting a single compound Tweedie model conflates frequency drivers with "
            "severity drivers, reducing pricing discrimination and increasing pure premium error."
        ),
    }

    metrics = {
        "project": config.PROJECT_NAME,
        "data_source": config.DATA_SOURCE,
        "fast_mode": fast_mode,
        "seed": config.SEED,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "data": {
            "n_train": n_train,
            "n_test": n_test,
            "train_exposure": round(float(np.sum(train_policies.exposure)), 2),
            "train_claims": int(np.sum(train_policies.claim_count)),
            "train_losses": round(float(np.sum(train_policies.claim_amount)), 2),
            "test_exposure": round(float(np.sum(test_policies.exposure)), 2),
            "test_claims": int(np.sum(test_policies.claim_count)),
            "test_losses": round(float(np.sum(test_policies.claim_amount)), 2),
            "sha256": config.CHECKSUM_FILE.read_text(encoding="utf-8").split()[0].strip(),
        },
        "assumptions": {
            "target_loss_ratio": config.TARGET_LOSS_RATIO,
            "max_top_decile_lr": config.MAX_TOP_DECILE_LR,
            "min_gini": config.MIN_GINI,
        },
        "policies": policy_metrics,
        "headline": {
            "policy": config.HEADLINE,
            "metric": "gini_coefficient",
            "value": headline_gini,
            "ci95": [round(ci_headline[0], 4), round(ci_headline[1], 4)],
            "ci_method": f"percentile bootstrap over policies, {n_boot} resamples",
            "minus_single_tweedie": {
                "value": round(headline_gini - policy_metrics["single_tweedie"]["gini"], 4),
                "ci95": [round(ci_diff[0], 6), round(ci_diff[1], 6)],
            },
            "top_decile_loss_ratio": headline_top_decile_lr,
        },
        "gate": {
            "rule": config.GATE_RULE,
            "checks": checks,
            "passed": gate_passed,
        },
        "traps": {
            "ignoring_exposure_time": trap_exposure,
            "loss_ratio_by_decile": trap_deciles,
            "one_model_for_frequency_and_size": trap_one_model,
        },
        "caveats": [
            "Everything is simulated. Driver risk parameters, vehicle values, and accident distributions are chosen by the author.",
            "Exposure is simulated uniformly from 0.1 to 1.0 year without seasonality or cancellation correlation.",
            "Individual claims are assumed independent and follow a Gamma distribution. Real severity often has heavy Pareto tails.",
            "Features are limited to five synthetic variables without interaction terms or telematics data.",
            "Intervals resample holdout policies. They do not cover model misspecification or a changing underlying environment.",
        ],
    }

    # Error analysis: segmentation study
    segments = {}
    for seg_name, mask in [
        ("young_driver_age_under_25", test_policies.driver_age < 25),
        (
            "middle_driver_age_25_to_55",
            (test_policies.driver_age >= 25) & (test_policies.driver_age <= 55),
        ),
        ("older_driver_age_over_55", test_policies.driver_age > 55),
        ("urban_driving", test_policies.urban == 1),
        ("rural_driving", test_policies.urban == 0),
        ("newer_vehicle_age_under_5", test_policies.vehicle_age < 5),
        ("older_vehicle_age_at_least_5", test_policies.vehicle_age >= 5),
        ("high_value_vehicle_over_30k", test_policies.vehicle_value > 30000),
        ("low_value_vehicle_under_30k", test_policies.vehicle_value <= 30000),
    ]:
        seg_loss = float(np.sum(test_policies.claim_amount[mask]))
        seg_exp = float(np.sum(test_policies.exposure[mask]))
        seg_tp_prem = float(np.sum(charged_premiums["freq_sev_two_part"][mask]))
        seg_flat_prem = float(np.sum(charged_premiums["flat_baseline"][mask]))
        seg_tw_prem = float(np.sum(charged_premiums["single_tweedie"][mask]))

        segments[seg_name] = {
            "n_policies": int(np.sum(mask)),
            "exposure": round(seg_exp, 2),
            "actual_loss": round(seg_loss, 2),
            "two_part_loss_ratio": round(seg_loss / max(seg_tp_prem, 1e-6), 4),
            "single_tweedie_loss_ratio": round(seg_loss / max(seg_tw_prem, 1e-6), 4),
            "flat_loss_ratio": round(seg_loss / max(seg_flat_prem, 1e-6), 4),
        }

    error_analysis = {
        "project": config.PROJECT_NAME,
        "segments": segments,
        "decile_tables": {
            "flat_baseline": flat_deciles,
            "two_part": two_part_deciles,
        },
    }

    # Save artifacts
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(res_dir / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2)

    # Generate plot: Decile loss ratios
    fig, ax = plt.subplots(figsize=(8, 5))
    decile_nums = np.arange(1, 11)
    flat_lrs = [d["loss_ratio"] for d in flat_deciles]
    tp_lrs = [d["loss_ratio"] for d in two_part_deciles]

    ax.plot(
        decile_nums,
        flat_lrs,
        marker="o",
        linewidth=2,
        color="#d95f02",
        label="Flat pricing (adverse selection)",
    )
    ax.plot(
        decile_nums,
        tp_lrs,
        marker="s",
        linewidth=2,
        color="#1b9e77",
        label="Two-part frequency-severity (headline)",
    )
    ax.axhline(
        config.TARGET_LOSS_RATIO,
        color="#7570b3",
        linestyle="--",
        label=f"Target loss ratio ({config.TARGET_LOSS_RATIO:.2f})",
    )
    ax.axhline(
        1.0,
        color="#e7298a",
        linestyle=":",
        label="Break-even line (1.00)",
    )

    ax.set_xlabel("Predicted Risk Decile (1 = Safest, 10 = Riskiest)")
    ax.set_ylabel("Actual Loss Ratio (Claims / Charged Premium)")
    ax.set_title("Project 16: Loss Ratio Across Risk Deciles")
    ax.set_xticks(decile_nums)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(res_dir / "loss_ratio_by_decile.png", dpi=150)
    plt.close(fig)

    return metrics, error_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run insurance claims pricing pipeline")
    parser.add_argument("--fast", action="store_true", help="Run fast mode for quick test")
    parser.add_argument("--write-checksum", action="store_true", help="Write data fingerprint file")
    args = parser.parse_args()

    if args.write_checksum:
        write_fingerprint(config.CHECKSUM_FILE)
        print("Wrote checksum to", config.CHECKSUM_FILE)
        return

    metrics, _ = run_pipeline(fast_mode=args.fast)
    print("Project:", metrics["project"])
    print(
        "Headline Gini:",
        metrics["headline"]["value"],
        metrics["headline"]["ci95"],
    )
    print("Gate passed:", metrics["gate"]["passed"])
    for check_name, passed in metrics["gate"]["checks"].items():
        print(f"  {check_name}: {passed}")


if __name__ == "__main__":
    main()
