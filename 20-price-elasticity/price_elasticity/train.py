"""Full training, evaluation, and reporting pipeline for price elasticity and profit optimization."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, data, metrics, models

CAVEATS = [
    "Demand is modeled as a constant elasticity log-log specification with multiplicative demand shocks.",
    "Wholesale cost shocks serve as a valid instrument assuming competitive commodity sourcing.",
    "Cross-price elasticity substitution between competing internal catalog items is assumed zero.",
    "Dynamic customer purchase deferral and price anticipation are not modeled in weekly horizons.",
]


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def run_pipeline(
    results_dir: Path | None = None,
    fast_mode: bool = False,
) -> tuple[dict, dict]:
    """Execute the full price elasticity evaluation pipeline."""
    out_dir = results_dir or config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train_prods = config.FAST_TRAIN_PRODUCTS if fast_mode else config.TRAIN_PRODUCTS
    n_test_prods = config.FAST_TEST_PRODUCTS if fast_mode else config.TEST_PRODUCTS
    n_weeks = config.FAST_WEEKS_PER_PRODUCT if fast_mode else config.WEEKS_PER_PRODUCT

    train_ds = data.generate_elasticity_data(
        n_products=n_train_prods, n_weeks=n_weeks, seed=config.SEED
    )
    test_ds = data.generate_elasticity_data(
        n_products=n_test_prods, n_weeks=n_weeks, seed=config.SEED + 100
    )

    # Checksum validation outside fast mode
    if not fast_mode and config.CHECKSUM_FILE.exists():
        expected_sha = config.CHECKSUM_FILE.read_text(encoding="utf-8").strip().split()[0]
        if train_ds.sha256 != expected_sha:
            raise RuntimeError(
                f"Data fingerprint {train_ds.sha256} does not match expected checksum {expected_sha}."
            )

    # Fit models on training data
    naive_model = models.NaiveOLS().fit(train_ds)
    two_sls_model = models.TwoStageLeastSquares().fit(train_ds)
    oracle_model = models.OracleModel().fit(train_ds)

    # Evaluate pricing policies on holdout test cohort
    naive_eval = metrics.simulate_pricing_policy(naive_model, test_ds)
    two_sls_eval = metrics.simulate_pricing_policy(two_sls_model, test_ds)
    oracle_eval = metrics.simulate_pricing_policy(oracle_model, test_ds)

    # Bootstrap 95% CI for 2SLS elasticity
    n_boot = 50 if fast_mode else 250
    ci_95 = metrics.bootstrap_elasticity_ci(train_ds, n_boot=n_boot, seed=config.SEED)

    # Decision gate check
    true_test_betas = [o.true_elasticity for o in test_ds.observations]
    true_avg_beta = float(np.mean(true_test_betas))

    gate_result = metrics.evaluate_decision_gate(
        two_sls_eval=two_sls_eval,
        naive_eval=naive_eval,
        oracle_eval=oracle_eval,
        ci_95=ci_95,
        true_elasticity=true_avg_beta,
    )

    # Measure traps
    traps = metrics.compute_traps(
        dataset=test_ds,
        naive_eval=naive_eval,
        two_sls_eval=two_sls_eval,
        oracle_eval=oracle_eval,
        f_stat=two_sls_model.first_stage_f_stat or 0.0,
    )

    # Headline block for metrics.json
    headline_block = {
        "model": config.HEADLINE_MODEL,
        "metric": "elasticity_estimation_error",
        "value": float(round(gate_result["elasticity_error_2sls"], 4)),
        "ci_95": [float(round(ci_95[0], 4)), float(round(ci_95[1], 4))],
        "ci_method": "empirical_bootstrap_percentile",
    }

    # Generate plot
    plot_path = out_dir / "elasticity_curve.png"
    _plot_demand_and_profit(naive_model, two_sls_model, oracle_model, plot_path)

    models_summary = {
        "oracle": {
            "elasticity": oracle_eval["estimated_elasticity"],
            "total_profit": oracle_eval["total_profit"],
            "mean_price": oracle_eval["mean_price"],
            "mean_quantity": oracle_eval["mean_quantity"],
            "category_profits": oracle_eval["category_profits"],
        },
        "naive_ols": {
            "elasticity": naive_eval["estimated_elasticity"],
            "estimation_error": gate_result["elasticity_error_naive"],
            "total_profit": naive_eval["total_profit"],
            "profit_recovery_ratio": gate_result["profit_recovery_ratio_naive"],
            "mean_price": naive_eval["mean_price"],
            "mean_quantity": naive_eval["mean_quantity"],
            "category_profits": naive_eval["category_profits"],
        },
        "two_stage_least_squares": {
            "elasticity": two_sls_eval["estimated_elasticity"],
            "estimation_error": gate_result["elasticity_error_2sls"],
            "first_stage_f_statistic": two_sls_model.first_stage_f_stat,
            "total_profit": two_sls_eval["total_profit"],
            "profit_recovery_ratio": gate_result["profit_recovery_ratio_2sls"],
            "mean_price": two_sls_eval["mean_price"],
            "mean_quantity": two_sls_eval["mean_quantity"],
            "category_profits": two_sls_eval["category_profits"],
        },
    }

    metrics_payload = {
        "project": "20-price-elasticity",
        "data_source": "simulated_known_truth",
        "fast_mode": fast_mode,
        "seed": config.SEED,
        "versions": _versions(),
        "data": {
            "train_products": n_train_prods,
            "test_products": n_test_prods,
            "weeks_per_product": n_weeks,
            "train_observations": train_ds.total_observations,
            "test_observations": test_ds.total_observations,
            "category_distribution": test_ds.category_counts,
            "mean_price": test_ds.mean_price,
            "mean_quantity": test_ds.mean_quantity,
            "sha256": train_ds.sha256,
        },
        "assumptions": {
            "max_elasticity_error": config.MAX_ELASTICITY_ERROR,
            "min_profit_recovery": config.MIN_PROFIT_RECOVERY,
            "max_naive_recovery": config.MAX_NAIVE_RECOVERY,
            "min_first_stage_f": config.MIN_FIRST_STAGE_F,
        },
        "headline": headline_block,
        "gate": gate_result,
        "models": models_summary,
        "traps": traps,
        "caveats": CAVEATS,
    }

    error_analysis_payload = {
        "project": "20-price-elasticity",
        "fast_mode": fast_mode,
        "category_breakdown": {
            cat: {
                "oracle_profit": round(oracle_eval["category_profits"][cat]["total_profit"], 2),
                "naive_profit": round(naive_eval["category_profits"][cat]["total_profit"], 2),
                "two_sls_profit": round(two_sls_eval["category_profits"][cat]["total_profit"], 2),
                "naive_recovery_pct": round(
                    naive_eval["category_profits"][cat]["total_profit"]
                    / oracle_eval["category_profits"][cat]["total_profit"]
                    * 100.0,
                    2,
                ),
                "two_sls_recovery_pct": round(
                    two_sls_eval["category_profits"][cat]["total_profit"]
                    / oracle_eval["category_profits"][cat]["total_profit"]
                    * 100.0,
                    2,
                ),
            }
            for cat in config.PRODUCT_CATEGORIES
        },
        "post_hoc_study": {
            "status": "designed_after_results_does_not_change_verdict",
            "observation": (
                "Endogeneity attenuation bias is strongest in product categories with volatile "
                "demand surges (e.g., Electronics), where dynamic markups produce severe OLS error."
            ),
        },
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (out_dir / "error_analysis.json").write_text(
        json.dumps(error_analysis_payload, indent=2), encoding="utf-8"
    )

    return metrics_payload, error_analysis_payload


def _plot_demand_and_profit(naive_m, two_sls_m, oracle_m, out_path: Path) -> None:
    """Generate side-by-side plot of estimated demand curves and gross profit response."""
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    prices = np.linspace(25.0, 90.0, 100)
    base_cost = 30.0

    # Demand curves: Q(P) = exp(alpha + beta * ln(P))
    q_oracle = np.exp(8.5 + oracle_m.elasticity * np.log(prices))
    q_naive = np.exp(8.5 + naive_m.elasticity * np.log(prices))
    q_2sls = np.exp(8.5 + two_sls_m.elasticity * np.log(prices))

    ax1.plot(
        prices,
        q_oracle,
        "k--",
        label=f"True Demand (beta={oracle_m.elasticity:.2f})",
        linewidth=2,
    )
    ax1.plot(
        prices,
        q_naive,
        "r-",
        label=f"Naive OLS (beta={naive_m.elasticity:.2f})",
        linewidth=1.8,
    )
    ax1.plot(
        prices,
        q_2sls,
        "g-",
        label=f"2SLS IV (beta={two_sls_m.elasticity:.2f})",
        linewidth=2,
    )
    ax1.set_xlabel("Retail Price ($)")
    ax1.set_ylabel("Expected Sales Volume")
    ax1.set_title("Demand Curve Estimation")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Gross Profit curves: Profit(P) = (P - cost) * True_Q(P)
    profit_actual = (prices - base_cost) * q_oracle
    ax2.plot(prices, profit_actual, "b-", label="True Gross Profit ($)", linewidth=2)

    p_opt_oracle = oracle_m.recommend_price(base_cost)
    p_opt_2sls = two_sls_m.recommend_price(base_cost)
    p_opt_naive = naive_m.recommend_price(base_cost)

    ax2.axvline(
        p_opt_oracle,
        color="k",
        linestyle="--",
        label=f"Oracle Price (${p_opt_oracle:.1f})",
    )
    ax2.axvline(
        p_opt_2sls,
        color="g",
        linestyle="-",
        label=f"2SLS Price (${p_opt_2sls:.1f})",
    )
    ax2.axvline(
        p_opt_naive,
        color="r",
        linestyle="-",
        label=f"Naive Price (${p_opt_naive:.1f})",
    )

    ax2.set_xlabel("Retail Price ($)")
    ax2.set_ylabel("True Gross Profit ($)")
    ax2.set_title("Profit-Maximizing Retail Pricing")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run price elasticity and pricing optimization pipeline."
    )
    parser.add_argument(
        "--fast", action="store_true", help="Run quick smoke test with reduced dataset."
    )
    parser.add_argument(
        "--write-checksum",
        action="store_true",
        help="Write dataset SHA-256 to CHECKSUMS.txt.",
    )
    args = parser.parse_args()

    metrics_res, _ = run_pipeline(fast_mode=args.fast)

    if args.write_checksum:
        sha = metrics_res["data"]["sha256"]
        config.CHECKSUM_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.CHECKSUM_FILE.write_text(
            f"{sha}  simulated_elasticity_transactions\n", encoding="utf-8"
        )
        print(f"Wrote checksum {sha} to {config.CHECKSUM_FILE}")

    print(f"Pipeline complete. Gate passed: {metrics_res['gate']['passed']}")


if __name__ == "__main__":
    main()
