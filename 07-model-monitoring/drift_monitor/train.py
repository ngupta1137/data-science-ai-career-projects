"""Calibrate monitors on null streams, score them on fresh drifting streams, write results.

    python -m drift_monitor.train           # full run, about a minute
    python -m drift_monitor.train --fast    # few streams, marked

Streams are simulated so the true drift onset is known. Thresholds come from null streams
only. The headline monitor is chosen on dev streams. Test streams are scored once.
"""

from __future__ import annotations

import argparse
import json
import platform
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn

from . import config, data
from .metrics import stream_stats, wilson_ci

CAVEATS = [
    "All data is simulated. Drift sizes and the model are assumptions, not measurements of a real system.",
    "Streams are independent weekly batches of 500 rows with one drift event. Real drift is messier and several things move at once.",
    "Monitors compare each week to a fixed reference. They do not adapt, retrain, or handle seasonality.",
    "The KS monitor covers numeric features only, so it cannot see the categorical shift. Part of the gap between PSI and KS is that design choice, not the statistics.",
    "Thresholds are calibrated for a 52 week horizon. A longer horizon needs a higher threshold for the same chance of any false alarm.",
    "Thresholds are calibrated to a 5% chance of any false alarm in a 52 week null stream. Choose your own budget for your own cost of a false alarm.",
    "The model is a logistic regression on five numeric features and one categorical feature. Monitor behaviour on high-dimensional or text inputs is not tested.",
]


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
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


def _rng(seed: int, label: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(label.encode("utf-8"))])


def collect(ref, seed, label, scenario, n, size=config.BATCH_SIZE, magnitude=None) -> dict:
    """Statistics for `n` independent streams. Each value has shape (n, weeks)."""
    rng = _rng(seed, label)
    rows = [
        stream_stats(ref, data.draw_stream(rng, scenario, size=size, magnitude=magnitude))
        for _ in range(n)
    ]
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def calibrate(null_stats: dict, budget: float = config.BUDGET) -> dict:
    """Threshold per monitor: the (1 - budget) quantile of each null stream's weekly maximum."""
    return {
        m: float(np.quantile(np.nanmax(null_stats[m], axis=1), 1 - budget, method="higher"))
        for m in config.MONITORS
    }


def monitor_specs(thresholds: dict) -> dict:
    """name -> (statistic key, threshold, label lag in weeks)."""
    specs = {
        m: (m, thresholds[m], config.LABEL_LAG if m == "auc_drop" else 0) for m in config.MONITORS
    }
    for t in config.FIXED_PSI_THRESHOLDS:
        specs[f"psi_fixed_{t}"] = ("psi_max", t, 0)
    return specs


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson_ci(int(k), int(n))
    return {"rate": k / n, "ci95": [lo, hi], "n": int(n)}


def evaluate(stat: np.ndarray, thr: float, lag: int, onset: int | None) -> dict:
    """Score one monitor on streams of one scenario. onset None means a null scenario."""
    alarm = stat > thr  # NaN compares False
    n = len(alarm)
    if onset is None:
        return {"false_alarm": _rate(alarm.any(axis=1).sum(), n)}
    post = alarm[:, onset:]
    detected = post.any(axis=1)
    delay = post.argmax(axis=1)[detected] + lag
    return {
        "detection": _rate(detected.sum(), n),
        "median_delay_weeks": float(np.median(delay)) if detected.any() else None,
        "pre_onset_false_alarm": _rate(alarm[:, :onset].any(axis=1).sum(), n),
    }


def _save_plot(sweep: list[dict], results_dir: Path) -> None:
    shifts = [r["shift_sd"] for r in sweep]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for key, label in (("headline", "calibrated headline"), ("psi_fixed_0.2", "PSI > 0.2")):
        ax.plot(shifts, [r[key]["detection"]["rate"] for r in sweep], marker="o", label=label)
    ax.axhline(
        config.DETECTION_FLOOR, color="black", linestyle="--", linewidth=1, label="gate floor"
    )
    ax.set_xlabel("Mean shift of one feature (standard deviations)")
    ax.set_ylabel("Detection rate within 26 weeks")
    ax.set_title("Smallest shift each monitor catches")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "detection_by_shift_size.png", dpi=120)
    plt.close(fig)


def run(
    results_dir: Path,
    fast: bool = False,
    seed: int = config.SEED,
    data_sha256: str | None = None,
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    n_cal, n_dev, n_test, n_sweep, n_trap, n_supp = (
        (20, 10, 30, 10, 30, 40)
        if fast
        else (
            config.N_CAL,
            config.N_DEV,
            config.N_TEST,
            config.N_SWEEP,
            config.N_TRAP,
            config.N_SUPP_NULL,
        )
    )
    onset = config.ONSET_WEEK
    ref = data.make_reference(seed)

    cal = collect(ref, seed, "cal", "null", n_cal)
    thresholds = calibrate(cal)
    specs = monitor_specs(thresholds)

    # Pick the headline label-free monitor on dev streams.
    dev_detection = {m: [] for m in config.LABEL_FREE}
    for sc in config.FEATURE_SCENARIOS:
        st = collect(ref, seed, f"dev|{sc}", sc, n_dev)
        for m in config.LABEL_FREE:
            key, thr, lag = specs[m]
            dev_detection[m].append(evaluate(st[key], thr, lag, onset)["detection"]["rate"])
    dev_mean = {m: float(np.mean(v)) for m, v in dev_detection.items()}
    best = max(dev_mean.values())
    headline = next(m for m in config.LABEL_FREE if dev_mean[m] == best)

    # Test.
    results: dict[str, dict] = {m: {} for m in specs}
    test_stats = {}
    for sc in config.SCENARIOS:
        st = collect(ref, seed, f"test|{sc}", sc, n_test)
        test_stats[sc] = st
        for m, (key, thr, lag) in specs.items():
            results[m][sc] = evaluate(st[key], thr, lag, None if sc == "null" else onset)

    head_null = results[headline]["null"]["false_alarm"]
    head_abrupt = results[headline]["covariate_abrupt"]["detection"]
    checks = {
        "false_alarm_within_limit": bool(head_null["ci95"][1] <= config.FALSE_ALARM_LIMIT),
        "abrupt_shift_detected": bool(head_abrupt["ci95"][0] >= config.DETECTION_FLOOR),
    }

    # Follow-up, run after the gate result was known. Not part of the gate.
    supp_stats = collect(ref, seed, "supp|null", "null", n_supp)
    supplementary = {
        m: evaluate(supp_stats[specs[m][0]], specs[m][1], specs[m][2], None)["false_alarm"]
        for m in (*config.MONITORS, "psi_fixed_0.1", "psi_fixed_0.2")
    }

    # Sweep: how small a shift can the monitor see?
    sweep = []
    for shift in config.SWEEP_SHIFTS:
        st = collect(ref, seed, f"sweep|{shift}", "covariate_abrupt", n_sweep, magnitude=shift)
        row = {"shift_sd": shift}
        for name, m in (("headline", headline), ("psi_fixed_0.2", "psi_fixed_0.2")):
            key, thr, lag = specs[m]
            row[name] = evaluate(st[key], thr, lag, onset)
        sweep.append(row)
    reaching = [
        r["shift_sd"] for r in sweep if r["headline"]["detection"]["rate"] >= config.DETECTION_FLOOR
    ]
    smallest = min(reaching) if reaching else None

    # Traps.
    small = collect(ref, seed, "trap|small_batch", "null", n_trap, size=config.SMALL_BATCH_SIZE)
    fixed = {}
    for t in config.FIXED_PSI_THRESHOLDS:
        name = f"psi_fixed_{t}"
        fixed[str(t)] = {
            "false_alarm_batch_500": results[name]["null"]["false_alarm"],
            "false_alarm_batch_100": evaluate(small["psi_max"], t, 0, None)["false_alarm"],
        }
    key, thr, lag = specs[headline]
    traps = {
        "fixed_psi_rule_of_thumb": {
            "by_threshold": fixed,
            "detection_of_small_shift_psi_0.2": results["psi_fixed_0.2"]["covariate_small"][
                "detection"
            ],
            "detection_of_small_shift_calibrated_headline": results[headline]["covariate_small"][
                "detection"
            ],
            "note": "A fixed PSI cutoff ignores batch size and misses small shifts.",
        },
        "threshold_moved_across_batch_size": {
            "headline_false_alarm_batch_500": head_null,
            "headline_false_alarm_batch_100": evaluate(small[key], thr, lag, None)["false_alarm"],
            "note": "A threshold calibrated at one batch size, used at another.",
        },
        "label_free_blind_to_concept_drift": {
            "headline_detection": results[headline]["concept_drift"]["detection"],
            "delayed_label_auc_detection": results["auc_drop"]["concept_drift"]["detection"],
            "delayed_label_median_delay_weeks": results["auc_drop"]["concept_drift"][
                "median_delay_weeks"
            ],
            "note": "The inputs did not move. Only the link between inputs and outcome did.",
        },
    }

    analysis = {
        "headline_monitor": headline,
        "detection_matrix": {
            m: {
                sc: results[m][sc].get("detection", {}).get("rate")
                for sc in config.SCENARIOS
                if sc != "null"
            }
            for m in results
        },
        "headline_misses": [
            {"scenario": sc, "detection": results[headline][sc]["detection"]}
            for sc in config.SCENARIOS
            if sc != "null" and results[headline][sc]["detection"]["rate"] < 0.5
        ],
        "smallest_shift_reaching_floor": smallest,
        "sweep": sweep,
    }
    metrics = {
        "project": config.PROJECT_NAME,
        "data_source": config.DATA_SOURCE,
        "fast_mode": bool(fast),
        "seed": seed,
        "versions": _versions(),
        "data": {
            "weeks": config.N_WEEKS,
            "batch_size": config.BATCH_SIZE,
            "onset_week": onset,
            "calibration_streams": n_cal,
            "dev_streams_per_scenario": n_dev,
            "test_streams_per_scenario": n_test,
            "reference_auc": ref.ref_auc,
            "sha256": data_sha256,
        },
        "thresholds": thresholds,
        "selection": {"dev_mean_detection": dev_mean, "headline": headline},
        "headline": {
            "monitor": headline,
            "metric": "abrupt_covariate_shift_detection_rate",
            "value": head_abrupt["rate"],
            "ci95": head_abrupt["ci95"],
            "ci_method": f"Wilson interval over {n_test} test streams",
            "false_alarm_rate": head_null,
        },
        "results": results,
        "supplementary_null_check": {
            "streams": n_supp,
            "status": "post hoc follow-up, run after the gate result was known; not part of the gate",
            "false_alarm": supplementary,
        },
        "gate": {"rule": config.GATE_RULE, "checks": checks, "passed": bool(all(checks.values()))},
        "traps": traps,
        "caveats": CAVEATS,
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(_clean(metrics), indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "error_analysis.json").write_text(
        json.dumps(_clean(analysis), indent=2) + "\n", encoding="utf-8"
    )
    _save_plot(sweep, results_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate and score drift monitors.")
    parser.add_argument("--fast", action="store_true", help="smoke-test mode, results are marked")
    parser.add_argument(
        "--write-checksum", action="store_true", help="write the generator fingerprint"
    )
    args = parser.parse_args()
    if args.write_checksum:
        data.write_fingerprint()
    digest = data.verify_fingerprint()
    metrics = run(config.RESULTS_DIR, fast=args.fast, data_sha256=digest)
    h = metrics["headline"]
    print(f"headline {h['monitor']}: abrupt shift detection {h['value']:.3f} {h['ci95']}")
    print(f"null false alarm {h['false_alarm_rate']['rate']:.3f} {h['false_alarm_rate']['ci95']}")
    print(f"gate: {metrics['gate']['checks']} passed: {metrics['gate']['passed']}")


if __name__ == "__main__":
    main()
