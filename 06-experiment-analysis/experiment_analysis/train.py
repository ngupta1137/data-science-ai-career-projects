"""Run every experiment and write every result file.

    python -m experiment_analysis.train           # full run
    python -m experiment_analysis.train --fast    # fewer simulated experiments, marked

Each "cell" is a set of simulated experiments with a known truth. We count how often an
analysis is wrong. Intervals describe simulation error, not uncertainty about the world.
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

from . import analysis, config, data
from .metrics import mean_ci, n_per_arm, variance_ratio_ci, wilson_ci

CAVEATS = [
    "All data is simulated with a known truth. Real experiments have effects that vary by user, interference between users, and data problems this project does not model.",
    "The outcome is one number per user with a clean pre-period metric. Real metrics are noisier and pre-period data is often missing.",
    "Tests use a normal approximation. Intervals are about the simulation, so they show how reliable a method is under these assumptions only.",
    "Only equally spaced looks and Bonferroni are measured. Alpha-spending and always-valid methods are designed, not built.",
    "No sample ratio mismatch checks, no novelty effects, no network effects, no variance from repeated measures.",
]


def _versions() -> dict:
    import pandas  # only for the version string

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
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
    """One independent, reproducible stream per named cell."""
    return np.random.default_rng([seed, zlib.crc32(label.encode("utf-8"))])


def _chunk_sizes(reps: int, chunk: int):
    done = 0
    while done < reps:
        step = min(chunk, reps - done)
        yield step
        done += step


def run_cell(seed, label, n, dist, rho, effect, reps, contamination=0.0) -> dict:
    """Simulate `reps` experiments and analyse each with and without CUPED."""
    rng = _rng(seed, label)
    keys = ("raw_diff", "raw_se", "cuped_diff", "cuped_se")
    acc = {k: [] for k in keys}
    for step in _chunk_sizes(reps, config.CHUNK):
        d = data.draw(rng, step, n, dist, rho, effect, contamination)
        raw_diff, raw_se = analysis.diff_and_se(d["y_t"], d["y_c"])
        yt, yc = analysis.cuped_adjust(d["x_t"], d["y_t"], d["x_c"], d["y_c"])
        cu_diff, cu_se = analysis.diff_and_se(yt, yc)
        for k, v in zip(keys, (raw_diff, raw_se, cu_diff, cu_se), strict=True):
            acc[k].append(v)
    return {k: np.concatenate(v) for k, v in acc.items()}


def _rate(flags: np.ndarray) -> dict:
    k, n = int(flags.sum()), int(flags.size)
    lo, hi = wilson_ci(k, n)
    return {"rate": k / n, "ci95": [lo, hi], "n": n}


def _summarize(cell: dict, effect: float) -> dict:
    z = analysis.z_crit(config.ALPHA)
    out = {}
    for name in ("raw", "cuped"):
        diff, se = cell[f"{name}_diff"], cell[f"{name}_se"]
        reject = np.abs(diff / se) > z
        covered = np.abs(diff - effect) <= z * se
        err_lo, err_hi = mean_ci(diff - effect)
        out[name] = {
            "reject_rate": _rate(reject),
            "coverage": _rate(covered),
            "mean_error": float(np.mean(diff - effect)),
            "mean_error_ci95": [err_lo, err_hi],
            "mean_ci_width": float(np.mean(2 * z * se)),
        }
    return out


def fixed_horizon_grid(seed, reps) -> list[dict]:
    cells = []
    for dist in config.GRID_DIST:
        for n in config.GRID_N:
            is_head = n == config.HEADLINE_N and dist == config.HEADLINE_DIST
            aa_reps = max(reps, config.HEADLINE_AA_REPS) if is_head and reps >= 1000 else reps
            aa = run_cell(seed, f"aa|{dist}|{n}", n, dist, config.RHO, 0.0, aa_reps)
            ab = run_cell(seed, f"ab|{dist}|{n}", n, dist, config.RHO, config.TRUE_EFFECT, reps)
            aa_s, ab_s = _summarize(aa, 0.0), _summarize(ab, config.TRUE_EFFECT)
            cells.append(
                {
                    "n_per_arm": n,
                    "dist": dist,
                    "rho": config.RHO,
                    "aa": {m: aa_s[m]["reject_rate"] for m in aa_s},
                    "with_effect": {
                        m: {
                            "power": ab_s[m]["reject_rate"],
                            "coverage": ab_s[m]["coverage"],
                            "mean_error": ab_s[m]["mean_error"],
                            "mean_error_ci95": ab_s[m]["mean_error_ci95"],
                            "mean_ci_width": ab_s[m]["mean_ci_width"],
                        }
                        for m in ab_s
                    },
                }
            )
    return cells


def cuped_by_rho(seed, reps) -> list[dict]:
    rows = []
    for rho in config.RHO_GRID:
        cell = run_cell(
            seed,
            f"rho|{rho}",
            config.HEADLINE_N,
            config.HEADLINE_DIST,
            rho,
            config.TRUE_EFFECT,
            reps,
        )
        point, lo, hi = variance_ratio_ci(
            cell["cuped_diff"], cell["raw_diff"], config.N_BOOT if reps >= 1000 else 200, seed
        )
        s = _summarize(cell, config.TRUE_EFFECT)
        rows.append(
            {
                "rho": rho,
                "variance_ratio": point,
                "variance_ratio_ci95": [lo, hi],
                "theory_ratio": 1.0 - rho**2,
                "cuped_mean_error": s["cuped"]["mean_error"],
                "cuped_mean_error_ci95": s["cuped"]["mean_error_ci95"],
            }
        )
    return rows


def power_check(seed, reps) -> dict:
    d, rho = config.TRUE_EFFECT, config.RHO
    n_raw = n_per_arm(d, config.POWER_TARGET, config.ALPHA)
    n_cuped = n_per_arm(d, config.POWER_TARGET, config.ALPHA, 1 - rho**2)
    z = analysis.z_crit(config.ALPHA)
    out = {"effect": d, "target_power": config.POWER_TARGET, "rho": rho}
    for name, n, key in (("raw", n_raw, "raw"), ("cuped", n_cuped, "cuped")):
        cell = run_cell(seed, f"power|{name}", n, "normal", rho, d, reps)
        flags = np.abs(cell[f"{key}_diff"] / cell[f"{key}_se"]) > z
        out[name] = {"n_per_arm": n, "simulated_power": _rate(flags)}
    out["users_saved_percent"] = 100.0 * (1.0 - n_cuped / n_raw)
    return out


def peeking(seed, reps) -> list[dict]:
    rows = []
    n = config.PEEK_N
    for k in config.PEEK_LOOKS:
        rng = _rng(seed, f"peek|{k}")
        naive, bonf = [], []
        for step in _chunk_sizes(reps, config.CHUNK):
            d = data.draw(rng, step, n, "normal", 0.0, 0.0)
            zs = analysis.looks_z(d["y_t"], d["y_c"], k)
            naive.append((np.abs(zs) > analysis.z_crit(config.ALPHA)).any(axis=1))
            bonf.append((np.abs(zs) > analysis.z_crit(config.ALPHA / k)).any(axis=1))
        rows.append(
            {
                "looks": k,
                "naive_false_positive": _rate(np.concatenate(naive)),
                "bonferroni_false_positive": _rate(np.concatenate(bonf)),
            }
        )
    return rows


def multiple_metrics(seed, reps) -> dict:
    m, n = config.MULTI_METRICS, config.MULTI_N
    out = {"metrics": m, "n_per_arm": n, "true_effect": config.MULTI_TRUE_EFFECT}
    for scenario, n_true in (("all_null", 0), ("two_true", config.MULTI_N_TRUE)):
        effects = [config.MULTI_TRUE_EFFECT] * n_true + [0.0] * (m - n_true)
        rng = _rng(seed, f"multi|{scenario}")
        flags = {"uncorrected": [], "bonferroni": [], "holm": []}
        for step in _chunk_sizes(reps, config.CHUNK // 2):
            t, c = data.draw_metrics(rng, step, n, effects)
            diff, se = analysis.diff_and_se(t, c)
            p = analysis.p_value(diff, se)
            flags["uncorrected"].append(p <= config.ALPHA)
            flags["bonferroni"].append(analysis.bonferroni_reject(p, config.ALPHA))
            flags["holm"].append(analysis.holm_reject(p, config.ALPHA))
        res = {}
        for name, parts in flags.items():
            rej = np.concatenate(parts)
            row = {"family_wise_error": _rate(rej[:, n_true:].any(axis=1))}
            if n_true:
                row["power_per_true_metric"] = _rate(rej[:, :n_true].ravel())
            res[name] = row
        out[scenario] = res
    return out


def contamination_trap(seed, reps) -> dict:
    kw = {"n": config.HEADLINE_N, "dist": "normal", "rho": config.RHO, "reps": reps}
    clean = run_cell(seed, "contam|clean", effect=config.TRUE_EFFECT, **kw)
    dirty = run_cell(
        seed,
        "contam|dirty",
        effect=config.TRUE_EFFECT,
        contamination=config.CONTAMINATION_SHIFT,
        **kw,
    )
    s_clean, s_dirty = _summarize(clean, config.TRUE_EFFECT), _summarize(dirty, config.TRUE_EFFECT)
    return {
        "true_effect": config.TRUE_EFFECT,
        "contamination_shift": config.CONTAMINATION_SHIFT,
        "cuped_mean_error_clean_covariate": s_clean["cuped"]["mean_error"],
        "cuped_mean_error_contaminated_covariate": s_dirty["cuped"]["mean_error"],
        "cuped_coverage_clean": s_clean["cuped"]["coverage"],
        "cuped_coverage_contaminated": s_dirty["cuped"]["coverage"],
        "note": (
            "The covariate moves with treatment, so CUPED removes part of the real effect. "
            "A covariate must be fixed before assignment."
        ),
    }


def _save_plot(peek: list[dict], results_dir: Path) -> None:
    ks = [r["looks"] for r in peek]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.38
    xs = np.arange(len(ks))
    ax.bar(xs - width / 2, [r["naive_false_positive"]["rate"] for r in peek], width, label="naive")
    ax.bar(
        xs + width / 2,
        [r["bonferroni_false_positive"]["rate"] for r in peek],
        width,
        label="Bonferroni",
    )
    ax.axhline(config.ALPHA, color="black", linewidth=1, linestyle="--", label="nominal 0.05")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Number of looks at the data")
    ax.set_ylabel("A/A false positive rate")
    ax.set_title("Peeking inflates false positives")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "peeking_false_positives.png", dpi=120)
    plt.close(fig)


def run(
    results_dir: Path,
    fast: bool = False,
    seed: int = config.SEED,
    data_sha256: str | None = None,
    reps: int | None = None,
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    reps = reps or (200 if fast else config.N_REPS)

    grid = fixed_horizon_grid(seed, reps)
    rho_rows = cuped_by_rho(seed, reps)
    power = power_check(seed, reps)
    peek = peeking(seed, reps)
    multi = multiple_metrics(seed, reps)
    contam = contamination_trap(seed, reps)

    head_cell = next(
        c for c in grid if c["n_per_arm"] == config.HEADLINE_N and c["dist"] == config.HEADLINE_DIST
    )
    head_rho = next(r for r in rho_rows if r["rho"] == config.RHO)
    fpr = head_cell["aa"]["cuped"]
    bias_ci = head_cell["with_effect"]["cuped"]["mean_error_ci95"]
    vr_ci = head_rho["variance_ratio_ci95"]
    holm_lo = multi["all_null"]["holm"]["family_wise_error"]["ci95"][0]
    checks = {
        "aa_false_positive_contains_nominal": bool(
            fpr["ci95"][0] <= config.ALPHA <= fpr["ci95"][1]
        ),
        "cuped_unbiased": bool(bias_ci[0] <= 0.0 <= bias_ci[1]),
        "cuped_variance_ratio_below_one": bool(vr_ci[1] < 1.0),
        "holm_family_wise_error_controlled": bool(holm_lo <= config.ALPHA),
    }

    # Error analysis: where does a method fail?
    flagged = []
    for c in grid:
        for method in ("raw", "cuped"):
            f = c["aa"][method]
            if not f["ci95"][0] <= config.ALPHA <= f["ci95"][1]:
                flagged.append(
                    {
                        "n_per_arm": c["n_per_arm"],
                        "dist": c["dist"],
                        "method": method,
                        "false_positive_rate": f["rate"],
                        "ci95": f["ci95"],
                    }
                )
    low_cov = []
    for c in grid:
        for method in ("raw", "cuped"):
            cov = c["with_effect"][method]["coverage"]
            if cov["ci95"][1] < 0.95:
                low_cov.append(
                    {
                        "n_per_arm": c["n_per_arm"],
                        "dist": c["dist"],
                        "method": method,
                        "coverage": cov["rate"],
                        "ci95": cov["ci95"],
                    }
                )
    analysis_out = {
        "false_positive_rate_outside_nominal": flagged,
        "interval_coverage_below_95": low_cov,
        "grid": grid,
    }

    traps = {
        "peeking": {
            "looks": 10,
            "naive_false_positive": next(r for r in peek if r["looks"] == 10)[
                "naive_false_positive"
            ],
            "bonferroni_false_positive": next(r for r in peek if r["looks"] == 10)[
                "bonferroni_false_positive"
            ],
            "note": "Stopping at the first significant look in an A/A test.",
        },
        "uncorrected_metrics": {
            "metrics": config.MULTI_METRICS,
            "uncorrected_family_wise_error": multi["all_null"]["uncorrected"]["family_wise_error"],
            "holm_family_wise_error": multi["all_null"]["holm"]["family_wise_error"],
            "note": "Reading ten metrics at the 0.05 level and acting on any that moves.",
        },
        "heavy_tail_small_sample": {
            "n_per_arm": min(config.GRID_N),
            "lognormal_raw_false_positive": next(
                c for c in grid if c["n_per_arm"] == min(config.GRID_N) and c["dist"] == "lognormal"
            )["aa"]["raw"],
            "normal_raw_false_positive": next(
                c for c in grid if c["n_per_arm"] == min(config.GRID_N) and c["dist"] == "normal"
            )["aa"]["raw"],
            "note": "A normal approximation on skewed data with few users.",
        },
        "contaminated_covariate": contam,
    }

    metrics = {
        "project": config.PROJECT_NAME,
        "data_source": config.DATA_SOURCE,
        "fast_mode": bool(fast),
        "seed": seed,
        "versions": _versions(),
        "data": {
            "experiments_per_cell": reps,
            "headline_aa_experiments": max(reps, config.HEADLINE_AA_REPS) if reps >= 1000 else reps,
            "sha256": data_sha256,
        },
        "headline": {
            "analysis": "cuped_fixed_horizon",
            "metric": "variance_ratio_cuped_over_raw",
            "n_per_arm": config.HEADLINE_N,
            "rho": config.RHO,
            "value": head_rho["variance_ratio"],
            "ci95": vr_ci,
            "theory": head_rho["theory_ratio"],
            "ci_method": f"bootstrap over {reps} simulated experiments",
        },
        "fixed_horizon_grid": grid,
        "cuped_by_rho": rho_rows,
        "power": power,
        "peeking": peek,
        "multiple_metrics": multi,
        "gate": {"rule": config.GATE_RULE, "checks": checks, "passed": bool(all(checks.values()))},
        "traps": traps,
        "caveats": CAVEATS,
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(_clean(metrics), indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "error_analysis.json").write_text(
        json.dumps(_clean(analysis_out), indent=2) + "\n", encoding="utf-8"
    )
    _save_plot(peek, results_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simulated experiments.")
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
    print(f"CUPED variance ratio {h['value']:.3f} (theory {h['theory']:.3f}), interval {h['ci95']}")
    print(f"gate: {metrics['gate']['checks']}")
    print(f"gate passed: {metrics['gate']['passed']}")
    print(f"wrote results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
