"""Fit scorers, choose thresholds on dev, audit on test, write every result file.

    python -m fairness_audit.train          # full run
    python -m fairness_audit.train --fast   # fewer bootstrap draws, marked

Group A is the larger group, group B the smaller. Gaps are B minus A.
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
import pandas as pd
import scipy
import sklearn

from . import config, data, models
from .metrics import auc, ci95, disparities, group_report, stratified_bootstrap

CAVEATS = [
    "All data is simulated with mechanisms chosen by the author. The sizes of the gaps describe this setup, not any real lending or hiring system.",
    "Two groups only. No intersections, no more than one protected attribute, and no small-subgroup problem beyond group B being a quarter of the data.",
    "The group label is observed and correct. In practice it is often missing, inferred, or legally off limits.",
    "The 0.10 tolerance and the cost of a bad approval are assumptions. Change them and the verdict can change; the sensitivity table shows how.",
    "Equal opportunity is one definition among several. Satisfying it does not make a policy fair in a legal or ethical sense. This project gives no legal advice and no compliance determination.",
    "The outcome label is taken as the truth. In real data it is often itself a product of past decisions, which this project does not model.",
]


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
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


def dev_size_study(scorer, s_test, y, g, seed: int, reps: int, big_n: int) -> list[dict]:
    """How well do dev-tuned group thresholds transfer, as the dev set grows?

    Post hoc: added after the main run showed the thresholds did not transfer. Each set of
    thresholds is scored twice: on the one fixed test set (which carries its own sampling
    error) and on a large fresh set that approximates the population.
    """
    big = data.generate(_rng(seed, "bigeval"), big_n)
    s_big, y_big, g_big = scorer(big), big["y"].to_numpy(), big["g"].to_numpy()
    rows = []
    for size in config.DEV_SIZES:
        gaps, utils, big_gaps = [], [], []
        for r in range(reps):
            dv = data.generate(_rng(seed, f"devsize|{size}|{r}"), size)
            s, yv, gv = scorer(dv), dv["y"].to_numpy(), dv["g"].to_numpy()
            t = models.best_threshold(s, yv)
            target = float((s >= t)[yv == 1].mean())
            thr = models.group_thresholds(s, yv, gv, target)
            st = policy_stats(y, models.apply_thresholds(s_test, g, thr), g)
            utils.append(st[0])
            gaps.append(st[1])
            big_gaps.append(
                policy_stats(y_big, models.apply_thresholds(s_big, g_big, thr), g_big)[1]
            )
        gaps, big_gaps = np.asarray(gaps), np.asarray(big_gaps)
        rows.append(
            {
                "dev_rows": size,
                "replications": reps,
                "mean_test_tpr_gap": float(gaps.mean()),
                "sd_test_tpr_gap": float(gaps.std(ddof=1)),
                "mean_large_eval_tpr_gap": float(big_gaps.mean()),
                "sd_large_eval_tpr_gap": float(big_gaps.std(ddof=1)),
                "share_within_tolerance_large_eval": float(
                    (np.abs(big_gaps) <= config.TPR_GAP_TOL).mean()
                ),
                "share_within_tolerance": float((np.abs(gaps) <= config.TPR_GAP_TOL).mean()),
                "mean_test_utility": float(np.mean(utils)),
            }
        )
    return rows


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def policy_stats(y, approve, g, cost_bad=config.COST_BAD) -> np.ndarray:
    """[utility, tpr_gap, fpr_gap, ppv_gap, selection_ratio] for one set of rows."""
    a, b = g == 0, g == 1
    tpr = [_mean(approve[m & (y == 1)]) for m in (a, b)]
    fpr = [_mean(approve[m & (y == 0)]) for m in (a, b)]
    ppv = [_mean(y[m & approve]) for m in (a, b)]
    sel = [approve[m].mean() for m in (a, b)]
    hi, lo = max(sel), min(sel)
    return np.array(
        [
            models.utility(y, approve, cost_bad),
            tpr[1] - tpr[0],
            fpr[1] - fpr[0],
            ppv[1] - ppv[0],
            lo / hi if hi > 0 else np.nan,
        ]
    )


STAT_NAMES = ("utility", "tpr_gap", "fpr_gap", "ppv_gap", "selection_ratio")


def calibration_by_group(score, y, g, bins: int = 5) -> dict:
    out = {}
    for name, gval in (("A", 0), ("B", 1)):
        m = g == gval
        s, yy = score[m], y[m]
        edges = np.quantile(s, np.linspace(0, 1, bins + 1)[1:-1])
        idx = np.searchsorted(edges, s)
        out[name] = [
            {
                "bin": int(k),
                "n": int((idx == k).sum()),
                "mean_score": float(s[idx == k].mean()),
                "observed_rate": float(yy[idx == k].mean()),
            }
            for k in range(bins)
            if (idx == k).any()
        ]
    return out


def _save_plot(table: dict, results_dir: Path) -> None:
    names = list(config.CANDIDATES)
    xs = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(xs - 0.2, [table[n]["groups"]["A"]["tpr"] for n in names], 0.4, label="group A")
    ax.bar(xs + 0.2, [table[n]["groups"]["B"]["tpr"] for n in names], 0.4, label="group B")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("True positive rate (test)")
    ax.set_title("Who gets approved among those who would repay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "tpr_by_group.png", dpi=120)
    plt.close(fig)


def run(
    results_dir: Path,
    fast: bool = False,
    seed: int = config.SEED,
    data_sha256: str | None = None,
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    n_boot = 100 if fast else config.N_BOOT
    splits = data.make_splits(seed)
    for df in splits.values():
        data.validate(df)
    fit, dev, test = splits["fit"], splits["dev"], splits["test"]
    y_dev, g_dev = dev["y"].to_numpy(), dev["g"].to_numpy()
    y, g = test["y"].to_numpy(), test["g"].to_numpy()

    feature_sets = {
        "aware": ["g", *config.FEATURES_ALL],
        "unaware": list(config.FEATURES_ALL),
        "blind": list(config.FEATURES_NO_PROXY),
    }
    scorers = {k: models.fit_scorer(fit, v, seed) for k, v in feature_sets.items()}
    s_dev = {k: f(dev) for k, f in scorers.items()}
    s_test = {k: f(test) for k, f in scorers.items()}

    def single(model: str, cost_bad: float = config.COST_BAD):
        t = models.best_threshold(s_dev[model], y_dev, cost_bad)
        return t, s_dev[model] >= t, s_test[model] >= t

    policies: dict[str, dict] = {}
    for name, model in (
        ("aware_single", "aware"),
        ("unaware_single", "unaware"),
        ("blind_single", "blind"),
    ):
        t, a_dev, a_test = single(model)
        policies[name] = {
            "model": model,
            "thresholds": {"all": t},
            "approve_dev": a_dev,
            "approve_test": a_test,
        }
    base = policies["unaware_single"]
    target = float(base["approve_dev"][y_dev == 1].mean())
    th = models.group_thresholds(s_dev["unaware"], y_dev, g_dev, target)
    policies["unaware_group_thresholds"] = {
        "model": "unaware",
        "thresholds": {"A": th[0], "B": th[1]},
        "target_tpr_on_dev": target,
        "approve_dev": models.apply_thresholds(s_dev["unaware"], g_dev, th),
        "approve_test": models.apply_thresholds(s_test["unaware"], g, th),
    }
    policies["reject_all"] = {
        "model": "none",
        "thresholds": {},
        "approve_dev": np.zeros(len(dev), dtype=bool),
        "approve_test": np.zeros(len(test), dtype=bool),
    }

    # Pick the headline on dev only.
    dev_rows = {}
    for name in config.CANDIDATES:
        st = policy_stats(y_dev, policies[name]["approve_dev"], g_dev)
        dev_rows[name] = {"utility": float(st[0]), "tpr_gap": float(st[1])}
    eligible = [
        n
        for n in config.CANDIDATES
        if dev_rows[n]["utility"] > 0 and abs(dev_rows[n]["tpr_gap"]) <= config.TPR_GAP_TOL
    ]
    if eligible:
        headline = max(
            eligible, key=lambda n: (dev_rows[n]["utility"], -config.CANDIDATES.index(n))
        )
        reason = "highest dev utility among candidates with positive utility and a dev TPR gap within tolerance"
    else:
        positive = [n for n in config.CANDIDATES if dev_rows[n]["utility"] > 0] or list(
            config.CANDIDATES
        )
        headline = min(positive, key=lambda n: abs(dev_rows[n]["tpr_gap"]))
        reason = (
            "no candidate met the tolerance; smallest dev TPR gap among those with positive utility"
        )

    # Test audit with paired, group-stratified bootstrap.
    table = {}
    boots = {}
    for name in config.CANDIDATES:
        pol = policies[name]
        score = s_test[pol["model"]] if pol["model"] != "none" else np.zeros(len(test))
        rep = group_report(y, score, pol["approve_test"], g)
        point = policy_stats(y, pol["approve_test"], g)
        draws = stratified_bootstrap(
            lambda idx, a=pol["approve_test"]: policy_stats(y[idx], a[idx], g[idx]), g, n_boot, seed
        )
        boots[name] = draws
        table[name] = {
            "model": pol["model"],
            "thresholds": pol["thresholds"],
            "groups": rep,
            "disparities": disparities(rep),
            **{
                s: {"value": float(point[i]), "ci95": list(ci95(draws[:, i]))}
                for i, s in enumerate(STAT_NAMES)
            },
            "abs_tpr_gap_ci95": list(ci95(np.abs(draws[:, 1]))),
        }
    h = table[headline]
    checks = {
        "tpr_gap_within_tolerance": bool(h["abs_tpr_gap_ci95"][1] <= config.TPR_GAP_TOL),
        "beats_rejecting_everyone": bool(h["utility"]["ci95"][0] > 0),
    }

    # Model quality, overall and by group.
    model_auc = {
        k: {
            "overall": auc(s_test[k], y),
            "group_A": auc(s_test[k][g == 0], y[g == 0]),
            "group_B": auc(s_test[k][g == 1], y[g == 1]),
        }
        for k in s_test
    }

    # Price of equal opportunity, paired.
    gt, us = boots["unaware_group_thresholds"], boots["unaware_single"]
    price = ci95(us[:, 0] - gt[:, 0])
    price_point = (
        table["unaware_single"]["utility"]["value"]
        - table["unaware_group_thresholds"]["utility"]["value"]
    )

    # Cost sensitivity: threshold re-chosen on dev for each cost, measured on test.
    sensitivity = []
    for c in config.COST_SENSITIVITY:
        t, _, a_single = single("unaware", c)
        t_dev_pol = s_dev["unaware"] >= t
        tgt = float(t_dev_pol[y_dev == 1].mean())
        thr = models.group_thresholds(s_dev["unaware"], y_dev, g_dev, tgt)
        a_group = models.apply_thresholds(s_test["unaware"], g, thr)
        row = {"cost_bad": c}
        for label, ap in (("single", a_single), ("group_thresholds", a_group)):
            st = policy_stats(y, ap, g, c)
            row[label] = {"utility": float(st[0]), "tpr_gap": float(st[1])}
        sensitivity.append(row)

    # Traps.
    t_grp = models.group_thresholds(s_test["unaware"], y, g, target)  # tuned on test: the leak
    leak = models.apply_thresholds(s_test["unaware"], g, t_grp)
    leak_stats = policy_stats(y, leak, g)
    tpr_ci = {}
    for name, gval in (("A", 0), ("B", 1)):
        d = stratified_bootstrap(
            lambda idx, v=gval: _mean(
                policies["unaware_single"]["approve_test"][idx][(g[idx] == v) & (y[idx] == 1)]
            ),
            g,
            n_boot,
            seed,
        )
        tpr_ci[name] = {"width": float(np.subtract(*ci95(d)[::-1])), "ci95": list(ci95(d))}
    gt_tab = table["unaware_group_thresholds"]
    us_tab = table["unaware_single"]
    traps = {
        "fairness_through_unawareness": {
            "tpr_gap_aware": table["aware_single"]["tpr_gap"],
            "tpr_gap_unaware": table["unaware_single"]["tpr_gap"],
            "tpr_gap_blind_no_proxy": table["blind_single"]["tpr_gap"],
            "note": "Dropping the group column, and even the proxy, leaves a gap. The gap comes from different base rates and a noisier measurement for group B.",
        },
        "aggregate_auc_hides_group_gap": {
            "overall_auc": model_auc["unaware"]["overall"],
            "group_A_auc": model_auc["unaware"]["group_A"],
            "group_B_auc": model_auc["unaware"]["group_B"],
            "note": "One overall number averages over a group the model serves worse.",
        },
        "group_thresholds_tuned_on_test": {
            "tpr_gap_with_leak": float(leak_stats[1]),
            "tpr_gap_honest_dev_tuned": gt_tab["tpr_gap"]["value"],
            "note": "Thresholds fitted on the rows they are scored on look perfect by construction.",
        },
        "gap_of_zero_is_cheap": {
            "reject_all_tpr_gap": table["reject_all"]["tpr_gap"]["value"],
            "reject_all_utility": table["reject_all"]["utility"]["value"],
            "note": "A policy that approves nobody has no gap. A fairness number needs a utility number beside it.",
        },
        "cannot_equalize_everything": {
            "unaware_single": {k: us_tab[k]["value"] for k in ("tpr_gap", "fpr_gap", "ppv_gap")},
            "group_thresholds": {k: gt_tab[k]["value"] for k in ("tpr_gap", "fpr_gap", "ppv_gap")},
            "base_rate_A": us_tab["groups"]["A"]["base_rate"],
            "base_rate_B": us_tab["groups"]["B"]["base_rate"],
            "note": "With different base rates, equalizing the true positive rate moves the other gaps. They cannot all be zero together.",
        },
        "small_group_wide_interval": {"tpr_ci_unaware_single": tpr_ci},
    }

    dev_study = dev_size_study(
        scorers["unaware"],
        s_test["unaware"],
        y,
        g,
        seed,
        10 if fast else config.DEV_REPS,
        20_000 if fast else config.BIG_EVAL,
    )
    analysis = {
        "headline_policy": headline,
        "post_hoc_dev_size_study": {
            "status": "added after the main run showed dev-tuned group thresholds did not transfer; not part of the gate",
            "rows": dev_study,
        },
        "selection_reason": reason,
        "dev_rows": dev_rows,
        "worst_off_group_by_tpr": min(h["groups"], key=lambda k: h["groups"][k]["tpr"]),
        "calibration_by_group": {
            k: calibration_by_group(s_test[k], y, g) for k in ("aware", "unaware")
        },
        "auc_by_group": model_auc,
        "price_of_equal_opportunity": {
            "utility_difference_single_minus_group_thresholds": price_point,
            "ci95": list(price),
        },
        "cost_sensitivity": sensitivity,
    }

    metrics = {
        "project": config.PROJECT_NAME,
        "data_source": config.DATA_SOURCE,
        "fast_mode": bool(fast),
        "seed": seed,
        "versions": _versions(),
        "data": {
            "fit": len(fit),
            "dev": len(dev),
            "test": len(test),
            "test_group_B_share": float((g == 1).mean()),
            "test_base_rate_A": float(y[g == 0].mean()),
            "test_base_rate_B": float(y[g == 1].mean()),
            "sha256": data_sha256,
        },
        "assumptions": {
            "benefit_good": config.BENEFIT_GOOD,
            "cost_bad": config.COST_BAD,
            "tpr_gap_tolerance": config.TPR_GAP_TOL,
        },
        "policies": table,
        "model_auc": model_auc,
        "headline": {
            "policy": headline,
            "selected_by": reason,
            "metric": "abs_tpr_gap",
            "value": abs(h["tpr_gap"]["value"]),
            "ci95": h["abs_tpr_gap_ci95"],
            "utility": h["utility"],
            "ci_method": f"group-stratified bootstrap, {n_boot} resamples",
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
    _save_plot({n: {"groups": table[n]["groups"]} for n in config.CANDIDATES}, results_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit decision policies by group.")
    parser.add_argument("--fast", action="store_true", help="smoke-test mode, results are marked")
    parser.add_argument(
        "--write-checksum", action="store_true", help="write the generator fingerprint"
    )
    args = parser.parse_args()
    if args.write_checksum:
        data.write_fingerprint()
    digest = data.verify_fingerprint()
    m = run(config.RESULTS_DIR, fast=args.fast, data_sha256=digest)
    h = m["headline"]
    print(f"headline {h['policy']}: |TPR gap| {h['value']:.3f} {h['ci95']}, utility {h['utility']}")
    print(f"gate: {m['gate']['checks']} passed: {m['gate']['passed']}")


if __name__ == "__main__":
    main()
