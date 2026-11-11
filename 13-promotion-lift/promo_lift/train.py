"""Estimate a promotion's incremental profit three ways and compare with the truth.

    python -m promo_lift.train          # full run
    python -m promo_lift.train --fast   # fewer simulated campaigns, marked

The headline campaign is one randomised trial. The method study repeats the trial many times,
which only a simulation allows, to measure bias, interval coverage and how often each method
makes the wrong call.
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

from . import config, data, metrics

CAVEATS = [
    "Everything is simulated. Store sizes, the season, the lift, cannibalisation, pull-forward and the margins are chosen by the author, so the sizes of the results describe this setup, not any retailer.",
    "One item, one sibling, one four-week promotion. No competing promotions, display or feature support, stock-outs, price ladders or halo on other categories.",
    "Stores are independent. Real stores share distribution centres, media and shoppers, which can leak a promotion into control stores.",
    "The promotion is randomised across stores in the headline campaign. The manager-picked variant shows what happens without that.",
    "Margins and the discount are assumptions. Profit here ignores supplier funding, which often pays part of the discount.",
    "Intervals assume stores are independent draws. They do not cover a mistaken choice of pre-period or post-period length.",
]


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
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


def _rng(seed: int, tag: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(tag.encode())])


def _truth(c: data.Campaign) -> float:
    return float(c.true_profit[c.treated == 1].mean())


def _study(seed: int, reps: int, n_stores: int, assignment: str, names) -> dict:
    rows = {n: {"est": [], "covers": [], "calls": []} for n in names}
    truths = []
    for r in range(reps):
        c = data.generate(_rng(seed, f"study|{assignment}|{n_stores}|{r}"), n_stores, assignment)
        truth = _truth(c)
        truths.append(truth)
        for n in names:
            e = metrics.estimate(c, n)
            rows[n]["est"].append(e["value"])
            rows[n]["covers"].append(metrics.covers(e["ci95"], truth))
            rows[n]["calls"].append(metrics.call(e["ci95"]))
    truths = np.array(truths)
    out = {}
    for n, v in rows.items():
        est = np.array(v["est"])
        calls = np.array(v["calls"])
        out[n] = {
            "mean_estimate": float(est.mean()),
            "mean_truth": float(truths.mean()),
            "bias": float((est - truths).mean()),
            "rmse": float(np.sqrt(((est - truths) ** 2).mean())),
            "coverage": float(np.mean(v["covers"])),
            "share_run_it": float((calls == "run it").mean()),
            "share_stop_it": float((calls == "stop it").mean()),
            "share_unclear": float((calls == "unclear").mean()),
        }
    return {"campaigns": reps, "stores": n_stores, "assignment": assignment, "estimators": out}


def run(
    results_dir: Path, fast: bool = False, seed: int = config.SEED, data_sha256: str | None = None
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    reps = 60 if fast else config.N_REPS
    power_reps = 30 if fast else config.POWER_REPS

    c = data.generate(_rng(seed, "headline"), config.N_STORES, "random")
    data.validate(c)
    truth = _truth(c)
    table = {}
    for n in config.ESTIMATORS:
        e = metrics.estimate(c, n)
        table[n] = {
            **e,
            "call": metrics.call(e["ci95"]),
            "error": e["value"] - truth,
            "covers_truth": metrics.covers(e["ci95"], truth),
        }
    h = table[config.HEADLINE]
    true_call = "run it" if truth > 0 else "stop it"

    study = _study(seed, reps, config.N_STORES, "random", config.ESTIMATORS)
    picked = _study(seed, reps, config.N_STORES, "picked", ("full_did",))
    power = []
    for n_stores in config.STORE_COUNTS:
        s = _study(seed, power_reps, n_stores, "random", ("full_did",))["estimators"]["full_did"]
        power.append(
            {
                "stores": n_stores,
                "campaigns": power_reps,
                **{k: s[k] for k in ("share_run_it", "share_stop_it", "share_unclear", "coverage")},
            }
        )

    checks = {
        "interval_coverage_at_least_0.90": study["estimators"]["full_did"]["coverage"]
        >= config.MIN_COVERAGE,
        "headline_interval_excludes_zero": h["call"] != "unclear",
        "headline_call_matches_truth": h["call"] == true_call,
    }

    t = c.treated == 1
    parts = {k: float(v[t].mean()) for k, v in c.true_parts.items()}
    segments = []
    for flag, name in ((1, "urban"), (0, "rural")):
        m = c.urban == flag
        sub = data.Campaign(c.item[m], c.sibling[m], c.treated[m], c.urban[m], c.true_profit[m], {})
        e = metrics.estimate(sub, "full_did")
        segments.append(
            {
                "segment": name,
                "stores": int(m.sum()),
                "promoted_stores": int((m & t).sum()),
                "estimate": e["value"],
                "ci95": e["ci95"],
                "truth": _truth(sub),
                "call": metrics.call(e["ci95"]),
            }
        )

    analysis = {
        "true_profit_parts_per_promoted_store": parts,
        "segments": segments,
        "method_study": study,
        "manager_picked_stores": picked,
        "stores_needed": power,
    }
    traps = {
        "before_and_after_without_control": {
            "headline_estimate": table["pre_post"]["value"],
            "bias_over_campaigns": study["estimators"]["pre_post"]["bias"],
            "note": "The promotion weeks fall on the seasonal peak, so a before-and-after view credits the season to the promotion.",
        },
        "counting_only_the_promoted_item_in_promo_weeks": {
            "headline_estimate": table["window_did"]["value"],
            "bias_over_campaigns": study["estimators"]["window_did"]["bias"],
            "missed_cannibalisation": parts["cannibalisation"],
            "missed_pull_forward": parts["pull_forward"],
        },
        "manager_picked_stores": {
            "bias_full_did_random": study["estimators"]["full_did"]["bias"],
            "bias_full_did_picked": picked["estimators"]["full_did"]["bias"],
            "coverage_picked": picked["estimators"]["full_did"]["coverage"],
            "note": "When the stores with the best trend get the promotion, their growth is credited to it.",
        },
        "units_up_profit_down": {
            "extra_units_per_promoted_store": float(
                ((c.true_parts["extra_units_sold"][t]) / config.MARGIN_ITEM).mean()
            ),
            "true_profit_per_promoted_store": truth,
        },
    }
    out = {
        "project": config.PROJECT_NAME,
        "data_source": config.DATA_SOURCE,
        "fast_mode": bool(fast),
        "seed": seed,
        "versions": _versions(),
        "data": {
            "stores": len(c),
            "promoted_stores": int(t.sum()),
            "weeks": config.WEEKS,
            "promo_weeks": data.weeks("promo").tolist(),
            "post_weeks": data.weeks("post").tolist(),
            "pre_weeks": data.weeks("pre").tolist(),
            "mean_weekly_units_item": float(c.item.mean()),
            "sha256": data_sha256,
        },
        "assumptions": {
            "margin_item": config.MARGIN_ITEM,
            "discount": config.DISCOUNT,
            "margin_sibling": config.MARGIN_SIBLING,
            "lift": config.LIFT,
            "cannibalisation": config.CANNIBALISATION,
            "pull_forward": config.PULL_FORWARD,
        },
        "estimators": table,
        "true_incremental_profit_per_promoted_store": truth,
        "true_call": true_call,
        "headline": {
            "estimator": config.HEADLINE,
            "metric": "incremental_profit_per_promoted_store",
            "value": h["value"],
            "ci95": h["ci95"],
            "ci_method": "Welch t interval over stores, promoted against control",
            "call": h["call"],
        },
        "gate": {"rule": config.GATE_RULE, "checks": checks, "passed": bool(all(checks.values()))},
        "traps": traps,
        "caveats": CAVEATS,
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(_clean(out), indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "error_analysis.json").write_text(
        json.dumps(_clean(analysis), indent=2) + "\n", encoding="utf-8"
    )
    _plot(c, results_dir)
    return out


def _plot(c: data.Campaign, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    w = np.arange(config.WEEKS)
    for flag, lab in ((1, "promoted stores"), (0, "control stores")):
        m = c.treated == flag
        ax.plot(w, c.item[m].mean(0), label=f"promoted item, {lab}")
        ax.plot(w, c.sibling[m].mean(0), "--", label=f"sibling item, {lab}")
    ax.axvspan(config.PROMO_START - 0.5, config.PROMO_START + config.PROMO_WEEKS - 0.5, alpha=0.15)
    ax.set_xlabel("week")
    ax.set_ylabel("mean units per store")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(results_dir / "weekly_sales.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure a promotion's incremental profit.")
    parser.add_argument("--fast", action="store_true", help="smoke-test mode, results are marked")
    parser.add_argument(
        "--write-checksum", action="store_true", help="write the generator fingerprint"
    )
    args = parser.parse_args()
    if args.write_checksum:
        data.write_fingerprint()
    digest = data.verify_fingerprint()
    m = run(config.RESULTS_DIR, fast=args.fast, data_sha256=digest)
    for n, e in m["estimators"].items():
        print(f"{n}: {e['value']:.1f} {e['ci95']} call {e['call']}")
    print(
        f"truth {m['true_incremental_profit_per_promoted_store']:.1f}; gate {m['gate']['checks']} passed {m['gate']['passed']}"
    )


if __name__ == "__main__":
    main()
