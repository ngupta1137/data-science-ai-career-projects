"""Run each visit policy on the same simulated chain and compare with no visits and an oracle.

python -m oos_detect.train          # full run
python -m oos_detect.train --fast   # fewer stores and resamples, marked
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

from . import config, data, detect, metrics

CAVEATS = [
    "Everything is simulated. Sales rates, how often shelves go empty, how long they stay empty, the margin and the visit cost are chosen by the author.",
    "Demand has no weekday pattern, season or promotions, which in real data create zero-sales runs that are not empty shelves.",
    "The prior share of empty shelves is set close to the simulator's true value. In practice it comes from shelf audits and is less accurate.",
    "A visit always finds and fixes an empty shelf. Real reps miss some, and some empty shelves cannot be refilled because there is no stock in the back room.",
    "Shelves are independent. Real empty shelves cluster by store, delivery day and supplier.",
    "Intervals resample stores. They do not cover a different simulator.",
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


def run(
    results_dir: Path, fast: bool = False, seed: int = config.SEED, data_sha256: str | None = None
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    n_stores = 12 if fast else config.N_STORES
    n_boot = 200 if fast else config.N_BOOT
    scale = n_stores / config.N_STORES
    capacity = max(1, round(config.CAPACITY * scale))
    days = config.HISTORY_DAYS + config.EVAL_DAYS
    w = data.generate(_rng(seed, "world"), n_stores, config.N_ITEMS, days)
    data.validate(w)

    outs = {p: detect.simulate(w, p, capacity) for p in config.POLICIES}
    base = outs["no_visits"]["lost_margin"]
    nv = {p: metrics.net_value_per_store_day(base, o, w.store, n_stores) for p, o in outs.items()}
    table = {}
    for p, o in outs.items():
        table[p] = {
            "lost_margin_per_store_day": float(
                o["lost_margin"].sum() / n_stores / config.EVAL_DAYS
            ),
            "visits_per_day": float(o["visits"].sum() / config.EVAL_DAYS),
            "flagged_per_day": o["flagged_per_day"],
            "precision": metrics.precision(o),
            "recall": metrics.recall(o),
            "median_days_to_fix": float(np.median(o["delays"])) if len(o["delays"]) else None,
            "net_value_per_store_day": metrics.boot_mean_ci(nv[p], _rng(seed, f"boot|{p}"), n_boot),
        }
    h = config.HEADLINE
    diff = metrics.boot_mean_ci(nv[h] - nv["zero_run_rule"], _rng(seed, "boot|diff"), n_boot)
    checks = {
        "beats_no_visits": table[h]["net_value_per_store_day"]["ci95"][0] > 0,
        "beats_zero_run_rule": diff["ci95"][0] > 0,
        "precision_at_least_0.5": table[h]["precision"] >= config.MIN_PRECISION,
    }

    # Rate estimates: the naive rate counts empty days as zero demand.
    rc, rn = outs[h]["rate_clean"], outs[h]["rate_naive"]
    rate_trap = {
        "mean_true_rate": float(w.rate.mean()),
        "mean_naive_rate": float(rn.mean()),
        "mean_clean_rate": float(rc.mean()),
        "median_ratio_naive_to_true": float(np.median(rn / w.rate)),
        "median_ratio_clean_to_true": float(np.median(rc / w.rate)),
    }

    bands = [
        ("slow, under 0.3 a day", 0, 0.3),
        ("medium, 0.3 to 2 a day", 0.3, 2),
        ("fast, over 2 a day", 2, np.inf),
    ]
    band_rows = []
    for name, lo, hi in bands:
        m = (w.rate >= lo) & (w.rate < hi)
        row = {
            "band": name,
            "share_of_shelves": float(m.mean()),
            "share_of_lost_margin_with_no_visits": float(base[m].sum() / base.sum()),
        }
        for p in ("zero_run_rule", "expected_value", "oracle"):
            o = outs[p]
            v = o["visits"][m].sum()
            row[p] = {
                "share_of_visits": float(v / max(o["visits"].sum(), 1)),
                "precision": float(o["useful_visits"][m].sum() / v) if v else None,
                "share_of_lost_margin_recovered": float(
                    (base[m] - o["lost_margin"][m]).sum() / base[m].sum()
                ),
            }
        band_rows.append(row)

    cap_rows = []
    for cap in config.CAPACITY_SENSITIVITY:
        c = max(1, round(cap * scale))
        row = {"capacity_per_day": cap}
        for p in ("zero_run_rule", "expected_value"):
            o = outs[p] if c == capacity else detect.simulate(w, p, c)
            row[p] = {
                "net_value_per_store_day": float(
                    metrics.net_value_per_store_day(base, o, w.store, n_stores).mean()
                ),
                "precision": metrics.precision(o),
                "recall": metrics.recall(o),
            }
        cap_rows.append(row)

    calib = outs[h]["visit_calibration"]
    edges = np.quantile(calib[:, 0], np.linspace(0, 1, 6))
    bins = np.clip(np.searchsorted(edges, calib[:, 0], side="right") - 1, 0, 4)
    calib_rows = [
        {
            "fifth_of_visits_by_predicted_chance": int(b) + 1,
            "mean_predicted_chance_empty": float(calib[bins == b, 0].mean()),
            "share_found_empty": float(calib[bins == b, 1].mean()),
            "visits": int((bins == b).sum()),
        }
        for b in range(5)
    ]
    analysis = {
        "demand_bands": band_rows,
        "capacity": cap_rows,
        "rate_estimates": rate_trap,
        "post_hoc_visit_calibration": {
            "status": "post hoc, added after the gate failed; does not change the verdict",
            "mean_predicted_chance_empty": float(calib[:, 0].mean()),
            "share_found_empty": float(calib[:, 1].mean()),
            "share_of_shelf_days_truly_empty_during_evaluation": float(outs[h]["empty_share"]),
            "rows": calib_rows,
        },
    }
    traps = {
        "zero_sales_read_as_zero_demand": {
            **rate_trap,
            "net_value_naive_rate": table["expected_value_naive_rate"]["net_value_per_store_day"][
                "value"
            ],
            "net_value_clean_rate": table["expected_value"]["net_value_per_store_day"]["value"],
        },
        "same_rule_for_every_shelf": {
            "zero_run_rule_flagged_per_day": table["zero_run_rule"]["flagged_per_day"],
            "capacity_per_day": capacity,
            "zero_run_rule_precision": table["zero_run_rule"]["precision"],
            "note": "Three days without a sale is normal for a slow seller and a long gap for a fast one, so one rule floods reps with slow-seller alerts.",
        },
        "alarm_without_value": {
            "poisson_alarm_precision": table["poisson_alarm"]["precision"],
            "poisson_alarm_net_value": table["poisson_alarm"]["net_value_per_store_day"]["value"],
            "note": "A shelf can be almost surely empty and still not worth a visit if it sells little.",
        },
    }
    out = {
        "project": config.PROJECT_NAME,
        "data_source": config.DATA_SOURCE,
        "fast_mode": bool(fast),
        "seed": seed,
        "versions": _versions(),
        "data": {
            "stores": n_stores,
            "shelves": len(w),
            "history_days": config.HISTORY_DAYS,
            "eval_days": config.EVAL_DAYS,
            "eval_empty_episodes": int(outs["no_visits"]["episodes"].sum()),
            "share_of_shelf_days_empty_no_visits": float(
                (outs["no_visits"]["lost_margin"] > 0).mean()
            ),
            "sha256": data_sha256,
        },
        "assumptions": {
            "margin": config.MARGIN,
            "visit_cost": config.VISIT_COST,
            "capacity_per_day": capacity,
            "prior_empty": config.PRIOR_EMPTY,
            "zero_run_rule_days": config.ZERO_RUN_RULE,
        },
        "policies": table,
        "headline": {
            "policy": h,
            "metric": "net_value_per_store_day",
            "value": table[h]["net_value_per_store_day"]["value"],
            "ci95": table[h]["net_value_per_store_day"]["ci95"],
            "ci_method": f"percentile bootstrap over stores, {n_boot} resamples",
            "minus_zero_run_rule": diff,
            "precision": table[h]["precision"],
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
    _plot(table, results_dir)
    return out


def _plot(table: dict, results_dir: Path) -> None:
    names = [p for p in config.POLICIES if p != "no_visits"]
    vals = [table[p]["net_value_per_store_day"]["value"] for p in names]
    errs = np.array(
        [
            [
                table[p]["net_value_per_store_day"]["value"]
                - table[p]["net_value_per_store_day"]["ci95"][0],
                table[p]["net_value_per_store_day"]["ci95"][1]
                - table[p]["net_value_per_store_day"]["value"],
            ]
            for p in names
        ]
    ).T
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, vals, xerr=errs)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("net value per store per day over no visits")
    fig.tight_layout()
    fig.savefig(results_dir / "net_value.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare out-of-stock visit policies.")
    parser.add_argument("--fast", action="store_true", help="smoke-test mode, results are marked")
    parser.add_argument(
        "--write-checksum", action="store_true", help="write the generator fingerprint"
    )
    args = parser.parse_args()
    if args.write_checksum:
        data.write_fingerprint()
    digest = data.verify_fingerprint()
    m = run(config.RESULTS_DIR, fast=args.fast, data_sha256=digest)
    for p, r in m["policies"].items():
        print(
            p,
            {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in r.items()
                if k != "net_value_per_store_day"
            },
            r["net_value_per_store_day"],
        )
    print("gate", m["gate"]["checks"], m["gate"]["passed"])


if __name__ == "__main__":
    main()
