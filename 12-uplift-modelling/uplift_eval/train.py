"""Fit uplift learners on a randomised trial, pick one on dev, test it, compare with the truth.

    python -m uplift_eval.train          # full run
    python -m uplift_eval.train --fast   # fewer bootstrap draws and replications, marked

'Estimated' profit uses the trial's outcomes and the known randomisation. 'True' profit uses the
simulator's individual effects on a large fresh population, which real data never provides.
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

from . import config, data, metrics, models

V, C = config.VALUE_PER_CONVERSION, config.COST_PER_OFFER

CAVEATS = [
    "Everything is simulated. The effect of an offer, the size of the hidden driver and the economics are chosen by the author, so the sizes of the results describe this setup, not any retailer.",
    "The trial is large (20,000 customers per split) and the effect is strong and smooth. Real uplift is often small against noise and needs far more data than a response model.",
    "The value of a conversion and the cost of an offer are assumptions. The cost sensitivity table shows how the targeted share moves.",
    "Only a one-off binary offer with a binary outcome. No dose, timing, fatigue, multiple offers or long-run effects.",
    "The class-transformation learner is valid only because the trial gives the offer to exactly half. It would be wrong for other randomisation rates.",
    "The learners are untuned, with fixed settings chosen before results.",
    "True profit uses the simulator's effects. In practice the nearest equivalent is a later randomised holdout, which this project does not run.",
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


def fit_all(train: data.Customers) -> dict:
    return {n: cls().fit(train.x, train.t, train.y) for n, cls in models.LEARNERS.items()}


def est_value(policy, c: data.Customers, rng, boots, cost: float = C) -> dict:
    terms = metrics.ipw_value(policy, c.t, c.y, c.propensity, V, cost)
    return {"value": float(terms.mean()), "ci95": list(metrics.boot_mean_ci(terms, rng, boots))}


def run(
    results_dir: Path,
    fast: bool = False,
    seed: int = config.SEED,
    data_sha256: str | None = None,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    boots = 100 if fast else config.N_BOOT
    splits = {
        "train": data.generate(_rng(seed, "train"), config.N_TRAIN),
        "dev": data.generate(_rng(seed, "dev"), config.N_DEV),
        "test": data.generate(_rng(seed, "test"), config.N_TEST),
    }
    for c in splits.values():
        data.validate(c)
    train, dev, test = splits["train"], splits["dev"], splits["test"]
    pop = data.generate(_rng(seed, "population"), 50_000 if fast else config.BIG_EVAL)
    fitted = fit_all(train)
    response = models.ResponseGbm().fit(train.x, train.t, train.y)

    s_dev = {n: m.uplift(dev.x) for n, m in fitted.items()}
    s_test = {n: m.uplift(test.x) for n, m in fitted.items()}
    s_pop = {n: m.uplift(pop.x) for n, m in fitted.items()}
    dev_auuc = {
        n: metrics.auuc(*metrics.qini_curve(s_dev[n], dev.t, dev.y)) for n in config.CANDIDATES
    }
    headline = max(config.CANDIDATES, key=lambda n: (dev_auuc[n], -config.CANDIDATES.index(n)))
    reason = "highest area under the uplift curve (AUUC) on dev among the four learners"

    thr = config.THRESHOLD
    rand_rng = _rng(seed, "random-policy")
    policies_test = {n: s_test[n] > thr for n in config.CANDIDATES}
    policies_test["treat_none"] = np.zeros(len(test), bool)
    policies_test["treat_all"] = np.ones(len(test), bool)
    policies_test["random_budget"] = rand_rng.random(len(test)) < config.BUDGET_SHARE
    policies_test["response_budget"] = metrics.top_share(
        response.uplift(test.x), config.BUDGET_SHARE
    )
    policies_test["oracle"] = test.tau * V > C
    policies_pop = {n: s_pop[n] > thr for n in config.CANDIDATES}
    policies_pop["treat_none"] = np.zeros(len(pop), bool)
    policies_pop["treat_all"] = np.ones(len(pop), bool)
    policies_pop["random_budget"] = np.full(len(pop), False)  # replaced below by its expectation
    policies_pop["response_budget"] = metrics.top_share(response.uplift(pop.x), config.BUDGET_SHARE)
    policies_pop["oracle"] = pop.tau * V > C

    table = {}
    for n, pol in policies_test.items():
        if n == "random_budget":
            true_pop = config.BUDGET_SHARE * metrics.true_value(
                np.ones(len(pop), bool), pop.tau, V, C
            )
        else:
            true_pop = metrics.true_value(policies_pop[n], pop.tau, V, C)
        row = {
            "treat_share": float(pol.mean()),
            "estimated_profit": est_value(pol, test, _rng(seed, f"ipw|{n}"), boots),
            "true_profit_test_rows": metrics.true_value(pol, test.tau, V, C),
            "true_profit_population": true_pop,
            "mean_true_effect_of_those_treated": float(test.tau[pol].mean()) if pol.any() else None,
            "share_treated_with_negative_effect": float((test.tau[pol] < 0).mean())
            if pol.any()
            else None,
        }
        if n in config.CANDIDATES:
            qb = metrics.boot_ci(
                lambda i, s=s_test[n]: metrics.qini_coefficient(s[i], test.t[i], test.y[i]),
                len(test),
                _rng(seed, f"qini|{n}"),
                boots,
            )
            x, q = metrics.qini_curve(s_test[n], test.t, test.y)
            row.update(
                {
                    "qini_coefficient": {
                        "value": metrics.qini_coefficient(s_test[n], test.t, test.y),
                        "ci95": list(qb),
                    },
                    "auuc_test": metrics.auuc(x, q),
                    "dev_auuc": dev_auuc[n],
                    "corr_with_true_effect": float(np.corrcoef(s_pop[n], pop.tau)[0, 1]),
                    "true_profit_top_budget_population": metrics.true_value(
                        metrics.top_share(s_pop[n], config.BUDGET_SHARE), pop.tau, V, C
                    ),
                }
            )
        table[n] = row

    terms_h = metrics.ipw_value(policies_test[headline], test.t, test.y, test.propensity, V, C)
    terms_all = metrics.ipw_value(policies_test["treat_all"], test.t, test.y, test.propensity, V, C)
    diff_ci = metrics.boot_mean_ci(terms_h - terms_all, _rng(seed, "gate-diff"), boots)
    hq = table[headline]["qini_coefficient"]
    checks = {
        "beats_treating_nobody": bool(table[headline]["estimated_profit"]["ci95"][0] > 0),
        "beats_treating_everybody": bool(diff_ci[0] > 0),
        "ranking_beats_random_targeting": bool(hq["ci95"][0] > 0),
    }

    # Traps, measured.
    resp_pop = response.uplift(pop.x)
    top_h = metrics.top_share(s_pop[headline], config.BUDGET_SHARE)
    top_r = metrics.top_share(resp_pop, config.BUDGET_SHARE)
    top_o = metrics.top_share(pop.tau, config.BUDGET_SHARE)
    budget_rows = []
    for share in (0.1, 0.2, 0.3, 0.5, 0.7):
        budget_rows.append(
            {
                "share_treated": share,
                "headline": metrics.true_value(
                    metrics.top_share(s_pop[headline], share), pop.tau, V, C
                ),
                "response_model": metrics.true_value(
                    metrics.top_share(resp_pop, share), pop.tau, V, C
                ),
                "oracle": metrics.true_value(metrics.top_share(pop.tau, share), pop.tau, V, C),
                "random": share * metrics.true_value(np.ones(len(pop), bool), pop.tau, V, C),
            }
        )
    response_trap = {
        "budget_share": config.BUDGET_SHARE,
        "true_profit_uplift_model": metrics.true_value(top_h, pop.tau, V, C),
        "true_profit_response_model": metrics.true_value(top_r, pop.tau, V, C),
        "true_profit_random": config.BUDGET_SHARE
        * metrics.true_value(np.ones(len(pop), bool), pop.tau, V, C),
        "true_profit_oracle": metrics.true_value(top_o, pop.tau, V, C),
        "mean_true_effect_targeted_uplift_model": float(pop.tau[top_h].mean()),
        "mean_true_effect_targeted_response_model": float(pop.tau[top_r].mean()),
        "mean_true_effect_everyone": float(pop.tau.mean()),
        "note": "A model of who will buy picks customers who buy anyway. It is not a model of who the offer changes.",
    }
    # (b) winner's curse: the same policy valued on the rows it was fitted on.
    in_sample = est_value(
        fitted[headline].uplift(train.x) > thr, train, _rng(seed, "ipw-train"), boots
    )
    dev_est = est_value(s_dev[headline] > thr, dev, _rng(seed, "ipw-dev"), boots)
    best_dev_of_all = max(
        est_value(s_dev[n] > thr, dev, _rng(seed, f"ipw-dev|{n}"), 20)["value"]
        for n in config.CANDIDATES
    )
    winners_curse = {
        "headline": headline,
        "estimated_on_training_rows": in_sample,
        "estimated_on_dev": dev_est,
        "best_dev_estimate_among_candidates": best_dev_of_all,
        "estimated_on_test": table[headline]["estimated_profit"],
        "true_on_test_rows": table[headline]["true_profit_test_rows"],
        "true_on_population": table[headline]["true_profit_population"],
        "treat_all_estimated_vs_true_population": [
            table["treat_all"]["estimated_profit"]["value"],
            table["treat_all"]["true_profit_population"],
        ],
        "oracle_estimated_vs_true_population": [
            table["oracle"]["estimated_profit"]["value"],
            table["oracle"]["true_profit_population"],
        ],
        "note": "An estimate from the rows a model was fitted on, or the best of several candidates on the rows used to choose, is expected to be optimistic. Compare each with the true value. Absolute estimates from one sample share that sample's luck, so compare policies by paired differences.",
    }
    # (c) offers that followed past buying signals, with an unmeasured driver.
    obs = data.generate(_rng(seed, "obs-train"), config.N_TRAIN, randomised=False)
    obs_fit = fit_all(obs)
    confounded = {
        "naive_effect_in_observational_data": float(
            obs.y[obs.t == 1].mean() - obs.y[obs.t == 0].mean()
        ),
        "naive_effect_in_trial": float(train.y[train.t == 1].mean() - train.y[train.t == 0].mean()),
        "true_average_effect": float(obs.tau.mean()),
        "share_offered_observational": float(obs.t.mean()),
        "true_profit_population": {
            n: {
                "trained_on_trial": table[n]["true_profit_population"],
                "trained_on_observational": metrics.true_value(
                    m.uplift(pop.x) > thr, pop.tau, V, C
                ),
            }
            for n, m in obs_fit.items()
        },
        "note": "When past offers went to likely buyers and some of what made them likely is not in the data, a learner reads that as the effect of the offer.",
    }
    # (d) how much data does uplift need?
    sizes = (2000, 5000) if fast else config.TRAIN_SIZES
    reps = 1 if fast else config.SIZE_REPS
    size_rows = []
    for n_rows in sizes:
        acc = {n: [] for n in config.CANDIDATES}
        for r in range(reps):
            sub = data.generate(_rng(seed, f"size|{n_rows}|{r}"), n_rows)
            for n, m in fit_all(sub).items():
                acc[n].append(metrics.true_value(m.uplift(pop.x) > thr, pop.tau, V, C))
        size_rows.append(
            {
                "train_customers": n_rows,
                "replications": reps,
                "mean_true_profit": {n: float(np.mean(v)) for n, v in acc.items()},
                "min_true_profit": {n: float(np.min(v)) for n, v in acc.items()},
            }
        )
    # (e) cost sensitivity: re-threshold the same predictions.
    cost_rows = []
    for cost in config.COST_SENSITIVITY:
        t_c = cost / V
        pol = s_pop[headline] > t_c
        cost_rows.append(
            {
                "cost_per_offer": cost,
                "threshold": t_c,
                "treat_share": float(pol.mean()),
                "true_profit_headline": metrics.true_value(pol, pop.tau, V, cost),
                "true_profit_treat_all": metrics.true_value(
                    np.ones(len(pop), bool), pop.tau, V, cost
                ),
                "true_profit_oracle": metrics.true_value(pop.tau * V > cost, pop.tau, V, cost),
            }
        )

    traps = {
        "response_model_instead_of_uplift": response_trap,
        "optimistic_profit_estimate": winners_curse,
        "offers_that_followed_buying_signals": confounded,
        "data_needed": size_rows,
        "budget_curve": budget_rows,
    }

    # Error analysis on the population.
    seg_edges = [(-np.inf, 0.0), (0.0, thr), (thr, np.inf)]
    seg_names = ("offer lowers buying", "offer helps but not enough to pay", "offer pays")
    pol_h = s_pop[headline] > thr
    segments = []
    for (lo, hi), nm in zip(seg_edges, seg_names, strict=True):
        m = (pop.tau > lo) & (pop.tau <= hi) if lo > -np.inf else (pop.tau <= hi)
        segments.append(
            {
                "segment": nm,
                "share_of_customers": float(m.mean()),
                "mean_true_effect": float(pop.tau[m].mean()),
                "headline_treats": float(pol_h[m].mean()),
                "profit_per_customer_if_treat_all": float(np.mean(m * (V * pop.tau - C))),
            }
        )
    deciles = np.digitize(s_pop[headline], np.quantile(s_pop[headline], np.linspace(0.1, 0.9, 9)))
    calib = [
        {
            "decile_of_predicted_uplift": int(d) + 1,
            "mean_predicted": float(s_pop[headline][deciles == d].mean()),
            "mean_true": float(pop.tau[deciles == d].mean()),
        }
        for d in range(10)
    ]
    analysis = {
        "headline_learner": headline,
        "selection_reason": reason,
        "dev_auuc": dev_auuc,
        "segments": segments,
        "calibration_by_decile": calib,
        "cost_sensitivity": cost_rows,
        "base_rates": {
            "p0_mean": float(pop.p0.mean()),
            "p1_mean": float(pop.p1.mean()),
            "share_negative_effect": float((pop.tau < 0).mean()),
        },
    }

    metrics_out = {
        "project": config.PROJECT_NAME,
        "data_source": config.DATA_SOURCE,
        "fast_mode": bool(fast),
        "seed": seed,
        "versions": _versions(),
        "data": {
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
            "population": len(pop),
            "treated_share_test": float(test.t.mean()),
            "conversion_rate_test": float(test.y.mean()),
            "sha256": data_sha256,
        },
        "assumptions": {
            "value_per_conversion": V,
            "cost_per_offer": C,
            "threshold": thr,
            "budget_share": config.BUDGET_SHARE,
            "treatment_share": config.TREATMENT_SHARE,
        },
        "policies": table,
        "headline": {
            "learner": headline,
            "selected_by": reason,
            "metric": "incremental_profit_per_customer",
            "value": table[headline]["estimated_profit"]["value"],
            "ci95": table[headline]["estimated_profit"]["ci95"],
            "ci_method": f"percentile bootstrap over test customers, {boots} resamples; inverse-probability weighted",
            "minus_treat_all": {
                "value": float((terms_h - terms_all).mean()),
                "ci95": list(diff_ci),
            },
            "qini_coefficient": hq,
            "true_profit_population": table[headline]["true_profit_population"],
        },
        "gate": {"rule": config.GATE_RULE, "checks": checks, "passed": bool(all(checks.values()))},
        "traps": traps,
        "caveats": CAVEATS,
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(_clean(metrics_out), indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "error_analysis.json").write_text(
        json.dumps(_clean(analysis), indent=2) + "\n", encoding="utf-8"
    )
    _save_plot(s_test[headline], test, pop, s_pop[headline], results_dir)
    return metrics_out


def _save_plot(s_test, test, pop, s_pop, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x, q = metrics.qini_curve(s_test, test.t, test.y)
    ax.plot(x, q * 100, label="headline (estimated on test)")
    ax.plot([0, 1], [0, q[-1] * 100], "--", label="random targeting")
    ax.set_xlabel("fraction of customers offered, best predicted first")
    ax.set_ylabel("incremental buyers per 100 customers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "qini.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate uplift learners on a randomised trial.")
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
    print(
        f"headline {h['learner']}: profit {h['value']:.3f} {h['ci95']}, true {h['true_profit_population']:.3f}"
    )
    print(f"gate: {m['gate']['checks']} passed: {m['gate']['passed']}")


if __name__ == "__main__":
    main()
