"""Fit recommenders on the past, evaluate on the future, compare offline metrics with true value.

    python -m recsys_eval.train          # full run
    python -m recsys_eval.train --fast   # fewer bootstrap draws and sweep points, marked

Windows: train on days before 70 and validate on 70 to 85 to pick the headline model; refit on days
before 85 and test on 85 to 100. 'Observed' metrics use held-out clicks, as in real life. 'True'
metrics use the simulator's affinities, which real logs never reveal.
"""

from __future__ import annotations

import argparse
import json
import platform
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn

from . import config, data, metrics, models
from .data import Events, World

CAVEATS = [
    "Everything is simulated. Taste is a low-rank signal plus drift, exposure is a popularity softmax, and clicks are Bernoulli. The sizes of the effects describe this setup, not any real catalogue.",
    "True value is only known because the simulator knows it. In real life the equivalent is an online experiment, which this project does not run.",
    "Clicks are the only signal. There are no dwell times, ratings, purchases, sessions or context.",
    "Models are small classical baselines (popularity, item-kNN, SVD). No neural, sequential or two-tower model, and no tuning of ranks or neighbourhoods.",
    "The inverse-popularity weight uses a fixed exponent of 0.5 chosen before results. A real exposure model would estimate propensities.",
    "The 0.80 Spearman bar is an assumption. With seven models a correlation has a wide interval.",
    "Feedback loops (today's recommendations changing tomorrow's logs) are not simulated.",
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


@dataclass
class Window:
    train: Events
    hold: Events
    end: float  # last day of the holdout window
    t_eval: float  # day at which true affinity is read


def dense(ev: Events, n_users: int, n_items: int) -> np.ndarray:
    m = np.zeros((n_users, n_items), bool)
    m[ev.user, ev.item] = True
    return m


def build_scores(
    world: World, win: Window, which: tuple[str, ...], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    u, i = world.n_users, world.n_items
    now = float(win.train.time.max())
    out = {}
    for name in which:
        if name == "random":
            out[name] = models.score_random(rng, u, i)
        elif name == "popularity":
            out[name] = models.score_popularity(win.train, u, i)
        elif name == "recent_popularity":
            out[name] = models.score_recent_popularity(win.train, u, i, now)
        elif name == "item_knn":
            out[name] = models.score_item_knn(win.train, u, i)
        elif name == "svd":
            out[name] = models.score_svd(win.train, u, i)
        elif name == "svd_ips":
            out[name] = models.score_svd(win.train, u, i, ips_alpha=config.IPS_ALPHA)
        elif name == "oracle":
            out[name] = models.score_oracle(world, win.t_eval)
    return out


def evaluate(
    world: World, win: Window, scores: dict[str, np.ndarray], seed: int, tag: str, k: int = config.K
) -> dict:
    """Per-user arrays and top-k lists for every model. Users must have a train and a holdout click."""
    u, i = world.n_users, world.n_items
    seen = dense(win.train, u, i)
    relevant = dense(win.hold, u, i) & ~seen
    avail = world.release <= win.end
    keep = seen.any(axis=1) & relevant.any(axis=1)
    true_prob = world.like_prob_matrix(win.t_eval)
    res: dict = {"keep": keep, "avail": avail, "seen": seen, "relevant": relevant, "models": {}}
    for name, s in scores.items():
        rng = _rng(seed, f"{tag}|topk|{name}")
        top = metrics.topk(s, seen, avail, k, rng)
        res["models"][name] = {
            "top": top,
            "ndcg": metrics.ndcg_at_k(top, relevant),
            "recall": metrics.recall_at_k(top, relevant),
            "hit": metrics.hit_rate_at_k(top, relevant),
            "true_like": metrics.true_like_rate(top, true_prob),
            "scores": s,
        }
    rand = res["models"].get("random")
    orac = res["models"].get("oracle")
    res["true_prob"] = true_prob
    res["random_true"] = float(rand["true_like"][keep].mean()) if rand else None
    res["oracle_true"] = float(orac["true_like"][keep].mean()) if orac else None
    return res


def mean_over_users(res: dict, name: str, key: str) -> float:
    x = res["models"][name][key][res["keep"]]
    return float(np.nanmean(x))


def run(
    results_dir: Path,
    fast: bool = False,
    seed: int = config.SEED,
    data_sha256: str | None = None,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    boots = 100 if fast else config.N_BOOT
    world = data.make_world(_rng(seed, "world"))
    events = data.simulate_events(_rng(seed, "events"), world)
    data.validate(world, events)
    t = events.time

    val = Window(
        events.select(t < config.VAL_START),
        events.select((t >= config.VAL_START) & (t < config.TEST_START)),
        config.TEST_START,
        (config.VAL_START + config.TEST_START) / 2,
    )
    test = Window(
        events.select(t < config.TEST_START),
        events.select(t >= config.TEST_START),
        config.HORIZON,
        (config.TEST_START + config.HORIZON) / 2,
    )

    # Validation picks the headline among the personalised candidates.
    val_res = evaluate(
        world,
        val,
        build_scores(world, val, config.CANDIDATES, _rng(seed, "val-build")),
        seed,
        "val",
    )
    val_ndcg = {n: mean_over_users(val_res, n, "ndcg") for n in config.CANDIDATES}
    headline = max(config.CANDIDATES, key=lambda n: (val_ndcg[n], -config.CANDIDATES.index(n)))
    reason = "highest NDCG@10 against held-out clicks on the validation window among item_knn, svd, svd_ips"

    res = evaluate(
        world,
        test,
        build_scores(world, test, config.ALL_MODELS, _rng(seed, "test-build")),
        seed,
        "test",
    )
    keep = res["keep"]
    n_avail = int(res["avail"].sum())
    span = res["oracle_true"] - res["random_true"]
    table = {}
    for name in config.ALL_MODELS:
        m = res["models"][name]
        tl = m["true_like"]
        table[name] = {
            "ndcg10": {
                "value": mean_over_users(res, name, "ndcg"),
                "ci95": list(
                    metrics.boot_mean_ci(m["ndcg"][keep], _rng(seed, f"ci|{name}"), boots)
                ),
            },
            "recall10": mean_over_users(res, name, "recall"),
            "hit10": mean_over_users(res, name, "hit"),
            "true_like10": {
                "value": float(tl[keep].mean()),
                "ci95": list(metrics.boot_mean_ci(tl[keep], _rng(seed, f"cit|{name}"), boots)),
            },
            "value_captured": (float(tl[keep].mean()) - res["random_true"]) / span,
            "coverage10": metrics.coverage(m["top"][keep], n_avail),
            "gini10": metrics.gini(m["top"][keep], world.n_items),
            "dev_ndcg10_validation": val_ndcg.get(name),
        }

    pop = res["models"]["popularity"]
    hm = res["models"][headline]
    d_obs = metrics.boot_diff_ci(hm["ndcg"][keep], pop["ndcg"][keep], _rng(seed, "gate-obs"), boots)
    d_true = metrics.boot_diff_ci(
        hm["true_like"][keep], pop["true_like"][keep], _rng(seed, "gate-true"), boots
    )
    obs_vec = [table[n]["ndcg10"]["value"] for n in config.ALL_MODELS]
    true_vec = [table[n]["true_like10"]["value"] for n in config.ALL_MODELS]
    rho = metrics.spearman(obs_vec, true_vec)
    checks = {
        "beats_popularity_on_observed_ndcg": bool(d_obs[0] > 0),
        "beats_popularity_on_true_like_rate": bool(d_true[0] > 0),
        "offline_metric_orders_models_like_truth": bool(rho >= config.SPEARMAN_FLOOR),
    }

    # Traps.
    u, i = world.n_users, world.n_items
    # (a) random split of events instead of a temporal split.
    rs_rng = _rng(seed, "random-split")
    in_train = rs_rng.random(len(events)) < 0.8
    rwin = Window(
        events.select(in_train), events.select(~in_train), config.HORIZON, config.HORIZON / 2
    )
    rres = evaluate(
        world,
        rwin,
        build_scores(world, rwin, config.ALL_MODELS, _rng(seed, "rs-build")),
        seed,
        "rs",
    )
    random_split = {n: mean_over_users(rres, n, "ndcg") for n in config.ALL_MODELS}
    # (b) sampled metrics.
    sampled = {}
    for n in config.ALL_MODELS:
        sh = metrics.sampled_hit_rate(
            res["models"][n]["scores"],
            res["relevant"],
            res["seen"],
            res["avail"],
            config.K,
            config.SAMPLED_NEGATIVES,
            _rng(seed, f"sampled|{n}"),
        )
        sampled[n] = {"sampled_hit10": float(np.nanmean(sh[keep])), "full_hit10": table[n]["hit10"]}
    s_vec = [sampled[n]["sampled_hit10"] for n in config.ALL_MODELS]
    f_vec = [sampled[n]["full_hit10"] for n in config.ALL_MODELS]
    # (d) share of test clicks on items the training window never saw.
    train_items = np.zeros(i, bool)
    train_items[test.train.item] = True
    hold_items = test.hold.item[keep[test.hold.user]]
    cold_share = float((~train_items[hold_items]).mean())
    # (e) exposure sweep.
    sweep = []
    for gamma in (0.0, 1.5) if fast else config.SWEEP_EXPOSURE:
        w2 = data.make_world(_rng(seed, f"sweep-world|{gamma}"), exposure_strength=gamma)
        e2 = data.simulate_events(_rng(seed, f"sweep-events|{gamma}"), w2)
        win2 = Window(
            e2.select(e2.time < config.TEST_START),
            e2.select(e2.time >= config.TEST_START),
            config.HORIZON,
            (config.TEST_START + config.HORIZON) / 2,
        )
        r2 = evaluate(
            w2,
            win2,
            build_scores(w2, win2, config.ALL_MODELS, _rng(seed, f"sweep-build|{gamma}")),
            seed,
            f"sweep|{gamma}",
        )
        o = [mean_over_users(r2, n, "ndcg") for n in config.ALL_MODELS]
        tr = [float(r2["models"][n]["true_like"][r2["keep"]].mean()) for n in config.ALL_MODELS]
        order_o = [config.ALL_MODELS[j] for j in np.argsort(o)[::-1]]
        sweep.append(
            {
                "exposure_strength": gamma,
                "spearman_observed_vs_true": metrics.spearman(o, tr),
                "observed_order": order_o,
                "true_order": [config.ALL_MODELS[j] for j in np.argsort(tr)[::-1]],
                "oracle_observed_ndcg": o[config.ALL_MODELS.index("oracle")],
                "popularity_observed_ndcg": o[config.ALL_MODELS.index("popularity")],
            }
        )

    order_obs = sorted(config.ALL_MODELS, key=lambda n: -table[n]["ndcg10"]["value"])
    order_true = sorted(config.ALL_MODELS, key=lambda n: -table[n]["true_like10"]["value"])
    traps = {
        "random_split_inflates_scores": {
            "ndcg10_random_split": random_split,
            "ndcg10_temporal_test": {n: table[n]["ndcg10"]["value"] for n in config.ALL_MODELS},
            "note": "A random split lets the model train on clicks from after the ones it is tested on. Scores rise and the gap between models changes.",
        },
        "sampled_metrics": {
            "by_model": sampled,
            "spearman_sampled_vs_full": metrics.spearman(s_vec, f_vec),
            "note": (
                f"Ranking each click against {config.SAMPLED_NEGATIVES} random items raises the hit rate of poorly ranked models most, and can reorder models (Krichene and Rendle, KDD 2020). "
                + (
                    "In this run it did not reorder them."
                    if metrics.spearman(s_vec, f_vec) > 0.999
                    else "In this run it did reorder them."
                )
            ),
        },
        "observed_metric_rewards_the_logging_policy": {
            "observed_order": order_obs,
            "true_order": order_true,
            "oracle_observed_ndcg10": table["oracle"]["ndcg10"]["value"],
            "popularity_observed_ndcg10": table["popularity"]["ndcg10"]["value"],
            "spearman_observed_vs_true": rho,
            "note": "Held-out clicks were exposed by a policy that favours popular items, so a model that shows popular items is rewarded whether or not users like them.",
        },
        "cold_start_ceiling": {
            "share_of_test_clicks_on_items_unseen_in_training": cold_share,
            "note": "No model trained on the past can recommend these. It caps recall.",
        },
        "exposure_sweep": sweep,
    }

    # Error analysis.
    act = np.asarray(test.train.user.size and np.bincount(test.train.user, minlength=u))
    segments = (
        ("1 to 4 clicks", (act >= 1) & (act <= 4)),
        ("5 to 14", (act >= 5) & (act <= 14)),
        ("15 or more", act >= 15),
    )
    seg_rows = []
    for label, m in segments:
        mk = keep & m
        seg_rows.append(
            {
                "segment": label,
                "users": int(mk.sum()),
                "headline_ndcg10": float(np.nanmean(hm["ndcg"][mk])),
                "popularity_ndcg10": float(np.nanmean(pop["ndcg"][mk])),
                "headline_true_like10": float(hm["true_like"][mk].mean()),
                "popularity_true_like10": float(pop["true_like"][mk].mean()),
            }
        )
    item_pop = np.bincount(test.train.item, minlength=i)
    avail_idx = np.nonzero(res["avail"])[0]
    tercile = np.digitize(item_pop, np.quantile(item_pop[avail_idx], [1 / 3, 2 / 3]))
    group_rows = []
    for g, label in enumerate(("least clicked third", "middle third", "most clicked third")):
        in_g = (tercile == g) & res["avail"]
        rel_g = res["relevant"] & in_g[None, :] & keep[:, None]
        n_pos = int(rel_g.sum())
        row = {"item_group": label, "held_out_clicks": n_pos}
        for nm in (headline, "popularity"):
            top = res["models"][nm]["top"]
            hits = rel_g[np.arange(u)[:, None], top].sum()
            row[f"{nm}_recall10"] = float(hits / n_pos) if n_pos else float("nan")
        group_rows.append(row)
    analysis = {
        "headline_model": headline,
        "selection_reason": reason,
        "validation_ndcg10": val_ndcg,
        "by_user_activity": seg_rows,
        "by_item_popularity": group_rows,
        "eligible_users": int(keep.sum()),
        "holdout_clicks": int(res["relevant"][keep].sum()),
    }

    metrics_out = {
        "project": config.PROJECT_NAME,
        "data_source": config.DATA_SOURCE,
        "fast_mode": bool(fast),
        "seed": seed,
        "versions": _versions(),
        "data": {
            "users": u,
            "items": i,
            "clicks": len(events),
            "train_clicks_test": len(test.train),
            "holdout_clicks_test": len(test.hold),
            "sha256": data_sha256,
        },
        "assumptions": {
            "exposure_strength": config.EXPOSURE_STRENGTH,
            "drift": config.DRIFT,
            "click_beta": config.BETA,
            "click_bias": config.BIAS,
            "ips_alpha": config.IPS_ALPHA,
            "spearman_floor": config.SPEARMAN_FLOOR,
            "k": config.K,
        },
        "models": table,
        "reference_true_like10": {"random": res["random_true"], "oracle": res["oracle_true"]},
        "headline": {
            "model": headline,
            "selected_by": reason,
            "metric": "ndcg10_against_held_out_clicks",
            "value": table[headline]["ndcg10"]["value"],
            "ci95": table[headline]["ndcg10"]["ci95"],
            "ci_method": f"percentile bootstrap over {int(keep.sum())} users, {boots} resamples",
            "minus_popularity_observed_ndcg10": {
                "value": table[headline]["ndcg10"]["value"]
                - table["popularity"]["ndcg10"]["value"],
                "ci95": list(d_obs),
            },
            "minus_popularity_true_like10": {
                "value": table[headline]["true_like10"]["value"]
                - table["popularity"]["true_like10"]["value"],
                "ci95": list(d_true),
            },
            "spearman_observed_vs_true": rho,
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
    _save_plot(table, results_dir)
    return metrics_out


def _save_plot(table: dict, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for n, v in table.items():
        ax.scatter(v["ndcg10"]["value"], v["true_like10"]["value"])
        ax.annotate(
            n,
            (v["ndcg10"]["value"], v["true_like10"]["value"]),
            fontsize=8,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax.set_xlabel("NDCG@10 against held-out clicks (observed)")
    ax.set_ylabel("true like rate at 10 (simulator only)")
    fig.tight_layout()
    fig.savefig(results_dir / "observed_vs_true.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate recommenders against held-out clicks and true value."
    )
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
        f"headline {h['model']}: NDCG@10 {h['value']:.4f}, Spearman observed vs true {h['spearman_observed_vs_true']:.2f}"
    )
    print(f"gate: {m['gate']['checks']} passed: {m['gate']['passed']}")


if __name__ == "__main__":
    main()
