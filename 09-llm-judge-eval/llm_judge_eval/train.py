"""Audit a simulated pairwise LLM judge, pick a headline judge on dev, test it, write every result.

    python -m llm_judge_eval.train          # full run
    python -m llm_judge_eval.train --fast   # fewer bootstrap draws, marked

X and Y are the two answers in a pair. A call shows one of them first. Agreement is measured
against the human consensus of three annotators, and also against the true quality, which only a
simulation can supply.
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
from scipy import stats

from . import config, data, judge
from .data import JudgeParams, Pairs
from .metrics import (
    agreement,
    boot_diff_ci,
    cluster_bias_ci,
    cohen_kappa,
    wilson_ci,
)

CAVEATS = [
    "Everything is simulated. The judge, the humans and the biases are chosen by the author, so the sizes of the effects describe this setup, not GPT-4, Claude, or any real judge.",
    "No real model was called. The design is built; a run against a real judge and real human labels is designed, not done. The audit module accepts that data.",
    "Humans are modelled as unbiased noisy raters of true quality. Real annotators have their own length and style preferences, which this project does not model.",
    "Self-preference (a judge favouring its own family's answers) is not simulated. It needs real model outputs.",
    "Judge calls are independent draws given the pair. A real judge's repeated calls at temperature zero are not independent, so swap averaging helps less than it does here when the bias is deterministic.",
    "The 0.80 bar and the 5% outranking limit are assumptions. Change them and the verdict can change.",
    "Only pairwise preference is studied. Pointwise scores, rubric grading and multi-turn judging are not.",
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
class Sim:
    pairs: Pairs
    votes: np.ndarray
    cons: np.ndarray
    v0: np.ndarray
    v1: np.ndarray


def simulate(
    seed: int,
    label: str,
    n: int,
    params: JudgeParams,
    rho: float = config.LENGTH_QUALITY_CORR,
    x: tuple[float, float] = (0.0, 0.0),
    y: tuple[float, float] = (0.0, 0.0),
) -> Sim:
    rng = _rng(seed, label)
    pairs = data.make_pairs(rng, n, x=x, y=y, rho=rho)
    votes = data.human_votes(rng, pairs)
    v0, v1 = data.judge_calls(rng, pairs, params)
    return Sim(pairs, votes, data.consensus(votes), v0, v1)


def decide_all(sim: Sim, seed: int, label: str, cal) -> dict[str, np.ndarray]:
    rng = _rng(seed, label)
    n = len(sim.pairs)
    return {
        "single_order": judge.decide_single(rng, sim.v0, sim.v1),
        "swap_coin": judge.decide_swap(rng, sim.v0, sim.v1),
        "swap_calibrated": judge.decide_calibrated(cal, sim.v0, sim.v1, sim.pairs.dl),
        "length_only": judge.decide_length_only(sim.pairs.dl),
        "oracle": judge.decide_oracle(sim.pairs.dq),
        "coin_flip": judge.decide_coin(rng, n),
    }


def _wilson_block(correct: np.ndarray) -> dict:
    k, n = int(correct.sum()), len(correct)
    return {"value": k / n, "ci95": list(wilson_ci(k, n))}


def system_study(seed: int, params: JudgeParams, cal, boots: int) -> dict:
    """Round robin of the six systems. Does a judge rank them in the true order?"""
    names = list(config.SYSTEMS)
    truth_q = np.array([config.SYSTEMS[k]["quality"] for k in names])
    deciders = ("single_order", "swap_coin", "swap_calibrated", "length_only", "oracle")
    wins = {d: np.full((len(names), len(names)), np.nan) for d in deciders}
    raw: dict[str, dict[tuple[str, str], np.ndarray]] = {d: {} for d in deciders}
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            sim = simulate(
                seed,
                f"systems|{a}|{b}",
                config.SYSTEM_N,
                params,
                x=(config.SYSTEMS[a]["quality"], config.SYSTEMS[a]["length"]),
                y=(config.SYSTEMS[b]["quality"], config.SYSTEMS[b]["length"]),
            )
            dec = decide_all(sim, seed, f"systems-dec|{a}|{b}", cal)
            for d in deciders:
                wins[d][i, j] = dec[d].mean()
                wins[d][j, i] = 1.0 - dec[d].mean()
                raw[d][(a, b)] = dec[d]
    out: dict = {"order_by_truth": [names[k] for k in np.argsort(-truth_q)]}
    iv = names.index("V")
    for d in deciders:
        score = np.nanmean(wins[d], axis=1)
        tau = stats.kendalltau(score, truth_q).statistic
        shares = {}
        for better in config.BETTER_THAN_V:
            # raw[d][(better, V)] is True where `better` wins; V wins where it is False.
            boot_rng = _rng(seed, f"outrank|{d}|{better}")
            dec = raw[d][(better, "V")]
            idx = boot_rng.integers(0, len(dec), size=(boots, len(dec)))
            v_win = 1.0 - dec.astype(float)[idx].mean(axis=1)
            shares[better] = float((v_win > 0.5).mean())
        out[d] = {
            "borda_score": dict(zip(names, map(float, score), strict=True)),
            "order": [names[k] for k in np.argsort(-score)],
            "kendall_tau_vs_truth": float(tau),
            "v_win_rate_vs": {
                b: float(1.0 - wins[d][names.index(b), iv]) for b in config.BETTER_THAN_V
            },
            "share_of_resamples_v_beats": shares,
            "max_share_v_beats_better_system": max(shares.values()),
        }
    return out


def n_needed(p: float, floor: float, step: int = 50, cap: int = 20000) -> int | None:
    """Smallest eval-set size whose Wilson lower bound at agreement p reaches the floor."""
    if p <= floor:
        return None
    for n in range(step, cap + 1, step):
        if wilson_ci(p * n, n)[0] >= floor:
            return n
    return None


def sweep(seed: int, reps: int, n_dev: int, n_test: int) -> list[dict]:
    """Planted slot bias by planted length bias. Not part of the gate."""
    rows = []
    for pos in config.SWEEP_POS:
        for ln in config.SWEEP_LEN:
            params = JudgeParams(pos=pos, length=ln)
            acc = {k: [] for k in ("single_order", "swap_coin", "swap_calibrated", "length_only")}
            v_swap, v_cal = [], []
            for r in range(reps):
                tag = f"sweep|{pos}|{ln}|{r}"
                dev = simulate(seed, tag + "|dev", n_dev, params)
                test = simulate(seed, tag + "|test", n_test, params)
                cal = judge.fit_calibrator(dev.v0, dev.v1, dev.pairs.dl, dev.cons)
                dec = decide_all(test, seed, tag + "|dec", cal)
                for k in acc:
                    acc[k].append(agreement(dec[k], test.cons))
                vs = simulate(
                    seed,
                    tag + "|vs",
                    config.SYSTEM_N,
                    params,
                    x=(config.SYSTEMS["S4"]["quality"], 0.0),
                    y=(config.SYSTEMS["V"]["quality"], config.SYSTEMS["V"]["length"]),
                )
                vd = decide_all(vs, seed, tag + "|vsdec", cal)
                v_swap.append(1.0 - vd["swap_coin"].mean())
                v_cal.append(1.0 - vd["swap_calibrated"].mean())
            rows.append(
                {
                    "pos_bias": pos,
                    "length_bias": ln,
                    "replications": reps,
                    "agreement": {k: float(np.mean(v)) for k, v in acc.items()},
                    "v_win_rate_vs_s4_swap_coin": float(np.mean(v_swap)),
                    "v_win_rate_vs_s4_calibrated": float(np.mean(v_cal)),
                }
            )
    return rows


def _save_plot(table: dict, floor: float, results_dir: Path) -> None:
    names = list(table)
    vals = [table[n]["agreement_consensus"]["value"] for n in names]
    err = [
        [v - table[n]["agreement_consensus"]["ci95"][0] for n, v in zip(names, vals, strict=True)],
        [table[n]["agreement_consensus"]["ci95"][1] - v for n, v in zip(names, vals, strict=True)],
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, vals, yerr=err, capsize=4, color="#4c78a8")
    ax.axhline(floor, color="#d62728", linestyle="--", label=f"floor {floor:.2f}")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("agreement with human consensus (test)")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(results_dir / "agreement.png", dpi=120)
    plt.close(fig)


def run(
    results_dir: Path,
    fast: bool = False,
    seed: int = config.SEED,
    data_sha256: str | None = None,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    boots = 100 if fast else config.N_BOOT
    boots_reg = 30 if fast else config.N_BOOT_REG
    params = JudgeParams()

    dev = simulate(seed, "dev", config.N_DEV, params)
    test = simulate(seed, "test", config.N_TEST, params)
    cal = judge.fit_calibrator(dev.v0, dev.v1, dev.pairs.dl, dev.cons)
    dev_dec = decide_all(dev, seed, "dev-dec", cal)
    test_dec = decide_all(test, seed, "test-dec", cal)

    dev_agree = {k: agreement(dev_dec[k], dev.cons) for k in config.CANDIDATES}
    headline = max(config.CANDIDATES, key=lambda k: (dev_agree[k], -config.CANDIDATES.index(k)))
    reason = (
        "higher agreement with the human consensus on dev between single_order and swap_calibrated"
    )

    cons, truth = test.cons, test.pairs.truth
    tie = judge.tie_rate(test.v0, test.v1)
    table = {}
    for name in (*config.CANDIDATES, *config.REFERENCES):
        dec = test_dec[name]
        table[name] = {
            "agreement_consensus": _wilson_block(dec == cons),
            "agreement_truth": _wilson_block(dec == truth),
            "kappa_vs_consensus": cohen_kappa(dec, cons),
            "dev_agreement_consensus": agreement(dev_dec[name], dev.cons),
        }
    table["swap_coin"]["split_verdict_rate"] = tie
    table["swap_calibrated"]["split_verdict_rate"] = tie
    verdict, decided = judge.decide_triage(test.v0, test.v1)
    triage = {
        "coverage": float(decided.mean()),
        "agreement_consensus_on_decided": _wilson_block(verdict[decided] == cons[decided]),
        "agreement_truth_on_decided": _wilson_block(verdict[decided] == truth[decided]),
        "agreement_consensus_on_split": _wilson_block(verdict[~decided] == cons[~decided]),
        "note": "Pairs where the two orders disagree are sent to humans. Agreement is on the pairs the judge keeps.",
    }

    # Bias recovery. The planted values are known; a practitioner only has the human proxy.
    pick0, pick1 = test.v0, ~test.v1
    bias_rng = _rng(seed, "bias")
    oracle_fit = cluster_bias_ci(pick0, pick1, test.pairs.dq, test.pairs.dl, bias_rng, boots_reg)
    proxy_fit = cluster_bias_ci(
        pick0, pick1, np.where(cons, 1.0, -1.0), test.pairs.dl, bias_rng, boots_reg
    )
    bias = {
        "planted": {
            "position": params.pos,
            "quality_slope": params.beta,
            "length_slope": params.length,
        },
        "recovered_with_true_quality": oracle_fit,
        "recovered_with_human_label_as_quality": proxy_fit,
        "p_first_slot_picked": float((pick0.mean() + pick1.mean()) / 2),
        "note": "The slot bias is recovered either way. The human label is a noisy stand-in for quality, so the length slope is overstated when length tracks quality.",
    }

    systems = system_study(seed, params, cal, boots)

    correct = {k: test_dec[k] == cons for k in table}
    h_corr = correct[headline]
    checks = {
        "agreement_lower_bound_at_least_floor": bool(
            table[headline]["agreement_consensus"]["ci95"][0] >= config.AGREEMENT_FLOOR
        ),
        "beats_longer_answer_baseline": bool(
            boot_diff_ci(h_corr, correct["length_only"], _rng(seed, "gate-diff"), boots)[0] > 0
        ),
        "verbose_system_not_over_ranked": bool(
            systems[headline]["max_share_v_beats_better_system"] <= config.OUTRANK_LIMIT
        ),
    }
    diff_lo, diff_hi = boot_diff_ci(h_corr, correct["length_only"], _rng(seed, "gate-diff"), boots)

    # Traps, measured.
    fixed = np.where(cons, test.v0, ~test.v1)
    balanced = np.concatenate([test.v0 == cons, (~test.v1) == ~cons])
    confound = {}
    for rho in (0.3, 0.0, -0.3):
        s = simulate(seed, f"rho|{rho}", config.N_TEST, params, rho=rho)
        d = decide_all(s, seed, f"rho-dec|{rho}", cal)
        confound[f"rho={rho:+.1f}"] = {
            "swap_coin": _wilson_block(d["swap_coin"] == s.cons),
            "length_only": _wilson_block(d["length_only"] == s.cons),
            "oracle": _wilson_block(d["oracle"] == s.cons),
        }
    p_head = table[headline]["agreement_consensus"]["value"]
    traps = {
        "human_label_is_not_truth": {
            "oracle_agreement_with_consensus": table["oracle"]["agreement_consensus"],
            "consensus_agreement_with_truth": _wilson_block(cons == truth),
            "single_annotator_agreement_with_truth": float(
                np.mean([(test.votes[:, k] == truth).mean() for k in range(config.N_ANNOTATORS)])
            ),
            "note": "A perfect judge only agrees with the consensus this often. Agreement near 1.0 is impossible.",
        },
        "fixed_slot_evaluation": {
            "agreement_when_consensus_winner_always_first": _wilson_block(fixed),
            "agreement_when_slots_balanced": _wilson_block(balanced),
            "note": "If the answer humans prefer always sits in the first slot, a judge that likes the first slot looks more accurate than it is.",
        },
        "length_confounded_eval_set": {
            "by_length_quality_correlation": confound,
            "note": "A length-biased judge scores higher when longer answers tend to be better, and lower when they do not.",
        },
        "eval_set_size": {
            "headline_agreement": p_head,
            "wilson_lower_bound_by_n": {
                str(n): wilson_ci(p_head * n, n)[0] for n in config.EVAL_SIZES
            },
            "n_needed_for_lower_bound_to_reach_floor": n_needed(p_head, config.AGREEMENT_FLOOR),
            "note": "Even a good judge cannot be certified on a small human-labelled set.",
        },
        "verbose_system_over_ranked": {
            d: {
                "order": systems[d]["order"],
                "v_win_rate_vs": systems[d]["v_win_rate_vs"],
                "share_of_resamples_v_beats": systems[d]["share_of_resamples_v_beats"],
            }
            for d in ("single_order", "swap_coin", "swap_calibrated", "length_only", "oracle")
        },
    }

    # Error analysis.
    dq = np.abs(test.pairs.dq)
    bands = (
        ("close, |dq| < 0.3", dq < 0.3),
        ("mid, 0.3 to 1.0", (dq >= 0.3) & (dq < 1.0)),
        ("clear, >= 1.0", dq >= 1.0),
    )
    h_dec = test_dec[headline]
    disagree = h_dec != cons
    swap_split = test.v0 != test.v1
    longer = test.pairs.dl > 0
    band_rows = []
    for label, m in bands:
        band_rows.append(
            {
                "band": label,
                "pairs": int(m.sum()),
                "headline_agreement_truth": float((h_dec[m] == truth[m]).mean()),
                "headline_agreement_consensus": float((h_dec[m] == cons[m]).mean()),
                "consensus_agreement_truth": float((cons[m] == truth[m]).mean()),
                "split_verdict_rate": float(swap_split[m].mean()),
            }
        )
    analysis = {
        "headline_judge": headline,
        "selection_reason": reason,
        "dev_agreement": dev_agree,
        "by_quality_gap": band_rows,
        "disagreements_with_consensus": {
            "count": int(disagree.sum()),
            "share_where_judge_matches_truth": float((h_dec[disagree] == truth[disagree]).mean()),
            "note": "Where the judge and the consensus differ, the share above is how often the judge still matched the true order, so the consensus was the one that erred.",
        },
        "raw_judge_follows_longer_answer": {
            "single_order_call_picks_longer": float(
                np.concatenate(
                    [np.where(longer, pick0, ~pick0), np.where(longer, ~pick1, pick1)]
                ).mean()
            ),
            "note": "Share of calls where the judge picked the longer of the two answers. 0.50 would mean no pull.",
        },
        "calibrator": {
            "swap_score_weight": float(cal.coef_[0][0]),
            "length_weight": float(cal.coef_[0][1]),
        },
        "systems": {
            k: systems[k]
            for k in (
                "order_by_truth",
                "single_order",
                "swap_coin",
                "swap_calibrated",
                "length_only",
                "oracle",
            )
        },
        "sweep": sweep(seed, 1 if fast else config.SWEEP_REPS, config.N_DEV, config.N_TEST),
    }

    metrics = {
        "project": config.PROJECT_NAME,
        "data_source": config.DATA_SOURCE,
        "fast_mode": bool(fast),
        "seed": seed,
        "versions": _versions(),
        "data": {
            "dev_pairs": len(dev.pairs),
            "test_pairs": len(test.pairs),
            "annotators": config.N_ANNOTATORS,
            "sha256": data_sha256,
        },
        "assumptions": {
            "human_beta": config.HUMAN_BETA,
            "judge_beta": params.beta,
            "position_bias": params.pos,
            "length_bias": params.length,
            "length_quality_corr": config.LENGTH_QUALITY_CORR,
            "agreement_floor": config.AGREEMENT_FLOOR,
            "outrank_limit": config.OUTRANK_LIMIT,
        },
        "candidates": table,
        "triage": triage,
        "headline": {
            "judge": headline,
            "selected_by": reason,
            "metric": "agreement_with_human_consensus",
            "value": table[headline]["agreement_consensus"]["value"],
            "ci95": table[headline]["agreement_consensus"]["ci95"],
            "ci_method": "Wilson interval over test pairs",
            "minus_longer_answer_baseline": {
                "value": agreement(h_dec, cons) - agreement(test_dec["length_only"], cons),
                "ci95": [diff_lo, diff_hi],
            },
            "max_share_v_beats_better_system": systems[headline]["max_share_v_beats_better_system"],
        },
        "gate": {"rule": config.GATE_RULE, "checks": checks, "passed": bool(all(checks.values()))},
        "bias": bias,
        "traps": traps,
        "caveats": CAVEATS,
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(_clean(metrics), indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "error_analysis.json").write_text(
        json.dumps(_clean(analysis), indent=2) + "\n", encoding="utf-8"
    )
    _save_plot(table, config.AGREEMENT_FLOOR, results_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a simulated pairwise LLM judge.")
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
    print(f"headline {h['judge']}: agreement {h['value']:.3f} {h['ci95']}")
    print(f"gate: {m['gate']['checks']} passed: {m['gate']['passed']}")


if __name__ == "__main__":
    main()
