"""Audit your own judge. Reads a CSV of paired judge calls and prints bias estimates.

    python -m llm_judge_eval.audit judgments.csv
    python -m llm_judge_eval.audit --example example.csv   # write a simulated example file

Required columns, one row per judge call, every pair_id present in both orders:
  pair_id             any id shared by the two calls on the same pair
  order               0 if answer X was shown first, 1 if answer Y was shown first
  judge_prefers_first 1 if the judge picked the answer in the first slot, else 0
  human_prefers_x     1 if the human consensus prefers X, else 0 (same on both rows of a pair)
  len_x, len_y        length of each answer (characters, tokens, anything monotone)

The audit uses the human label as a stand-in for quality. That stand-in is noisy, which biases
the length estimate when length tracks quality. The length slope is per standard deviation of a
single answer's length. The simulated run in this project measures by
how much (see the model card).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data
from .metrics import boot_mean_ci, cluster_bias_ci, cohen_kappa

MIN_PAIRS = 30
REQUIRED = ("pair_id", "order", "judge_prefers_first", "human_prefers_x", "len_x", "len_y")


class AuditInputError(ValueError):
    """Raised when the CSV cannot be audited."""


def _paired_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise AuditInputError(f"missing columns: {missing}")
    if df[list(REQUIRED)].isna().any().any():
        raise AuditInputError("missing values")
    if not set(df["order"].unique()) <= {0, 1}:
        raise AuditInputError("order must be 0 or 1")
    if not set(df["judge_prefers_first"].unique()) <= {0, 1}:
        raise AuditInputError("judge_prefers_first must be 0 or 1")
    if not set(df["human_prefers_x"].unique()) <= {0, 1}:
        raise AuditInputError("human_prefers_x must be 0 or 1")
    counts = df.groupby("pair_id")["order"].agg(["nunique", "size"])
    if (counts["nunique"] != 2).any() or (counts["size"] != 2).any():
        raise AuditInputError("every pair_id needs exactly one call in each order")
    a = df[df["order"] == 0].set_index("pair_id").sort_index()
    b = df[df["order"] == 1].set_index("pair_id").sort_index()
    if not (a["human_prefers_x"] == b["human_prefers_x"]).all():
        raise AuditInputError("human_prefers_x must match across the two rows of a pair")
    out = pd.DataFrame(
        {
            "pick0": a["judge_prefers_first"].astype(bool),
            "pick1": b["judge_prefers_first"].astype(bool),
            "human_x": a["human_prefers_x"].astype(bool),
            "dl": (a["len_x"] - a["len_y"]).astype(float),
            "len_scale_sample": a["len_x"].astype(float),
            "len_y": a["len_y"].astype(float),
        }
    )
    return out


def basic_stats(p: pd.DataFrame) -> dict:
    """Counting statistics from the paired frame. No fitting, so small examples can be checked by hand."""
    v0 = p["pick0"].to_numpy()  # order 0 shows X first, so picking first means X
    v1 = ~p["pick1"].to_numpy()  # order 1 shows Y first, so picking first means Y
    human = p["human_x"].to_numpy()
    dl = p["dl"].to_numpy()
    swap = v0.astype(int) + v1.astype(int)
    decided = swap != 1
    return {
        "pairs": len(p),
        "position_bias_p_first_minus_half": float(
            (p["pick0"].mean() + p["pick1"].mean()) / 2 - 0.5
        ),
        "swap_inconsistency_rate": float((v0 != v1).mean()),
        "agreement_single_order": float(((v0 == human).mean() + (v1 == human).mean()) / 2),
        "agreement_swap_ties_half": float(np.where(decided, (swap == 2) == human, 0.5).mean()),
        "agreement_on_decided_pairs": float(((swap == 2)[decided] == human[decided]).mean())
        if decided.any()
        else float("nan"),
        "coverage_when_splits_go_to_humans": float(decided.mean()),
        "agreement_longer_answer": float(((dl > 0) == human).mean()),
        "kappa_decided_pairs_vs_human": cohen_kappa((swap == 2)[decided], human[decided])
        if decided.any()
        else float("nan"),
    }


def audit(df: pd.DataFrame, boots: int = config.N_BOOT_REG, seed: int = config.SEED) -> dict:
    """Bias and agreement estimates from paired judge calls. Needs at least 30 pairs."""
    p = _paired_frame(df)
    if len(p) < MIN_PAIRS:
        raise AuditInputError(f"need at least {MIN_PAIRS} pairs, got {len(p)}")
    rng = np.random.default_rng(seed)
    human = p["human_x"].to_numpy()
    v0, v1 = p["pick0"].to_numpy(), ~p["pick1"].to_numpy()
    swap = v0.astype(int) + v1.astype(int)
    swap_correct = np.where(swap == 1, 0.5, (swap == 2) == human)
    # Human label as the quality stand-in, signed to the first slot of each call.
    sign0 = np.where(human, 1.0, -1.0)
    # Length is scaled by the sd of a single answer's length, so the slope is per standard deviation.
    scale = float(np.concatenate([p["len_scale_sample"], p["len_y"]]).std())
    if scale == 0:
        raise AuditInputError("all answers have the same length")
    reg = cluster_bias_ci(
        p["pick0"].to_numpy(), p["pick1"].to_numpy(), sign0, p["dl"].to_numpy() / scale, rng, boots
    )
    out = basic_stats(p)
    out["agreement_swap_ci95"] = list(boot_mean_ci(swap_correct, rng, boots))
    out["regression_with_human_label_as_quality"] = reg
    return out


def write_example_csv(path: Path, n: int = 400, seed: int = config.SEED) -> None:
    """A simulated file in the required format, for trying the audit."""
    rng = np.random.default_rng([seed, 99])
    pairs = data.make_pairs(rng, n)
    human = data.consensus(data.human_votes(rng, pairs))
    v0, v1 = data.judge_calls(rng, pairs, data.JudgeParams())
    ids = np.arange(n)
    rows = []
    for order, pick in ((0, v0), (1, ~v1)):
        rows.append(
            pd.DataFrame(
                {
                    "pair_id": ids,
                    "order": order,
                    "judge_prefers_first": pick.astype(int),
                    "human_prefers_x": human.astype(int),
                    "len_x": np.round(500 + 100 * pairs.lx),
                    "len_y": np.round(500 + 100 * pairs.ly),
                }
            )
        )
    pd.concat(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit paired LLM judge calls.")
    parser.add_argument("csv", nargs="?", help="CSV of paired judge calls")
    parser.add_argument("--example", help="write a simulated example CSV to this path and exit")
    args = parser.parse_args()
    if args.example:
        write_example_csv(Path(args.example))
        print(f"wrote {args.example}")
        return
    if not args.csv:
        parser.error("give a CSV path or --example PATH")
    print(json.dumps(audit(pd.read_csv(args.csv)), indent=2))


if __name__ == "__main__":
    main()
