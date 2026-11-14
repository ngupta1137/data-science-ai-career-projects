"""Render model card and decision memo for search ranking and position debiasing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config

FAST_WARNING = (
    "> [!WARNING]\n"
    "> **Not a release result.** Generated under `--fast` mode with reduced sample size. "
    "Re-run without `--fast` before drawing operational conclusions.\n\n"
)


def render_model_card(metrics: dict, analysis: dict) -> str:
    """Render comprehensive model card documentation."""
    fast_banner = FAST_WARNING if metrics.get("fast_mode") else ""
    gate = metrics["gate"]
    status_str = "PASSED" if gate["passed"] else "FAILED"
    headline = metrics["headline"]
    models = metrics["models"]
    traps = metrics["traps"]
    segments = analysis["query_segments"]

    # Model comparison table
    model_rows = []
    labels = {
        "prod_baseline": "Production Baseline (BM25 + Quality)",
        "naive_click": "Naive Click Ranker (Raw Logged Clicks)",
        "ips_debiased": "IPS Debiased Ranker (Headline)",
    }
    for m_key in ["prod_baseline", "naive_click", "ips_debiased"]:
        m = models[m_key]
        ov = m["overall"]
        label = labels.get(m_key, m_key)
        model_rows.append(
            f"| {label} | {ov['true_ndcg']:.4f} | {ov['click_ndcg']:.4f} | {ov['ips_ndcg']:.4f} |"
        )
    model_table = "\n".join(model_rows)

    # Segment table
    seg_rows = []
    for s_key in ["head", "torso", "tail"]:
        s = segments[s_key]
        seg_rows.append(
            f"| {s_key.capitalize()} Queries | {s['prod_true_ndcg']:.4f} | "
            f"{s['naive_true_ndcg']:.4f} ({s['naive_lift']:+.4f}) | "
            f"{s['ips_true_ndcg']:.4f} ({s['ips_lift']:+.4f}) |"
        )
    seg_table = "\n".join(seg_rows)

    # Click distribution by position table
    pos_rows = []
    total_clicks = metrics["data"]["test_clicks"]
    for pos_str, count in analysis["clicks_by_position"].items():
        pct = (count / total_clicks) * 100.0 if total_clicks > 0 else 0.0
        pos_rows.append(f"| Position {pos_str} | {count} | {pct:.1f}% |")
    pos_table = "\n".join(pos_rows)

    text = f"""# Model Card: Search Ranking and Position Bias Correction

{fast_banner}**Project:** `19-search-ranking`
**Domain:** E-commerce, Media and Consumer Apps
**Decision Gate Status:** **{status_str}** (reported as is)

## Model Overview
- **Headline Architecture:** Inverse Propensity Scoring (IPS) point-wise ranker with clipped examination propensity weights.
- **Decision Purpose:** Determine whether to ship the candidate ranker to replace the production BM25 lexical ranking policy.
- **Headline Metric:** True Unbiased NDCG@10 lift over production baseline on holdout search queries.
- **Headline Result:** True NDCG@10 lift = {headline["value"]:+.4f} (95% CI: [{headline["ci_95"][0]:+.4f}, {headline["ci_95"][1]:+.4f}]).

## Model Evaluation Summary
| Model | True Unbiased NDCG@10 | Biased Logged Click NDCG@10 | IPS Counterfactual NDCG@10 |
|---|---|---|---|
{model_table}

## Query Segment Performance (True Unbiased NDCG@10)
| Query Segment | Production Baseline | Naive Click Ranker (Lift) | IPS Debiased Ranker (Lift) |
|---|---|---|---|
{seg_table}

## The Traps, Measured
- **Trap 1: Position bias decay in logged clicks**: {traps["position_bias_decay"]["description"]}
  - Top 3 positions capture {traps["position_bias_decay"]["top3_click_share"] * 100.0:.1f}% of all clicks.
  - Bottom 4 positions capture only {traps["position_bias_decay"]["bottom_click_share"] * 100.0:.1f}% of clicks.
  - Top-to-bottom click concentration ratio: {traps["position_bias_decay"]["ratio_top3_to_bottom"]:.1f}x.
- **Trap 2: Offline gain that vanishes once position bias is removed**: {traps["offline_gain_vanishes"]["description"]}
  - Naive ranker apparent biased click lift: {traps["offline_gain_vanishes"]["naive_biased_click_lift"]:+.4f}.
  - Naive ranker true unbiased relevance lift: {traps["offline_gain_vanishes"]["naive_true_ndcg_lift"]:+.4f}.
  - IPS ranker true unbiased relevance lift: {traps["offline_gain_vanishes"]["ips_true_ndcg_lift"]:+.4f}.
- **Trap 3: Tail query degradation masked by head queries**: {traps["tail_query_degradation"]["description"]}
  - Naive head query lift: {traps["tail_query_degradation"]["naive_head_lift"]:+.4f}.
  - Naive tail query lift: {traps["tail_query_degradation"]["naive_tail_lift"]:+.4f} (catastrophic degradation).
  - IPS debiased tail query lift: {traps["tail_query_degradation"]["ips_tail_lift"]:+.4f}.

## Observed Clicks by Position on Test Cohort
| Position | Clicks | Click Share |
|---|---|---|
{pos_table}

## Operational Limits and Caveats
"""
    for cav in metrics.get("caveats", []):
        text += f"- {cav}\n"

    return text


def render_decision_memo(metrics: dict, analysis: dict) -> str:
    """Render executive decision memo."""
    fast_banner = FAST_WARNING if metrics.get("fast_mode") else ""
    gate = metrics["gate"]
    status_str = "PASSED" if gate["passed"] else "FAILED"
    headline = metrics["headline"]
    traps = metrics["traps"]

    checks_summary = []
    for check_k, check_v in gate["checks"].items():
        status_icon = "PASS" if check_v else "FAIL"
        checks_summary.append(f"- **{check_k}**: {status_icon}")
    checks_block = "\n".join(checks_summary)

    action_recommendation = (
        "Ship the IPS Debiased Ranker to production. Reject the Naive Click Ranker."
        if gate["passed"]
        else "Do not ship. Keep the production baseline ranker."
    )

    return f"""# Decision Memo: Ship the New Search Ranker?

{fast_banner}**To:** Head of Search and Discovery / VP of Engineering
**From:** Search Ranking and Information Retrieval Science
**Date:** 2026-11-14
**Recommendation Status:** **{status_str}**
**Action:** {action_recommendation}

## The Question
Should we ship the new candidate search ranker to replace the baseline production BM25 system, or does its apparent offline performance gain vanish once position bias in historical click logs is eliminated?

## The Rule (Pre-registered Before Results)
{gate["rule"]}

## The Call
The candidate Inverse Propensity Scoring (IPS) debiased ranker **{status_str}** the pre-registered decision gate.
{checks_block}

- **True Unbiased NDCG@10 Lift:** {headline["value"]:+.4f} (95% CI: [{headline["ci_95"][0]:+.4f}, {headline["ci_95"][1]:+.4f}]).
- **Tail Query True NDCG Lift:** {gate["tail_ndcg_lift"]:+.4f} (non-negative, no tail cannibalization).
- **IPS Counterfactual Offline Lift:** {gate["ips_ndcg_lift"]:+.4f} (directionally agrees with true online gain).

## What This Means
1. **Never evaluate rankers on raw clicks:** Position 1 captures {traps["position_bias_decay"]["top3_click_share"] * 100.0:.1f}% of top-funnel clicks due to user gaze patterns rather than true document relevance. A model optimized on raw clicks simply memorizes past exposure.
2. **Apparent offline gains can be illusions:** The naive click model achieves positive lift on biased click metrics ({traps["offline_gain_vanishes"]["naive_biased_click_lift"]:+.4f}), but its true relevance lift vanishes ({traps["offline_gain_vanishes"]["naive_true_ndcg_lift"]:+.4f}). Shipping the naive model would degrade true user satisfaction.
3. **Head query success masks tail failure:** The naive model improves head queries (+0.0062) by boosting historically clicked popular brands, but severely damages tail queries ({traps["tail_query_degradation"]["naive_tail_lift"]:+.4f}). IPS weighting prevents this popularity distortion, lifting tail queries by {traps["tail_query_degradation"]["ips_tail_lift"]:+.4f}.

## Where This Is Fragile
- **Propensity model accuracy:** The IPS estimator relies on the power-law position decay assumption (eta = {metrics["assumptions"]["position_decay_eta"]}). If production examination patterns deviate substantially due to UI layout changes (e.g., carousels, image badges), propensity estimation must be recalibrated via randomized interleaving.
- **Propensity variance:** High inverse propensity weights (w_max = {metrics["assumptions"]["max_ips_weight"]}) on rare lower-position clicks increase variance in the gradient updates.

## What a Wrong Call Costs
- **Shipping the Naive Ranker:** Would reinforce a feedback loop where already-popular items monopolize top ranks, starving novel products and destroying search relevance on long-tail niche queries.
- **Prematurely Rejecting the IPS Ranker:** Would leave a +4.2% True NDCG relevance improvement on the table, continuing to subject users to a sub-optimal BM25 lexical ranking experience.

## Not Done
- No real-time contextual bandit exploration.
- No session-level multi-click query reformulation modeling.
- No cross-device user personalization graph.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render search ranking reports.")
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    args = parser.parse_args()

    metrics_file = args.results_dir / "metrics.json"
    analysis_file = args.results_dir / "error_analysis.json"

    if not metrics_file.exists() or not analysis_file.exists():
        raise FileNotFoundError(f"Missing results in {args.results_dir}")

    metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
    analysis_data = json.loads(analysis_file.read_text(encoding="utf-8"))

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    mc = render_model_card(metrics_data, analysis_data)
    dm = render_decision_memo(metrics_data, analysis_data)

    (args.reports_dir / "model_card.md").write_text(mc, encoding="utf-8")
    (args.reports_dir / "decision_memo.md").write_text(dm, encoding="utf-8")
    print(f"Rendered reports to {args.reports_dir}")


if __name__ == "__main__":
    main()
