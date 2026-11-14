"""Render model card and decision memo for multi-touch attribution."""

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
    budget = metrics["budget_allocation"]
    traps = metrics["traps"]

    # Channel shares table
    rows = []
    for m_name in [
        "last_touch",
        "first_touch",
        "linear",
        "position_based",
        "time_decay",
        "markov",
        "oracle",
    ]:
        m = models[m_name]
        sh = m["channel_shares"]
        mae_str = f"{m['share_mae']:.4f}" if m_name != "oracle" else "0.0000"
        spearman_str = f"{m['share_spearman']:.4f}" if m_name != "oracle" else "1.0000"
        disp = f"{sh['display'] * 100.1:.1f}%"
        soc = f"{sh['social'] * 100.0:.1f}%"
        srch = f"{sh['search'] * 100.0:.1f}%"
        em = f"{sh['email'] * 100.0:.1f}%"
        rows.append(
            f"| {m_name.replace('_', ' ').title()} | {disp} | {soc} | {srch} | {em} | {mae_str} | {spearman_str} |"
        )
    shares_table = "\n".join(rows)

    # Segments table
    seg_rows = []
    for k, v in analysis["journey_length_segments"].items():
        label = k.replace("length_", "Touches: ")
        seg_rows.append(
            f"| {label} | {v['total_journeys']} | {v['conversions']} | {v['conversion_rate'] * 100.0:.1f}% |"
        )
    seg_table = "\n".join(seg_rows)

    text = f"""# Model Card: Multi-Touch Attribution for Budget Allocation

{fast_banner}**Project:** `18-multi-touch-attribution`
**Domain:** Marketing and Growth
**Decision Gate Status:** **{status_str}** (reported as is)

## Model Overview
- **Headline Architecture:** First-order Markov chain multi-touch attribution model with counterfactual channel removal effects.
- **Decision Purpose:** Determine optimal channel credit shares to guide marketing acquisition spend across Display, Social, Paid Search, and Email.
- **Headline Metric:** Mean Absolute Error (MAE) across channel credit shares relative to true counterfactual Shapley credit on holdout customer journeys.
- **Headline Result:** Share MAE = {headline["value"]:.4f} (95% CI: [{headline["ci_95"][0]:.4f}, {headline["ci_95"][1]:.4f}]).

## Attributed Channel Credit Shares
| Model | Display | Social | Search | Email | Share MAE | Spearman Rank Corr |
|---|---|---|---|---|---|---|
{shares_table}

## Budget Reallocation Impact
Evaluating acquisition spend of ${budget["reallocation_budget"]:,.0f} reallocated according to channel credit weights:
- **Last-Touch Allocation Conversions:** {budget["last_touch_conversions"]:,.1f} incremental conversions.
- **Markov Allocation Conversions:** {budget["markov_conversions"]:,.1f} incremental conversions.
- **Relative Lift:** {budget["relative_gain_pct"]:+.2f}% incremental conversion yield.

## The Traps, Measured
- **Trap 1: Last-touch attribution bias**: {traps["last_touch_bias"]["finding"]}
  - Display true share: {traps["last_touch_bias"]["display_true_credit_share"] * 100.0:.1f}%, Last-Touch share: {traps["last_touch_bias"]["display_last_touch_share"] * 100.0:.1f}%.
  - Search true share: {traps["last_touch_bias"]["search_true_credit_share"] * 100.0:.1f}%, Last-Touch share: {traps["last_touch_bias"]["search_last_touch_share"] * 100.0:.1f}%.
- **Trap 2: Ignoring organic baseline conversion**: {traps["ignoring_organic_baseline"]["finding"]}
  - Organic conversion share: {traps["ignoring_organic_baseline"]["organic_conversion_share"] * 100.0:.1f}% of total conversions ({traps["ignoring_organic_baseline"]["expected_organic_conversions"]:,.0f} out of {traps["ignoring_organic_baseline"]["total_test_conversions"]:,}).
- **Trap 3: Linear attribution dilution**: {traps["linear_dilution"]["finding"]}
  - Linear share MAE: {traps["linear_dilution"]["linear_share_mae"]:.4f}.

## Journey Length Error Analysis
| Segment | Total Journeys | Conversions | Conversion Rate |
|---|---|---|---|
{seg_table}

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
    budget = metrics["budget_allocation"]
    traps = metrics["traps"]

    checks_summary = []
    for check_k, check_v in gate["checks"].items():
        status_icon = "PASS" if check_v else "FAIL"
        checks_summary.append(f"- **{check_k}**: {status_icon}")
    checks_block = "\n".join(checks_summary)

    return f"""# Decision Memo: Multi-Touch Attribution Channel Reallocation

{fast_banner}**To:** Chief Marketing Officer / Head of Growth
**From:** Marketing Science and Analytics
**Date:** 2026-11-14
**Recommendation Status:** **{status_str}**

## The Question
Which marketing channels truly drive incremental conversions, and should we abandon heuristic last-touch attribution in favor of a first-order Markov chain model to reallocate our acquisition budget?

## The Rule (Pre-registered Before Results)
{gate["rule"]}

## The Call
The headline Markov attribution model **{status_str}** the pre-registered decision gate.
{checks_block}

- **Headline Model Share MAE:** {headline["value"]:.4f} (95% CI: [{headline["ci_95"][0]:.4f}, {headline["ci_95"][1]:.4f}]).
- **Last-Touch Share MAE:** {metrics["models"]["last_touch"]["share_mae"]:.4f}.
- **Budget Reallocation Yield:** {budget["markov_conversions"]:,.1f} conversions vs {budget["last_touch_conversions"]:,.1f} under last touch ({budget["relative_gain_pct"]:+.2f}% gain).

## What This Means
1. **First-order Markov attribution improves top-of-funnel visibility:** Last touch severely under-rewards Display ({traps["last_touch_bias"]["display_last_touch_share"] * 100.0:.1f}% vs {traps["last_touch_bias"]["display_true_credit_share"] * 100.0:.1f}% true causal contribution) because Display acts as an early awareness touchpoint.
2. **Bottom-of-funnel channels absorb unearned credit:** Last-touch awards Search and Email outsized shares because they sit immediately prior to checkout.
3. **Organic conversions distort overall ROI:** Roughly {traps["ignoring_organic_baseline"]["organic_conversion_share"] * 100.0:.1f}% of conversions represent organic customer demand that would occur even with zero ad exposure. Rule-based models that assign 100% of revenue to paid media overstate true marketing incrementality.

## Where This Is Fragile
- **Memoryless assumption:** First-order Markov chains treat transition probabilities as dependent only on the immediate prior state, ignoring journey duration and cumulative multi-touch exposure effects.
- **Static auction pricing:** The budget simulation assumes fixed unit costs per touch ($0.50 for Display, $2.50 for Search), ignoring programmatic bidding competition that drives up marginal CPMs as spend scales.

## What a Wrong Call Costs
- If we stay on naive Last Touch, we will continue starving top-of-funnel customer acquisition, resulting in shrinking top-funnel audience pools over time.
- If we prematurely deploy an uncalibrated algorithmic model without randomized geo-lift or holdout validation, we risk misallocating media budget based on observational correlation rather than causal incrementality.

## Not Done
- No cookie-less identity resolution or cross-device identity graphs.
- No dynamic bidding response curves.
- No long-term brand equity carryover modeled past the multi-week purchase cycle.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render attribution reports.")
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    args = parser.parse_args()

    metrics_file = args.results_dir / "metrics.json"
    analysis_file = args.results_dir / "error_analysis.json"

    if not metrics_file.exists() or not analysis_file.exists():
        raise FileNotFoundError(f"Missing results in {args.results_dir}")

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_file.read_text(encoding="utf-8"))

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    mc = render_model_card(metrics, analysis)
    dm = render_decision_memo(metrics, analysis)

    (args.reports_dir / "model_card.md").write_text(mc, encoding="utf-8")
    (args.reports_dir / "decision_memo.md").write_text(dm, encoding="utf-8")
    print(f"Rendered reports to {args.reports_dir}")


if __name__ == "__main__":
    main()
