"""Ranking metrics, counterfactual evaluation, and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np

from . import config
from .data import RankingDataset


def compute_dcg_at_k(
    gains: list[float] | np.ndarray, k: int = config.CANDIDATES_PER_QUERY
) -> float:
    """Compute Discounted Cumulative Gain at rank K."""
    arr = np.asarray(gains, dtype=float)[:k]
    if len(arr) == 0:
        return 0.0
    discounts = np.log2(np.arange(1, len(arr) + 1) + 1.0)
    return float(np.sum((2.0**arr - 1.0) / discounts))


def compute_ndcg_at_k(
    gains: list[float] | np.ndarray,
    ideal_gains: list[float] | np.ndarray,
    k: int = config.CANDIDATES_PER_QUERY,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K."""
    dcg = compute_dcg_at_k(gains, k=k)
    idcg = compute_dcg_at_k(sorted(ideal_gains, reverse=True), k=k)
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def evaluate_query(
    ranked_candidates: list,
    k: int = config.CANDIDATES_PER_QUERY,
) -> dict[str, float]:
    """Evaluate True NDCG, Logged Click NDCG, and IPS Counterfactual NDCG for a query."""
    # Ideal gains
    ideal_true = [c.true_rel for c in ranked_candidates]
    ideal_click = [c.click for c in ranked_candidates]
    ideal_ips = [c.click / c.exam_prob for c in ranked_candidates]

    # Model ranked gains
    true_gains = [c.true_rel for c in ranked_candidates]
    click_gains = [c.click for c in ranked_candidates]
    ips_gains = [c.click / c.exam_prob for c in ranked_candidates]

    return {
        "true_ndcg": compute_ndcg_at_k(true_gains, ideal_true, k=k),
        "click_ndcg": compute_ndcg_at_k(click_gains, ideal_click, k=k),
        "ips_ndcg": compute_ndcg_at_k(ips_gains, ideal_ips, k=k),
    }


def evaluate_ranker(
    model,
    dataset: RankingDataset,
    k: int = config.CANDIDATES_PER_QUERY,
) -> dict:
    """Evaluate ranking model across all queries in a dataset."""
    true_scores: list[float] = []
    click_scores: list[float] = []
    ips_scores: list[float] = []

    seg_true: dict[str, list[float]] = {qt: [] for qt in config.QUERY_PROBS}
    seg_click: dict[str, list[float]] = {qt: [] for qt in config.QUERY_PROBS}
    seg_ips: dict[str, list[float]] = {qt: [] for qt in config.QUERY_PROBS}

    for query in dataset.queries:
        ranked = model.rank_candidates(query)
        res = evaluate_query(ranked, k=k)

        true_scores.append(res["true_ndcg"])
        click_scores.append(res["click_ndcg"])
        ips_scores.append(res["ips_ndcg"])

        qt = query.query_type
        if qt in seg_true:
            seg_true[qt].append(res["true_ndcg"])
            seg_click[qt].append(res["click_ndcg"])
            seg_ips[qt].append(res["ips_ndcg"])

    return {
        "name": getattr(model, "name", "unknown"),
        "overall": {
            "true_ndcg": float(np.mean(true_scores)),
            "click_ndcg": float(np.mean(click_scores)),
            "ips_ndcg": float(np.mean(ips_scores)),
        },
        "query_true_ndcg": true_scores,
        "query_click_ndcg": click_scores,
        "query_ips_ndcg": ips_scores,
        "segments": {
            qt: {
                "count": len(seg_true[qt]),
                "true_ndcg": float(np.mean(seg_true[qt])) if seg_true[qt] else 0.0,
                "click_ndcg": float(np.mean(seg_click[qt])) if seg_click[qt] else 0.0,
                "ips_ndcg": float(np.mean(seg_ips[qt])) if seg_ips[qt] else 0.0,
            }
            for qt in config.QUERY_PROBS
        },
    }


def bootstrap_paired_lift_ci(
    candidate_scores: list[float],
    baseline_scores: list[float],
    n_boot: int = 500,
    seed: int = config.SEED,
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval on paired NDCG lift."""
    diffs = np.array(candidate_scores, dtype=float) - np.array(baseline_scores, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(diffs[indices], axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def evaluate_decision_gate(
    headline_eval: dict,
    baseline_eval: dict,
    seed: int = config.SEED,
) -> dict:
    """Pre-registered decision gate check."""
    head_true = headline_eval["overall"]["true_ndcg"]
    base_true = baseline_eval["overall"]["true_ndcg"]
    true_lift = head_true - base_true

    ci_low, ci_high = bootstrap_paired_lift_ci(
        headline_eval["query_true_ndcg"],
        baseline_eval["query_true_ndcg"],
        seed=seed,
    )

    tail_head = headline_eval["segments"]["tail"]["true_ndcg"]
    tail_base = baseline_eval["segments"]["tail"]["true_ndcg"]
    tail_lift = tail_head - tail_base

    head_ips = headline_eval["overall"]["ips_ndcg"]
    base_ips = baseline_eval["overall"]["ips_ndcg"]
    ips_lift = head_ips - base_ips

    # Directional agreement: sign of offline IPS lift matches sign of true lift
    directional_agreement = (ips_lift > 0) == (true_lift > 0)

    checks = {
        "unbiased_ndcg_lift_meets_threshold": bool(true_lift >= config.MIN_TRUE_NDCG_LIFT),
        "ci_lower_bound_positive": bool(ci_low > 0.0),
        "no_tail_query_degradation": bool(tail_lift >= config.MIN_TAIL_NDCG_LIFT),
        "debiased_agreement_ratio": bool(directional_agreement),
    }

    return {
        "rule": config.GATE_RULE,
        "headline_model": config.HEADLINE_MODEL,
        "true_ndcg_lift": float(true_lift),
        "true_ndcg_lift_ci_95": [float(ci_low), float(ci_high)],
        "tail_ndcg_lift": float(tail_lift),
        "ips_ndcg_lift": float(ips_lift),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def compute_traps(
    dataset: RankingDataset,
    prod_eval: dict,
    naive_eval: dict,
    ips_eval: dict,
) -> dict:
    """Measure the three pre-registered traps."""
    # Trap 1: Position bias decay in logged clicks
    total_clicks = dataset.total_clicks
    top3_clicks = sum(dataset.clicks_by_position.get(p, 0) for p in [1, 2, 3])
    bottom_clicks = sum(dataset.clicks_by_position.get(p, 0) for p in range(7, 11))
    top3_share = float(top3_clicks / total_clicks) if total_clicks > 0 else 0.0
    bottom_share = float(bottom_clicks / total_clicks) if total_clicks > 0 else 0.0

    # Trap 2: Offline gain that vanishes once position bias is removed
    naive_click_lift = naive_eval["overall"]["click_ndcg"] - prod_eval["overall"]["click_ndcg"]
    naive_true_lift = naive_eval["overall"]["true_ndcg"] - prod_eval["overall"]["true_ndcg"]
    ips_true_lift = ips_eval["overall"]["true_ndcg"] - prod_eval["overall"]["true_ndcg"]
    gain_vanished = bool(naive_click_lift > 0 and naive_true_lift <= 0.005)

    # Trap 3: Tail query degradation masked by head queries
    naive_head_lift = (
        naive_eval["segments"]["head"]["true_ndcg"] - prod_eval["segments"]["head"]["true_ndcg"]
    )
    naive_tail_lift = (
        naive_eval["segments"]["tail"]["true_ndcg"] - prod_eval["segments"]["tail"]["true_ndcg"]
    )
    ips_tail_lift = (
        ips_eval["segments"]["tail"]["true_ndcg"] - prod_eval["segments"]["tail"]["true_ndcg"]
    )

    return {
        "position_bias_decay": {
            "top3_click_share": float(top3_share),
            "bottom_click_share": float(bottom_share),
            "ratio_top3_to_bottom": float(top3_share / bottom_share) if bottom_share > 0 else 0.0,
            "description": (
                "Clicks are overwhelmingly concentrated at top positions due to examination "
                "bias, with the top 3 positions capturing the vast majority of clicks."
            ),
        },
        "offline_gain_vanishes": {
            "naive_biased_click_lift": float(naive_click_lift),
            "naive_true_ndcg_lift": float(naive_true_lift),
            "ips_true_ndcg_lift": float(ips_true_lift),
            "gain_vanished": gain_vanished,
            "description": (
                "Naive rankers trained on raw clicks appear competitive on biased click metrics, "
                "but their apparent offline gain vanishes or reverses on true relevance."
            ),
        },
        "tail_query_degradation": {
            "naive_head_lift": float(naive_head_lift),
            "naive_tail_lift": float(naive_tail_lift),
            "ips_tail_lift": float(ips_tail_lift),
            "description": (
                "Historical popularity features cause naive rankers to degrade tail and niche queries "
                "by promoting irrelevant brand distractors, while IPS debiasing preserves tail relevance."
            ),
        },
    }
