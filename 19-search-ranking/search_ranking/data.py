"""Simulated search impressions with position-biased clicks and known true relevance."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from . import config


class DataValidationError(ValueError):
    """Raised when ranking dataset fails schema or integrity validation."""


@dataclass
class CandidateDoc:
    query_id: int
    doc_id: int
    pos: int  # 1-indexed presentation position in logged session
    exam_prob: float  # P(examine | pos)
    true_rel: int  # 0, 1, 2, or 3
    semantic: float  # Deep semantic embedding match score
    bm25: float  # Lexical match score
    quality: float  # Merchant/product quality rating
    pop_bias: float  # Historical popularity/exposure bias
    prod_score: float  # Production ranker score
    log_score: float  # Logged score with exploration noise
    click: int  # 0 or 1 observed click


@dataclass
class SearchQuery:
    query_id: int
    query_type: str  # "head", "torso", or "tail"
    candidates: list[CandidateDoc]


@dataclass
class RankingDataset:
    queries: list[SearchQuery]
    total_queries: int
    total_impressions: int
    total_clicks: int
    clicks_by_position: dict[int, int]
    sha256: str

    def __len__(self) -> int:
        return len(self.queries)


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))


def get_examination_curve(
    k: int = config.CANDIDATES_PER_QUERY, eta: float = config.POSITION_DECAY_ETA
) -> np.ndarray:
    """Compute position examination curve under power law decay."""
    ranks = np.arange(1, k + 1, dtype=float)
    return 1.0 / (ranks**eta)


def generate_ranking_data(
    n_queries: int,
    seed: int = config.SEED,
    k: int = config.CANDIDATES_PER_QUERY,
) -> RankingDataset:
    """Generate simulated search ranking impressions with known ground truth."""
    rng = np.random.default_rng(seed)
    exam_curve = get_examination_curve(k=k)

    q_types = list(config.QUERY_PROBS.keys())
    q_probs = [config.QUERY_PROBS[qt] for qt in q_types]

    queries: list[SearchQuery] = []
    total_clicks = 0
    clicks_by_pos: dict[int, int] = dict.fromkeys(range(1, k + 1), 0)

    for qid in range(n_queries):
        q_type = str(rng.choice(q_types, p=q_probs))

        # True latent relevance distribution depends on query type
        if q_type == "tail":
            p_rel = [0.65, 0.22, 0.10, 0.03]
        elif q_type == "torso":
            p_rel = [0.55, 0.26, 0.14, 0.05]
        else:
            p_rel = [0.45, 0.30, 0.18, 0.07]

        true_rels = rng.choice(config.RELEVANCE_LEVELS, size=k, p=p_rel)
        if int(np.max(true_rels)) == 0:
            # Guarantee at least one marginally relevant item per query
            true_rels[int(rng.integers(0, k))] = 1

        # Feature synthesis
        semantic = 1.0 * true_rels + rng.normal(0, 0.8, size=k)
        bm25 = 0.5 * true_rels + rng.normal(0, 0.9, size=k)
        quality = 0.4 * true_rels + rng.normal(0, 0.7, size=k)

        # Baseline production ranker: BM25 lexical match (65%) and Quality (35%)
        prod_score = 0.65 * bm25 + 0.35 * quality

        # Historical exposure / popularity bias:
        # High-popularity items in old catalog were heavily exposed at top positions.
        # In tail queries, popular items act as irrelevant distractors.
        pop_bias = np.zeros(k, dtype=float)
        prod_ranks = np.argsort(-prod_score)
        for p_idx, d_idx in enumerate(prod_ranks):
            distractor = (
                -0.5 * float(true_rels[d_idx])
                if q_type == "tail"
                else 0.1 * float(true_rels[d_idx])
            )
            pop_bias[d_idx] = 2.2 * exam_curve[p_idx] + distractor + float(rng.normal(0, 0.2))

        # Logging policy applies rank exploration noise
        log_score = prod_score + rng.normal(0, 0.7, size=k)
        presented_order = np.argsort(-log_score)

        candidates: list[CandidateDoc] = []
        for pos_idx, doc_idx in enumerate(presented_order):
            pos = pos_idx + 1
            rel = int(true_rels[doc_idx])

            # User attraction: depends on true relevance and brand recognition
            attract_logit = -2.6 + 1.2 * rel + 0.55 * pop_bias[doc_idx]
            attract_prob = float(sigmoid(attract_logit))
            exam_prob = float(exam_curve[pos_idx])
            click_prob = exam_prob * attract_prob
            click = int(rng.random() < click_prob)

            total_clicks += click
            clicks_by_pos[pos] += click

            candidates.append(
                CandidateDoc(
                    query_id=qid,
                    doc_id=int(doc_idx),
                    pos=pos,
                    exam_prob=exam_prob,
                    true_rel=rel,
                    semantic=float(semantic[doc_idx]),
                    bm25=float(bm25[doc_idx]),
                    quality=float(quality[doc_idx]),
                    pop_bias=float(pop_bias[doc_idx]),
                    prod_score=float(prod_score[doc_idx]),
                    log_score=float(log_score[doc_idx]),
                    click=click,
                )
            )

        queries.append(SearchQuery(query_id=qid, query_type=q_type, candidates=candidates))

    # Compute deterministic SHA256 fingerprint
    hasher = hashlib.sha256()
    hasher.update(f"queries={n_queries},seed={seed},k={k},clicks={total_clicks}".encode())
    for pos in range(1, k + 1):
        hasher.update(f"pos{pos}:{clicks_by_pos[pos]}".encode())
    sha = hasher.hexdigest()

    dataset = RankingDataset(
        queries=queries,
        total_queries=n_queries,
        total_impressions=n_queries * k,
        total_clicks=total_clicks,
        clicks_by_position=clicks_by_pos,
        sha256=sha,
    )
    validate(dataset)
    return dataset


def validate(dataset: RankingDataset) -> None:
    """Validate ranking dataset schema and logical constraints."""
    if dataset.total_queries <= 0:
        raise DataValidationError("Dataset must contain at least one query.")
    if dataset.total_impressions != dataset.total_queries * config.CANDIDATES_PER_QUERY:
        raise DataValidationError("Total impressions mismatch query count.")
    if dataset.total_clicks <= 0:
        raise DataValidationError("Dataset has zero recorded clicks.")
    if dataset.total_clicks > dataset.total_impressions:
        raise DataValidationError("Clicks cannot exceed total impressions.")
    for pos, count in dataset.clicks_by_position.items():
        if count < 0:
            raise DataValidationError(f"Negative click count for position {pos}.")
    for q in dataset.queries:
        if len(q.candidates) != config.CANDIDATES_PER_QUERY:
            raise DataValidationError(
                f"Query {q.query_id} has {len(q.candidates)} candidates, expected {config.CANDIDATES_PER_QUERY}."
            )
        positions = [c.pos for c in q.candidates]
        if sorted(positions) != list(range(1, config.CANDIDATES_PER_QUERY + 1)):
            raise DataValidationError(f"Query {q.query_id} has malformed candidate positions.")
        for c in q.candidates:
            if not (0.0 <= c.exam_prob <= 1.0):
                raise DataValidationError(f"Invalid examination probability {c.exam_prob}.")
            if c.click not in (0, 1):
                raise DataValidationError(f"Invalid click flag {c.click}.")
            if c.true_rel not in config.RELEVANCE_LEVELS:
                raise DataValidationError(f"Invalid relevance grade {c.true_rel}.")
