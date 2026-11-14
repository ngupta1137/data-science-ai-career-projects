"""Ranking models for search: Production baseline, Naive click, and IPS debiased."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import config
from .data import CandidateDoc, RankingDataset, SearchQuery


class ProductionRanker:
    """Production baseline ranker driven by lexical BM25 match and merchant quality."""

    name: str = "prod_baseline"

    def fit(self, dataset: RankingDataset) -> ProductionRanker:
        # Pre-existing production policy; fixed heuristics
        return self

    def score_candidates(self, query: SearchQuery) -> np.ndarray:
        return np.array([c.prod_score for c in query.candidates], dtype=float)

    def rank_candidates(self, query: SearchQuery) -> list[CandidateDoc]:
        scores = self.score_candidates(query)
        order = np.argsort(-scores)
        return [query.candidates[i] for i in order]


class NaiveClickRanker:
    """Candidate ranker trained on raw logged clicks without propensity weighting."""

    name: str = "naive_click"
    feature_names: ClassVar[list[str]] = ["bm25", "quality", "pop_bias"]

    def __init__(self, c_reg: float = 0.1, seed: int = config.SEED) -> None:
        self.c_reg = c_reg
        self.seed = seed
        self.model = LogisticRegression(C=c_reg, max_iter=500, random_state=seed)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def _extract_features(self, candidates: list[CandidateDoc]) -> np.ndarray:
        return np.array(
            [[getattr(c, col) for col in self.feature_names] for c in candidates],
            dtype=float,
        )

    def fit(self, dataset: RankingDataset) -> NaiveClickRanker:
        all_candidates = [c for q in dataset.queries for c in q.candidates]
        X = self._extract_features(all_candidates)
        y = np.array([c.click for c in all_candidates], dtype=int)

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        X_std = (X - self.mean) / self.std

        self.model.fit(X_std, y)
        return self

    def score_candidates(self, query: SearchQuery) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Model must be fitted before scoring.")
        X = self._extract_features(query.candidates)
        X_std = (X - self.mean) / self.std
        return self.model.decision_function(X_std)

    def rank_candidates(self, query: SearchQuery) -> list[CandidateDoc]:
        scores = self.score_candidates(query)
        order = np.argsort(-scores)
        return [query.candidates[i] for i in order]


class IPSDebiasedRanker:
    """Candidate headline ranker trained with Inverse Propensity Scoring (IPS)."""

    name: str = "ips_debiased"
    feature_names: ClassVar[list[str]] = ["semantic", "bm25", "quality", "pop_bias"]

    def __init__(
        self,
        c_reg: float = 0.1,
        max_weight: float = config.MAX_IPS_WEIGHT,
        seed: int = config.SEED,
    ) -> None:
        self.c_reg = c_reg
        self.max_weight = max_weight
        self.seed = seed
        self.model = LogisticRegression(C=c_reg, max_iter=500, random_state=seed)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def _extract_features(self, candidates: list[CandidateDoc]) -> np.ndarray:
        return np.array(
            [[getattr(c, col) for col in self.feature_names] for c in candidates],
            dtype=float,
        )

    def fit(self, dataset: RankingDataset) -> IPSDebiasedRanker:
        all_candidates = [c for q in dataset.queries for c in q.candidates]
        X = self._extract_features(all_candidates)
        y = np.array([c.click for c in all_candidates], dtype=int)

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        X_std = (X - self.mean) / self.std

        # Inverse propensity sample weights: w_i = 1 / P(examine | pos_i)
        raw_weights = np.array([1.0 / c.exam_prob for c in all_candidates], dtype=float)
        clipped_weights = np.clip(raw_weights, 1.0, self.max_weight)
        # Scale weights to keep average weight 1.0
        sample_weights = clipped_weights / float(np.mean(clipped_weights))

        self.model.fit(X_std, y, sample_weight=sample_weights)
        return self

    def score_candidates(self, query: SearchQuery) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Model must be fitted before scoring.")
        X = self._extract_features(query.candidates)
        X_std = (X - self.mean) / self.std
        return self.model.decision_function(X_std)

    def rank_candidates(self, query: SearchQuery) -> list[CandidateDoc]:
        scores = self.score_candidates(query)
        order = np.argsort(-scores)
        return [query.candidates[i] for i in order]
