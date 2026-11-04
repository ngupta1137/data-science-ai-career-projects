"""Retrievers. All are deterministic and need no network or model download.

  bm25    : lexical scoring, implemented here from the standard formula
  tfidf   : cosine similarity of TF-IDF vectors
  lsa     : TF-IDF reduced with truncated SVD. A classical stand-in for dense retrieval:
            it can link words that co-occur in the corpus, but it is not a neural embedder
  hybrid  : reciprocal rank fusion of BM25 and LSA
  random  : seeded random order, the floor every retriever must beat

Ties are broken by document id, so rankings never depend on platform floating point noise.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import RRF_K, STOPWORDS

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens without stopwords. No stemming, on purpose."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def rank_order(scores: np.ndarray, doc_ids: list[str]) -> list[int]:
    """Indices from best to worst. Scores are rounded so near-ties fall back to doc id."""
    return sorted(range(len(doc_ids)), key=lambda i: (-round(float(scores[i]), 10), doc_ids[i]))


def idf_table(doc_tokens: list[list[str]]) -> dict[str, float]:
    n = len(doc_tokens)
    df: Counter = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))
    return {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}


class BM25:
    name = "bm25"

    def __init__(self, docs: list[dict], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        tokens = [tokenize(d["text"]) for d in docs]
        self.tf = [Counter(t) for t in tokens]
        self.lengths = np.array([len(t) for t in tokens], dtype=float)
        self.avg_len = float(self.lengths.mean())
        self.idf = idf_table(tokens)

    def scores(self, question: str) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for term in set(tokenize(question)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                    out[i] += idf * f * (self.k1 + 1) / norm
        return out


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, docs: list[dict]):
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in docs])

    def scores(self, question: str) -> np.ndarray:
        query = self.vectorizer.transform([question])
        return np.asarray((self.matrix @ query.T).todense()).ravel()


class LSARetriever:
    name = "lsa"

    def __init__(self, docs: list[dict], dims: int = 20, seed: int = 42):
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tfidf = self.vectorizer.fit_transform([d["text"] for d in docs])
        n_components = max(2, min(dims, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_components, algorithm="arpack", random_state=seed)
        self.matrix = normalize(self.svd.fit_transform(tfidf))
        self.dims = n_components

    def scores(self, question: str) -> np.ndarray:
        query = normalize(self.svd.transform(self.vectorizer.transform([question])))
        return (self.matrix @ query.T).ravel()


class RRFRetriever:
    name = "hybrid_rrf"

    def __init__(self, retrievers: list, doc_ids: list[str], k: int = RRF_K):
        self.retrievers, self.doc_ids, self.k = retrievers, doc_ids, k

    def scores(self, question: str) -> np.ndarray:
        fused = np.zeros(len(self.doc_ids))
        for retriever in self.retrievers:
            order = rank_order(retriever.scores(question), self.doc_ids)
            for position, index in enumerate(order, start=1):
                fused[index] += 1.0 / (self.k + position)
        return fused


class RandomRetriever:
    name = "random"

    def __init__(self, n_docs: int, seed: int = 42):
        self.n_docs, self.seed = n_docs, seed

    def scores(self, question: str) -> np.ndarray:
        rng = np.random.default_rng(self.seed + zlib.crc32(question.encode("utf-8")))
        return rng.random(self.n_docs)
