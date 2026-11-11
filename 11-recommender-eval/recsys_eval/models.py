"""Recommenders. Each takes training events and returns a score for every (user, item).

Scores of items a user already clicked are masked later, in the evaluation.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from . import config
from .data import Events, World


def interaction_matrix(ev: Events, n_users: int, n_items: int) -> sp.csr_matrix:
    """Binary user by item matrix of clicks. Repeat clicks count once."""
    m = sp.coo_matrix((np.ones(len(ev)), (ev.user, ev.item)), shape=(n_users, n_items)).tocsr()
    m.data[:] = 1.0
    return m


def item_counts(ev: Events, n_items: int, since: float | None = None) -> np.ndarray:
    mask = np.ones(len(ev), bool) if since is None else ev.time >= since
    return np.bincount(ev.item[mask], minlength=n_items).astype(float)


def score_random(rng: np.random.Generator, n_users: int, n_items: int) -> np.ndarray:
    return rng.random((n_users, n_items))


def score_popularity(ev: Events, n_users: int, n_items: int) -> np.ndarray:
    return np.tile(item_counts(ev, n_items), (n_users, 1))


def score_recent_popularity(ev: Events, n_users: int, n_items: int, now: float) -> np.ndarray:
    since = now - config.RECENT_WINDOW
    return np.tile(item_counts(ev, n_items, since=since), (n_users, 1))


def score_item_knn(ev: Events, n_users: int, n_items: int) -> np.ndarray:
    """Cosine similarity between items from co-clicks. Score = similarity to the user's clicked items."""
    x = interaction_matrix(ev, n_users, n_items)
    norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    xn = x @ sp.diags(1.0 / norms)
    sim = (xn.T @ xn).toarray()
    np.fill_diagonal(sim, 0.0)
    return np.asarray(x @ sim)


def score_svd(ev: Events, n_users: int, n_items: int, ips_alpha: float = 0.0) -> np.ndarray:
    """Truncated SVD of the click matrix. With ips_alpha > 0, clicks are weighted by the inverse
    of the item's popularity to the power alpha, to undo some of the exposure bias."""
    x = interaction_matrix(ev, n_users, n_items).astype(float)
    if ips_alpha > 0:
        pop = np.asarray(x.sum(axis=0)).ravel()
        scale = np.maximum(pop, 1.0) / max(pop.mean(), 1.0)
        x = x @ sp.diags(scale ** (-ips_alpha))
    k = min(config.SVD_RANK, min(x.shape) - 1)
    u, s, vt = svds(x, k=k, random_state=0)
    return (u * s) @ vt


def score_oracle(world: World, t: float) -> np.ndarray:
    """The true like probability at time t. Not a real model: the ceiling for true value."""
    return world.like_prob_matrix(t)
