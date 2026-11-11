"""A simulated catalogue, users with drifting taste, and clicks logged under a popularity-biased policy.

The code knows every user's true affinity for every item at every time. That makes it possible to
compare an offline metric (computed on logged clicks, as in real life) with the true value of a
recommendation (what a user would actually like), which real logs never show.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .config import CHECKSUM_FILE


class DataValidationError(ValueError):
    """Raised on a schema problem or a fingerprint mismatch."""


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class World:
    p: np.ndarray  # user taste, (users, dim)
    d: np.ndarray  # direction in which taste drifts, (users, dim)
    q: np.ndarray  # item factors, (items, dim)
    pop: np.ndarray  # item base popularity logit
    trend: np.ndarray  # item popularity trend over the horizon
    release: np.ndarray  # day each item appears
    activity: np.ndarray  # relative activity of each user
    exposure_strength: float

    @property
    def n_users(self) -> int:
        return len(self.p)

    @property
    def n_items(self) -> int:
        return len(self.q)

    def affinity_matrix(self, t: float) -> np.ndarray:
        """True affinity of every user for every item at day t. Shape (users, items)."""
        taste = self.p + config.DRIFT * (t / config.HORIZON) * self.d
        return taste @ self.q.T / np.sqrt(config.DIM)

    def like_prob_matrix(self, t: float) -> np.ndarray:
        return sigmoid(config.BETA * self.affinity_matrix(t) + config.BIAS)


@dataclass
class Events:
    user: np.ndarray
    item: np.ndarray
    time: np.ndarray

    def __len__(self) -> int:
        return len(self.user)

    def select(self, mask: np.ndarray) -> Events:
        return Events(self.user[mask], self.item[mask], self.time[mask])


def make_world(
    rng: np.random.Generator, exposure_strength: float = config.EXPOSURE_STRENGTH
) -> World:
    n_new = int(config.NEW_ITEM_SHARE * config.N_ITEMS)
    release = np.zeros(config.N_ITEMS)
    release[:n_new] = rng.uniform(1.0, config.HORIZON * 0.95, n_new)
    rng.shuffle(release)
    return World(
        p=rng.standard_normal((config.N_USERS, config.DIM)),
        d=rng.standard_normal((config.N_USERS, config.DIM)),
        q=rng.standard_normal((config.N_ITEMS, config.DIM)),
        pop=config.POP_SD * rng.standard_normal(config.N_ITEMS),
        trend=config.TREND_SD * rng.standard_normal(config.N_ITEMS),
        release=release,
        activity=np.exp(config.ACTIVITY_SD * rng.standard_normal(config.N_USERS)),
        exposure_strength=exposure_strength,
    )


def simulate_events(
    rng: np.random.Generator, world: World, n_exposures: int = config.N_EXPOSURES
) -> Events:
    """Show items under the logging policy, keep the clicks.

    The policy shows popular items more often (strength `exposure_strength`), so the logged clicks
    over-represent popular items even when they are no better liked.
    """
    n_bins = 50
    edges = np.linspace(0.0, config.HORIZON, n_bins + 1)
    users, items, times = [], [], []
    user_p = world.activity / world.activity.sum()
    user_cdf = np.cumsum(user_p)
    per_bin = n_exposures // n_bins
    for b in range(n_bins):
        t0, t1 = edges[b], edges[b + 1]
        t_mid = 0.5 * (t0 + t1)
        logits = world.exposure_strength * (world.pop + world.trend * t_mid / config.HORIZON)
        logits = np.where(world.release <= t0, logits, -np.inf)
        p = np.exp(logits - logits.max())
        cdf = np.cumsum(p / p.sum())
        u = np.minimum(np.searchsorted(user_cdf, rng.random(per_bin)), world.n_users - 1)
        i = np.minimum(np.searchsorted(cdf, rng.random(per_bin)), world.n_items - 1)
        t = rng.uniform(t0, t1, per_bin)
        taste = world.p[u] + config.DRIFT * (t / config.HORIZON)[:, None] * world.d[u]
        aff = (taste * world.q[i]).sum(axis=1) / np.sqrt(config.DIM)
        click = rng.random(per_bin) < sigmoid(config.BETA * aff + config.BIAS)
        users.append(u[click])
        items.append(i[click])
        times.append(t[click])
    ev = Events(np.concatenate(users), np.concatenate(items), np.concatenate(times))
    order = np.argsort(ev.time, kind="stable")
    return Events(ev.user[order], ev.item[order], ev.time[order])


def validate(world: World, events: Events) -> None:
    if len(events) == 0:
        raise DataValidationError("no events")
    if (events.item < 0).any() or (events.item >= world.n_items).any():
        raise DataValidationError("item out of range")
    if (events.time < world.release[events.item]).any():
        raise DataValidationError("a click happened before its item was released")
    if not (np.diff(events.time) >= 0).all():
        raise DataValidationError("events are not sorted by time")


def fingerprint(seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    w = make_world(rng)
    ev = simulate_events(rng, w, 5000)
    blob = np.concatenate([ev.user.astype(float), ev.item.astype(float), np.round(ev.time, 6)])
    return hashlib.sha256(blob.tobytes()).hexdigest()


def write_fingerprint(path: Path = CHECKSUM_FILE) -> str:
    digest = fingerprint()
    path.write_text(f"{digest}  generator_probe(seed=0)\n", encoding="utf-8")
    return digest


def verify_fingerprint(path: Path = CHECKSUM_FILE) -> str:
    digest = fingerprint()
    if path.exists():
        expected = path.read_text(encoding="utf-8").split()[0]
        if expected != digest:
            raise DataValidationError(
                f"generator fingerprint mismatch: expected {expected}, got {digest}"
            )
    return digest
