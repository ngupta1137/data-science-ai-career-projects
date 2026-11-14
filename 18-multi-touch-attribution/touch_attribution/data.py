"""Simulated customer journeys with known counterfactual attribution ground truth."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from . import config


class DataValidationError(ValueError):
    """Raised when journey dataset fails schema or integrity checks."""


@dataclass
class Journey:
    user_id: int
    path: list[str]
    converted: int  # 0 or 1
    conversion_prob: float
    organic_prob: float
    true_channel_lifts: dict[str, float]  # channel -> incremental probability lift


@dataclass
class JourneyDataset:
    journeys: list[Journey]
    conversions: int
    total_journeys: int
    channel_touch_counts: dict[str, int]
    sha256: str

    def __len__(self) -> int:
        return len(self.journeys)


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# Transition matrix between marketing touchpoints
TRANSITION_PROBS = {
    "start": {"display": 0.40, "social": 0.35, "search": 0.20, "email": 0.05},
    "display": {"display": 0.10, "social": 0.40, "search": 0.40, "email": 0.10},
    "social": {"display": 0.10, "social": 0.20, "search": 0.50, "email": 0.20},
    "search": {"display": 0.10, "social": 0.20, "search": 0.30, "email": 0.40},
    "email": {"display": 0.10, "social": 0.20, "search": 0.40, "email": 0.30},
}


def _draw_path(rng: np.random.Generator, max_touches: int = 5) -> list[str]:
    """Draw a sequential path of touches using Markov state transitions."""
    n_touches = int(rng.integers(1, max_touches + 1))
    path = []

    # Choose first touch
    start_probs = [TRANSITION_PROBS["start"][ch] for ch in config.CHANNELS]
    current = str(rng.choice(config.CHANNELS, p=start_probs))
    path.append(current)

    for _ in range(n_touches - 1):
        step_probs = [TRANSITION_PROBS[current][ch] for ch in config.CHANNELS]
        current = str(rng.choice(config.CHANNELS, p=step_probs))
        path.append(current)

    return path


def _compute_logit(channels: set[str], latent_u: float) -> float:
    """Compute conversion log-odds given active touchpoints and user predisposition."""
    base = config.BASE_LOGIT + config.LATENT_INTENT_SCALE * latent_u
    ad_lift = sum(config.CHANNEL_LIFTS[ch] for ch in channels)

    synergy_lift = 0.0
    for (ch1, ch2), syn_value in config.SYNERGIES.items():
        if ch1 in channels and ch2 in channels:
            synergy_lift += syn_value

    return float(base + ad_lift + synergy_lift)


def generate_journeys(n_journeys: int, seed: int = config.SEED) -> JourneyDataset:
    """Generate simulated customer journeys with known causal counterfactuals."""
    rng = np.random.default_rng(seed)
    journeys: list[Journey] = []
    conversions = 0
    touch_counts: dict[str, int] = dict.fromkeys(config.CHANNELS, 0)

    for user_id in range(n_journeys):
        path = _draw_path(rng)
        channels_in_path = set(path)
        for ch in path:
            touch_counts[ch] += 1

        latent_u = float(rng.standard_normal())
        full_logit = _compute_logit(channels_in_path, latent_u)
        prob = float(sigmoid(full_logit))
        converted = int(rng.random() < prob)
        conversions += converted

        # Organic baseline conversion prob (no ad touches)
        org_logit = _compute_logit(set(), latent_u)
        org_prob = float(sigmoid(org_logit))

        # Counterfactual channel removal lifts: P(convert|path) - P(convert|path \ {c})
        lifts: dict[str, float] = {}
        for ch in config.CHANNELS:
            if ch in channels_in_path:
                cf_channels = channels_in_path - {ch}
                cf_logit = _compute_logit(cf_channels, latent_u)
                cf_prob = float(sigmoid(cf_logit))
                lifts[ch] = max(0.0, prob - cf_prob)
            else:
                lifts[ch] = 0.0

        journeys.append(
            Journey(
                user_id=user_id,
                path=path,
                converted=converted,
                conversion_prob=prob,
                organic_prob=org_prob,
                true_channel_lifts=lifts,
            )
        )

    # Compute deterministic SHA256
    hasher = hashlib.sha256()
    hasher.update(f"journeys={n_journeys},seed={seed},conv={conversions}".encode())
    for ch in sorted(config.CHANNELS):
        hasher.update(f"{ch}:{touch_counts[ch]}".encode())
    sha = hasher.hexdigest()

    dataset = JourneyDataset(
        journeys=journeys,
        conversions=conversions,
        total_journeys=n_journeys,
        channel_touch_counts=touch_counts,
        sha256=sha,
    )
    validate(dataset)
    return dataset


def calculate_true_causal_credit(dataset: JourneyDataset) -> dict[str, float]:
    """Calculate the population true counterfactual incremental conversion credit per channel."""
    channel_totals: dict[str, float] = dict.fromkeys(config.CHANNELS, 0.0)
    for j in dataset.journeys:
        if j.converted:
            for ch in config.CHANNELS:
                channel_totals[ch] += j.true_channel_lifts[ch]

    total_incremental = sum(channel_totals.values())
    if total_incremental <= 0:
        return {ch: 1.0 / len(config.CHANNELS) for ch in config.CHANNELS}

    return {ch: channel_totals[ch] / total_incremental for ch in config.CHANNELS}


def validate(dataset: JourneyDataset) -> None:
    """Validate journey schema and integrity."""
    if dataset.total_journeys <= 0:
        raise DataValidationError("Dataset must contain at least one journey.")
    if dataset.conversions <= 0:
        raise DataValidationError("Dataset must have at least one conversion.")
    if dataset.conversions > dataset.total_journeys:
        raise DataValidationError("Conversions cannot exceed total journeys.")
    for ch in config.CHANNELS:
        if dataset.channel_touch_counts.get(ch, 0) == 0:
            raise DataValidationError(f"Channel {ch} has zero recorded touches.")
    for j in dataset.journeys:
        if not j.path:
            raise DataValidationError(f"Journey {j.user_id} has empty touchpoint path.")
        if not (0.0 <= j.conversion_prob <= 1.0):
            raise DataValidationError(f"Invalid conversion probability for user {j.user_id}.")
        if j.converted not in (0, 1):
            raise DataValidationError(f"Invalid converted flag for user {j.user_id}.")
