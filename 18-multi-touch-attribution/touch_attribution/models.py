"""Attribution models: heuristic baselines, Markov chains, and causal oracle."""

from __future__ import annotations

import numpy as np

from . import config, data


class LastTouchAttribution:
    """Attributes 100% credit to the final touchpoint before conversion."""

    def __init__(self) -> None:
        self.channel_shares_: dict[str, float] = {}

    def fit(self, dataset: data.JourneyDataset) -> LastTouchAttribution:
        credits: dict[str, float] = dict.fromkeys(config.CHANNELS, 0.0)
        total = 0.0

        for j in dataset.journeys:
            if j.converted and j.path:
                last_ch = j.path[-1]
                credits[last_ch] += 1.0
                total += 1.0

        if total > 0:
            self.channel_shares_ = {ch: credits[ch] / total for ch in config.CHANNELS}
        else:
            self.channel_shares_ = dict.fromkeys(config.CHANNELS, 1.0 / len(config.CHANNELS))
        return self


class FirstTouchAttribution:
    """Attributes 100% credit to the first touchpoint in the journey."""

    def __init__(self) -> None:
        self.channel_shares_: dict[str, float] = {}

    def fit(self, dataset: data.JourneyDataset) -> FirstTouchAttribution:
        credits: dict[str, float] = dict.fromkeys(config.CHANNELS, 0.0)
        total = 0.0

        for j in dataset.journeys:
            if j.converted and j.path:
                first_ch = j.path[0]
                credits[first_ch] += 1.0
                total += 1.0

        if total > 0:
            self.channel_shares_ = {ch: credits[ch] / total for ch in config.CHANNELS}
        else:
            self.channel_shares_ = dict.fromkeys(config.CHANNELS, 1.0 / len(config.CHANNELS))
        return self


class LinearAttribution:
    """Attributes equal credit (1/K) across all touches in converting journeys."""

    def __init__(self) -> None:
        self.channel_shares_: dict[str, float] = {}

    def fit(self, dataset: data.JourneyDataset) -> LinearAttribution:
        credits: dict[str, float] = dict.fromkeys(config.CHANNELS, 0.0)
        total = 0.0

        for j in dataset.journeys:
            if j.converted and j.path:
                weight = 1.0 / len(j.path)
                for ch in j.path:
                    credits[ch] += weight
                    total += weight

        if total > 0:
            self.channel_shares_ = {ch: credits[ch] / total for ch in config.CHANNELS}
        else:
            self.channel_shares_ = dict.fromkeys(config.CHANNELS, 1.0 / len(config.CHANNELS))
        return self


class PositionBasedAttribution:
    """Attributes 40% to first, 40% to last, and 20% divided across middle touches."""

    def __init__(self) -> None:
        self.channel_shares_: dict[str, float] = {}

    def fit(self, dataset: data.JourneyDataset) -> PositionBasedAttribution:
        credits: dict[str, float] = dict.fromkeys(config.CHANNELS, 0.0)
        total = 0.0

        for j in dataset.journeys:
            if j.converted and j.path:
                k = len(j.path)
                if k == 1:
                    credits[j.path[0]] += 1.0
                    total += 1.0
                elif k == 2:
                    credits[j.path[0]] += 0.5
                    credits[j.path[1]] += 0.5
                    total += 1.0
                else:
                    credits[j.path[0]] += 0.40
                    credits[j.path[-1]] += 0.40
                    mid_weight = 0.20 / (k - 2)
                    for ch in j.path[1:-1]:
                        credits[ch] += mid_weight
                    total += 1.0

        if total > 0:
            self.channel_shares_ = {ch: credits[ch] / total for ch in config.CHANNELS}
        else:
            self.channel_shares_ = dict.fromkeys(config.CHANNELS, 1.0 / len(config.CHANNELS))
        return self


class TimeDecayAttribution:
    """Attributes exponentially increasing credit to touches closer to conversion."""

    def __init__(self, half_life_steps: float = 2.0) -> None:
        self.half_life_steps = half_life_steps
        self.channel_shares_: dict[str, float] = {}

    def fit(self, dataset: data.JourneyDataset) -> TimeDecayAttribution:
        credits: dict[str, float] = dict.fromkeys(config.CHANNELS, 0.0)
        total = 0.0

        for j in dataset.journeys:
            if j.converted and j.path:
                k = len(j.path)
                # Touch t in 0..k-1 has distance (k - 1 - t) steps from conversion
                distances = np.array([k - 1 - t for t in range(k)], dtype=float)
                weights = np.power(2.0, -distances / self.half_life_steps)
                weights = weights / np.sum(weights)

                for ch, w in zip(j.path, weights, strict=True):
                    credits[ch] += float(w)
                    total += float(w)

        if total > 0:
            self.channel_shares_ = {ch: credits[ch] / total for ch in config.CHANNELS}
        else:
            self.channel_shares_ = dict.fromkeys(config.CHANNELS, 1.0 / len(config.CHANNELS))
        return self


class MarkovAttribution:
    """First-order Markov chain multi-touch attribution with channel removal effects.

    Estimates transition matrix between states (START, channels, CONV, NULL).
    Computes overall conversion probability, then removes each channel to measure
    counterfactual removal effect: R(c) = 1 - P(Conv | without c) / P(Conv).
    """

    def __init__(self) -> None:
        self.channel_shares_: dict[str, float] = {}
        self.removal_effects_: dict[str, float] = {}
        self.base_conversion_rate_: float = 0.0

    def fit(self, dataset: data.JourneyDataset) -> MarkovAttribution:
        states = ["START", *config.CHANNELS, "CONV", "NULL"]
        state_idx = {s: i for i, s in enumerate(states)}
        n_states = len(states)
        trans_counts = np.zeros((n_states, n_states), dtype=float)

        for j in dataset.journeys:
            if not j.path:
                continue
            # START -> first touch
            trans_counts[state_idx["START"], state_idx[j.path[0]]] += 1.0
            # Steps along path
            for t in range(len(j.path) - 1):
                from_s = state_idx[j.path[t]]
                to_s = state_idx[j.path[t + 1]]
                trans_counts[from_s, to_s] += 1.0
            # Last touch -> CONV or NULL
            last_s = state_idx[j.path[-1]]
            outcome_s = state_idx["CONV"] if j.converted else state_idx["NULL"]
            trans_counts[last_s, outcome_s] += 1.0

        # Terminal absorbing states
        trans_counts[state_idx["CONV"], state_idx["CONV"]] = 1.0
        trans_counts[state_idx["NULL"], state_idx["NULL"]] = 1.0

        # Normalize row-stochastic transition matrix
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        trans_matrix = trans_counts / row_sums

        def compute_conv_prob(t_mat: np.ndarray) -> float:
            """Compute probability of reaching CONV from START via absorbing Markov chain."""
            # Transient states: START and marketing channels
            transient = ["START", *config.CHANNELS]
            tr_idx = [state_idx[s] for s in transient]
            q_mat = t_mat[np.ix_(tr_idx, tr_idx)]
            r_mat = t_mat[np.ix_(tr_idx, [state_idx["CONV"], state_idx["NULL"]])]

            # Fundamental matrix N = (I - Q)^(-1)
            eye = np.eye(len(transient))
            try:
                n_mat = np.linalg.inv(eye - q_mat)
            except np.linalg.LinAlgError:
                n_mat = np.linalg.pinv(eye - q_mat)

            # Absorption probabilities B = N * R
            b_mat = n_mat @ r_mat
            # From START (index 0) to CONV (index 0 of R)
            return float(b_mat[0, 0])

        base_p = compute_conv_prob(trans_matrix)
        self.base_conversion_rate_ = base_p

        # Compute removal effects
        removal_effects: dict[str, float] = {}
        for ch in config.CHANNELS:
            ch_i = state_idx[ch]
            t_removed = trans_matrix.copy()
            # Routing transitions through ch into NULL (as if channel is dropped)
            t_removed[ch_i, :] = 0.0
            t_removed[ch_i, state_idx["NULL"]] = 1.0
            # Any transition to ch gets redirected to NULL
            for from_s in range(n_states):
                if from_s != ch_i and t_removed[from_s, ch_i] > 0:
                    prob_to_ch = t_removed[from_s, ch_i]
                    t_removed[from_s, ch_i] = 0.0
                    t_removed[from_s, state_idx["NULL"]] += prob_to_ch

            p_removed = compute_conv_prob(t_removed)
            removal = max(0.0, (base_p - p_removed) / base_p) if base_p > 0 else 0.0
            removal_effects[ch] = removal

        self.removal_effects_ = removal_effects
        total_removal = sum(removal_effects.values())
        if total_removal > 0:
            self.channel_shares_ = {
                ch: removal_effects[ch] / total_removal for ch in config.CHANNELS
            }
        else:
            self.channel_shares_ = dict.fromkeys(config.CHANNELS, 1.0 / len(config.CHANNELS))

        return self
