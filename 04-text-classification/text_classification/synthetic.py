"""Synthetic text with the 20 Newsgroups schema. Used by tests and smoke runs only.

Each class has topical words. The raw text also carries class-specific header tokens
that are absent from the cleaned text, so the metadata trap can be measured.
Nothing measured on this data says anything about real-data performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_text(
    n_per_class: int = 120,
    n_classes: int = 6,
    seed: int = 0,
    empty_rate: float = 0.02,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shared = [f"common{i}" for i in range(150)]
    rows = []
    for c in range(n_classes):
        topical = [f"topic{c}_{i}" for i in range(25)]
        for k in range(n_per_class):
            n_words = int(rng.integers(15, 120))
            pool = [topical if rng.random() < 0.18 else shared for _ in range(n_words)]
            words = [str(rng.choice(p)) for p in pool]
            clean = " ".join(words)
            if rng.random() < empty_rate:
                clean = ""
            header = f"From: user{c} Organization: org{c} Lines: {n_words}\n\n"
            rows.append(
                {
                    "text": clean,
                    "text_raw": header + clean,
                    "label": f"class_{c}",
                    "split": "train" if k < int(n_per_class * 0.6) else "test",
                }
            )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
