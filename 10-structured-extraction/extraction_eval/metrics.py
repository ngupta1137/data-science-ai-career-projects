"""Field-level and document-level scoring, error taxonomy, and intervals."""

from __future__ import annotations

import numpy as np

from . import config
from .data import Doc
from .extractors import candidates
from .normalize import normalize


def wilson_ci(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def boot_diff_ci(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, boots: int
) -> tuple[float, float]:
    """Paired bootstrap interval for mean(a) - mean(b) over the same documents."""
    idx = rng.integers(0, len(a), size=(boots, len(a)))
    diffs = (a.astype(float) - b.astype(float))[idx].mean(axis=1)
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def score(docs: list[Doc], preds: list[dict]) -> dict:
    """Per-field boolean arrays: extracted, parseable, correct. Plus the document-level array."""
    n = len(docs)
    extracted = {f: np.zeros(n, bool) for f in config.FIELDS}
    parseable = {f: np.zeros(n, bool) for f in config.FIELDS}
    correct = {f: np.zeros(n, bool) for f in config.FIELDS}
    for i, (d, p) in enumerate(zip(docs, preds, strict=True)):
        for f in config.FIELDS:
            raw = p.get(f)
            extracted[f][i] = raw is not None
            norm = normalize(f, raw)
            parseable[f][i] = norm is not None
            correct[f][i] = norm is not None and norm == d.truth[f]
    doc_correct = np.logical_and.reduce([correct[f] for f in config.FIELDS])
    return {
        "extracted": extracted,
        "parseable": parseable,
        "correct": correct,
        "doc_correct": doc_correct,
    }


def prf(correct: np.ndarray, extracted: np.ndarray) -> dict:
    """Precision over extracted values, recall over all documents. A wrong value costs both."""
    tp = int(correct.sum())
    n_ext = int(extracted.sum())
    p = tp / n_ext if n_ext else float("nan")
    r = tp / len(correct)
    f1 = 2 * p * r / (p + r) if n_ext and (p + r) else float("nan")
    return {"precision": p, "recall": r, "f1": f1, "extracted": n_ext}


def micro_f1(sc: dict) -> dict:
    correct = np.concatenate([sc["correct"][f] for f in config.FIELDS])
    extracted = np.concatenate([sc["extracted"][f] for f in config.FIELDS])
    return prf(correct, extracted)


def classify_error(field: str, raw: str | None, truth, distractors: dict, text: str) -> str | None:
    """Why a prediction is wrong. None if it is right."""
    if raw is None:
        return "missing"
    norm = normalize(field, raw)
    if norm is None:
        return "unparseable"
    if norm == truth:
        return None
    for role, d in distractors.get(field, {}).items():
        if normalize(field, d) == norm:
            return f"distractor:{role.rstrip('0123456789')}"
    return "other_value_in_text" if raw in text else "not_in_text"


def taxonomy(docs: list[Doc], preds: list[dict]) -> dict:
    out: dict = {f: {} for f in config.FIELDS}
    for d, p in zip(docs, preds, strict=True):
        for f in config.FIELDS:
            why = classify_error(f, p.get(f), d.truth[f], d.distractors, d.text)
            if why:
                out[f][why] = out[f].get(why, 0) + 1
    return out


def silent_error_share(sc: dict) -> dict:
    """Of the wrong values, how many parse cleanly and so would pass a schema check?"""
    wrong = {f: sc["extracted"][f] & ~sc["correct"][f] for f in config.FIELDS}
    n_wrong = int(sum(w.sum() for w in wrong.values()))
    silent = int(sum((wrong[f] & sc["parseable"][f]).sum() for f in config.FIELDS))
    return {"wrong_values": n_wrong, "schema_valid_but_wrong": silent}


def recoverable(docs: list[Doc], texts: list[str]) -> np.ndarray:
    """Documents where every true value can still be read from the (noisy) text as some candidate.

    This is a ceiling for any extractor that copies spans.
    """
    out = np.zeros(len(docs), bool)
    for i, (d, t) in enumerate(zip(docs, texts, strict=True)):
        out[i] = all(
            any(normalize(f, c.text) == d.truth[f] for c in candidates(f, t)) for f in config.FIELDS
        )
    return out
