"""A deliberately simple extractive reader.

It is NOT a stand-in for a language model. It exists so that failures are mechanical and
inspectable: it picks the retrieved sentence that holds a value and best overlaps the question (weighted
by rarity), then returns the first value (a number with a unit, or a dollar amount) in it.

Known weaknesses, measured rather than hidden:
  - a sentence with two values returns the first, even if the question asks for the second
  - an archived page that looks like the current one can win the sentence contest
  - question words that overlap a sentence say nothing about whether it holds the answer
"""

from __future__ import annotations

import re

from .retrieval import tokenize

_UNIT = r"(?:business days?|weeks?|days?|months?|hours?|years?)"
VALUE_RE = re.compile(rf"\$\d[\d,]*(?:\.\d+)?|\b\d+(?:\.\d+)?\s+{_UNIT}\b", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def extract_values(sentence: str) -> list[str]:
    return [m.group(0) for m in VALUE_RE.finditer(sentence)]


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")


class Reader:
    def __init__(self, idf: dict[str, float]):
        self.idf = idf

    def answer(self, question: str, docs: list[dict]) -> dict:
        """docs are in retrieval order. Returns answer, source document id, confidence."""
        q_tokens = set(tokenize(question))
        q_mass = sum(self.idf.get(t, 0.0) for t in q_tokens)
        best: tuple[float, str, str] | None = None
        for doc in docs:
            for sentence in split_sentences(doc["text"]):
                if not extract_values(sentence):
                    continue  # this reader only answers with a value, so skip sentences without one
                matched = sum(self.idf.get(t, 0.0) for t in q_tokens & set(tokenize(sentence)))
                if best is None or matched > best[0] + 1e-12:
                    best = (matched, doc["id"], sentence)
        if best is None:
            return {"answer": None, "source": None, "confidence": 0.0, "sentence": None}
        values = extract_values(best[2])
        return {
            "answer": values[0] if values else None,
            "source": best[1],
            "confidence": best[0] / q_mass if q_mass > 0 else 0.0,
            "sentence": best[2],
        }
