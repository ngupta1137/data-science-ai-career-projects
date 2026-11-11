"""Three extractors: a first-match baseline, hand-written rules, and a learned span ranker.

Each takes the document text and returns a dict of raw strings (or None) for the five fields.
None means 'nothing found'. Normalising and scoring happen elsewhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from . import config
from .normalize import normalize

DATE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\d{1,2} [A-Z][a-z]{2} \d{4}|[A-Z][a-z]+ \d{1,2}, \d{4}|\d{2}/\d{2}/\d{4}|\d{2}\.\d{2}\.\d{4}"
)
MONEY_RE = re.compile(r"\d[\d.,]*[.,]\d{2}\b")
ID_RE = re.compile(
    r"(?<![\w/-])(?=[A-Z0-9/-]*\d)(?=[A-Z0-9/-]*[A-Z/-])[A-Z0-9][A-Z0-9/-]{4,}(?![\w/-])"
)
CURRENCY_RE = re.compile(r"EUR|USD|GBP|€|\$|£")
VENDOR_SEPARATORS = (": ", " | ")


@dataclass(frozen=True)
class Cand:
    text: str
    line: int
    start: int


def _lines(text: str) -> list[str]:
    return text.split("\n")


def candidates(field: str, text: str) -> list[Cand]:
    """Every span that could plausibly be the field, found without knowing the template."""
    out: list[Cand] = []
    lines = _lines(text)
    for i, ln in enumerate(lines):
        if field == "date":
            out += [Cand(m.group(), i, m.start()) for m in DATE_RE.finditer(ln)]
        elif field == "total":
            out += [Cand(m.group(), i, m.start()) for m in MONEY_RE.finditer(ln)]
        elif field == "currency":
            out += [Cand(m.group(), i, m.start()) for m in CURRENCY_RE.finditer(ln)]
        elif field == "invoice_number":
            for m in ID_RE.finditer(ln):
                if not DATE_RE.fullmatch(m.group()):
                    out.append(Cand(m.group(), i, m.start()))
        else:  # vendor: the whole line, or what follows a separator
            spans = [(ln.strip(), 0)]
            for sep in VENDOR_SEPARATORS:
                if sep in ln:
                    k = ln.index(sep) + len(sep)
                    spans.append((ln[k:].strip(), k))
            if gap := re.search(r"\S\s{2,}(?=\S)", ln):  # columns separated by runs of spaces
                spans.append((ln[gap.end() :].strip(), gap.end()))
            for t, k in spans:
                if 3 <= len(t) <= 60 and not any(ch.isdigit() for ch in t) and t[0].isalpha():
                    out.append(Cand(t, i, k))
    return out


def first_match(text: str) -> dict:
    """Baseline: take the first candidate of each type. No labels, no learning."""
    out = {}
    for f in config.FIELDS:
        c = candidates(f, text)
        out[f] = c[0].text if c else None
    return out


_CUR = r"(?:EUR|USD|GBP|€|\$|£)"
_RULES = {
    "invoice_number": re.compile(r"(?:Invoice No:|Invoice Number:|INVOICE #|Ref \|)\s*(\S+)", re.M),
    "date": re.compile(
        r"(?:^Date:|^Invoice Date:|^Issued|^Invoice date \|)\s*(" + DATE_RE.pattern + ")", re.M
    ),
    "total": re.compile(
        r"(?:^Total:|^Amount Due:|^TOTAL \|)\s*(?:" + _CUR + r"\s?)?(" + MONEY_RE.pattern + ")",
        re.M,
    ),
    "currency": re.compile(
        r"(?:^Total:|^Amount Due:|^TOTAL \|)\s*(?:("
        + _CUR
        + r")\s?)?"
        + MONEY_RE.pattern
        + r"(?:\s("
        + _CUR
        + "))?",
        re.M,
    ),
}


def template_regex(text: str) -> dict:
    """Hand-written rules for templates A, B and C. They cover label variants that occur in at
    least 30% of those templates' documents, and say nothing about the rest."""
    out: dict = dict.fromkeys(config.FIELDS)
    for f in ("invoice_number", "date", "total"):
        if m := _RULES[f].search(text):
            out[f] = m.group(1)
    if m := _RULES["currency"].search(text):
        out["currency"] = m.group(1) or m.group(2)
    lines = _lines(text)
    if m := re.search(r"^Bill from \| (.+)$", text, re.M):
        out["vendor"] = m.group(1).strip()
    elif lines[0].startswith("INVOICE #") and len(lines) > 1:
        out["vendor"] = lines[1].strip()
    elif len(lines) > 1 and re.match(r"Invoice (?:No|Number):", lines[1]):
        out["vendor"] = lines[0].strip()  # a recognised template; anything else gets no vendor
    return out


def _features(field: str, text: str, cand: Cand, cands: list[Cand], k: int) -> dict:
    lines = _lines(text)
    ln = lines[cand.line]
    before = re.findall(r"[a-zA-Z]+", ln[: cand.start].lower())
    feats: dict[str, float] = {f"p{j + 1}={w}": 1.0 for j, w in enumerate(reversed(before[-3:]))}
    feats.update({f"bag={w}": 1.0 for w in before[-6:]})
    if cand.line > 0:
        prev = re.findall(r"[a-zA-Z]+", lines[cand.line - 1].lower())
        feats.update({f"pl={w}": 1.0 for w in prev[:2]})
    feats[f"line_rel={int(5 * cand.line / max(len(lines), 1))}"] = 1.0
    feats[f"line_from_end={min(len(lines) - 1 - cand.line, 3)}"] = 1.0
    feats[f"line_from_start={min(cand.line, 3)}"] = 1.0
    feats[f"rank={min(k, 3)}"] = 1.0
    feats[f"rank_from_end={min(len(cands) - 1 - k, 3)}"] = 1.0
    feats[f"lastw={cand.text.split()[-1].lower() if cand.text.split() else ''}"] = 1.0
    feats[f"n_tokens={min(len(cand.text.split()), 4)}"] = 1.0
    feats["after_line_text"] = float(bool(ln[cand.start + len(cand.text) :].strip()))
    return feats


class SpanRanker:
    """For each field, score every candidate span from the words around it and keep the best.

    Trained on documents with known truth. It never writes a value that is not in the text, so
    it cannot invent an amount, but it can pick the wrong one.
    """

    def __init__(self, abstain_below: float = config.ABSTAIN_BELOW) -> None:
        self.abstain_below = abstain_below
        self.models: dict[str, tuple[DictVectorizer, LogisticRegression]] = {}

    def fit(self, docs: list) -> SpanRanker:
        for f in config.FIELDS:
            rows, labels = [], []
            for d in docs:
                cs = candidates(f, d.text)
                for k, c in enumerate(cs):
                    rows.append(_features(f, d.text, c, cs, k))
                    labels.append(int(normalize(f, c.text) == d.truth[f]))
            vec = DictVectorizer()
            x = vec.fit_transform(rows)
            self.models[f] = (vec, LogisticRegression(C=1.0, max_iter=2000).fit(x, labels))
        return self

    def extract(self, text: str) -> dict:
        out: dict = {}
        for f in config.FIELDS:
            cs = candidates(f, text)
            if not cs:
                out[f] = None
                continue
            vec, model = self.models[f]
            x = vec.transform([_features(f, text, c, cs, k) for k, c in enumerate(cs)])
            p = model.predict_proba(x)[:, 1]
            best = int(np.argmax(p))
            out[f] = cs[best].text if p[best] >= self.abstain_below else None
        return out
