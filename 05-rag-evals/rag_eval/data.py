"""Load and validate the generated corpus and questions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import (
    ANSWERABLE_TYPES,
    CHECKSUM_FILE,
    CORPUS_FILE,
    QUESTION_TYPES,
    QUESTIONS_FILE,
    UNANSWERABLE_TYPES,
)

DOC_KINDS = {"policy", "faq", "archive"}


class DataValidationError(ValueError):
    """Raised with every problem found, not just the first."""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m rag_eval.generate")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_corpus(path: Path = CORPUS_FILE) -> list[dict]:
    return read_jsonl(path)


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    return read_jsonl(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checksums(
    corpus_path: Path = CORPUS_FILE,
    questions_path: Path = QUESTIONS_FILE,
    checksum_file: Path = CHECKSUM_FILE,
) -> str:
    """Return one digest covering both files. Raise if a recorded checksum does not match."""
    actual = {"corpus.jsonl": _sha(corpus_path), "questions.jsonl": _sha(questions_path)}
    if checksum_file.exists():
        recorded = {
            line.split()[1]: line.split()[0]
            for line in checksum_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        for name, digest in actual.items():
            if recorded.get(name) != digest:
                raise DataValidationError(f"checksum mismatch for {name}")
    return hashlib.sha256((actual["corpus.jsonl"] + actual["questions.jsonl"]).encode()).hexdigest()


def validate(docs: list[dict], questions: list[dict]) -> None:
    """Structure and answer checks. Collects all problems."""
    problems: list[str] = []
    ids = [d.get("id") for d in docs]
    if len(set(ids)) != len(ids):
        problems.append("duplicate document ids")
    for d in docs:
        for key in ("id", "kind", "topic", "title", "text"):
            if not d.get(key):
                problems.append(f"document {d.get('id')} missing {key}")
        if d.get("kind") not in DOC_KINDS:
            problems.append(f"document {d.get('id')} has unknown kind {d.get('kind')!r}")
    by_id = {d["id"]: d for d in docs if "id" in d}
    faq_text = " ".join(d["text"].lower() for d in docs if d.get("kind") == "faq")

    qids = [q.get("id") for q in questions]
    if len(set(qids)) != len(qids):
        problems.append("duplicate question ids")
    for q in questions:
        qid = q.get("id")
        if q.get("type") not in QUESTION_TYPES:
            problems.append(f"{qid}: unknown type {q.get('type')!r}")
            continue
        if q.get("split") not in {"dev", "test"}:
            problems.append(f"{qid}: split must be dev or test")
        if not q.get("question"):
            problems.append(f"{qid}: empty question")
        gold = q.get("gold_docs", [])
        if q["type"] in UNANSWERABLE_TYPES:
            if gold or q.get("answer") is not None:
                problems.append(f"{qid}: unanswerable question has gold documents or an answer")
            continue
        answer = q.get("answer")
        if not gold or not answer:
            problems.append(f"{qid}: answerable question needs gold documents and an answer")
            continue
        missing = [g for g in gold if g not in by_id]
        if missing:
            problems.append(f"{qid}: unknown gold documents {missing}")
            continue
        if not any(answer.lower() in by_id[g]["text"].lower() for g in gold):
            problems.append(f"{qid}: answer {answer!r} is not in its gold documents")
        if answer.lower() in faq_text:
            problems.append(f"{qid}: answer {answer!r} appears in a FAQ page")
        if q["type"] == "stale_version":
            topic = by_id[gold[0]]["topic"]
            archive = by_id.get(f"archive_{topic}")
            if archive is None:
                problems.append(f"{qid}: stale_version question has no archive document")
            elif answer.lower() in archive["text"].lower():
                problems.append(f"{qid}: current answer also appears in the archived document")

    for qtype in QUESTION_TYPES:
        for split in ("dev", "test"):
            if not any(q.get("type") == qtype and q.get("split") == split for q in questions):
                problems.append(f"no {split} questions of type {qtype}")
    if problems:
        raise DataValidationError("; ".join(problems))


def split_questions(questions: list[dict]) -> tuple[list[dict], list[dict]]:
    dev = [q for q in questions if q["split"] == "dev"]
    test = [q for q in questions if q["split"] == "test"]
    return dev, test


def is_answerable(question: dict) -> bool:
    return question["type"] in ANSWERABLE_TYPES
