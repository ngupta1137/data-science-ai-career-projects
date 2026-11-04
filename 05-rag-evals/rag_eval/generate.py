"""Build the teaching corpus and question set from rag_eval/topics.py.

    python -m rag_eval.generate

The output is fully determined by topics.py, so regenerating gives byte-identical files.
A test checks the committed files against a fresh build.
"""

from __future__ import annotations

import hashlib
import json

from . import config
from .topics import COMPANY, FILLERS, OFFTOPIC_QUESTIONS, TOPICS


def _policy_sentence(topic: dict, old: bool = False) -> str:
    values = {"v1": topic["v1_old"] if old else topic["v1"], "v2": topic.get("v2", "")}
    return topic["sentence"].format(**values)


def build_documents() -> list[dict]:
    docs = []
    for i, topic in enumerate(TOPICS):
        filler = FILLERS[i % len(FILLERS)]
        name = topic["name"]
        docs.append(
            {
                "id": f"policy_{topic['id']}",
                "kind": "policy",
                "topic": topic["id"],
                "title": f"{name} Policy",
                "text": f"{COMPANY} {name} Policy (effective 2024). {_policy_sentence(topic)} {filler}",
            }
        )
        docs.append(
            {
                "id": f"faq_{topic['id']}",
                "kind": "faq",
                "topic": topic["id"],
                "title": f"FAQ: {name}",
                "text": (
                    f"FAQ: {name}. This page is also known as {topic['alias']}. "
                    "To make a request, send the form to your manager. "
                    "Questions go to the People Team. "
                    "This page does not list amounts, limits, or deadlines."
                ),
            }
        )
        if "v1_old" in topic:
            docs.append(
                {
                    "id": f"archive_{topic['id']}",
                    "kind": "archive",
                    "topic": topic["id"],
                    "title": f"{name} Policy (archived)",
                    "text": (
                        f"{COMPANY} {name} Policy (archived, in effect from 2021 to 2023). "
                        f"{_policy_sentence(topic, old=True)} {filler}"
                    ),
                }
            )
    return docs


def build_questions() -> list[dict]:
    rows: list[tuple[str, str, list[str], str | None]] = []  # type, question, gold, answer
    for topic in TOPICS:
        policy = f"policy_{topic['id']}"
        for q in topic["q_lex"]:
            rows.append(("lexical", q, [policy], topic["v1"]))
    for topic in TOPICS:
        policy = f"policy_{topic['id']}"
        for q in topic["q_para"]:
            rows.append(("paraphrase", q, [policy], topic["v1"]))
    for topic in TOPICS:
        if "q2" in topic:
            rows.append(("numeric_detail", topic["q2"], [f"policy_{topic['id']}"], topic["v2"]))
    for topic in TOPICS:
        if "q_current" in topic:
            rows.append(
                ("stale_version", topic["q_current"], [f"policy_{topic['id']}"], topic["v1"])
            )
    for topic in TOPICS:
        rows.append(("unanswerable_attribute", topic["q_unans"], [], None))
    for q in OFFTOPIC_QUESTIONS:
        rows.append(("unanswerable_offtopic", q, [], None))

    # Alternate dev and test within each type, in order. No randomness.
    seen: dict[str, int] = {}
    questions = []
    for n, (qtype, question, gold, answer) in enumerate(rows, start=1):
        k = seen.get(qtype, 0)
        seen[qtype] = k + 1
        questions.append(
            {
                "id": f"q{n:03d}",
                "type": qtype,
                "question": question,
                "gold_docs": gold,
                "answer": answer,
                "split": "dev" if k % 2 == 0 else "test",
            }
        )
    return questions


def to_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)


def checksum_text(corpus_text: str, questions_text: str) -> str:
    def sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    return f"{sha(corpus_text)}  corpus.jsonl\n{sha(questions_text)}  questions.jsonl\n"


def main() -> None:
    corpus_text = to_jsonl(build_documents())
    questions_text = to_jsonl(build_questions())
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.CORPUS_FILE.write_text(corpus_text, encoding="utf-8")
    config.QUESTIONS_FILE.write_text(questions_text, encoding="utf-8")
    config.CHECKSUM_FILE.write_text(checksum_text(corpus_text, questions_text), encoding="utf-8")
    print(
        f"wrote {config.CORPUS_FILE.name}, {config.QUESTIONS_FILE.name}, {config.CHECKSUM_FILE.name}"
    )


if __name__ == "__main__":
    main()
