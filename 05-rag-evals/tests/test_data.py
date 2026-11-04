import copy

import pytest

from rag_eval import config, data
from rag_eval.generate import build_documents, build_questions, checksum_text, to_jsonl


def test_generated_data_validates(docs, questions):
    data.validate(docs, questions)
    assert len({d["id"] for d in docs}) == len(docs)
    assert len({q["id"] for q in questions}) == len(questions)


def test_generation_is_deterministic():
    assert build_documents() == build_documents()
    assert build_questions() == build_questions()


def test_every_type_has_dev_and_test(questions):
    for t in config.QUESTION_TYPES:
        splits = {q["split"] for q in questions if q["type"] == t}
        assert splits == {"dev", "test"}, t


def test_unanswerable_questions_have_no_gold_and_no_answer(questions):
    for q in questions:
        if q["type"] in config.UNANSWERABLE_TYPES:
            assert q["gold_docs"] == [] and q["answer"] is None
        else:
            assert q["gold_docs"] and q["answer"]


def test_validate_rejects_answer_missing_from_gold_doc(docs, questions):
    bad = copy.deepcopy(questions)
    first = next(q for q in bad if q["type"] == "lexical")
    first["answer"] = "999 centuries"
    with pytest.raises(data.DataValidationError):
        data.validate(docs, bad)


def test_validate_rejects_unknown_gold_doc(docs, questions):
    bad = copy.deepcopy(questions)
    next(q for q in bad if q["type"] == "lexical")["gold_docs"] = ["nope"]
    with pytest.raises(data.DataValidationError):
        data.validate(docs, bad)


def test_committed_files_match_regenerated(docs, questions):
    assert config.CORPUS_FILE.read_text(encoding="utf-8") == to_jsonl(docs)
    assert config.QUESTIONS_FILE.read_text(encoding="utf-8") == to_jsonl(questions)
    assert config.CHECKSUM_FILE.read_text(encoding="utf-8") == checksum_text(
        to_jsonl(docs), to_jsonl(questions)
    )


def test_verify_checksums_detects_tampering(tmp_path):
    corpus, qs, sums = tmp_path / "corpus.jsonl", tmp_path / "questions.jsonl", tmp_path / "S.txt"
    corpus.write_text("a\n")
    qs.write_text("b\n")
    sums.write_text(checksum_text("a\n", "b\n"))
    assert data.verify_checksums(corpus, qs, sums)
    corpus.write_text("changed\n")
    with pytest.raises(data.DataValidationError):
        data.verify_checksums(corpus, qs, sums)
