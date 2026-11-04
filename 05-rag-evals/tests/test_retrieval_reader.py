import numpy as np

from rag_eval.reader import Reader, extract_values, normalize_answer, split_sentences
from rag_eval.retrieval import (
    BM25,
    LSARetriever,
    RandomRetriever,
    RRFRetriever,
    TfidfRetriever,
    rank_order,
    tokenize,
)

TOY = [
    {"id": "d1", "text": "Cats purr when happy. Cats sleep 16 hours a day."},
    {"id": "d2", "text": "Dogs bark at strangers. Dogs need 2 hours of exercise."},
    {"id": "d3", "text": "Fish swim in water. Fish need clean water."},
]


def test_tokenize_drops_stopwords_and_does_not_stem():
    assert tokenize("The Cats are sleeping") == ["cats", "sleeping"]


def test_rank_order_breaks_ties_by_doc_id():
    assert rank_order(np.array([1.0, 1.0, 2.0]), ["b", "a", "c"]) == [2, 1, 0]


def test_bm25_ranks_the_matching_document_first():
    scores = BM25(TOY).scores("how many hours do cats sleep")
    assert int(np.argmax(scores)) == 0
    assert BM25(TOY).scores("zebra")[0] == 0.0


def test_tfidf_and_lsa_rank_the_matching_document_first():
    assert int(np.argmax(TfidfRetriever(TOY).scores("dogs exercise"))) == 1
    assert int(np.argmax(LSARetriever(TOY, 2).scores("dogs exercise"))) == 1


def test_rrf_scores_sum_reciprocal_ranks():
    ids = [d["id"] for d in TOY]
    bm = BM25(TOY)
    fused = RRFRetriever([bm, bm], ids, k=60).scores("cats sleep")
    assert fused[0] == 2 / 61  # rank 1 in both lists


def test_random_retriever_is_seeded_per_question():
    r = RandomRetriever(5, seed=3)
    assert np.array_equal(r.scores("same"), r.scores("same"))
    assert not np.array_equal(r.scores("same"), r.scores("other"))


def test_extract_values_and_first_value_rule():
    assert extract_values("Pay is $3,000 after 6 months of work.") == ["$3,000", "6 months"]
    assert extract_values("No numbers here.") == []
    assert normalize_answer(" 16  Weeks. ") == "16 weeks"
    assert normalize_answer(None) == ""


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_reader_returns_value_from_best_overlapping_sentence():
    bm = BM25(TOY)
    out = Reader(bm.idf).answer("how many hours do dogs need for exercise", TOY[:2])
    assert out["answer"] == "2 hours"
    assert out["source"] == "d2"
    assert 0 < out["confidence"] <= 1


def test_reader_gives_no_answer_when_no_sentence_has_a_value():
    docs = [{"id": "x", "text": "Nothing numeric lives here."}]
    out = Reader({"nothing": 1.0}).answer("what is nothing", docs)
    assert out["answer"] is None and out["confidence"] == 0.0
