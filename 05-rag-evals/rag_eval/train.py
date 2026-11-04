"""Select, evaluate, and write every result file.

    python -m rag_eval.generate     # (re)build the committed corpus and questions
    python -m rag_eval.train        # full run
    python -m rag_eval.train --fast # smoke test, results are marked

Design:
  - Dev questions choose retriever settings, the headline retriever, and the abstention
    threshold. Test questions are scored once.
  - End-to-end correctness counts an unanswerable question as correct only if the system
    abstains. Always abstaining is therefore a real baseline, not a zero.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from . import config, data
from .metrics import bootstrap_ci, ndcg_at_k, recall_at_k, reciprocal_rank
from .reader import Reader, normalize_answer
from .retrieval import (
    BM25,
    LSARetriever,
    RandomRetriever,
    RRFRetriever,
    TfidfRetriever,
    rank_order,
)

K = config.K_CONTEXT
TIE_ORDER = ("bm25", "tfidf", "lsa", "hybrid_rrf")

CAVEATS = [
    "The corpus is generated from templates about a fictional company. It teaches how to evaluate retrieval and answering. It says nothing about real documents.",
    "The test set is small (see test_questions), so every interval is wide. Differences of a few points are not evidence.",
    "Questions are written from the same templates as the documents, so lexical retrieval is easier here than on real text.",
    "The reader is a rule-based extractor, not a language model. LSA is a classical stand-in for dense retrieval, not a neural embedder.",
    "No language model generates or judges answers in this project. That comparison is designed, not built.",
]


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def rank_ids(retriever, question: str, doc_ids: list[str]) -> list[str]:
    return [doc_ids[i] for i in rank_order(retriever.scores(question), doc_ids)]


def retrieval_scores(retriever, questions: list[dict], doc_ids: list[str]) -> dict:
    """Per-question retrieval scores on answerable questions."""
    rows = [q for q in questions if data.is_answerable(q)]
    out = {"rr": [], "r1": [], "r3": [], "ndcg5": [], "ids": [q["id"] for q in rows]}
    for q in rows:
        ranked = rank_ids(retriever, q["question"], doc_ids)
        out["rr"].append(reciprocal_rank(ranked, q["gold_docs"]))
        out["r1"].append(recall_at_k(ranked, q["gold_docs"], 1))
        out["r3"].append(recall_at_k(ranked, q["gold_docs"], 3))
        out["ndcg5"].append(ndcg_at_k(ranked, q["gold_docs"], 5))
    return {k: (np.array(v) if k != "ids" else v) for k, v in out.items()}


def answer_all(retriever, reader: Reader, questions, docs, doc_ids) -> list[dict]:
    by_id = {d["id"]: d for d in docs}
    outputs = []
    for q in questions:
        ranked = rank_ids(retriever, q["question"], doc_ids)[:K]
        result = reader.answer(q["question"], [by_id[i] for i in ranked])
        outputs.append({"question": q, "retrieved": ranked, **result})
    return outputs


def is_correct(out: dict, threshold: float) -> bool:
    abstain = out["answer"] is None or out["confidence"] < threshold
    q = out["question"]
    if not data.is_answerable(q):
        return abstain
    return (not abstain) and normalize_answer(out["answer"]) == normalize_answer(q["answer"])


def choose_threshold(outputs: list[dict]) -> float:
    """Smallest threshold that maximizes dev correctness. Candidates are the observed
    confidences plus one value above all of them (abstain on everything)."""
    confs = sorted({round(o["confidence"], 10) for o in outputs})
    candidates = [0.0, *confs, (confs[-1] + 1e-6) if confs else 1.0]
    scored = [(sum(is_correct(o, t) for o in outputs), t) for t in candidates]
    best = max(s for s, _ in scored)
    return min(t for s, t in scored if s == best)


def failure_category(out: dict, threshold: float) -> str:
    q = out["question"]
    abstain = out["answer"] is None or out["confidence"] < threshold
    if not data.is_answerable(q):
        return "correct_refusal" if abstain else "answered_unanswerable"
    gold_in_context = bool(set(q["gold_docs"]) & set(out["retrieved"]))
    if is_correct(out, threshold):
        return "correct"
    if not gold_in_context:
        return "retrieval_miss"
    if abstain:
        return "false_abstain"
    if out["source"] is not None and out["source"].startswith("archive_"):
        return "stale_source_chosen"
    if out["source"] not in q["gold_docs"]:
        return "wrong_doc_used"
    return "wrong_span"


def _save_plot(by_type: dict, results_dir: Path) -> None:
    names = list(by_type)
    vals = [by_type[n]["end_to_end_correct"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(range(len(names)), vals)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("End-to-end correct (test)")
    ax.set_title("Correctness by question type")
    fig.tight_layout()
    fig.savefig(results_dir / "correctness_by_type.png", dpi=120)
    plt.close(fig)


def run(
    docs: list[dict],
    questions: list[dict],
    results_dir: Path,
    fast: bool = False,
    seed: int = config.SEED,
    data_source: str = "generated_teaching_corpus",
    data_sha256: str | None = None,
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data.validate(docs, questions)
    doc_ids = [d["id"] for d in docs]
    dev, test = data.split_questions(questions)
    n_boot = 50 if fast else config.N_BOOT
    bm25_grid = config.BM25_GRID[:2] if fast else config.BM25_GRID
    lsa_dims = config.LSA_DIMS[:1] if fast else config.LSA_DIMS

    def dev_mrr(retriever) -> float:
        return float(retrieval_scores(retriever, dev, doc_ids)["rr"].mean())

    # Select settings on dev only. First best in grid order wins ties.
    bm25_scored = [(dev_mrr(BM25(docs, k1, b)), (k1, b)) for k1, b in bm25_grid]
    bm25_best = max(s for s, _ in bm25_scored)
    k1, b = next(p for s, p in bm25_scored if s == bm25_best)
    lsa_scored = [(dev_mrr(LSARetriever(docs, d, seed)), d) for d in lsa_dims]
    lsa_best = max(s for s, _ in lsa_scored)
    dims = next(d for s, d in lsa_scored if s == lsa_best)

    bm25 = BM25(docs, k1, b)
    tfidf = TfidfRetriever(docs)
    lsa = LSARetriever(docs, dims, seed)
    hybrid = RRFRetriever([bm25, lsa], doc_ids)
    systems = {"bm25": bm25, "tfidf": tfidf, "lsa": lsa, "hybrid_rrf": hybrid}
    random_baseline = RandomRetriever(len(docs), seed)

    dev_scores = {n: dev_mrr(r) for n, r in systems.items()}
    best_dev = max(dev_scores.values())
    headline = next(n for n in TIE_ORDER if dev_scores[n] == best_dev)

    test_ret = {n: retrieval_scores(r, test, doc_ids) for n, r in systems.items()}
    test_ret["random"] = retrieval_scores(random_baseline, test, doc_ids)
    n_ans = len(test_ret[headline]["rr"])
    retrieval_results = {
        n: {
            "dev_mrr": dev_scores.get(n),
            "test_recall_at_1": float(s["r1"].mean()),
            "test_recall_at_3": float(s["r3"].mean()),
            "test_mrr": float(s["rr"].mean()),
            "test_ndcg_at_5": float(s["ndcg5"].mean()),
        }
        for n, s in test_ret.items()
    }
    rr_h = test_ret[headline]["rr"]
    mrr_ci = bootstrap_ci(lambda i: rr_h[i].mean(), n_ans, n_boot, seed)
    rr_b = test_ret["bm25"]["rr"]
    gain_ci = bootstrap_ci(lambda i: (rr_h[i] - rr_b[i]).mean(), n_ans, n_boot, seed)

    # End to end with the headline retriever.
    idf = bm25.idf
    reader = Reader(idf)
    dev_out = answer_all(systems[headline], reader, dev, docs, doc_ids)
    test_out = answer_all(systems[headline], reader, test, docs, doc_ids)
    threshold = choose_threshold(dev_out)
    correct = np.array([is_correct(o, threshold) for o in test_out], dtype=float)
    unans = np.array([not data.is_answerable(o["question"]) for o in test_out], dtype=float)
    always_abstain = float(unans.mean())
    e2e_ci = bootstrap_ci(lambda i: correct[i].mean(), len(test_out), n_boot, seed)
    no_abstain = float(np.mean([is_correct(o, 0.0) and o["answer"] is not None for o in test_out]))

    # Reader alone, given the gold documents (an upper bound for the retrieval stage).
    by_id = {d["id"]: d for d in docs}
    gold_em = []
    for q in test:
        if data.is_answerable(q):
            r = reader.answer(q["question"], [by_id[g] for g in q["gold_docs"]])
            gold_em.append(float(normalize_answer(r["answer"]) == normalize_answer(q["answer"])))
    gold_context_em = float(np.mean(gold_em))

    # Taxonomy and by-type tables.
    cats = [failure_category(o, threshold) for o in test_out]
    taxonomy = dict(sorted(Counter(cats).items()))
    by_type: dict[str, dict] = {}
    for t in config.QUESTION_TYPES:
        idx = [i for i, o in enumerate(test_out) if o["question"]["type"] == t]
        if not idx:
            continue
        row = {"n": len(idx), "end_to_end_correct": float(correct[idx].mean())}
        if t in config.ANSWERABLE_TYPES:
            hit = [
                bool(set(test_out[i]["question"]["gold_docs"]) & set(test_out[i]["retrieved"]))
                for i in idx
            ]
            row["gold_in_top_k"] = float(np.mean(hit))
        row["categories"] = dict(sorted(Counter(cats[i] for i in idx).items()))
        by_type[t] = row
    failures = [
        {
            "id": o["question"]["id"],
            "type": o["question"]["type"],
            "category": c,
            "question": o["question"]["question"],
            "gold_answer": o["question"]["answer"],
            "predicted": o["answer"],
            "source": o["source"],
            "confidence": o["confidence"],
        }
        for o, c in zip(test_out, cats, strict=True)
        if c not in ("correct", "correct_refusal")
    ]

    # Traps, measured on the test questions.
    thr_test = choose_threshold(test_out)
    tuned_on_test = float(np.mean([is_correct(o, thr_test) for o in test_out]))
    ans_idx = [i for i, o in enumerate(test_out) if data.is_answerable(o["question"])]
    answerable_only = float(correct[ans_idx].mean())
    traps = {
        "threshold_tuned_on_test": {
            "correct_with_leak": tuned_on_test,
            "correct_honest": float(correct.mean()),
            "difference": tuned_on_test - float(correct.mean()),
            "note": "The threshold was chosen on the questions it is scored on. Optimistic by construction.",
        },
        "answer_everything": {
            "correct_without_abstention": no_abstain,
            "correct_honest": float(correct.mean()),
            "difference": float(correct.mean()) - no_abstain,
            "note": "A system that never abstains gets every unanswerable question wrong.",
        },
        "answerable_only": {
            "correct_answerable_only": answerable_only,
            "correct_all_questions": float(correct.mean()),
            "difference": answerable_only - float(correct.mean()),
            "note": "Scoring only answerable questions hides the cost of answering what should be refused.",
        },
    }

    gate_e2e = bool(e2e_ci[0] > always_abstain)
    gate_gain = True if headline == "bm25" else bool(gain_ci[0] > 0)
    gate_passed = gate_e2e and gate_gain

    metrics = {
        "project": config.PROJECT_NAME,
        "data_source": data_source,
        "fast_mode": bool(fast),
        "seed": seed,
        "versions": _versions(),
        "data": {
            "documents": len(docs),
            "questions": len(questions),
            "dev_questions": len(dev),
            "test_questions": len(test),
            "test_answerable": n_ans,
            "sha256": data_sha256,
        },
        "selection": {
            "bm25": {"k1": k1, "b": b},
            "lsa_dims": dims,
            "context_documents": K,
            "abstention_threshold": threshold,
            "selected_by": "dev MRR on answerable questions; threshold by dev end-to-end correctness",
        },
        "retrieval": retrieval_results,
        "headline": {
            "retriever": headline,
            "metric": "test_mrr",
            "value": float(rr_h.mean()),
            "ci95": [mrr_ci[0], mrr_ci[1]],
            "ci_method": f"bootstrap over {n_ans} answerable test questions, {n_boot} resamples",
        },
        "mrr_gain_over_bm25_ci95": [gain_ci[0], gain_ci[1]],
        "end_to_end": {
            "correct": float(correct.mean()),
            "ci95": [e2e_ci[0], e2e_ci[1]],
            "always_abstain_correct": always_abstain,
            "reader_given_gold_context_exact_match": gold_context_em,
            "taxonomy": taxonomy,
        },
        "gate": {
            "rule": config.GATE_RULE,
            "end_to_end_beats_always_abstain": gate_e2e,
            "retrieval_gain_over_bm25": gate_gain,
            "passed": gate_passed,
        },
        "traps": traps,
        "caveats": CAVEATS,
    }
    analysis = {"retriever": headline, "by_type": by_type, "failures": failures}

    (results_dir / "metrics.json").write_text(
        json.dumps(_clean(metrics), indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "error_analysis.json").write_text(
        json.dumps(_clean(analysis), indent=2) + "\n", encoding="utf-8"
    )
    _save_plot(by_type, results_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answering.")
    parser.add_argument("--fast", action="store_true", help="smoke-test mode, results are marked")
    args = parser.parse_args()
    digest = data.verify_checksums()
    docs, questions = data.load_corpus(), data.load_questions()
    metrics = run(docs, questions, config.RESULTS_DIR, fast=args.fast, data_sha256=digest)
    head, e2e = metrics["headline"], metrics["end_to_end"]
    print(
        f"headline {head['retriever']}: test MRR {head['value']:.3f}, 95% interval {head['ci95']}"
    )
    print(f"end to end {e2e['correct']:.3f} (always abstain {e2e['always_abstain_correct']:.3f})")
    print(f"gate passed: {metrics['gate']['passed']}")
    print(f"wrote results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
