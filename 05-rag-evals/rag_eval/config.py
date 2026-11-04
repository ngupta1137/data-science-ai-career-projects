"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "05-rag-evals"
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"

CORPUS_FILE = DATA_DIR / "corpus.jsonl"
QUESTIONS_FILE = DATA_DIR / "questions.jsonl"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

SEED = 42
K_CONTEXT = 3  # documents passed to the reader
RRF_K = 60  # reciprocal rank fusion constant
BM25_GRID = [(k1, b) for k1 in (0.9, 1.2, 1.5) for b in (0.4, 0.75)]
LSA_DIMS = (10, 20, 30)
N_BOOT = 1000

ANSWERABLE_TYPES = ("lexical", "paraphrase", "numeric_detail", "stale_version")
UNANSWERABLE_TYPES = ("unanswerable_attribute", "unanswerable_offtopic")
QUESTION_TYPES = ANSWERABLE_TYPES + UNANSWERABLE_TYPES

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "under",
        "what",
        "when",
        "which",
        "who",
        "with",
        "within",
    }
)

GATE_RULE = (
    "Pass only if both hold on the test questions: (1) the lower bound of the 95% bootstrap "
    "interval on end-to-end correctness is above the correctness of the always-abstain policy "
    "(which equals the share of unanswerable questions); (2) unless the headline retriever is "
    "BM25 itself, the lower bound of the 95% paired bootstrap interval on the MRR gain over "
    "BM25 is above zero, so that any extra complexity is shown to pay for itself."
)
