"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "10-structured-extraction"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

DATA_SOURCE = "simulated_known_truth"
SEED = 42
N_TRAIN, N_DEV, N_TEST_SEEN, N_TEST_UNSEEN = 2000, 600, 1200, 600  # unseen is per template
N_BOOT = 1000
FIELDS = ("vendor", "invoice_number", "date", "total", "currency")

SEEN_TEMPLATES = ("A", "B", "C")
UNSEEN_TEMPLATES = ("E", "G")  # E: new label words in English. G: German labels and number format.

ABSTAIN_BELOW = 0.5  # the learned extractor returns nothing when its best candidate scores lower

OCR_RATES = (0.0, 0.005, 0.01, 0.02)
TRAIN_SIZES = (50, 200, 800, 2000)
NEW_TEMPLATE_LABELS = (0, 10, 30, 100)  # post hoc study

EXTRACTORS = ("template_regex", "span_ranker")
REFERENCES = ("first_match",)

DOC_FLOOR_SEEN = 0.90
DOC_FLOOR_UNSEEN = 0.70

GATE_RULE = (
    "Pass only if all three hold on the test sets for the headline extractor, which is picked on "
    "dev as the one with the higher share of documents with all five fields correct: (1) on "
    "documents from templates seen in training, the lower 95% bound on that share is at least "
    "0.90; (2) on documents from the two unseen templates, pooled, the lower 95% bound is at "
    "least 0.70; (3) on seen templates, the 95% paired interval on that share minus the share of "
    "the first-match baseline has a lower bound above zero. The 0.90 and 0.70 bars are "
    "assumptions fixed before results."
)
