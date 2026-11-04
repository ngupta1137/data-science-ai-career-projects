"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "04-text-classification"
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"

RAW_CSV = RAW_DIR / "newsgroups.csv"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

# 20 Newsgroups through scikit-learn. The official train/test split is by posting date,
# so the test set is later than the training set.
EXPECTED_ROWS = 18846
EXPECTED_TRAIN = 11314
EXPECTED_TEST = 7532
EXPECTED_CLASSES = 20
METADATA_REMOVED = ("headers", "footers", "quotes")

SEED = 42
VAL_FRAC = 0.20  # stratified random share of the training rows held out to choose a model
N_BOOT = 1000
TOP_FEATURES = 8

GATE_RULE = (
    "Pass if both hold on the test set: (1) the lower bound of the headline model's 95% "
    "bootstrap interval on macro-F1 is above the majority-class baseline's macro-F1; "
    "(2) unless the headline is the Naive Bayes model itself, the lower bound of the 95% "
    "paired bootstrap interval on macro-F1 gain over Naive Bayes is above zero, so that any "
    "extra complexity is shown to pay for itself."
)
