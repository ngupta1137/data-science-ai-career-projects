"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "02-fraud-detection"
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"

RAW_CSV = RAW_DIR / "creditcard.csv"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

# OpenML id for the ULB credit card fraud dataset. Verify at download time: the script
# checks the column names and, when set, the documented row and fraud counts.
OPENML_DATA_ID = 1597

SEED = 42
TARGET = "Class"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
EXPECTED_ROWS = 284807
EXPECTED_FRAUD = 492

# Time-ordered split: the last TEST_FRAC of transactions is the test set. The last
# VAL_FRAC of the remaining rows is validation (model choice, calibration, threshold).
TEST_FRAC = 0.30
VAL_FRAC = 0.20
MIN_POSITIVES = 20  # every split must hold at least this many fraud rows
N_BOOT = 1000

# ASSUMPTIONS, not facts from the data. Cost of a missed fraud is its transaction
# amount. Cost of flagging any transaction is a fixed manual review cost, in the
# dataset's currency units. The base value is a placeholder; the sensitivity list
# shows how the answer moves. Replace with real review costs before using in a decision.
REVIEW_COST_BASE = 5.0
REVIEW_COSTS = (1.0, 5.0, 20.0)

GATE_RULE = (
    "Pass only if both hold on the test set: (1) the lower bound of the headline model's "
    "95% bootstrap interval on PR-AUC is above the PR-AUC of the constant baseline "
    "(which equals the test fraud rate); (2) at the base review cost, the per-transaction "
    "expected-cost policy costs less than both trivial policies (flag nothing, flag everything)."
)
