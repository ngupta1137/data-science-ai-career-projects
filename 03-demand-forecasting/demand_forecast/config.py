"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "03-demand-forecasting"
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"

RAW_CSV = RAW_DIR / "day.csv"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

# UCI Bike Sharing dataset, daily file. Verify the URL and terms before release.
SOURCE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
SOURCE_MEMBER = "day.csv"
EXPECTED_ROWS = 731

SEED = 42
TARGET = "cnt"
HORIZON = 7  # days ahead. Every feature must be known this far in advance.
INITIAL_TRAIN = 365  # days of history before the first forecast origin
STEP = 7  # days between forecast origins

# The first half of the origins tunes hyperparameters. The second half is the test.
# Test origins are all later than tuning origins.
N_BOOT = 1000

GATE_RULE = (
    "Pass if the lower bound of the 95% block-bootstrap interval on the test MAE gap "
    "(best baseline MAE minus headline model MAE) is above zero. The headline model and "
    "the best baseline are both chosen on the tuning folds, never on the test folds."
)
