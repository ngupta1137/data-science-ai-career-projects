"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "01-house-price-regression"
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"

RAW_CSV = RAW_DIR / "california_housing.csv"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

SEED = 42

TARGET = "MedHouseVal"  # median house value, units of $100,000 (1990 dollars)
TARGET_CAP = 5.0  # values at or above this are censored in the source data
FEATURES_RAW = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
EXPECTED_ROWS = 20640

# Spatial leakage control: rows in the same 0.5 degree cell never straddle train and test.
GROUP_CELL_DEG = 0.5
TEST_SIZE = 0.2
CV_FOLDS = 5
N_BOOT = 1000

# Decision gate, fixed before any result was seen: the headline model passes if the
# upper end of its 95% interval on test RMSE is below the best baseline's test RMSE.
GATE_RULE = (
    "Pass if the upper bound of the headline model's 95% cluster-bootstrap interval "
    "on test RMSE is below the best baseline's test RMSE."
)
