"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "16-claims-frequency-severity"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

DATA_SOURCE = "simulated_known_truth"
SEED = 42

# The world. Assumptions about an auto insurance portfolio.
N_TRAIN = 30000  # historical written policies
N_TEST = 20000  # holdout policies for evaluation
TARGET_LOSS_RATIO = 0.70  # target claims / premium ratio (70% claims, 30% expense and margin)

# Economics and risk bounds.
MAX_TOP_DECILE_LR = (
    0.95  # maximum sustainable loss ratio in highest-risk decile (break-even is 1.0)
)
MIN_GINI = 0.25  # minimum acceptable Gini index for risk segmentation

# Candidate pricing methods.
POLICIES = (
    "flat_baseline",
    "no_exposure",
    "single_tweedie",
    "freq_sev_two_part",
    "oracle",
)
HEADLINE = "freq_sev_two_part"  # fixed before results
N_BOOT = 500

GATE_RULE = (
    "Pass only if all three hold for the two-part Poisson-Gamma model (Poisson frequency with "
    "exposure offset times Gamma severity on positive claims), fixed before results as the "
    "headline: (1) the 95% bootstrap interval on its pure premium Gini index against holdout losses "
    "has a lower bound above 0.25; (2) the 95% paired bootstrap interval on its Gini index minus "
    "that of the single compound Tweedie model has a lower bound above zero; (3) the actual loss "
    "ratio in the highest predicted risk decile is below 0.95."
)
