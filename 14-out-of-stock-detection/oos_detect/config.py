"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "14-out-of-stock-detection"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

DATA_SOURCE = "simulated_known_truth"
SEED = 42

# The world. All are assumptions about a made-up chain.
N_STORES = 50
N_ITEMS = 200  # per store, so 10,000 shelves
HISTORY_DAYS = 60  # no visits, used to estimate each shelf's sales rate
EVAL_DAYS = 60  # the policy runs on these days
RATE_LOG_MEAN, RATE_LOG_SD = -0.5, 1.2  # units per day; many slow sellers
OOS_START_RATE = (
    0.004  # chance per day that an in-stock shelf goes empty while the system shows stock
)
OOS_MEAN_DAYS = 14  # an empty shelf stays empty this long on average unless someone fixes it

# Economics. Assumptions.
MARGIN = 1.0  # per unit
VISIT_COST = 3.0  # cost of sending a rep to one shelf
CAPACITY = 150  # shelf visits per day across the chain
CAPACITY_SENSITIVITY = (50, 150, 300)

# Detector settings fixed before results.
ZERO_RUN_RULE = 3  # the common rule: flag after this many days with no sales
CLEAN_RUN_MIN = (
    5  # history zero runs at least this long are treated as likely empty shelf when estimating rate
)
PRIOR_EMPTY = 0.05  # prior share of shelves that are empty; in practice it comes from shelf audits
EXPECTED_REMAINING_DAYS = (
    OOS_MEAN_DAYS  # with a constant end rate, remaining days do not depend on age
)

POLICIES = (
    "no_visits",
    "zero_run_rule",
    "poisson_alarm",
    "expected_value",
    "expected_value_naive_rate",
    "oracle",
)
HEADLINE = "expected_value"  # fixed before results
N_BOOT = 1000
MIN_PRECISION = 0.5

GATE_RULE = (
    "Pass only if all three hold for the expected value policy (rank shelves by the chance they "
    "are empty times the margin they lose, visit only when that beats the visit cost, within the "
    "daily capacity), fixed before results as the headline: (1) the 95% store-bootstrap interval "
    "on its net value per store per day over no visits has a lower bound above zero; (2) the 95% "
    "paired store-bootstrap interval on its net value minus that of the three-zero-day rule has a "
    "lower bound above zero; (3) at least half of its visits find an empty shelf."
)
