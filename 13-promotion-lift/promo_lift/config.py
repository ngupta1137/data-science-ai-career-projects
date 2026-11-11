"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "13-promotion-lift"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

DATA_SOURCE = "simulated_known_truth"
SEED = 42

# The world. All are assumptions about a made-up grocery chain.
N_STORES = 200  # half get the promotion
WEEKS = 52
PROMO_START = 30  # weeks 30 to 33 are on promotion
PROMO_WEEKS = 4
POST_WEEKS = 4  # pull-forward window after the promotion
PRE_WEEKS = 8  # the comparison window before the promotion
BASE_LOG_MEAN, BASE_LOG_SD = 4.0, 0.5  # weekly units of the promoted item, per store
SIBLING_RATIO = 0.8  # the sibling item sells about 0.8 of the promoted item
SEASON_AMPLITUDE = 0.15  # peaks in the promotion weeks, which flatters a before-and-after view
SEASON_PEAK_WEEK = 33
TREND_MEAN, TREND_SD = 0.002, 0.003  # weekly growth, per store
NOISE_CV = 0.15  # week-to-week demand noise on top of Poisson counts
URBAN_SHARE = 0.5

# True effects of the promotion.
LIFT = {"urban": 0.60, "rural": 0.35}  # extra units of the promoted item during the promotion
CANNIBALISATION = 0.10  # share of sibling sales lost during the promotion
PULL_FORWARD = 0.08  # share of promoted-item sales lost in each post week

# Economics per unit. Assumptions.
MARGIN_ITEM = 2.0
DISCOUNT = 0.6  # paid on every promoted unit, including those that would have sold anyway
MARGIN_SIBLING = 1.5

ESTIMATORS = ("pre_post", "window_did", "full_did")
HEADLINE = "full_did"  # chosen before results as the method this project argues for

N_REPS = 400  # simulated campaigns in the method study
STORE_COUNTS = (40, 100, 200, 400)
POWER_REPS = 200
MIN_COVERAGE = 0.90

GATE_RULE = (
    "Pass only if all three hold for the full difference-in-differences estimator (promoted item "
    "plus sibling, promotion plus post weeks, randomised control stores), fixed before results as "
    "the headline: (1) across 400 simulated campaigns its 95% interval covers the true "
    "incremental profit in at least 90% of them; (2) on the headline campaign its 95% interval "
    "excludes zero, so it makes a definite call; (3) that call has the same sign as the true "
    "incremental profit."
)
