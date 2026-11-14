"""Project constants and decision gate configuration.

Change them here, nowhere else.
"""

from pathlib import Path

PROJECT_NAME = "17-marketing-mix-model"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

DATA_SOURCE = "simulated_known_truth"
SEED = 42

# Time series dimensions (weekly marketing observations)
N_WEEKS_TRAIN = 104  # 2 years historical data
N_WEEKS_TEST = 52  # 1 year holdout evaluation
N_WEEKS_TOTAL = N_WEEKS_TRAIN + N_WEEKS_TEST

N_WEEKS_TRAIN_FAST = 52
N_WEEKS_TEST_FAST = 26
N_BOOT_FAST = 50
N_BOOT = 500

# Marketing channels
CHANNELS = ("tv", "search", "social")

# True simulator baseline parameters
TRUE_BASE_SALES = 100_000.0  # weekly organic baseline sales ($)
TRUE_TREND_SLOPE = 50.0  # organic trend ($ per week)
TRUE_SEASONALITY_AMP = 15_000.0  # seasonal sinusoidal amplitude ($)
TRUE_HOLIDAY_SURGE = 25_000.0  # Q4 holiday surge peak ($)
TRUE_NOISE_STD = 4_000.0  # observation noise ($)

# True channel response curves:
# - adstock decay alpha in [0, 1): carryover retention rate per week
# - Hill half-saturation K: spend ($) where response reaches 50% of maximum
# - max lift beta ($): asymptotic maximum weekly incremental revenue
TRUE_CHANNEL_PARAMS = {
    "tv": {
        "alpha": 0.60,  # long carryover brand building
        "half_sat": 40_000.0,  # high saturation ceiling
        "max_lift": 50_000.0,  # highest asymptotic potential
    },
    "search": {
        "alpha": 0.15,  # immediate performance capture
        "half_sat": 18_000.0,  # saturates at moderate spend
        "max_lift": 28_000.0,  # limited non-brand query volume
    },
    "social": {
        "alpha": 0.35,  # moderate carryover
        "half_sat": 25_000.0,  # medium saturation ceiling
        "max_lift": 35_000.0,  # scalable visual consideration
    },
}

# Decision thresholds and gate
MIN_HOLDOUT_R2 = 0.75  # minimum acceptable out-of-sample weekly revenue R^2
MIN_ROI_SPEARMAN = 0.80  # minimum Spearman rank correlation on marginal ROI
MIN_ALLOCATION_GAIN = 0.05  # 5% incremental revenue gain over naive linear allocation

MODELS = (
    "naive_linear",
    "adstock_only",
    "saturation_only",
    "adstock_hill_mmm",
    "oracle",
)
HEADLINE = "adstock_hill_mmm"

GATE_RULE = (
    "Pass only if all three hold for the adstock-saturation MMM (geometric adstock carryover "
    "with Hill saturation), fixed before results as the headline: (1) the 95% bootstrap interval "
    "on holdout weekly revenue R^2 has a lower bound above 0.75; (2) the rank correlation (Spearman rho) "
    "between estimated channel marginal ROIs and true marginal ROIs is at least 0.80; (3) the holdout "
    "incremental revenue under the headline model's recommended budget allocation beats the naive linear "
    "allocation by at least 5%."
)
