"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "12-uplift-modelling"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

DATA_SOURCE = "simulated_known_truth"
SEED = 42
N_TRAIN, N_DEV, N_TEST = 20_000, 10_000, 20_000
BIG_EVAL = 200_000  # fresh population used to read the true value of a policy
N_BOOT = 500
N_FEATURES = 8

# Economics. Assumptions about a made-up retailer.
VALUE_PER_CONVERSION = 50.0
COST_PER_OFFER = 2.0
THRESHOLD = COST_PER_OFFER / VALUE_PER_CONVERSION  # treat when predicted uplift exceeds this
BUDGET_SHARE = 0.30  # for the fixed-budget comparisons
COST_SENSITIVITY = (1.0, 2.0, 5.0, 10.0)

# Population. Assumptions.
TREATMENT_SHARE = 0.5  # randomised trial
HIDDEN_CONFOUNDER_ON_BASELINE = 0.8  # an unmeasured driver of buying
OBS_PROPENSITY = (1.0, 0.8, 1.0)  # how much past offers followed x4, x5 and the hidden driver

TRAIN_SIZES = (2000, 5000, 20_000)
SIZE_REPS = 3

CANDIDATES = ("s_linear", "s_gbm", "t_gbm", "ct_gbm")
REFERENCES = ("treat_none", "treat_all", "random_budget", "response_budget", "oracle")

GATE_RULE = (
    "Pass only if all three hold on the randomised test set for the headline model, which is "
    "picked on dev as the candidate (s_linear, s_gbm, t_gbm, ct_gbm) with the highest area under "
    "the uplift curve (AUUC): (1) the 95% bootstrap interval on the incremental profit per customer "
    "of treating when predicted uplift exceeds cost over value has a lower bound above zero, so it "
    "beats treating nobody; (2) the 95% paired interval on that profit minus the profit of "
    "treating everyone has a lower bound above zero, so it beats blanket offers; (3) the 95% "
    "interval on the Qini coefficient has a lower bound above zero, so the ranking beats random "
    "targeting. Profit is estimated by inverse-probability weighting with the known randomisation. "
    "The cost and value are assumptions fixed before results."
)
