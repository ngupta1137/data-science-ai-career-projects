"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "07-model-monitoring"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

DATA_SOURCE = "simulated_known_truth"
SEED = 42

# Population. Five numeric features, one categorical feature with three levels.
N_NUM = 5
COEF = (1.0, 0.8, -0.6, 0.4, 0.0)  # true log-odds weights
CAT_EFFECT = (0.0, 0.5, -0.5)
BASE_CAT = (0.5, 0.3, 0.2)
REF_SIZE = 20000  # reference (training) rows, also used as the monitoring reference
REF_VAL_SIZE = 20000  # separate rows for the reference AUC

# Stream design. One batch per week.
N_WEEKS = 52
ONSET_WEEK = 26
BATCH_SIZE = 500
SMALL_BATCH_SIZE = 100  # used for the batch-size trap
LABEL_LAG = 4  # weeks until labels arrive for a batch

# Streams.
N_CAL = 1000  # null streams that set thresholds. 200 left about 1.5 points of noise on the achieved false alarm rate (see README, process note)
N_DEV = 100  # per scenario, used to pick the headline monitor
N_TEST = 300  # per scenario, scored once
N_SWEEP = 100  # per shift size
N_TRAP = 300
N_SUPP_NULL = 2000  # follow-up null streams, run after the gate result was known (post hoc, not part of the gate)

BUDGET = 0.05  # target chance of any false alarm in a 52 week null stream
FALSE_ALARM_LIMIT = 0.10  # gate: upper bound of the measured stream false alarm rate
DETECTION_FLOOR = 0.80  # gate: lower bound of detection for the abrupt covariate shift
FIXED_PSI_THRESHOLDS = (0.1, 0.2)  # common rules of thumb, tested as written

# Assumed drift sizes. These are choices, not facts about any real system.
SCENARIOS = {
    "null": {"kind": "none"},
    "covariate_abrupt": {"kind": "mean_shift", "magnitude": 1.0},
    "covariate_small": {"kind": "mean_shift", "magnitude": 0.3},
    "covariate_gradual": {"kind": "mean_shift", "magnitude": 1.0, "ramp_weeks": 12},
    "variance_shift": {"kind": "variance_shift", "magnitude": 1.8},
    "category_shift": {"kind": "category_shift", "magnitude": (0.2, 0.3, 0.5)},
    "concept_drift": {"kind": "concept", "magnitude": -0.5},
}
FEATURE_SCENARIOS = (
    "covariate_abrupt",
    "covariate_small",
    "covariate_gradual",
    "variance_shift",
    "category_shift",
)
SWEEP_SHIFTS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)

LABEL_FREE = ("psi_max", "ks_max", "score_psi")  # tie order for headline selection
MONITORS = (*LABEL_FREE, "auc_drop")

GATE_RULE = (
    "Pass only if both hold for the headline label-free monitor on fresh test streams: "
    "(1) the stream-level false alarm rate on null streams has a 95% interval whose upper "
    "bound is at most 0.10, which is twice the 0.05 budget the threshold was calibrated to; "
    "(2) the detection rate for the abrupt covariate shift has a 95% interval whose lower "
    "bound is at least 0.80. Both limits are design choices made before results were seen."
)
