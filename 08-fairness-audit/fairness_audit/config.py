"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "08-fairness-audit"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

DATA_SOURCE = "simulated_known_truth"
SEED = 42
N_FIT, N_DEV, N_TEST = 6000, 3000, 6000
N_BOOT = 1000
DEV_SIZES = (3000, 10000, 30000)  # post hoc study of how much dev data subgroup thresholds need
DEV_REPS = 100
BIG_EVAL = 100_000  # large fresh set that approximates the population gap

# Population. All of these are assumptions about a made-up lending-like setting.
MINORITY_SHARE = 0.25  # group B
LATENT_MEAN = {"A": 0.0, "B": -0.4}  # group B starts with a lower average latent score
SIGNAL = 1.5  # strength of the latent score in the outcome
NOISE_SD = {"A": 1.0, "B": 1.8}  # group B's second measurement is noisier
PROXY_STRENGTH = 0.8  # x3 carries group membership and nothing about the outcome

FEATURES_ALL = ("x1", "x2", "x3", "x4")
FEATURES_NO_PROXY = ("x1", "x2", "x4")

# Decision economics. Assumptions, with a sensitivity table.
BENEFIT_GOOD = 1.0
COST_BAD = 2.0
COST_SENSITIVITY = (1.0, 2.0, 5.0)

TPR_GAP_TOL = 0.10  # an assumed tolerance for a learning project, not a legal standard
SELECTION_RATIO_SCREEN = 0.8  # the four-fifths screening convention, not a legal determination

CANDIDATES = (
    "aware_single",
    "unaware_single",
    "blind_single",
    "unaware_group_thresholds",
    "reject_all",
)

GATE_RULE = (
    "Pass only if both hold on the test set for the headline policy, which is picked on dev "
    "as the highest-utility candidate with positive utility and a dev true positive rate gap "
    "no larger than 0.10 (or, if none qualifies, the smallest gap): (1) the 95% interval on "
    "the absolute true positive rate gap between groups has an upper bound at most 0.10; "
    "(2) the 95% interval on mean utility per applicant has a lower bound above zero, so the "
    "policy beats rejecting everyone. The 0.10 tolerance is an assumption made before results."
)
