"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "15-credit-scorecard"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

DATA_SOURCE = "simulated_known_truth"
SEED = 42

N_HISTORY = 40_000  # past applications: outcomes seen only for approved ones
N_TEST = 40_000  # fresh applications with every outcome known (only a simulation has this)
N_FEATURES = 5

# The world. Assumptions.
HIDDEN_ON_DEFAULT = 0.6  # an applicant trait the underwriter saw but the data did not record
OLD_POLICY_HIDDEN = 0.7  # how much the old approval decision used that trait
OLD_POLICY_CUTOFF = -1.6  # old policy approves below this risk index, softly
OLD_POLICY_SHARPNESS = 3.0
RANDOM_APPROVE_SHARE = 0.05  # share of applicants approved at random, whatever the old policy said
RANDOM_SHARES = (0.01, 0.02, 0.05, 0.10)  # study of how big that sample needs to be
RANDOM_SHARE_REPS = 5

# Economics per unit lent. Assumptions.
MARGIN = 0.15  # earned on a loan that is repaid
LOSS = 0.60  # lost on a loan that defaults
APPROVE_BELOW = MARGIN / (MARGIN + LOSS)  # approve when predicted default chance is below this

MODELS = ("accepted_only", "ipw", "random_sample_only", "recalibrated")
HEADLINE = (
    "recalibrated"  # fixed before results: accepted-only scores recalibrated on the random sample
)
CALIBRATION_BAND = (0.8, 1.25)  # predicted over true default rate among approved
N_BOOT = 1000

GATE_RULE = (
    "Pass only if all three hold on fresh applications with every outcome known, for the "
    "recalibrated model (scores from approved applicants, recalibrated on a random sample of "
    "approvals), fixed before results as the headline, approving when the predicted default "
    "chance is below margin over margin plus loss: (1) the 95% paired bootstrap interval on its "
    "profit per applicant minus the old policy's has a lower bound above zero; (2) the same "
    "against the model trained on approved applicants only has a lower bound above zero; (3) "
    "among the applicants it approves, its mean predicted default chance divided by the true "
    "default rate lies between 0.8 and 1.25."
)
