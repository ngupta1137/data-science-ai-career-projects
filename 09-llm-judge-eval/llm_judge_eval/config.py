"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "09-llm-judge-eval"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

DATA_SOURCE = "simulated_known_truth"
SEED = 42
N_DEV, N_TEST = 1500, 3000
N_BOOT = 1000
N_BOOT_REG = 300  # cluster bootstrap fits of the bias regression
SWEEP_REPS = 5
N_ANNOTATORS = 3  # odd, so the consensus never ties

# Population. Every number below is an assumption about a made-up judge and made-up humans.
HUMAN_BETA = 2.5  # humans see quality sharply
J_BETA = 1.5  # the judge sees quality less sharply than humans
POS_BIAS = 0.4  # logit added to "first slot wins"
LEN_BIAS = 0.5  # logit per standard deviation of length advantage
LENGTH_QUALITY_CORR = 0.3  # longer answers really are a bit better, which confounds length

# Systems for the ranking exercise. Quality is the mean of a per-prompt quality draw.
SYSTEMS = {
    "S1": {"quality": -0.6, "length": 0.0},
    "S2": {"quality": -0.3, "length": 0.0},
    "S3": {"quality": 0.0, "length": 0.0},
    "S4": {"quality": 0.3, "length": 0.0},
    "S5": {"quality": 0.6, "length": 0.0},
    "V": {
        "quality": 0.15,
        "length": 1.5,
    },  # verbose and mid-ranked: the system a biased judge over-rates
}
SYSTEM_N = 2000  # prompts per system pair
BETTER_THAN_V = ("S4", "S5")
OUTRANK_LIMIT = 0.05

AGREEMENT_FLOOR = 0.80  # the level Zheng et al. 2023 report for strong judges, an assumed bar here
EVAL_SIZES = (50, 100, 300, 1000, 3000)
SWEEP_POS = (0.0, 0.4, 0.8)
SWEEP_LEN = (0.0, 0.5, 1.0)

CANDIDATES = ("single_order", "swap_calibrated")  # full coverage: every pair gets a verdict
REFERENCES = ("swap_coin", "length_only", "oracle", "coin_flip")

GATE_RULE = (
    "Pass only if all three hold on the test set for the headline judge, which is picked on dev "
    "as the full-coverage judge candidate (single_order or swap_calibrated) with the higher agreement with the human consensus: (1) the lower "
    "95% bound on agreement with the human consensus is at least 0.80, the level Zheng et al. "
    "2023 report for strong judges; (2) the 95% interval on agreement minus the agreement of "
    "always picking the longer answer has a lower bound above zero; (3) in the system ranking, "
    "the verbose system V, whose true quality is below S4 and S5, beats S4 or S5 head to head "
    "in at most 5% of prompt-level bootstrap resamples. The 0.80 bar and the 5% limit are "
    "assumptions fixed before results."
)
