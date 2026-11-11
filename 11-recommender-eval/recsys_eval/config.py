"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "11-recommender-eval"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

DATA_SOURCE = "simulated_known_truth"
SEED = 42
N_BOOT = 1000

# World. Every number is an assumption about a made-up catalogue and made-up users.
N_USERS, N_ITEMS, DIM = 1500, 600, 8
HORIZON = 100.0  # days
DRIFT = 0.5  # how far a user's taste moves over the horizon
BETA, BIAS = 1.2, -1.5  # P(click | shown) = sigmoid(BETA * affinity + BIAS)
EXPOSURE_STRENGTH = 1.5  # how strongly the logging policy favours popular items
TREND_SD = 0.5  # items rise or fall in popularity over the horizon
POP_SD = 1.0
N_EXPOSURES = 250_000
NEW_ITEM_SHARE = 0.4  # items released after day 0
ACTIVITY_SD = 0.8

# Temporal protocol.
VAL_START, TEST_START = 70.0, 85.0
RECENT_WINDOW = 14.0
K = 10
SAMPLED_NEGATIVES = 100
IPS_ALPHA = 0.5
SVD_RANK = 16

CANDIDATES = ("item_knn", "svd", "svd_ips")
REFERENCES = ("random", "popularity", "recent_popularity", "oracle")
ALL_MODELS = (*REFERENCES[:3], *CANDIDATES, "oracle")

SWEEP_EXPOSURE = (0.0, 0.75, 1.5, 2.5)
SPEARMAN_FLOOR = 0.8

GATE_RULE = (
    "Pass only if all three hold on the test window for the headline model, which is picked on "
    "the validation window as the personalised candidate (item_knn, svd, svd_ips) with the "
    "highest NDCG@10 against held-out clicks: (1) the 95% paired bootstrap interval over users "
    "on its NDCG@10 minus popularity's has a lower bound above zero; (2) the 95% paired "
    "interval on its true like rate at 10 minus popularity's has a lower bound above zero, "
    "where the true like rate is known only because this is a simulation; (3) across all seven "
    "models, the Spearman correlation between NDCG@10 on held-out clicks and the true like rate "
    "is at least 0.80, meaning the offline metric can be trusted to choose between models. The "
    "0.80 bar is an assumption fixed before results."
)
