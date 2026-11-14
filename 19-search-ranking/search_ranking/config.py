"""Configuration and pre-registered decision gate for search ranking."""

from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

# Dataset sizes
TRAIN_QUERIES = 3_000
TEST_QUERIES = 1_000
FAST_TRAIN_QUERIES = 500
FAST_TEST_QUERIES = 200
CANDIDATES_PER_QUERY = 10  # K = 10 items per impression

# Random seed
SEED = 42

# Position examination curve parameters (Joachims eye-tracking power law)
# P(examine | k) = 1 / (k ** POSITION_DECAY_ETA)
POSITION_DECAY_ETA = 1.35
MAX_IPS_WEIGHT = 15.0  # Propensity weight clipping to control variance

# Query type distribution
QUERY_PROBS = {
    "head": 0.25,
    "torso": 0.45,
    "tail": 0.30,
}

# Graded relevance levels: 0 (irrelevant) to 3 (highly relevant)
RELEVANCE_LEVELS = [0, 1, 2, 3]

# Pre-registered Decision Gate thresholds
HEADLINE_MODEL = "ips_debiased"
MIN_TRUE_NDCG_LIFT = 0.030  # Minimum required True NDCG@10 lift over production baseline
MIN_TAIL_NDCG_LIFT = 0.000  # Non-negative lift on tail queries (no degradation)

GATE_RULE = (
    "Pass only if all three hold for the candidate Inverse Propensity Scoring ranker "
    "(ips_debiased), fixed before results as the headline: (1) the True Unbiased NDCG@10 "
    "lift over the baseline production ranker is at least +0.030 (with 95% bootstrap CI lower "
    "bound strictly greater than 0); (2) the True NDCG@10 lift on tail queries is non-negative "
    "(>= 0.000), ensuring head query popularity does not mask tail query degradation; (3) the "
    "offline IPS-counterfactual NDCG lift directionally agrees with true unbiased NDCG lift "
    "(sign matching), confirming that the offline metric does not vanish once position bias "
    "is removed."
)
