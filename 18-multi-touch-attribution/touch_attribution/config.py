"""Configuration and pre-registered decision gate for multi-touch attribution."""

from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

# Marketing channels
CHANNELS = ["display", "social", "search", "email"]

# Dataset size
TRAIN_JOURNEYS = 40_000
TEST_JOURNEYS = 10_000
FAST_TRAIN_JOURNEYS = 2_000
FAST_TEST_JOURNEYS = 1_000

# Random seed
SEED = 42

# True data-generating parameters (ground truth for simulation)
BASE_LOGIT = -2.2  # Organic base conversion rate ~ 9.9%
LATENT_INTENT_SCALE = 0.5

# Channel incremental lifts (additive log-odds shift when touched)
CHANNEL_LIFTS = {
    "display": 0.35,  # Top-of-funnel brand awareness
    "social": 0.45,  # Mid-funnel consideration
    "search": 0.65,  # Bottom-of-funnel high-intent conversion
    "email": 0.50,  # Direct nurture / retargeting
}

# Cross-channel synergies (e.g., display awareness amplifies paid search intent)
SYNERGIES = {
    ("display", "search"): 0.25,
    ("social", "search"): 0.15,
}

# Channel unit acquisition cost per touch (in USD)
CHANNEL_COSTS = {
    "display": 0.50,
    "social": 1.20,
    "search": 2.50,
    "email": 0.15,
}

# Total marketing budget for reallocation test (in USD)
REALLOCATION_BUDGET = 50_000.0

# Pre-registered Decision Gate thresholds
HEADLINE_MODEL = "markov"
MAX_CREDIT_SHARE_MAE = 0.05  # Maximum allowable credit share error vs true causal credit (5 pp)
MIN_REALLOCATION_GAIN = 0.05  # At least 5% increase in true incremental conversions vs last touch

GATE_RULE = (
    "Pass only if all three hold for the first-order Markov attribution model, fixed before "
    "results as the headline: (1) the mean absolute error (MAE) across channel credit shares "
    "against true counterfactual Shapley credit is below 0.05 (5 percentage points); (2) the "
    "headline model achieves strictly lower share MAE than the naive last-touch baseline; (3) "
    "budget reallocation using headline credit shares yields at least a 5% increase in true "
    "incremental conversions over last-touch budget allocation on holdout test journeys."
)
