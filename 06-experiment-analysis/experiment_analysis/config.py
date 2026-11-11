"""Project constants. Change them here, nowhere else."""

from pathlib import Path

PROJECT_NAME = "06-experiment-analysis"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

DATA_SOURCE = "simulated_known_truth"
SEED = 42
ALPHA = 0.05
N_REPS = 2000  # simulated experiments per cell
CHUNK = 250  # experiments generated at a time, to bound memory
N_BOOT = 1000

# Fixed-horizon grid: sample size per arm, outcome distribution.
GRID_N = (200, 1000, 5000)
GRID_DIST = ("normal", "lognormal")
RHO = 0.6  # correlation between pre-period metric and outcome
TRUE_EFFECT = 0.1  # in standard deviations of the outcome

# Headline cell: the one the decision gate is written about.
HEADLINE_N = 1000
HEADLINE_DIST = "normal"
# The A/A check at the headline cell needs a tight interval: with 2,000 experiments the
# interval is about +/- 1 point and cannot separate 0.05 from 0.06. See README, process note.
HEADLINE_AA_REPS = 20000

RHO_GRID = (0.0, 0.3, 0.6, 0.8)
PEEK_N = 2000
PEEK_LOOKS = (1, 5, 10, 20)

MULTI_METRICS = 10
MULTI_N = 500
MULTI_TRUE_EFFECT = 0.2
MULTI_N_TRUE = 2

POWER_TARGET = 0.8
CONTAMINATION_SHIFT = 0.5  # a covariate that moves with treatment (the CUPED trap)

GATE_RULE = (
    "Pass only if all four hold, each judged on 95% intervals from the simulation: "
    "(1) at the headline cell, the A/A false positive rate of CUPED with a fixed horizon "
    "has an interval that contains 0.05; "
    "(2) at the headline cell with a true effect, the CUPED estimate has no detectable bias "
    "(the interval on the mean error contains 0); "
    "(3) the variance of the CUPED estimate divided by the variance of the raw estimate has "
    "an interval whose upper bound is below 1; "
    "(4) the Holm family-wise error rate over the multiple-metric A/A set has an interval "
    "whose lower bound is at or below 0.05."
)
