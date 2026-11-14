"""Configuration, clinical decision thresholds, and pre-registered gate for clinical risk scoring."""

from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
CHECKSUM_FILE = DATA_DIR / "CHECKSUMS.txt"

# Dataset sizes
TRAIN_PATIENTS = 6000
TEST_PATIENTS = 3000
FAST_TRAIN_PATIENTS = 1500
FAST_TEST_PATIENTS = 800

# Random seed
SEED = 42

# Hospital site distributions and baseline logits
HOSPITAL_SITES = {
    "tertiary_academic": {
        "share": 0.45,
        "intercept": -2.8,
        "description": "High-acuity tertiary academic medical center with complex surgical referral mix",
    },
    "community_regional": {
        "share": 0.40,
        "intercept": -3.5,
        "description": "Regional general acute care community hospital",
    },
    "rural_access": {
        "share": 0.15,
        "intercept": -4.0,
        "description": "Critical access rural hospital with lower inpatient acuity",
    },
}

# Clinical decision parameters for Decision Curve Analysis
DECISION_THRESHOLD = 0.15  # 15% 30-day adverse event risk threshold
THRESHOLD_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
OUTREACH_COST_USD = 150.0  # Transitional care coordinator outreach cost per patient
AVOIDED_ADMISSION_VALUE_USD = 1000.0  # Net financial and clinical benefit of prevented readmission

# Pre-registered Decision Gate thresholds
HEADLINE_MODEL = "calibrated_clinical_risk"
MAX_BRIER_SCORE = 0.150
MAX_ECE = 0.035
MIN_NET_BENEFIT = 0.085
MIN_AUC = 0.720
MIN_AUC_CI_LOWER = 0.700
MIN_BRIER_IMPROVEMENT = 0.030
MIN_ECE_IMPROVEMENT = 0.100

GATE_RULE = (
    "Pass only if all four hold for the headline Calibrated Clinical Risk Model, fixed before results: "
    "(1) Brier score on holdout admissions is below 0.150 (and strictly lower than uncalibrated baseline "
    "by at least 0.030); (2) Expected Calibration Error (ECE across 10 bins) is below 0.035; "
    "(3) Net Benefit at clinical intervention threshold pt = 0.15 is at least 0.085 and strictly exceeds "
    "both Treat All and the uncalibrated model; (4) Discrimination AUC is at least 0.720 with 95% bootstrap "
    "confidence interval lower bound above 0.700."
)

HEALTHCARE_DISCLAIMER = (
    "No patient data is used in this repository. All clinical observations are synthetically generated. "
    "Nothing herein constitutes medical advice, clinical guidance, or regulatory determination."
)
