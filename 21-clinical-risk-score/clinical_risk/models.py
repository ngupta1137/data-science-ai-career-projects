"""Clinical risk prediction models: Calibrated Logistic, Uncalibrated Balanced, and Baselines."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import config
from .data import ClinicalRiskDataset


class CalibratedClinicalRiskModel:
    """Headline well-calibrated clinical risk estimator with hospital site controls."""

    name: str = "calibrated_clinical_risk"

    def __init__(self, c_reg: float = 1.0) -> None:
        self.c_reg = c_reg
        self.classifier: LogisticRegression | None = None
        self.feature_names: list[str] = []

    def fit(self, dataset: ClinicalRiskDataset) -> CalibratedClinicalRiskModel:
        X, y, _ = dataset.to_arrays(include_site=True)
        clf = LogisticRegression(
            C=self.c_reg,
            max_iter=1000,
            random_state=config.SEED,
            solver="lbfgs",
        )
        clf.fit(X, y)
        self.classifier = clf
        return self

    def predict_proba(self, dataset: ClinicalRiskDataset) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Model must be fitted before predicting.")
        X, _, _ = dataset.to_arrays(include_site=True)
        return self.classifier.predict_proba(X)[:, 1]


class UncalibratedRiskModel:
    """Uncalibrated model trained with class reweighting (common medical ML trap).

    Achieves identical or competitive ranking discrimination (AUC) but severely
    inflates predicted probabilities, destroying calibration and Net Benefit.
    """

    name: str = "uncalibrated_risk"

    def __init__(self, c_reg: float = 1.0) -> None:
        self.c_reg = c_reg
        self.classifier: LogisticRegression | None = None

    def fit(self, dataset: ClinicalRiskDataset) -> UncalibratedRiskModel:
        X, y, _ = dataset.to_arrays(include_site=True)
        # class_weight='balanced' artificially boosts positive class odds
        clf = LogisticRegression(
            class_weight="balanced",
            C=self.c_reg,
            max_iter=1000,
            random_state=config.SEED,
            solver="lbfgs",
        )
        clf.fit(X, y)
        self.classifier = clf
        return self

    def predict_proba(self, dataset: ClinicalRiskDataset) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Model must be fitted before predicting.")
        X, _, _ = dataset.to_arrays(include_site=True)
        return self.classifier.predict_proba(X)[:, 1]


class UnadjustedSiteRiskModel:
    """Model omitting hospital site controls, causing cross-facility case-mix calibration drift."""

    name: str = "unadjusted_site_risk"

    def __init__(self, c_reg: float = 1.0) -> None:
        self.c_reg = c_reg
        self.classifier: LogisticRegression | None = None

    def fit(self, dataset: ClinicalRiskDataset) -> UnadjustedSiteRiskModel:
        X_no_site, y, _ = dataset.to_arrays(include_site=False)
        clf = LogisticRegression(
            C=self.c_reg,
            max_iter=1000,
            random_state=config.SEED,
            solver="lbfgs",
        )
        clf.fit(X_no_site, y)
        self.classifier = clf
        return self

    def predict_proba(self, dataset: ClinicalRiskDataset) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Model must be fitted before predicting.")
        X_no_site, _, _ = dataset.to_arrays(include_site=False)
        return self.classifier.predict_proba(X_no_site)[:, 1]


class SimpleTriageBaseline:
    """Rule-based clinical triage heuristic using age and comorbidity thresholds."""

    name: str = "simple_triage_baseline"

    def __init__(self) -> None:
        self.base_prob: float = 0.18

    def fit(self, dataset: ClinicalRiskDataset) -> SimpleTriageBaseline:
        _, y, _ = dataset.to_arrays(include_site=False)
        self.base_prob = float(np.mean(y))
        return self

    def predict_proba(self, dataset: ClinicalRiskDataset) -> np.ndarray:
        obs = dataset.observations
        probs: list[float] = []
        for o in obs:
            # High risk score if elderly with multiple prior admissions or comorbidities
            score = 0
            if o.age >= 75.0:
                score += 1
            if o.prior_admissions >= 2:
                score += 2
            if o.comorbidity_charlson >= 3:
                score += 2
            if o.length_of_stay >= 5.0:
                score += 1
            if o.abnormal_lab_count >= 2:
                score += 1

            # Approximate probability scale from heuristic score (0 to 7)
            prob = 1.0 / (1.0 + np.exp(-(float(score) * 0.45 - 2.5)))
            probs.append(float(np.clip(prob, 0.02, 0.95)))
        return np.array(probs, dtype=float)


class OracleModel:
    """Theoretical ceiling knowing exact data-generating risk probabilities."""

    name: str = "oracle"

    def fit(self, dataset: ClinicalRiskDataset) -> OracleModel:
        del dataset
        return self

    def predict_proba(self, dataset: ClinicalRiskDataset) -> np.ndarray:
        _, _, p = dataset.to_arrays(include_site=True)
        return np.array(p, dtype=float)
