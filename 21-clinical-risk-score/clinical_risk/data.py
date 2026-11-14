"""Simulated clinical inpatient admissions with known true 30-day adverse event risk."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from . import config


class DataValidationError(ValueError):
    """Raised when clinical dataset fails schema or biological integrity checks."""


@dataclass
class PatientAdmissionObservation:
    patient_id: int
    hospital_site: str
    age: float
    prior_admissions: int
    comorbidity_charlson: int
    length_of_stay: float
    polypharmacy_med_count: int
    abnormal_lab_count: int
    social_vulnerability_index: float
    true_risk_probability: float
    readmitted_30d: int


@dataclass
class ClinicalRiskDataset:
    observations: list[PatientAdmissionObservation]
    total_admissions: int
    n_sites: int
    site_counts: dict[str, int]
    prevalence: float
    mean_age: float
    sha256: str

    def __len__(self) -> int:
        return len(self.observations)

    def to_arrays(self, include_site: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract design matrix X, binary outcomes y, and true risk probabilities."""
        obs = self.observations

        base_features = [
            np.array([o.age for o in obs], dtype=float),
            np.array([o.prior_admissions for o in obs], dtype=float),
            np.array([o.comorbidity_charlson for o in obs], dtype=float),
            np.array([o.length_of_stay for o in obs], dtype=float),
            np.array([o.polypharmacy_med_count for o in obs], dtype=float),
            np.array([o.abnormal_lab_count for o in obs], dtype=float),
            np.array([o.social_vulnerability_index for o in obs], dtype=float),
        ]

        if include_site:
            # Reference site: tertiary_academic (first site)
            sites = sorted(config.HOSPITAL_SITES.keys())
            for s in sites[1:]:
                site_indicator = np.array(
                    [1.0 if o.hospital_site == s else 0.0 for o in obs], dtype=float
                )
                base_features.append(site_indicator)

        X = np.column_stack(base_features)
        y = np.array([o.readmitted_30d for o in obs], dtype=int)
        p = np.array([o.true_risk_probability for o in obs], dtype=float)
        return X, y, p


def generate_clinical_data(
    n_patients: int,
    seed: int = config.SEED,
) -> ClinicalRiskDataset:
    """Generate simulated inpatient admissions with known ground-truth adverse event risk."""
    rng = np.random.default_rng(seed)
    sites = list(config.HOSPITAL_SITES.keys())
    site_shares = [config.HOSPITAL_SITES[s]["share"] for s in sites]

    assigned_sites = rng.choice(sites, size=n_patients, p=site_shares)
    site_counts = dict.fromkeys(sites, 0)
    for s in assigned_sites:
        site_counts[s] += 1

    ages = rng.normal(68.0, 11.5, size=n_patients).clip(21.0, 95.0)
    priors = rng.poisson(1.0, size=n_patients).clip(0, 8)
    charlsons = rng.poisson(2.2, size=n_patients).clip(0, 9)
    los_days = rng.exponential(3.2, size=n_patients).clip(1.0, 21.0)
    meds = rng.normal(9.5, 3.2, size=n_patients).clip(1.0, 28.0)
    labs = rng.poisson(1.3, size=n_patients).clip(0, 7)
    svis = rng.uniform(0.05, 0.95, size=n_patients)

    observations: list[PatientAdmissionObservation] = []
    events: list[int] = []

    for i in range(n_patients):
        site_name = assigned_sites[i]
        site_int = float(config.HOSPITAL_SITES[site_name]["intercept"])

        # True latent log-odds of 30-day readmission
        logit = (
            site_int
            + 0.025 * (float(ages[i]) - 65.0)
            + 0.42 * float(priors[i])
            + 0.34 * float(charlsons[i])
            + 0.18 * float(np.log(los_days[i]))
            + 0.05 * (float(meds[i]) - 10.0)
            + 0.26 * float(labs[i])
            + 0.48 * (float(svis[i]) - 0.5)
        )

        prob = float(1.0 / (1.0 + np.exp(-logit)))
        readmitted = int(rng.binomial(1, prob))

        obs = PatientAdmissionObservation(
            patient_id=i,
            hospital_site=site_name,
            age=float(round(ages[i], 1)),
            prior_admissions=int(priors[i]),
            comorbidity_charlson=int(charlsons[i]),
            length_of_stay=float(round(los_days[i], 1)),
            polypharmacy_med_count=round(meds[i]),
            abnormal_lab_count=int(labs[i]),
            social_vulnerability_index=float(round(svis[i], 3)),
            true_risk_probability=prob,
            readmitted_30d=readmitted,
        )
        observations.append(obs)
        events.append(readmitted)

    # Compute deterministic SHA-256 fingerprint
    hasher = hashlib.sha256()
    hasher.update(f"clinical_risk,patients={n_patients},seed={seed},events={sum(events)}".encode())
    for s in sorted(sites):
        hasher.update(f"{s}:{site_counts[s]}".encode())
    sha = hasher.hexdigest()

    dataset = ClinicalRiskDataset(
        observations=observations,
        total_admissions=n_patients,
        n_sites=len(sites),
        site_counts=site_counts,
        prevalence=float(np.mean(events)),
        mean_age=float(np.mean(ages)),
        sha256=sha,
    )
    validate(dataset)
    return dataset


def validate(dataset: ClinicalRiskDataset) -> None:
    """Validate clinical risk dataset schema and medical boundary conditions."""
    if dataset.total_admissions <= 0:
        raise DataValidationError("Dataset must contain at least one observation.")
    if dataset.total_admissions != len(dataset.observations):
        raise DataValidationError("Total admissions mismatch observation count.")
    for s, count in dataset.site_counts.items():
        if count <= 0:
            raise DataValidationError(f"Hospital site {s} has zero admissions.")
    for obs in dataset.observations:
        if obs.age < 18.0 or obs.age > 115.0:
            raise DataValidationError(f"Patient age out of valid range: {obs.age}.")
        if obs.prior_admissions < 0:
            raise DataValidationError(
                f"Prior admissions cannot be negative: {obs.prior_admissions}."
            )
        if obs.comorbidity_charlson < 0:
            raise DataValidationError(
                f"Charlson comorbidity cannot be negative: {obs.comorbidity_charlson}."
            )
        if obs.length_of_stay < 1.0:
            raise DataValidationError(
                f"Length of stay must be at least 1 day: {obs.length_of_stay}."
            )
        if obs.polypharmacy_med_count < 1:
            raise DataValidationError(
                f"Discharge medication count must be positive: {obs.polypharmacy_med_count}."
            )
        if obs.abnormal_lab_count < 0:
            raise DataValidationError(
                f"Abnormal lab count cannot be negative: {obs.abnormal_lab_count}."
            )
        if obs.social_vulnerability_index < 0.0 or obs.social_vulnerability_index > 1.0:
            raise DataValidationError(
                f"Social vulnerability index must be in [0, 1]: {obs.social_vulnerability_index}."
            )
        if obs.true_risk_probability <= 0.0 or obs.true_risk_probability >= 1.0:
            raise DataValidationError(
                f"Risk probability must be in (0, 1): {obs.true_risk_probability}."
            )
        if obs.readmitted_30d not in (0, 1):
            raise DataValidationError(
                f"30-day readmission outcome must be binary: {obs.readmitted_30d}."
            )
        if obs.hospital_site not in config.HOSPITAL_SITES:
            raise DataValidationError(f"Unknown hospital site: {obs.hospital_site}.")
