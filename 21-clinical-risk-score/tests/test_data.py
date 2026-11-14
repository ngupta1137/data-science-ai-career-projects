import pytest

from clinical_risk import config, data


def test_dataset_generation_shape():
    ds = data.generate_clinical_data(n_patients=500, seed=42)
    assert len(ds) == 500
    assert ds.total_admissions == 500
    assert set(ds.site_counts.keys()) == set(config.HOSPITAL_SITES.keys())
    assert 0.10 <= ds.prevalence <= 0.35
    assert 50.0 <= ds.mean_age <= 85.0


def test_data_validation_catches_invalid_values():
    ds = data.generate_clinical_data(n_patients=100, seed=42)
    data.validate(ds)

    # Corrupt age below minimum
    corrupt_obs = list(ds.observations)
    corrupt_obs[0] = data.PatientAdmissionObservation(
        patient_id=0,
        hospital_site="tertiary_academic",
        age=12.0,  # below 18
        prior_admissions=1,
        comorbidity_charlson=2,
        length_of_stay=3.0,
        polypharmacy_med_count=8,
        abnormal_lab_count=1,
        social_vulnerability_index=0.5,
        true_risk_probability=0.20,
        readmitted_30d=0,
    )
    bad_ds = data.ClinicalRiskDataset(
        observations=corrupt_obs,
        total_admissions=len(corrupt_obs),
        n_sites=ds.n_sites,
        site_counts=ds.site_counts,
        prevalence=ds.prevalence,
        mean_age=ds.mean_age,
        sha256="test",
    )
    with pytest.raises(data.DataValidationError, match="Patient age out of valid range"):
        data.validate(bad_ds)

    # Corrupt length of stay < 1.0
    corrupt_obs[0] = data.PatientAdmissionObservation(
        patient_id=0,
        hospital_site="tertiary_academic",
        age=65.0,
        prior_admissions=1,
        comorbidity_charlson=2,
        length_of_stay=0.5,  # invalid < 1.0
        polypharmacy_med_count=8,
        abnormal_lab_count=1,
        social_vulnerability_index=0.5,
        true_risk_probability=0.20,
        readmitted_30d=0,
    )
    bad_ds_los = data.ClinicalRiskDataset(
        observations=corrupt_obs,
        total_admissions=len(corrupt_obs),
        n_sites=ds.n_sites,
        site_counts=ds.site_counts,
        prevalence=ds.prevalence,
        mean_age=ds.mean_age,
        sha256="test",
    )
    with pytest.raises(data.DataValidationError, match="Length of stay must be at least 1 day"):
        data.validate(bad_ds_los)


def test_invalid_probability_or_outcome():
    ds = data.generate_clinical_data(n_patients=50, seed=42)
    corrupt_obs = list(ds.observations)
    corrupt_obs[0] = data.PatientAdmissionObservation(
        patient_id=0,
        hospital_site="community_regional",
        age=65.0,
        prior_admissions=1,
        comorbidity_charlson=2,
        length_of_stay=3.0,
        polypharmacy_med_count=8,
        abnormal_lab_count=1,
        social_vulnerability_index=0.5,
        true_risk_probability=1.5,  # probability > 1.0
        readmitted_30d=0,
    )
    bad_ds_prob = data.ClinicalRiskDataset(
        observations=corrupt_obs,
        total_admissions=len(corrupt_obs),
        n_sites=ds.n_sites,
        site_counts=ds.site_counts,
        prevalence=ds.prevalence,
        mean_age=ds.mean_age,
        sha256="test",
    )
    with pytest.raises(data.DataValidationError, match="Risk probability must be in"):
        data.validate(bad_ds_prob)


def test_deterministic_generation():
    ds1 = data.generate_clinical_data(n_patients=300, seed=99)
    ds2 = data.generate_clinical_data(n_patients=300, seed=99)
    assert ds1.sha256 == ds2.sha256
    assert ds1.prevalence == pytest.approx(ds2.prevalence)
    assert ds1.mean_age == pytest.approx(ds2.mean_age)


def test_hospital_site_case_mix():
    ds = data.generate_clinical_data(n_patients=2000, seed=42)
    obs = ds.observations
    tertiary_probs = [
        o.true_risk_probability for o in obs if o.hospital_site == "tertiary_academic"
    ]
    rural_probs = [o.true_risk_probability for o in obs if o.hospital_site == "rural_access"]

    mean_tertiary = sum(tertiary_probs) / len(tertiary_probs)
    mean_rural = sum(rural_probs) / len(rural_probs)
    # Tertiary academic centers have significantly higher baseline acuity and risk
    assert mean_tertiary > mean_rural + 0.05
