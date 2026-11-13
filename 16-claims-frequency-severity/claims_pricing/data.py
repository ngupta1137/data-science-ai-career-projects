"""A simulated auto insurance portfolio with claims frequency and severity.

Each policy has an exposure duration in years and risk features (driver age, annual mileage,
vehicle age, vehicle value, urban indicator). Frequency follows a Poisson process scaled by
exposure. Severity follows a Gamma distribution for policies that experience a claim.
True frequency and severity parameters are known, so model estimates can be verified against
the ground truth.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CHECKSUM_FILE


class DataValidationError(ValueError):
    """Raised on a schema problem or a fingerprint mismatch."""


@dataclass
class Policies:
    driver_age: np.ndarray  # (n,) driver age in years (18 to 75)
    annual_mileage: np.ndarray  # (n,) annual mileage (4,000 to 25,000)
    vehicle_age: np.ndarray  # (n,) vehicle age in years (0 to 15)
    vehicle_value: np.ndarray  # (n,) vehicle value in dollars ($5k to $80k)
    urban: np.ndarray  # (n,) binary indicator (1 = urban, 0 = rural/suburban)
    exposure: np.ndarray  # (n,) policy duration in force in years (0.1 to 1.0)
    claim_count: np.ndarray  # (n,) realized number of claims (Poisson draws)
    claim_amount: np.ndarray  # (n,) realized total claim dollars (sum of Gamma draws)
    true_freq: np.ndarray  # (n,) true annualized claim frequency lambda_i
    true_sev: np.ndarray  # (n,) true average severity mu_i

    @property
    def true_pure_premium(self) -> np.ndarray:
        return self.true_freq * self.true_sev

    def __len__(self) -> int:
        return len(self.exposure)

    def features(self) -> np.ndarray:
        return np.column_stack(
            [
                (self.driver_age - 45.0) / 15.0,
                np.log(self.annual_mileage / 10000.0),
                self.vehicle_age / 5.0,
                np.log(self.vehicle_value / 20000.0),
                self.urban.astype(float),
            ]
        )


def generate(rng: np.random.Generator, n_policies: int) -> Policies:
    driver_age = rng.uniform(18.0, 75.0, n_policies)
    annual_mileage = rng.uniform(4000.0, 25000.0, n_policies)
    vehicle_age = rng.uniform(0.0, 15.0, n_policies)
    vehicle_value = np.exp(rng.normal(10.0, 0.5, n_policies))
    urban = rng.binomial(1, 0.40, n_policies)
    exposure = rng.uniform(0.10, 1.0, n_policies)

    # Frequency: younger drivers, higher mileage, and urban driving increase claim rate
    freq_log = (
        -2.20 - 0.020 * (driver_age - 35.0) + 0.60 * np.log(annual_mileage / 10000.0) + 0.50 * urban
    )
    true_freq = np.exp(freq_log)
    claim_count = rng.poisson(exposure * true_freq)

    # Severity: vehicle value increases claim size; older cars and urban speeds reduce it
    sev_log = 7.70 + 0.70 * np.log(vehicle_value / 20000.0) - 0.050 * vehicle_age - 0.40 * urban
    true_sev = np.exp(sev_log)

    claim_amount = np.zeros(n_policies, dtype=float)
    has_claim = claim_count > 0
    indices = np.where(has_claim)[0]
    for idx in indices:
        k = int(claim_count[idx])
        claim_amount[idx] = float(np.sum(rng.gamma(shape=1.5, scale=true_sev[idx] / 1.5, size=k)))

    return Policies(
        driver_age=driver_age,
        annual_mileage=annual_mileage,
        vehicle_age=vehicle_age,
        vehicle_value=vehicle_value,
        urban=urban,
        exposure=exposure,
        claim_count=claim_count,
        claim_amount=claim_amount,
        true_freq=true_freq,
        true_sev=true_sev,
    )


def validate(p: Policies) -> None:
    problems: list[str] = []
    n = len(p)
    for name, arr in [
        ("driver_age", p.driver_age),
        ("annual_mileage", p.annual_mileage),
        ("vehicle_age", p.vehicle_age),
        ("vehicle_value", p.vehicle_value),
        ("urban", p.urban),
        ("exposure", p.exposure),
        ("claim_count", p.claim_count),
        ("claim_amount", p.claim_amount),
        ("true_freq", p.true_freq),
        ("true_sev", p.true_sev),
    ]:
        if len(arr) != n:
            problems.append(f"{name} length mismatch")

    if (p.exposure <= 0).any() or (p.exposure > 1.0).any():
        problems.append("exposure out of bounds (0, 1]")
    if (p.claim_count < 0).any():
        problems.append("negative claim counts")
    if (p.claim_amount < 0).any():
        problems.append("negative claim amounts")
    if ((p.claim_count == 0) & (p.claim_amount > 0)).any():
        problems.append("positive loss with zero claims")
    if ((p.claim_count > 0) & (p.claim_amount <= 0)).any():
        problems.append("non-positive loss with positive claims")
    if (p.driver_age < 16).any() or (p.driver_age > 100).any():
        problems.append("driver age out of plausible range")

    if problems:
        raise DataValidationError("; ".join(problems))


def fingerprint(seed: int = 0) -> str:
    p = generate(np.random.default_rng(seed), 100)
    blob = np.concatenate(
        [
            np.round(p.driver_age, 4),
            np.round(p.annual_mileage, 2),
            np.round(p.vehicle_age, 4),
            np.round(p.vehicle_value, 2),
            p.urban.astype(float),
            np.round(p.exposure, 4),
            p.claim_count.astype(float),
            np.round(p.claim_amount, 2),
            np.round(p.true_freq, 6),
            np.round(p.true_sev, 2),
        ]
    )
    return hashlib.sha256(blob.astype(float).tobytes()).hexdigest()


def write_fingerprint(path: Path = CHECKSUM_FILE) -> str:
    digest = fingerprint()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{digest}  generator_probe(seed=0)\n", encoding="utf-8")
    return digest


def verify_fingerprint(path: Path = CHECKSUM_FILE) -> str:
    digest = fingerprint()
    if not path.exists():
        raise DataValidationError(f"Missing checksum file: {path}")
    stored = path.read_text(encoding="utf-8").split()[0].strip()
    if digest != stored:
        raise DataValidationError(f"Fingerprint mismatch: computed {digest}, stored {stored}")
    return digest
