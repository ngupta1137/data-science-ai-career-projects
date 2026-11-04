import numpy as np
import pytest

from demand_forecast import config, data


def test_valid_frame_passes(synthetic_frame):
    data.validate_frame(synthetic_frame)


def test_missing_column_fails(synthetic_frame):
    with pytest.raises(data.DataValidationError, match="missing columns"):
        data.validate_frame(synthetic_frame.drop(columns=["cnt"]))


def test_missing_values_fail(synthetic_frame):
    synthetic_frame.loc[3, "cnt"] = np.nan
    with pytest.raises(data.DataValidationError, match="cnt"):
        data.validate_frame(synthetic_frame)


def test_a_gap_in_dates_fails(synthetic_frame):
    gapped = synthetic_frame.drop(index=10).reset_index(drop=True)
    with pytest.raises(data.DataValidationError, match="consecutive"):
        data.validate_frame(gapped)


def test_unordered_dates_fail(synthetic_frame):
    shuffled = synthetic_frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    with pytest.raises(data.DataValidationError, match="consecutive"):
        data.validate_frame(shuffled)


def test_broken_component_identity_fails(synthetic_frame):
    synthetic_frame.loc[0, "casual"] += 5
    with pytest.raises(data.DataValidationError, match="casual"):
        data.validate_frame(synthetic_frame)


def test_negative_counts_fail(synthetic_frame):
    synthetic_frame.loc[0, "cnt"] = -1
    with pytest.raises(data.DataValidationError, match="negative"):
        data.validate_frame(synthetic_frame)


def test_row_count_is_enforced_when_requested(synthetic_frame):
    with pytest.raises(data.DataValidationError, match="expected"):
        data.validate_frame(synthetic_frame.iloc[:500], expected_rows=config.EXPECTED_ROWS)


def test_checksum_mismatch_is_detected(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    checks = tmp_path / "CHECKSUMS.txt"
    digest = data.write_checksum(csv, checks)
    assert data.verify_checksum(csv, checks) == digest
    csv.write_text("a,b\n1,3\n", encoding="utf-8")
    with pytest.raises(data.DataValidationError, match="checksum mismatch"):
        data.verify_checksum(csv, checks)
