import numpy as np
import pytest

from text_classification import config, data, splits


def test_fit_val_split_is_disjoint_and_complete():
    labels = np.array(["a", "b"] * 50)
    fit, val = splits.fit_val_split(labels, 0.2, seed=1)
    splits.assert_disjoint(fit, val)
    assert len(fit) + len(val) == 100 and len(val) == 20


def test_fit_val_split_is_stratified():
    labels = np.array(["a"] * 80 + ["b"] * 20)
    _, val = splits.fit_val_split(labels, 0.25, seed=2)
    assert (labels[val] == "b").sum() == 5


def test_fit_val_split_is_deterministic():
    labels = np.array(["a", "b", "c", "d"] * 25)
    a = splits.fit_val_split(labels, 0.2, seed=3)
    b = splits.fit_val_split(labels, 0.2, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_assert_disjoint_detects_overlap():
    with pytest.raises(AssertionError):
        splits.assert_disjoint([1, 2, 3], [3, 4])


def test_valid_frame_passes_and_allows_empty_text(synthetic_frame):
    assert (synthetic_frame["text"] == "").any()
    data.validate_frame(synthetic_frame)


def test_missing_column_fails(synthetic_frame):
    with pytest.raises(data.DataValidationError, match="missing columns"):
        data.validate_frame(synthetic_frame.drop(columns=["label"]))


def test_missing_label_fails(synthetic_frame):
    synthetic_frame.loc[0, "label"] = ""
    with pytest.raises(data.DataValidationError, match="label"):
        data.validate_frame(synthetic_frame)


def test_bad_split_values_fail(synthetic_frame):
    synthetic_frame.loc[0, "split"] = "validation"
    with pytest.raises(data.DataValidationError, match="split"):
        data.validate_frame(synthetic_frame)


def test_a_class_missing_from_test_fails(synthetic_frame):
    mask = (synthetic_frame["label"] == "class_0") & (synthetic_frame["split"] == "test")
    with pytest.raises(data.DataValidationError, match="every class"):
        data.validate_frame(synthetic_frame[~mask])


def test_expected_counts_are_enforced(synthetic_frame):
    with pytest.raises(data.DataValidationError, match="expected"):
        data.validate_frame(synthetic_frame, expected_rows=config.EXPECTED_ROWS)
    with pytest.raises(data.DataValidationError, match="classes"):
        data.validate_frame(synthetic_frame, expected_classes=config.EXPECTED_CLASSES)


def test_checksum_mismatch_is_detected(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    checks = tmp_path / "CHECKSUMS.txt"
    digest = data.write_checksum(csv, checks)
    assert data.verify_checksum(csv, checks) == digest
    csv.write_text("a,b\n1,3\n", encoding="utf-8")
    with pytest.raises(data.DataValidationError, match="checksum mismatch"):
        data.verify_checksum(csv, checks)


def test_load_keeps_empty_text_as_empty_string(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("text,text_raw,label,split\n,hdr,a,train\n", encoding="utf-8")
    assert data.load(path).loc[0, "text"] == ""
