import pandas as pd
import pytest

from llm_judge_eval import audit


def frame(rows):
    return pd.DataFrame(
        rows,
        columns=["pair_id", "order", "judge_prefers_first", "human_prefers_x", "len_x", "len_y"],
    )


HAND = frame(
    [
        # pair 1: judge says X in both orders, human says X
        (1, 0, 1, 1, 300, 100),
        (1, 1, 0, 1, 300, 100),
        # pair 2: judge picks the first slot both times (a split), human says Y
        (2, 0, 1, 0, 200, 400),
        (2, 1, 1, 0, 200, 400),
    ]
)


def test_basic_stats_hand_example():
    out = audit.basic_stats(audit._paired_frame(HAND))
    assert out["pairs"] == 2
    assert out["position_bias_p_first_minus_half"] == pytest.approx(0.25)  # picks: 1, 0, 1, 1
    assert out["swap_inconsistency_rate"] == 0.5
    assert out["agreement_single_order"] == pytest.approx(0.75)
    assert out["agreement_swap_ties_half"] == pytest.approx(0.75)  # (1 + 0.5) / 2
    assert out["agreement_on_decided_pairs"] == 1.0
    assert out["coverage_when_splits_go_to_humans"] == 0.5
    assert out["agreement_longer_answer"] == 1.0  # pair 1 longer is X (human X), pair 2 longer is Y


def test_rejects_missing_column():
    with pytest.raises(audit.AuditInputError):
        audit._paired_frame(HAND.drop(columns=["len_y"]))


def test_rejects_unpaired_calls():
    with pytest.raises(audit.AuditInputError):
        audit._paired_frame(HAND.iloc[:3])


def test_rejects_bad_values():
    bad = HAND.copy()
    bad.loc[0, "order"] = 2
    with pytest.raises(audit.AuditInputError):
        audit._paired_frame(bad)


def test_rejects_human_label_that_differs_within_a_pair():
    bad = HAND.copy()
    bad.loc[1, "human_prefers_x"] = 0
    with pytest.raises(audit.AuditInputError):
        audit._paired_frame(bad)


def test_too_few_pairs():
    with pytest.raises(audit.AuditInputError):
        audit.audit(HAND)


def test_example_file_round_trip(tmp_path):
    path = tmp_path / "example.csv"
    audit.write_example_csv(path, n=1500)
    out = audit.audit(pd.read_csv(path), boots=30)
    reg = out["regression_with_human_label_as_quality"]
    assert out["pairs"] == 1500
    assert reg["position"]["value"] > 0.1  # the simulated judge likes the first slot
    assert reg["length_slope"]["value"] > 0.2  # and longer answers
    assert 0.6 < out["agreement_single_order"] < 0.9
