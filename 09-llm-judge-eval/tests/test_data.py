import numpy as np
import pytest

from llm_judge_eval import config, data


def test_length_quality_correlation_matches_setting():
    rng = np.random.default_rng(1)
    q, length = data.draw_answers(rng, 80000, 0.0, 0.0, 0.3)
    assert np.corrcoef(q, length)[0, 1] == pytest.approx(0.3, abs=0.02)
    assert length.std() == pytest.approx(1.0, abs=0.02)


def test_length_offset_shifts_mean():
    rng = np.random.default_rng(1)
    _, length = data.draw_answers(rng, 50000, 0.0, 1.5, 0.3)
    assert length.mean() == pytest.approx(1.5, abs=0.03)


def test_consensus_is_majority():
    votes = np.array([[True, True, False], [False, False, True], [True, True, True]])
    assert data.consensus(votes).tolist() == [True, False, True]


def test_humans_have_no_slot_effect_by_construction():
    """Human votes depend only on dq, so relabelling X and Y flips the vote rate exactly."""
    rng = np.random.default_rng(2)
    pairs = data.make_pairs(rng, 40000)
    p_x = data.sigmoid(config.HUMAN_BETA * pairs.dq)
    p_y = data.sigmoid(config.HUMAN_BETA * -pairs.dq)
    assert np.allclose(p_x + p_y, 1.0)


def test_unbiased_judge_picks_first_half_the_time():
    rng = np.random.default_rng(3)
    pairs = data.make_pairs(rng, 60000)
    v0, v1 = data.judge_calls(rng, pairs, data.JudgeParams(pos=0.0, length=0.0))
    pick_first = (v0.mean() + (1 - v1.mean())) / 2
    assert pick_first == pytest.approx(0.5, abs=0.01)


def test_slot_biased_judge_prefers_first_slot():
    rng = np.random.default_rng(3)
    pairs = data.make_pairs(rng, 60000)
    v0, v1 = data.judge_calls(rng, pairs, data.JudgeParams(pos=0.8, length=0.0))
    assert (v0.mean() + (1 - v1.mean())) / 2 > 0.58


def test_length_biased_judge_prefers_longer_answer():
    rng = np.random.default_rng(3)
    pairs = data.make_pairs(rng, 60000, rho=0.0)
    v0, _ = data.judge_calls(rng, pairs, data.JudgeParams(beta=0.0, pos=0.0, length=1.0))
    longer = pairs.dl > 0
    assert v0[longer].mean() > 0.6 and v0[~longer].mean() < 0.4


def test_truth_follows_quality():
    rng = np.random.default_rng(4)
    pairs = data.make_pairs(rng, 100)
    assert (pairs.truth == (pairs.qx > pairs.qy)).all()


def test_splits_are_independent_and_deterministic():
    a, b = data.make_splits(1), data.make_splits(1)
    assert np.array_equal(a["test"].qx, b["test"].qx)
    assert not np.array_equal(a["dev"].qx[:100], a["test"].qx[:100])
    assert len(a["dev"]) == config.N_DEV and len(a["test"]) == config.N_TEST


def test_validate_rejects_nonfinite():
    rng = np.random.default_rng(0)
    pairs = data.make_pairs(rng, 10)
    pairs.qx[0] = np.nan
    with pytest.raises(data.DataValidationError):
        data.validate(pairs)


def test_fingerprint_roundtrip_and_mismatch(tmp_path):
    f = tmp_path / "sum.txt"
    digest = data.write_fingerprint(f)
    assert data.verify_fingerprint(f) == digest
    f.write_text("0" * 64 + "  generator_probe(seed=0)\n")
    with pytest.raises(data.DataValidationError):
        data.verify_fingerprint(f)


def test_committed_fingerprint_matches_generator():
    assert data.verify_fingerprint() == data.fingerprint()
