import numpy as np
import pytest

from llm_judge_eval import data, judge, metrics


def test_swap_score_values():
    v0 = np.array([True, True, False, False])
    v1 = np.array([True, False, True, False])
    assert judge.swap_score(v0, v1).tolist() == [1, 0, 0, -1]


def test_tie_rate_hand_example():
    v0 = np.array([True, True, False, False])
    v1 = np.array([True, False, False, True])
    assert judge.tie_rate(v0, v1) == 0.5


def test_swap_coin_keeps_unanimous_verdicts(rng):
    v0 = np.array([True, False] * 50)
    dec = judge.decide_swap(rng, v0, v0.copy())
    assert (dec == v0).all()


def test_triage_marks_splits():
    v0 = np.array([True, True, False])
    v1 = np.array([True, False, False])
    verdict, decided = judge.decide_triage(v0, v1)
    assert decided.tolist() == [True, False, True]
    assert verdict[decided].tolist() == [True, False]


def test_swap_with_coin_equals_one_random_call_in_expectation():
    """The coin throws away the split signal, so the expected agreement is the single-call mean."""
    rng = np.random.default_rng(3)
    pairs = data.make_pairs(rng, 60000)
    cons = data.consensus(data.human_votes(rng, pairs))
    v0, v1 = data.judge_calls(rng, pairs, data.JudgeParams())
    swap = judge.decide_swap(rng, v0, v1)
    single_mean = ((v0 == cons).mean() + (v1 == cons).mean()) / 2
    assert metrics.agreement(swap, cons) == pytest.approx(single_mean, abs=0.01)


def test_triage_beats_coin_swap_on_decided_pairs():
    rng = np.random.default_rng(4)
    pairs = data.make_pairs(rng, 30000)
    cons = data.consensus(data.human_votes(rng, pairs))
    v0, v1 = data.judge_calls(rng, pairs, data.JudgeParams())
    verdict, decided = judge.decide_triage(v0, v1)
    assert (verdict[decided] == cons[decided]).mean() > (
        judge.decide_swap(rng, v0, v1) == cons
    ).mean() + 0.05
    assert 0.5 < decided.mean() < 0.95


def test_calibrator_is_odd_under_relabelling(rng):
    n = 3000
    dl = rng.standard_normal(n)
    v0 = rng.random(n) < 0.5
    v1 = rng.random(n) < 0.5
    label = rng.random(n) < 0.5
    model = judge.fit_calibrator(v0, v1, dl, label)
    a = judge.decide_calibrated(model, v0, v1, dl)
    b = judge.decide_calibrated(model, ~v1, ~v0, -dl)  # swap the names X and Y
    assert (a != b).all()


def test_calibrator_learns_to_discount_a_length_pull():
    """Humans ignore length. A judge that likes length splits toward the long answer, so the
    fitted length weight should push back (be negative) once the swap score is in the model."""
    rng = np.random.default_rng(5)
    params = data.JudgeParams(beta=1.5, pos=0.0, length=1.5)
    pairs = data.make_pairs(rng, 20000, rho=0.0)
    cons = data.consensus(data.human_votes(rng, pairs))
    v0, v1 = data.judge_calls(rng, pairs, params)
    model = judge.fit_calibrator(v0, v1, pairs.dl, cons)
    assert model.coef_[0][1] < -0.05
    assert model.coef_[0][0] > 0


def test_references():
    assert judge.decide_length_only(np.array([1.0, -1.0])).tolist() == [True, False]
    assert judge.decide_oracle(np.array([0.2, -0.1])).tolist() == [True, False]
