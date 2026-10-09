"""Reward-accounting checks for recovery diagnostics, not policy-quality tests."""

import pytest
from scripts.deploy.diagnose_unilab_g1_recovery_policy import accumulate_reward_terms


def test_reward_accounting_applies_dt_once_to_already_weighted_terms():
    totals = {"positive": 0.0, "penalty": 0.0}
    log = {"reward/positive": 5.0, "reward/penalty": -2.0}
    for _ in range(2):
        accumulate_reward_terms(totals, log, ctrl_dt=0.02, step_reward=0.06)
    assert totals == pytest.approx({"positive": 0.2, "penalty": -0.08})
    assert sum(totals.values()) == pytest.approx(0.12)


@pytest.mark.parametrize(
    ("log", "actual_reward", "exception"),
    [
        ({"reward/positive": 5.0}, 0.1, KeyError),
        ({"reward/positive": float("nan"), "reward/penalty": -2.0}, 0.06, FloatingPointError),
        ({"reward/positive": 5.0, "reward/penalty": -2.0}, 0.5, AssertionError),
    ],
)
def test_invalid_reward_accounting_fails_without_partial_accumulation(
    log, actual_reward, exception
):
    totals = {"positive": 0.5, "penalty": -0.2}
    before = totals.copy()
    with pytest.raises(exception):
        accumulate_reward_terms(totals, log, ctrl_dt=0.02, step_reward=actual_reward)
    assert totals == before
