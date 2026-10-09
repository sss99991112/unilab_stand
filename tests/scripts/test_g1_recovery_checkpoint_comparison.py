"""Comparison accounting must distinguish transient success from ending stable."""

import pytest
from scripts.deploy.compare_unilab_g1_recovery_checkpoints import summarize


def _episode(success, final_stable, longest, first=None):
    return {
        "full_episode": True,
        "assistance_fraction": 0.0,
        "assisted_steps": 0,
        "success": success,
        "stable_at_episode_end": final_stable,
        "longest_stable_seconds": longest,
        "first_success_step": first,
        "ctrl_dt_seconds": 0.02,
        "max_base_height_m": 0.7,
        "episode_reward": 12.0,
    }


def test_comparison_distinguishes_once_successful_from_stable_at_end():
    rows = [
        _episode(True, False, 1.2, 100),
        _episode(True, True, 3.2, 50),
        _episode(False, False, 0.4),
    ]
    stats = summarize(rows)
    assert stats["successes"] == 2
    assert stats["success_rate"] == pytest.approx(2 / 3)
    assert stats["successful_and_stable_at_end"] == 1
    assert stats["mean_longest_stable_seconds"] == pytest.approx(1.6)
    assert stats["mean_success_time_seconds"] == pytest.approx(1.5)


def test_no_success_does_not_invent_a_success_time():
    assert summarize([_episode(False, False, 0.0)])["mean_success_time_seconds"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [("full_episode", False), ("assistance_fraction", 0.4), ("assisted_steps", 1)],
)
def test_comparison_rejects_partial_or_assisted_episodes(field, value):
    row = _episode(False, False, 0.0)
    row[field] = value
    with pytest.raises(ValueError):
        summarize([row])
