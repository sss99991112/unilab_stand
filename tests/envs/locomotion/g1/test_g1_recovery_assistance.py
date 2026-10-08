from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.g1.recovery import G1RecoveryEnv
from unilab.envs.locomotion.g1.recovery_assistance import (
    RecoveryAssistanceCfg,
    assistance_force_rows,
)


def test_force_gating_respects_initial_fall_and_upright_orientation():
    force = assistance_force_rows(
        np.asarray([0.0, 0.9, 1.0, 0.9, 0.9]),
        np.asarray([99, 0, 30, 29, 30]),
        force_z=100.0,
        start_steps=30,
        min_upright_cos=0.8,
    )
    assert force.shape == (5, 1, 3)
    np.testing.assert_array_equal(force[:, 0, :2], 0)
    np.testing.assert_array_equal(force[:, 0, 2], [0, 0, 100, 0, 100])


def test_schedule_decreases_to_zero_and_evaluation_ignores_training_stage():
    for stage, expected in enumerate([0.4, 0.2, 0]):
        cfg = RecoveryAssistanceCfg(stage=stage)
        cfg.validate()
        assert cfg.effective_fraction == expected
        cfg.evaluation = True
        assert cfg.effective_fraction == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"force_fractions": [0.2, 0.4, 0]},
        {"force_fractions": [0.4, 0.2]},
        {"force_fractions": [1.0, 0]},
        {"force_fractions": [float("nan"), 0]},
        {"stage": -1},
        {"stage": True},
        {"start_after_seconds": -1},
        {"min_upright_cos": 1.5},
        {"evaluation": "false"},
    ],
)
def test_invalid_schedule_is_rejected(changes):
    with pytest.raises(ValueError):
        RecoveryAssistanceCfg(**changes).validate()


def _state(count):
    return NpEnvState(
        {},
        np.zeros(count),
        np.zeros(count, dtype=bool),
        np.zeros(count, dtype=bool),
        {"steps": np.full(count, 30), "recovery_assistance_force_z": np.full(count, -1.0)},
    )


def test_zero_assistance_does_not_call_backend_or_leave_diagnostic_force():
    env = object.__new__(G1RecoveryEnv)
    env._assistance_force_z = 0
    env._backend = object()  # No sensor/force methods: the OFF path must not need them.
    state = _state(2)
    env._apply_recovery_assistance(state)
    np.testing.assert_array_equal(state.info["recovery_assistance_force_z"], 0)


def test_assistance_uses_public_backend_interface_and_masks_each_row():
    calls = []
    env = object.__new__(G1RecoveryEnv)
    env._assistance_force_z = 80
    env._assistance_start_steps = 30
    env._assistance_min_up = 0.8
    env._assistance_body_ids = np.asarray([7], dtype=np.int32)
    env._cfg = SimpleNamespace(sensor=SimpleNamespace(upvector="up"))
    env._backend = SimpleNamespace(
        get_sensor_data=lambda name: np.asarray([[0, 0, 1.0], [1.0, 0, 0]]),
        apply_body_force=lambda ids, force: calls.append((ids.copy(), force.copy())),
    )
    state = _state(2)
    env._apply_recovery_assistance(state)
    np.testing.assert_array_equal(calls[0][0], [7])
    np.testing.assert_array_equal(calls[0][1][:, 0, 2], [80, 0])
    np.testing.assert_array_equal(state.info["recovery_assistance_force_z"], [80, 0])
