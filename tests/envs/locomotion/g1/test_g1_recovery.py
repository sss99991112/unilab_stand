from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dr import ResetPlan
from unilab.envs.locomotion.g1.joystick import G1WalkDomainRandomizationProvider
from unilab.envs.locomotion.g1.recovery import (
    G1RecoveryEnv,
    G1RecoveryResetProvider,
    RecoveryConfig,
    recovery_reward_terms,
    recovery_stable_mask,
    update_recovery_stability,
)
from unilab.training.backend_adapter import BackendAdapter

ROOT = Path(__file__).resolve().parents[4]


def _info(count=1):
    return {
        "steps": np.zeros(count, dtype=np.uint32),
        "current_actions": np.zeros((count, 29)),
        "last_actions": np.zeros((count, 29)),
        "recovery_stable_steps": np.zeros(count, dtype=np.int32),
        "recovery_last_step": np.full(count, -1, dtype=np.int64),
        "recovery_succeeded": np.zeros(count, dtype=bool),
        "recovery_frame": np.zeros(count, dtype=np.int64),
        "recovery_assistance_force_z": np.zeros(count, dtype=np.float64),
    }


def test_recovery_owner_composes_and_cli_routes():
    from unilab.cli import build_route

    registry.ensure_registries()
    route = build_route("sac", "g1_recovery", "mujoco")
    assert route.owner_task == "sac/g1_recovery/mujoco.yaml"
    with initialize_config_dir(config_dir=str(ROOT / "conf/offpolicy"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["algo=sac", "task=sac/g1_recovery/mujoco"])
    assert cfg.training.task_name == "G1Recovery"
    assert registry.contains("G1Recovery")
    overrides = BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override()
    assert set(overrides["reward_config"]["scales"]) == {
        "recovery_progress",
        "recovery_upright",
        "recovery_stability",
        "recovery_stand_height",
        "recovery_stand_motion",
        "recovery_action_rate",
        "recovery_joint_speed",
    }
    from unilab.envs.locomotion.g1.recovery import G1RecoveryCfg

    resolved = G1RecoveryCfg()
    registry.apply_cfg_overrides(resolved, overrides)
    resolved.recovery.validate(resolved.commands.default_height, resolved.ctrl_dt)
    assert resolved.reward_config.base_height_target == 0.754
    assert overrides["commands"]["observe_height_command"]
    assert not overrides["curriculum"]["enabled"]


def test_supine_reset_preserves_standing_reference_and_only_requested_rows(monkeypatch):
    reference = np.zeros(36)
    reference[2] = 0.754
    reference[3] = 1
    reference[7:] = np.linspace(-0.1, 0.2, 29)
    original = reference.copy()
    ids = np.asarray([1, 3], dtype=np.int32)

    def parent_reset(self, env, env_ids):
        return ResetPlan(env_ids, np.tile(reference, (2, 1)), np.ones((2, 35)), {})

    monkeypatch.setattr(G1WalkDomainRandomizationProvider, "build_reset_plan", parent_reset)
    starts = []
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            recovery=RecoveryConfig(reset_joint_noise=0),
            commands=SimpleNamespace(default_height=0.754),
        ),
        _num_action=29,
        _init_qpos=reference,
        _recovery_joint_limits=np.tile([-3, 3], (29, 1)),
        _spawn=SimpleNamespace(
            record_episode_start=lambda rows, pos: starts.append((rows.copy(), pos.copy()))
        ),
    )
    plan = G1RecoveryResetProvider().build_reset_plan(env, ids)
    np.testing.assert_array_equal(reference, original)
    np.testing.assert_array_equal(plan.env_ids, ids)
    np.testing.assert_allclose(plan.qpos[:, 7:], np.tile(original[7:], (2, 1)))
    np.testing.assert_allclose(plan.qpos[:, 2], 0.30)
    np.testing.assert_allclose(np.linalg.norm(plan.qpos[:, 3:7], axis=1), 1)
    # A -90 degree rotation about Y maps the robot's forward +X to world +Z.
    q = plan.qpos[:, 3:7]
    np.testing.assert_allclose(2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2]), 1)
    assert not np.any(plan.qvel)
    assert plan.info_updates["height_commands"].shape == (2, 1)
    assert not np.any(plan.info_updates["commands"])
    assert len(starts) == 1


def test_progress_reward_distinguishes_upright_from_upside_down():
    gravity = np.asarray([[0, 0, 1], [0, 0, -1], [1, 0, 0]], dtype=float)
    terms = recovery_reward_terms(
        np.full(3, 0.754),
        gravity,
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        np.zeros((3, 29)),
        np.zeros((3, 29)),
        np.zeros((3, 29)),
        np.zeros(3, dtype=bool),
        RecoveryConfig(),
        0.754,
    )
    assert (
        terms["recovery_progress"][0]
        > terms["recovery_progress"][2]
        > terms["recovery_progress"][1]
    )
    np.testing.assert_array_equal(terms["recovery_stand_height"] > 0, [True, False, False])


def test_success_requires_both_feet_orientation_and_quiet_motion():
    count = 6
    height = np.full(count, 0.754)
    gravity = np.tile([0, 0, 1.0], (count, 1))
    linvel, gyro = np.zeros((count, 3)), np.zeros((count, 3))
    feet = np.ones((count, 2), dtype=bool)
    height[1] = 0.3
    gravity[2, 2] = -1
    linvel[3, 0] = 1
    gyro[4, 0] = 1
    feet[5, 0] = False
    np.testing.assert_array_equal(
        recovery_stable_mask(height, gravity, linvel, gyro, feet, RecoveryConfig()),
        [True, False, False, False, False, False],
    )


def test_stability_counts_physics_transitions_not_refresh_calls():
    info = _info(2)
    update_recovery_stability(info, np.asarray([True, True]), info["steps"])
    update_recovery_stability(info, np.asarray([True, True]), info["steps"])
    np.testing.assert_array_equal(info["recovery_stable_steps"], [1, 1])
    info["steps"] += 1
    update_recovery_stability(info, np.asarray([True, False]), info["steps"])
    np.testing.assert_array_equal(info["recovery_stable_steps"], [2, 0])


def _fake_env(height, upright):
    env = object.__new__(G1RecoveryEnv)
    env._num_envs = 1
    env._cfg = SimpleNamespace(
        recovery=RecoveryConfig(stable_seconds=0.04),
        ctrl_dt=0.02,
        commands=SimpleNamespace(default_height=0.754),
        control_config=SimpleNamespace(action_scale=1.0, simulate_action_latency=False),
    )
    env._reward_cfg = SimpleNamespace(scales={"recovery_progress": 5.0})
    env._assistance_force_z = 0.0
    env._assistance_stage = 2
    env.default_angles = np.full(29, 0.2)
    env.get_local_linvel = lambda: np.zeros((1, 3))
    env.get_gyro = lambda: np.zeros((1, 3))
    env.get_dof_pos = lambda: np.zeros((1, 29))
    env.get_dof_vel = lambda: np.zeros((1, 29))
    env._terrain_relative_base_height = lambda: np.asarray([height])
    env._compute_obs = lambda *args: {"obs": np.zeros((1, 99)), "critic": np.zeros((1, 102))}
    env._backend = SimpleNamespace(
        get_sensor_data=lambda name: (
            np.asarray([[0, 0, upright]]) if name == "up" else np.ones((1, 1))
        )
    )
    env._cfg.sensor = SimpleNamespace(upvector="up")
    return env


def test_fallen_state_is_not_terminated_and_action_keeps_reference_semantics():
    env = _fake_env(0.15, -1)
    info = _info()
    state = NpEnvState(
        {"obs": np.zeros((1, 99)), "critic": np.zeros((1, 102))},
        np.zeros(1),
        np.zeros(1, dtype=bool),
        np.zeros(1, dtype=bool),
        info,
    )
    action = np.full((1, 29), 0.1)
    np.testing.assert_allclose(env.apply_action(action, state), 0.3)
    updated = env.update_state(state)
    assert not updated.terminated[0]
    assert np.all(np.isfinite(updated.reward))
    assert not info["recovery_succeeded"][0]
    action.fill(9)
    np.testing.assert_allclose(info["current_actions"], 0.1)


def test_success_is_latched_only_after_continuous_stability():
    env = _fake_env(0.754, 1)
    info = _info()
    state = NpEnvState({}, np.zeros(1), np.zeros(1, dtype=bool), np.zeros(1, dtype=bool), info)
    info["recovery_last_step"].fill(0)
    env.update_state(state)  # Refresh before physics must not count.
    assert info["recovery_stable_steps"][0] == 0
    env.apply_action(np.zeros((1, 29)), state)
    env.update_state(state)
    info["steps"] += 1  # NpEnv increments this after update_state.
    env.update_state(state)  # Refresh after step must not count twice.
    assert info["recovery_stable_steps"][0] == 1
    assert not info["recovery_succeeded"][0]
    env.apply_action(np.zeros((1, 29)), state)
    env.update_state(state)
    assert info["recovery_succeeded"][0]


@pytest.mark.parametrize(
    "changes",
    [
        {"stable_seconds": 0},
        {"reset_joint_noise": -1},
        {"stable_tilt_deg": 180},
        {"standing_height": 0.8},
        {"max_joint_speed": float("nan")},
    ],
)
def test_invalid_recovery_parameters_fail_closed(changes):
    with pytest.raises(ValueError):
        RecoveryConfig(**changes).validate(0.754, 0.02)
