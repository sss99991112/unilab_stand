"""Formal playback owns one action commit, previews and resets own no time."""

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.distill.moe_student import MoEStudentPolicy
from unilab.algos.torch.distill.recovery_integration import RECOVERY_ROUTING_CONTRACT
from unilab.training import BackendAdapter, create_env, ensure_registries
from unilab.training.rsl_rl import RslRlVecEnvWrapper
from unilab.visualization.interactive_playback import (
    PlaybackControls,
    RslRlPlaybackConfig,
    RslRlPlaybackSession,
    create_distill_playback_session,
)
from unilab.visualization.standing_assist_playback import StandingAssistPlaybackSession

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def context():
    ensure_registries()
    cfg = OmegaConf.load(ROOT / "conf/distill/config.yaml")
    del cfg["defaults"]
    cfg = OmegaConf.merge(
        cfg, OmegaConf.load(ROOT / "conf/distill/task/g1_recovery_combined/mujoco.yaml")
    )
    cfg.env.noise_config.level = 0.0
    env = create_env(
        cfg,
        num_envs=1,
        sim_backend="mujoco",
        task_name="G1RecoveryCombined",
        env_cfg_override=BackendAdapter(
            cfg, root_dir=ROOT, algo_name="distill"
        ).build_task_env_cfg_override(),
    )
    model = MoEStudentPolicy(
        obs_dim=99,
        action_dim=29,
        num_experts=3,
        expert_hidden_dims=(8,),
        router_hidden_dims=(),
        routing_mode="hard",
    ).eval()
    # Mild finite actions make this a lifecycle test, not a policy-quality test.
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    try:
        yield cfg, env, model
    finally:
        if env._backend._pool is not None:
            env._backend._pool.close()
        env.close()


def session_for(env, model):
    wrapped = RslRlVecEnvWrapper(env, device="cpu", policy_obs_mode="actor")
    session = StandingAssistPlaybackSession(
        env=env,
        wrapped_env=wrapped,
        device="cpu",
        action_mode="policy",
        policy=lambda obs: None,
        num_envs=1,
        student=model,
    )
    session.reset()
    return session


def test_formal_step_applies_once_and_advances_actual_history_and_phase_once(context):
    _, env, model = context
    session = session_for(env, model)
    phase = env.state.info["gait_phase"].copy()
    with patch.object(env, "apply_action", wraps=env.apply_action) as apply:
        session.step_once()
    assert apply.call_count == 1
    np.testing.assert_array_equal(env.state.info["current_actions"], session.actions.numpy())
    np.testing.assert_array_equal(env.state.info["executed_actions"], session.actions.numpy())
    np.testing.assert_array_equal(
        env.state.info["gait_phase"], (phase + env._gait_phase_delta) % (2 * np.pi)
    )
    assert session.step_count == 1 and session.controller.last_frame == 0


def test_pause_refresh_and_noisy_dryrun_do_not_consume_route_phase_history_or_rng(context):
    _, env, model = context
    session = session_for(env, model)
    env.cfg.noise_config.level = 1.0
    before = copy.deepcopy(env.state.info)
    rng = np.random.get_state()
    with patch.object(env, "apply_action", wraps=env.apply_action) as apply:
        assert not session.advance(PlaybackControls(paused=True))
        a, b = session._build_actions(), session._build_actions()
        assert apply.call_count == 0
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    after_rng = np.random.get_state()
    assert rng[0] == after_rng[0] and rng[2:] == after_rng[2:]
    np.testing.assert_array_equal(rng[1], after_rng[1])
    for key in (
        "gait_phase",
        "current_actions",
        "recovery_route_last_step",
        "recovery_route_stable_steps",
    ):
        np.testing.assert_array_equal(env.state.info[key], before[key])
    assert session.controller.last_frame == -1 and not session.controller.attempted
    env.refresh_state()
    session.refresh_observation()
    assert session.controller.last_frame == -1 and session.step_count == 0
    controls = PlaybackControls(paused=True)
    controls.request_single_step()
    assert session.advance(controls)
    assert not session.advance(controls)
    assert session.step_count == 1


def test_reset_and_supine_reset_clear_attempt_and_capture_live_randomized_gains(context):
    _, env, model = context
    session = session_for(env, model)
    session.step_once()
    session.controller.attempted = True
    session.controller.assisting = True
    session.controller.assist_age = 7
    session.reset_recovery_pose(supine=True)
    assert session.controller.last_frame == -1 and session.step_count == 0
    assert not session.controller.attempted and not session.controller.assisting
    assert session.last_decision is None
    np.testing.assert_array_equal(
        session.controller.info["current_actions"], np.zeros((1, 29), np.float32)
    )
    actual = env.get_playback_model(0)
    assert not np.array_equal(
        actual.actuator_gainprm[:, 0], env.get_playback_model().actuator_gainprm[:, 0]
    )
    np.testing.assert_array_equal(session.controller.actuation["kp"], actual.actuator_gainprm[:, 0])
    session.step_once()
    assert session.last_decision["expert"] == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_formal_factory_single_flag_selects_whole_owner_without_altering_off(
    context, tmp_path, enabled
):
    cfg, env, model = context
    cfg.interactive.standing_assist = enabled
    path = tmp_path / "student.pt"
    torch.save({}, path)
    runtime = dict(
        student_model_type="moe",
        recovery_integration=True,
        role_expert_targets={"walk": 0, "stand_height": 1, "recovery": 2},
        expert_behavior_loss_source="role",
        recovery_routing_contract=RECOVERY_ROUTING_CONTRACT,
    )
    loaded = SimpleNamespace(
        policy=model, distill_runtime_cfg=runtime, obs_dim=99, action_dim=29, agent_steps=0
    )
    session, _, _ = create_distill_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="g1_recovery_combined",
            load_run="-1",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="distill",
            log_root=None,
        ),
        cfg=cfg,
        root_dir=ROOT,
        device="cpu",
        deps={
            "resolve_checkpoint": lambda *args: path,
            "create_env": lambda *args, **kwargs: env,
            "wrapper_cls": RslRlVecEnvWrapper,
            "load_student_policy": lambda *args, **kwargs: loaded,
        },
        log=lambda text: None,
    )
    assert type(session) is (StandingAssistPlaybackSession if enabled else RslRlPlaybackSession)
    assert env._playback_recovery_route_deferred == enabled
    session.reset()
    session.step_once()


@pytest.mark.parametrize("field,value", [("action_mode", "zero"), ("num_envs", 2)])
def test_invalid_flag_combinations_fail_before_environment_creation(context, field, value):
    cfg, _, _ = context
    cfg.interactive.standing_assist = True
    values = dict(
        task="g1_recovery_combined",
        load_run="-1",
        checkpoint=None,
        action_mode="policy",
        policy_obs_mode="actor",
        algo_log_name="distill",
        log_root=None,
    )
    values[field] = value
    with pytest.raises(ValueError, match="CPU policy"):
        create_distill_playback_session(
            playback_cfg=RslRlPlaybackConfig(**values),
            cfg=cfg,
            root_dir=ROOT,
            device="cpu",
            deps={},
        )


def test_formal_crossfade_does_not_count_toward_full_window_and_refresh_does_not_reset_phase(
    context,
):
    _, env, model = context
    session = session_for(env, model)
    fixed = env.recovery_playback_state()
    fixed.update(
        height=np.float32(0.754),
        upvector=np.array([0, 0, 1], np.float32),
        linvel=np.zeros(3, np.float32),
        gyro=np.zeros(3, np.float32),
        feet=np.ones(2, bool),
    )
    decisions = []
    for frame in range(80):
        measured = {**fixed, "height": np.float32(0.30 if frame == 0 else 0.754)}
        with patch.object(env, "recovery_playback_state", return_value=measured):
            session.step_once()
        decisions.append(session.last_decision)
        before = env.state.info["gait_phase"].copy()
        env.refresh_state()
        session.refresh_observation()
        np.testing.assert_array_equal(before, env.state.info["gait_phase"])
    assert decisions[10]["event"] == "assist_start"
    assert decisions[10]["alpha"] == 0.05 and decisions[29]["alpha"] == 1
    assert all(d["active"] and d["stable_count"] == 0 for d in decisions[10:30])
    assert decisions[78]["active"] and decisions[79]["event"] == "release"
    assert decisions[79]["settle"] == 100


@pytest.mark.parametrize("enabled", [False, True])
def test_keyboard_launcher_flag_is_opt_in_and_preserves_existing_entry(tmp_path, enabled):
    import os
    import subprocess

    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"dry-run-only")
    command = [
        "bash",
        str(ROOT / "play_g1_recovery_keyboard.sh"),
        "--checkpoint-path",
        str(checkpoint),
        "--dry-run",
    ]
    if enabled:
        command.append("--standing-assist")
    environment = dict(os.environ)
    environment.pop("UNILAB_LOG_FILE", None)
    result = subprocess.run(
        command, cwd=ROOT, env=environment, text=True, capture_output=True, check=True
    )
    assert ("interactive.standing_assist=true" in result.stdout) == enabled
    assert "task=g1_recovery_combined sim=mujoco algo=distill" in result.stdout
    assert "interactive.start_paused=true" in result.stdout
