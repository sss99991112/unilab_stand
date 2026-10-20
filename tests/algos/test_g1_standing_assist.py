from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.algos.torch.distill.standing_assist import (
    StandingAssistConfig,
    StandingAssistController,
)
from unilab.envs.locomotion.g1.recovery import RecoveryConfig
from unilab.envs.locomotion.g1.recovery_combined import RecoveryRoutingConfig, update_recovery_route


def controller(enabled=True):
    env = SimpleNamespace(
        num_envs=1,
        cfg=SimpleNamespace(
            ctrl_dt=0.02,
            recovery=RecoveryConfig(),
            recovery_routing=RecoveryRoutingConfig(),
            commands=SimpleNamespace(default_height=0.754),
        ),
    )
    env.default_angles = np.full(29, 0.1, np.float32)
    actuation = dict(
        kp=np.ones(29),
        kd=np.zeros(29),
        joint_ranges=np.tile([-2.0, 2.0], (29, 1)),
        force_limits=np.tile([-100.0, 100.0], (29, 1)),
    )
    return StandingAssistController(
        env, None, StandingAssistConfig(enabled=enabled), actuation=actuation
    )


def state(height=0.754, gyro=0.0, feet=(True, True)):
    return dict(
        height=np.float32(height),
        upvector=np.array([0, 0, 1], np.float32),
        linvel=np.zeros(3, np.float32),
        gyro=np.array([0, 0, gyro], np.float32),
        feet=np.asarray(feet),
        q=np.zeros(29, np.float32),
        dq=np.zeros(29, np.float32),
    )


def start(c):
    c._route(0, state(height=0.3))
    for frame in range(1, 11):
        c._route(frame, state())


def test_default_off_reproduces_original_route_including_duplicate_reads():
    c, original = controller(False), {}
    for frame in range(230):
        s = state(height=0.3 if frame in (0, 190) else 0.754, gyro=0.31 if frame == 30 else 0.0)
        update_recovery_route(
            original,
            height=np.array([s["height"]]),
            gravity=s["upvector"][None],
            linvel=s["linvel"][None],
            gyro=s["gyro"][None],
            feet=s["feet"][None],
            steps=np.array([frame]),
            recovery=c.env.cfg.recovery,
            routing=c.env.cfg.recovery_routing,
            ctrl_dt=0.02,
        )
        c._route(frame, s)
        before = {k: v.copy() for k, v in c.info.items()}
        assert c._route(frame, s) is False
        for key in original:
            np.testing.assert_array_equal(c.info[key], original[key])
        for key in before:
            np.testing.assert_array_equal(c.info[key], before[key])
        assert not c.attempted and not c.assisting


def test_entry_blend_and_new_full_window_are_separate():
    c = controller()
    start(c)
    assert c.assisting and c.attempted and c.alpha == 0.05
    assert c.info["recovery_route_stable_steps"][0] == 0
    for frame in range(11, 30):
        c._route(frame, state())
        assert c.info["recovery_route_stable_steps"][0] == 0
        assert c.info["recovery_active"][0]
        assert c.alpha == min((frame - 9) / 20, 1)
    for frame in range(30, 79):
        c._route(frame, state())
        assert c.info["recovery_active"][0]
    c._route(79, state())
    assert not c.info["recovery_active"][0]
    assert c.event == "release"
    assert c.info["recovery_settle_remaining"][0] == 100
    for frame in range(80, 179):
        c._route(frame, state())
        assert c.info["recovery_settle_remaining"][0] > 0
    c._route(179, state())
    assert c.info["recovery_settle_remaining"][0] == 0


@pytest.mark.parametrize("bad", [state(gyro=0.31), state(feet=(True, False)), state(height=0.64)])
def test_any_original_condition_breaks_formal_window(bad):
    c = controller()
    start(c)
    for frame in range(11, 50):
        c._route(frame, state())
    assert c.info["recovery_route_stable_steps"][0] == 20
    c._route(50, bad)
    assert c.info["recovery_route_stable_steps"][0] == 0
    assert c.info["recovery_active"][0]


def test_timeout_once_then_normal_recovery_still_works_and_reset_clears_attempt():
    c = controller()
    start(c)
    for frame in range(11, 110):
        c._route(frame, state(gyro=0.31))
    c._route(110, state())
    assert c.event == "fallback_timeout" and not c.assisting and c.attempted
    assert c.info["recovery_route_stable_steps"][0] == 0
    for frame in range(111, 160):
        c._route(frame, state())
        assert c.info["recovery_active"][0] and not c.assisting
    c._route(160, state())
    assert not c.info["recovery_active"][0]
    c.reset()
    start(c)
    assert c.assisting


def test_refall_clears_stability_and_phase_without_second_assistance():
    c = controller()
    start(c)
    c.info["gait_phase"][:] = 1.0
    c._route(11, state(height=0.3))
    assert c.event == "fallback_refall"
    assert not c.assisting and c.info["recovery_active"][0]
    assert c.info["recovery_route_stable_steps"][0] == 0
    np.testing.assert_array_equal(c.info["gait_phase"], [[0, 0]])
    for frame in range(12, 25):
        c._route(frame, state())
        assert not c.assisting


def test_invalid_frame_and_configuration_are_rejected():
    c = controller()
    with pytest.raises(ValueError, match="contiguous"):
        c._route(2, state())
    with pytest.raises(TypeError):
        StandingAssistConfig(enabled="false")
    with pytest.raises(ValueError, match="actuation"):
        StandingAssistController(c.env, None, StandingAssistConfig(enabled=True))


@pytest.mark.parametrize("reject", [False, "joint", "force"])
def test_blended_action_history_is_actual_executed_action_and_phase_is_role_specific(reject):
    c = controller()
    start(c)
    seen = []

    class Expert(torch.nn.Module):
        def __init__(self, action):
            super().__init__()
            self.action = action

        def forward(self, obs):
            seen.append(obs.numpy().copy())
            return torch.full((1, 29), self.action)

    c.student = SimpleNamespace(experts=[Expert(0.0), Expert(0.8), Expert(0.2)])
    c.info["gait_phase"][:] = 0.188
    c.info["current_actions"][:] = 0.37

    def obs(info, *args):
        info["recovery_effective_commands"] = np.zeros((1, 3), np.float32)
        value = np.zeros((1, 99), np.float32)
        value[:, 64:93] = info["current_actions"]
        value[:, 97:99] = info["gait_phase"]
        return {"obs": value}

    def apply(action, sim_state):
        sim_state.info["current_actions"] = action.copy()
        return action + np.float32(0.1)

    c.env._compute_obs, c.env.apply_action = obs, apply
    if reject == "joint":
        c.actuation["joint_ranges"][:] = [-0.3, 0.32]
    elif reject == "force":
        c.actuation["force_limits"][:] = [-0.3, 0.32]
    target, extra = c.decide(11, state())
    np.testing.assert_allclose(extra["action"], 0.2 if reject else 0.26, atol=1e-7)
    np.testing.assert_allclose(target, 0.3 if reject else 0.36, atol=1e-7)
    assert extra["load_rejected"] == bool(reject)
    if reject:
        assert c.event == "fallback_load" and c.attempted and not c.assisting
        assert c.info["recovery_route_stable_steps"][0] == 0
    np.testing.assert_array_equal(seen[0][:, 64:93], np.full((1, 29), 0.37, np.float32))
    np.testing.assert_array_equal(seen[1][:, 64:93], seen[0][:, 64:93])
    np.testing.assert_array_equal(seen[0][:, 97:99], [[0, 0]])
    np.testing.assert_array_equal(seen[1][:, 97:99], np.full((1, 2), 0.188, np.float32))
    np.testing.assert_array_equal(c.info["current_actions"][0], extra["action"])
    with pytest.raises(ValueError):
        c.decide(11, state())
