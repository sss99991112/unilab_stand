"""Physical boundary, pre-clipping PD and selective contact reward semantics."""

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from unilab.envs.locomotion.g1.recovery_constraints import (
    RecoveryConstraintsCfg,
    constraint_reward_terms,
)
from unilab.training import BackendAdapter, create_env, ensure_registries

ROOT = Path(__file__).resolve().parents[4]


def metrics():
    return dict(
        q=np.zeros((1, 3, 29)),
        dq=np.zeros((1, 3, 29)),
        raw_pd=np.zeros((1, 3, 29)),
        pair_normal_force=np.zeros((1, 3, 2)),
        joint_ranges=np.tile([-1.0, 1.0], (1, 29, 1)),
        force_ranges=np.tile([-5.0, 5.0], (1, 29, 1)),
        valid=np.ones(1, bool),
        replay_error=0.0,
    )


def test_off_never_reads_privileged_metrics():
    class Forbidden:
        def __getitem__(self, key):
            raise AssertionError("OFF touched metrics")

    terms, log = constraint_reward_terms(Forbidden(), RecoveryConstraintsCfg(), 2)
    assert not log and all(np.array_equal(t, np.zeros(2)) for t in terms.values())


def test_actual_boundary_and_outward_velocity_are_not_target_clipping():
    m = metrics()
    c = RecoveryConstraintsCfg(enabled=True)
    m["q"][:, :, 0] = 0.98
    m["dq"][:, :, 0] = 2
    a, _ = constraint_reward_terms(m, c, 1)
    m["dq"][:, :, 0] = -2
    b, _ = constraint_reward_terms(m, c, 1)
    assert a["recovery_joint_boundary"][0] > 0
    assert a["recovery_outward_speed"][0] > b["recovery_outward_speed"][0] == 0
    m["q"][:, :, 0] = 1.1
    d, log = constraint_reward_terms(m, c, 1)
    assert d["recovery_joint_boundary"][0] > a["recovery_joint_boundary"][0]
    assert log["constraint/max_joint_excess_rad"] == pytest.approx(0.1)


def test_raw_pd_excess_is_visible_before_force_clipping_and_all_three_substeps_count():
    m = metrics()
    m["raw_pd"][0, 0, 0] = 10
    terms, _ = constraint_reward_terms(m, RecoveryConstraintsCfg(enabled=True), 1)
    assert terms["recovery_pd_excess"][0] == pytest.approx(1 / 3)
    assert terms["recovery_joint_boundary"][0] == 0
    m["pair_normal_force"][0, 1, 0] = 100
    terms, _ = constraint_reward_terms(m, RecoveryConstraintsCfg(enabled=True), 1)
    assert terms["recovery_self_collision"][0] == pytest.approx(1 / 3)
    m["valid"][:] = False
    terms, _ = constraint_reward_terms(m, RecoveryConstraintsCfg(enabled=True), 1)
    assert all(t[0] == 0 for t in terms.values())


@pytest.mark.parametrize(
    "field,value", [("boundary_margin_rad", 0), ("outward_speed_scale", -1), ("version", "unknown")]
)
def test_invalid_constraint_config_fails(field, value):
    c = RecoveryConstraintsCfg()
    setattr(c, field, value)
    with pytest.raises(ValueError):
        c.validate()


def config(profile):
    with initialize_config_dir(config_dir=str(ROOT / "conf/offpolicy"), version_base="1.3"):
        return compose(config_name="config", overrides=[f"task=sac/g1_recovery/{profile}"])


def rollout(c, enabled, randomized=True, action=None):
    ensure_registries()
    np.random.seed(12)
    ov = BackendAdapter(c, root_dir=ROOT, algo_name="sac").build_task_env_cfg_override()
    if enabled is not None:
        ov["recovery_constraints"]["enabled"] = enabled
    ov["noise_config"] = {"level": 0.0}
    ov["domain_rand"] = {"randomize_kp": randomized, "randomize_kd": randomized}
    env = create_env(
        c, num_envs=2, sim_backend="mujoco", task_name="G1Recovery", env_cfg_override=ov
    )
    states, rewards, observations = [], [], []
    try:
        state = env.init_state()
        env.set_autoreset(False)
        for i in range(12):
            state = env.step(np.zeros((2, 29), np.float32) if action is None else action)
            states.append(env.get_physics_state_snapshot().copy())
            rewards.append(state.reward.copy())
            observations.append(state.obs["obs"].copy())
            if enabled:
                m = env._backend.get_recovery_load_metrics()
                assert m["q"].shape == (2, 3, 29) and m["replay_error"] == 0
                assert state.obs["obs"].shape == (2, 99)
        if enabled:
            model = env.get_playback_model(0)
            from unilab.base.backend.mujoco.recovery_load_metrics import RecoveryLoadSampler

            with pytest.raises(ValueError, match="never world"):
                RecoveryLoadSampler([model], (("world", "torso_link"),))
            np.testing.assert_array_equal(m["force_ranges"][0], model.actuator_forcerange)
            # Sparse reset invalidates only the reset row's old interval metrics.
            env.reset(np.array([0], np.int32))
            m = env._backend.get_recovery_load_metrics()
            np.testing.assert_array_equal(m["valid"], [False, True])
            state = env.step(np.zeros((2, 29), np.float32))
            assert env._backend.get_recovery_load_metrics()["valid"].all()
        else:
            assert env._backend.get_recovery_load_metrics() is None
        return np.asarray(states), np.asarray(rewards), np.asarray(observations)
    finally:
        env._backend._pool.close()
        env.close()


def test_default_off_preserves_legacy_rewards_observations_and_native_physics():
    old = rollout(config("mujoco_unassisted_v2"), None)
    new = rollout(config("mujoco_constrained_v1"), False)
    for a, b in zip(old, new):
        np.testing.assert_array_equal(a, b)


def test_on_observer_preserves_actual_dynamics_with_reset_randomized_models():
    c = config("mujoco_constrained_v1")
    action = np.full((2, 29), 0.9, np.float32)
    off = rollout(c, False, action=action)
    on = rollout(c, True, action=action)
    np.testing.assert_array_equal(off[0], on[0])
    np.testing.assert_array_equal(off[2], on[2])
    assert np.all(on[1] <= off[1]) and np.any(on[1] < off[1])


def test_off_ignores_unused_new_weights_even_when_they_would_produce_nan_times_zero():
    old = rollout(config("mujoco_unassisted_v2"), None)
    c = config("mujoco_constrained_v1")
    c.reward.scales.recovery_pd_excess = float("nan")
    new = rollout(c, False)
    for a, b in zip(old, new):
        np.testing.assert_array_equal(a, b)
