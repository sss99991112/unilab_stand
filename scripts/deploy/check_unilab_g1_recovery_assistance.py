"""Tiny same-state physics differential for recovery assistance; no policy training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.training.backend_adapter import BackendAdapter


def _branch(stage: int, evaluation: bool):
    np.random.seed(1)
    with initialize_config_dir(config_dir=str(ROOT / "conf/offpolicy"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algo=sac",
                "task=sac/g1_recovery/mujoco",
                f"env.assistance.stage={stage}",
                f"training.play_only={str(evaluation).lower()}",
                "env.assistance.start_after_seconds=0.0",
            ],
        )
    overrides = BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override()
    env = registry.make("G1Recovery", num_envs=1, sim_backend="mujoco", env_cfg_override=overrides)
    try:
        env.set_autoreset(False)
        env.init_state()
        # Test-only intervention: put the same robot upright above the floor so
        # contact cannot hide or amplify the external-force acceleration.
        backend = env._backend
        pose = backend.get_keyframe_qpos("stand")[None, :].copy()
        pose[:, 2] = 1.5
        backend.set_state(np.asarray([0], dtype=np.int32), pose, backend.get_init_qvel()[None, :])
        env.refresh_state()
        initial = env.get_physics_state_snapshot().copy()
        state = env.step(np.zeros((1, 29), dtype=np.float32))
        if np.any(state.terminated | state.truncated) or not np.all(np.isfinite(state.reward)):
            raise AssertionError("invalid assistance probe transition")
        applied = float(state.info["recovery_assistance_force_z"][0])
        velocity = float(env.get_local_linvel()[0, 2])
        final = env.get_physics_state_snapshot().copy()
        # Test internal consumption boundary: no force remains staged after step.
        if np.any(backend._pending_xfrc_applied):
            raise AssertionError("assistance force was not consumed/cleared")
        state.info["steps"].fill(0)
        env._assistance_start_steps = 30
        env.step(np.zeros((1, 29), dtype=np.float32))
        if np.any(state.info["recovery_assistance_force_z"]):
            raise AssertionError("initial-fall gate left a stale assistance force")
        return (
            initial,
            final,
            {
                "stage": stage,
                "evaluation": evaluation,
                "force_z_newtons": applied,
                "vertical_velocity": velocity,
            },
        )
    finally:
        env.close()


def main():
    registry.ensure_registries()
    initial_off, final_off, off = _branch(2, False)
    initial_full, _, full = _branch(0, False)
    initial_half, _, half = _branch(1, False)
    initial_eval, final_eval, evaluation = _branch(0, True)
    for initial in (initial_full, initial_half, initial_eval):
        np.testing.assert_array_equal(initial, initial_off)
    np.testing.assert_array_equal(final_eval, final_off)
    if not full["force_z_newtons"] > half["force_z_newtons"] > 0:
        raise AssertionError("stage force did not decrease toward zero")
    np.testing.assert_allclose(full["force_z_newtons"], 2 * half["force_z_newtons"])
    if not full["vertical_velocity"] > half["vertical_velocity"] > off["vertical_velocity"]:
        raise AssertionError("upward force did not produce the expected physical effect")
    if evaluation["force_z_newtons"] != 0 or off["force_z_newtons"] != 0:
        raise AssertionError("evaluation/final stage was not unassisted")
    print(
        json.dumps(
            {
                "status": "PASS",
                "branches": [full, half, off, evaluation],
                "same_initial_state": True,
                "evaluation_off_equivalence": True,
                "pending_force_cleared": True,
                "scope": "four two-step physics probes; no training",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
