"""Bounded, headless G1 recovery task smoke probe; never trains a policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.training.backend_adapter import BackendAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--profile", choices=["unassisted_v2"], default=None)
    parser.add_argument(
        "--check-runner",
        action="store_true",
        help="construct CPU SAC runner and check actor/critic I/O without learning",
    )
    args = parser.parse_args()
    if not 1 <= args.steps <= 400 or not 1 <= args.num_envs <= 16:
        parser.error("probe requires 1..400 steps and 1..16 environments")
    np.random.seed(args.seed)
    registry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "conf/offpolicy"), version_base="1.3"):
        task = "mujoco_unassisted_v2" if args.profile else "mujoco"
        cfg = compose(config_name="config", overrides=["algo=sac", f"task=sac/g1_recovery/{task}"])
    overrides = BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override()
    env = registry.make(
        "G1Recovery", num_envs=args.num_envs, sim_backend="mujoco", env_cfg_override=overrides
    )
    try:
        runner_facts = {}
        if args.check_runner:
            import torch
            from scripts.train_offpolicy import build_runner

            from unilab.training.g1_recovery import validate_recovery_training_config

            torch.set_num_threads(1)
            cfg.training.device = "cpu"
            cfg.algo.algo_params.use_compile = False
            validate_recovery_training_config(cfg)
            runner = build_runner("sac", cfg)
            with torch.inference_mode():
                action = runner.learner.actor.explore(torch.zeros((2, 99)), deterministic=True)
                critic_logits = runner.learner.qnet(torch.zeros((2, 102)), action)
            if action.shape != (2, 29) or not torch.isfinite(action).all():
                raise AssertionError("actor connectivity failed")
            if (
                critic_logits.shape[-2:] != (2, cfg.algo.num_atoms)
                or not torch.isfinite(critic_logits).all()
            ):
                raise AssertionError("critic connectivity failed")
            if runner.train_start_threshold > runner.num_envs * runner.replay_buffer_n:
                raise AssertionError("learning threshold exceeds replay capacity")
            runner_facts = {
                "runner": type(runner).__name__,
                "learner": type(runner.learner).__name__,
                "sync_collection": runner.sync_collection,
                "runner_device_checked": "cpu",
                "learning_called": False,
            }
        env.set_autoreset(False)
        state = env.init_state()
        if state.obs["obs"].shape != (args.num_envs, 99) or state.obs["critic"].shape != (
            args.num_envs,
            102,
        ):
            raise AssertionError("recovery observation contract mismatch")
        if env.action_space.shape != (29,):
            raise AssertionError("recovery action contract mismatch")
        # Actor gravity occupies indices 3:6 with the same sign as existing policies.
        initial_gravity = -state.obs["obs"][:, 3:6].copy()
        if np.any(np.abs(initial_gravity[:, 2]) > 0.1):
            raise AssertionError("recovery did not initialize horizontal")
        reference = env.default_angles.copy()
        initial_obs = state.obs["obs"].copy()
        max_joint_speed = 0.0
        for step in range(args.steps):
            state = env.step(np.zeros((args.num_envs, 29), dtype=np.float32))
            if not all(np.all(np.isfinite(value)) for value in state.obs.values()) or not np.all(
                np.isfinite(state.reward)
            ):
                raise AssertionError(f"non-finite recovery output at step {step + 1}")
            if np.any(state.terminated | state.truncated):
                raise AssertionError(f"recovery unexpectedly ended at step {step + 1}")
            stable_counts = state.info["recovery_stable_steps"].copy()
            env.refresh_state()
            np.testing.assert_array_equal(state.info["recovery_stable_steps"], stable_counts)
            max_joint_speed = max(max_joint_speed, float(np.max(np.abs(env.get_dof_vel()))))
        np.testing.assert_array_equal(env.default_angles, reference)
        # A partial reset must not move or clear the other live environments.
        if args.num_envs > 1:
            before = env.get_physics_state_snapshot().copy()
            env.reset(np.asarray([0], dtype=np.int32))
            after = env.get_physics_state_snapshot()
            np.testing.assert_allclose(after[1:], before[1:])
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "seed": args.seed,
                    "profile": args.profile or "baseline",
                    "height_gated_upright": env.cfg.recovery.height_gated_upright,
                    "assistance_fraction": env.cfg.assistance.effective_fraction,
                    **runner_facts,
                    "steps": args.steps,
                    "num_envs": args.num_envs,
                    "actor_obs_dim": 99,
                    "critic_obs_dim": 102,
                    "action_dim": 29,
                    "initial_gravity": initial_gravity.tolist(),
                    "initial_target_height": initial_obs[:, 96].tolist(),
                    "max_joint_speed": max_joint_speed,
                    "partial_reset_checked": args.num_envs > 1,
                    "scope": "task lifecycle and optional CPU runner construction only; no learning, CUDA or recovery-success claim",
                },
                indent=2,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
