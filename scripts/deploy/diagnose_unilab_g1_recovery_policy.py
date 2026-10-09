"""Headless, paired recovery-policy diagnostics; never constructs a learner or trains.

The assisted branch is a diagnostic intervention, not deployment acceptance.
Use the checkpoint's saved config and the same deterministic actor as playback.
With --zero-action-baseline, both branches are unassisted. Zero actions mean
zero policy offsets, not disabled motors: reference-angle PD targets remain active.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from scripts.train_offpolicy import extract_play_obs, resolve_play_actor_spec

from unilab.algos.torch.common.actor_factory import build_actor
from unilab.base import registry
from unilab.base.observations import get_obs_dims
from unilab.envs.locomotion.g1.joystick import (
    LEFT_FOOT_CONTACT_SENSORS,
    RIGHT_FOOT_CONTACT_SENSORS,
    compute_aggregated_foot_contact,
)
from unilab.training.backend_adapter import BackendAdapter


def accumulate_reward_terms(totals, log, *, ctrl_dt, step_reward):
    """Env log terms are already weighted, but need the same dt as state.reward."""
    contributions = {}
    for name in totals:
        value = float(log[f"reward/{name}"]) * ctrl_dt
        if not np.isfinite(value):
            raise FloatingPointError(f"non-finite reward contribution: {name}")
        contributions[name] = value
    if not np.isclose(sum(contributions.values()), step_reward, atol=1e-7, rtol=1e-5):
        raise AssertionError("reward terms do not reconstruct the actual environment reward")
    for name, value in contributions.items():
        totals[name] += value


def run_branch(saved_config, checkpoint, *, assisted, seed, device, steps, zero_action=False):
    if assisted and zero_action:
        raise ValueError("zero-action baseline must be unassisted")
    cfg = OmegaConf.create(saved_config)
    if (cfg.training.task_name, cfg.training.sim_backend, cfg.algo.algo) != (
        "G1Recovery",
        "mujoco",
        "sac",
    ):
        raise ValueError("requires a MuJoCo G1Recovery SAC run_config.json")
    # Deliberate diagnostic-only intervention: ordinary play still forces assistance OFF.
    cfg.training.play_only = False
    cfg.env.assistance.evaluation = not assisted
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    overrides = BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override()
    env = registry.make("G1Recovery", num_envs=1, sim_backend="mujoco", env_cfg_override=overrides)
    try:
        env.set_autoreset(False)
        state = env.init_state()
        initial = env.get_physics_state_snapshot().copy()
        initial_obs = np.asarray(extract_play_obs(state.obs)).copy()
        obs_dim, critic_dim = get_obs_dims(env.obs_groups_spec)
        algo_type, kwargs = resolve_play_actor_spec(
            "sac", cfg, obs_dim=int(obs_dim), critic_obs_dim=int(critic_dim)
        )
        if algo_type != "sac":
            raise ValueError("this probe only supports the standard SACActor")
        if checkpoint.get("obs_normalizer") is not None:
            raise ValueError("checkpoint has external obs_normalizer; verify playback parity first")
        actor = build_actor(
            algo_type,
            obs_dim,
            env.action_space.shape[0],
            cfg.algo.actor_hidden_dim,
            cfg.algo.use_layer_norm,
            device,
            **kwargs,
        )
        actor.load_state_dict(checkpoint["actor"], strict=True)
        actor.eval()
        record = {
            "branch": "zero_actions"
            if zero_action
            else ("training_assistance" if assisted else "unassisted"),
            "controller": "zero_policy_offsets"
            if zero_action
            else "deterministic_checkpoint_actor",
            "seed": seed,
            "configured_stage": int(cfg.env.assistance.stage),
            "assistance_fraction": float(env.cfg.assistance.effective_fraction),
            "max_base_height_m": float(env._backend.get_base_pos()[0, 2]),
            "max_up_z": float(env._backend.get_sensor_data(env.cfg.sensor.upvector)[0, 2]),
            "assisted_steps": 0,
            "max_assistance_force_z_N": 0.0,
            "both_feet_contact_steps": 0,
            "stable_steps": 0,
            "max_continuous_stable_steps": 0,
            "ctrl_dt_seconds": float(env.cfg.ctrl_dt),
            "max_action_abs": 0.0,
            "episode_reward": 0.0,
            "reward_term_totals": {str(name): 0.0 for name in cfg.reward.scales},
            "success": False,
            "first_success_step": None,
        }
        limit = int(env.cfg.max_episode_steps)
        if limit <= 0:
            raise ValueError("diagnostic requires a finite episode length")
        with torch.inference_mode():
            for step in range(1, min(steps or limit, limit) + 1):
                obs = np.asarray(extract_play_obs(state.obs), dtype=np.float32)
                if not np.isfinite(obs).all():
                    raise FloatingPointError("non-finite actor input")
                action = (
                    np.zeros((1, int(env.action_space.shape[0])), dtype=np.float32)
                    if zero_action
                    else actor.explore(
                        torch.as_tensor(obs, dtype=torch.float32, device=device), deterministic=True
                    )
                    .cpu()
                    .numpy()
                )
                if not np.isfinite(action).all():
                    raise FloatingPointError("non-finite actor action")
                state = env.step(action)
                if not np.isfinite(state.reward).all():
                    raise FloatingPointError("non-finite reward")
                height = float(env._backend.get_base_pos()[0, 2])
                up = float(env._backend.get_sensor_data(env.cfg.sensor.upvector)[0, 2])
                feet = [
                    bool(compute_aggregated_foot_contact(env._backend, names)[0] > 0)
                    for names in (LEFT_FOOT_CONTACT_SENSORS, RIGHT_FOOT_CONTACT_SENSORS)
                ]
                force = float(state.info["recovery_assistance_force_z"][0])
                record["max_base_height_m"] = max(record["max_base_height_m"], height)
                record["max_up_z"] = max(record["max_up_z"], up)
                record["max_action_abs"] = max(
                    record["max_action_abs"], float(np.max(np.abs(action)))
                )
                record["assisted_steps"] += int(force > 0)
                record["max_assistance_force_z_N"] = max(record["max_assistance_force_z_N"], force)
                record["both_feet_contact_steps"] += int(all(feet))
                record["stable_steps"] += int(state.info["recovery_stable"][0])
                record["max_continuous_stable_steps"] = max(
                    record["max_continuous_stable_steps"],
                    int(state.info["recovery_stable_steps"][0]),
                )
                accumulate_reward_terms(
                    record["reward_term_totals"],
                    state.info["log"],
                    ctrl_dt=float(env.cfg.ctrl_dt),
                    step_reward=float(state.reward[0]),
                )
                record["episode_reward"] += float(state.reward[0])
                success = bool(state.info["recovery_succeeded"][0])
                if success and record["first_success_step"] is None:
                    record["first_success_step"] = step
                record["success"] |= success
                record.update(
                    {
                        "steps": step,
                        "final_base_height_m": height,
                        "final_up_z": up,
                        "final_feet_contact": feet,
                        "terminated": bool(state.terminated[0]),
                        "truncated": bool(state.truncated[0]),
                    }
                )
                if state.terminated[0] or state.truncated[0]:
                    break
        record["full_episode"] = bool(record["terminated"] or record["truncated"])
        record["longest_stable_seconds"] = (
            record["max_continuous_stable_steps"] * record["ctrl_dt_seconds"]
        )
        record["stable_at_episode_end"] = bool(state.info["recovery_stable"][0])
        record["reward_reconstruction_error"] = record["episode_reward"] - sum(
            record["reward_term_totals"].values()
        )
        return initial, initial_obs, record
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=0, help="0: one full episode per seed")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--zero-action-baseline",
        action="store_true",
        help="compare checkpoint actor vs zero policy offsets, both with assistance OFF",
    )
    args = parser.parse_args()
    if not 1 <= args.episodes <= 32 or args.steps < 0 or args.seed < 0:
        parser.error("require 1..32 episodes, nonnegative steps and seed")
    torch.set_num_threads(1)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    saved_config = json.loads((checkpoint_path.parent / "run_config.json").read_text())["config"]
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    registry.ensure_registries()
    records = []
    second_branch = "zero_actions" if args.zero_action_baseline else "training_assistance"
    branches = ("unassisted", second_branch)
    for seed in range(args.seed, args.seed + args.episodes):
        initial_off, obs_off, off = run_branch(
            saved_config,
            checkpoint,
            assisted=False,
            seed=seed,
            device=args.device,
            steps=args.steps,
        )
        initial_on, obs_on, on = run_branch(
            saved_config,
            checkpoint,
            assisted=not args.zero_action_baseline,
            seed=seed,
            device=args.device,
            steps=args.steps,
            zero_action=args.zero_action_baseline,
        )
        np.testing.assert_array_equal(initial_off, initial_on)
        np.testing.assert_array_equal(obs_off, obs_on)
        records.extend([off, on])
        print(
            f"seed={seed}: unassisted_success={off['success']}, "
            f"{second_branch}_success={on['success']}, "
            f"policy_reward={off['episode_reward']:.4f}, "
            f"{second_branch}_reward={on['episode_reward']:.4f}",
            file=sys.stderr,
            flush=True,
        )
    report = {
        "status": "DIAGNOSTIC_COMPLETED",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "same_initial_states_and_actor_observations": True,
        "episodes_per_branch": args.episodes,
        "comparison_mode": "zero_action_baseline" if args.zero_action_baseline else "assistance",
        "zero_actions_meaning": "zero policy offsets; reference-angle PD targets active; motors enabled",
        "mean_episode_reward": {
            branch: float(
                np.mean([row["episode_reward"] for row in records if row["branch"] == branch])
            )
            for branch in branches
        },
        "mean_reward_term_totals": {
            branch: {
                term: float(
                    np.mean(
                        [
                            row["reward_term_totals"][term]
                            for row in records
                            if row["branch"] == branch
                        ]
                    )
                )
                for term in records[0]["reward_term_totals"]
            }
            for branch in branches
        },
        "full_episodes": all(row["full_episode"] for row in records),
        "success_rate": {
            branch: sum(row["success"] for row in records if row["branch"] == branch)
            / args.episodes
            for branch in branches
        },
        "scope": "paired deterministic policy diagnosis; weighted reward terms include ctrl_dt; no training; assisted success is not deployment acceptance",
        "episodes": records,
    }
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        # Explicit output only, never overwrite a checkpoint or an existing report.
        with args.output.open("x") as stream:
            stream.write(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
