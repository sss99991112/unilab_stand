"""CPU readiness sentinel for actor-only constrained recovery; never trains.

Uses the formal config, environment/reward owner and teacher loading path.
This is a bounded prefix check, not teacher quality or hardware acceptance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scripts.train_offpolicy import apply_configured_actor_warm_start

from unilab.algos.torch.distill.teacher import DistillationTeacherSpec, load_sac_teacher_policy
from unilab.algos.torch.fast_sac.learner import FastSACLearner
from unilab.ipc.memory_budget import estimate_offpolicy_bytes
from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.training import BackendAdapter, create_env, ensure_registries
from unilab.training.g1_recovery import validate_recovery_training_config
from unilab.training.g1_recovery_refinement import validate_fresh_refinement_output

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent", type=Path, default=ROOT.parent / "2026-10-09_19-37-31_mujoco/model_12000.pt"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()
    if not 1 <= args.steps <= 750:
        parser.error("steps must be within the original 15s budget (1..750)")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    ensure_registries()
    parent_sha = hashlib.sha256(args.parent.read_bytes()).hexdigest()
    with initialize_config_dir(config_dir=str(ROOT / "conf/offpolicy"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=sac/g1_recovery/mujoco_constrained_v1",
                "env.recovery_constraints.enabled=true",
                f"algo.actor_warm_start_checkpoint={args.parent.resolve()}",
                f"training.log_dir={args.output.resolve()}/TRAINING_NOT_STARTED",
            ],
        )
    validate_recovery_training_config(cfg)
    validate_fresh_refinement_output(cfg)
    OmegaConf.save(cfg, args.output / "resolved_config.yaml")
    # Small fresh critic for a CPU loader sentinel only; formal config stays768.
    learner = FastSACLearner(
        obs_dim=99,
        action_dim=29,
        critic_obs_dim=102,
        device="cpu",
        actor_hidden_dim=512,
        critic_hidden_dim=32,
        num_atoms=11,
        use_compile=False,
    )
    initial_critic = copy.deepcopy(learner.qnet.state_dict())
    metadata = apply_configured_actor_warm_start("sac", cfg, SimpleNamespace(learner=learner))
    assert all(torch.equal(v, learner.qnet.state_dict()[k]) for k, v in initial_critic.items())
    assert learner.update_count == 0
    official = load_sac_teacher_policy(
        args.parent,
        DistillationTeacherSpec(obs_dim=99, action_dim=29, obs_normalization=True),
        device="cpu",
    )
    assert official.obs_normalizer is None and getattr(learner, "obs_normalizer", None) is None
    replay = ReplayBuffer(capacity=16, obs_dim=99, action_dim=29, critic_dim=102, device="cpu")
    assert int(replay.size[0]) == 0
    cases = []
    zero_start_error = 0.0
    replay_rewards = []
    for seed in (0, 2):
        traces = {}
        for enabled in (False, True):
            np.random.seed(seed)
            torch.manual_seed(seed)
            c = copy.deepcopy(cfg)
            c.env.recovery_constraints.enabled = enabled
            overrides = BackendAdapter(
                c, root_dir=ROOT, algo_name="sac"
            ).build_task_env_cfg_override()
            overrides.update(
                noise_config={"level": 0.0},
                domain_rand={"randomize_kp": False, "randomize_kd": False},
                max_episode_seconds=15.0,
            )
            env = create_env(
                c,
                num_envs=1,
                sim_backend="mujoco",
                task_name="G1Recovery",
                env_cfg_override=overrides,
            )
            states = []
            cost_rows = []
            obs_rows = []
            action_rows = []
            times = []
            rewards = []
            try:
                state = env.init_state()
                env.set_autoreset(False)
                for frame in range(args.steps):
                    obs = state.obs["obs"].copy()
                    critic = state.obs["critic"].copy()
                    with torch.inference_mode():
                        action = learner.actor(torch.from_numpy(obs))[0]
                        reference = official(torch.from_numpy(obs))
                    err = float(torch.max(torch.abs(action - reference)))
                    zero_start_error = max(zero_start_error, err)
                    torch.testing.assert_close(action, reference, rtol=0, atol=0)
                    begin = time.perf_counter()
                    state = env.step(action.numpy())
                    times.append(time.perf_counter() - begin)
                    states.append(env.get_physics_state_snapshot().copy())
                    obs_rows.append(obs[0])
                    action_rows.append(action.numpy()[0])
                    rewards.append(float(state.reward[0]))
                    log = state.info["log"]
                    cost_rows.append(
                        [
                            log.get(k, 0.0)
                            for k in (
                                "constraint/max_joint_excess_rad",
                                "constraint/max_pd_excess_nm",
                                "constraint/max_pair_normal_force_n",
                                "constraint/replay_error",
                            )
                        ]
                    )
                    if enabled:
                        # The exact reward owner -> fresh replay boundary, no learner update.
                        replay.add(
                            torch.from_numpy(obs),
                            action,
                            torch.from_numpy(state.reward.copy()),
                            torch.from_numpy(state.obs["obs"].copy()),
                            torch.from_numpy((state.terminated | state.truncated).copy()),
                            torch.from_numpy(state.truncated.copy()),
                            critic=torch.from_numpy(critic),
                            next_critic=torch.from_numpy(state.obs["critic"].copy()),
                        )
                        replay_rewards.append(float(state.reward[0]))
                        assert torch.isin(
                            replay.sample(8)["rewards"],
                            torch.tensor(replay_rewards[-16:], dtype=torch.float32),
                        ).all()
                    if state.terminated.any():
                        raise RuntimeError(
                            "original velocity termination reached during readiness sentinel"
                        )
                trace = dict(
                    physics=np.asarray(states),
                    obs=np.asarray(obs_rows),
                    action=np.asarray(action_rows),
                    reward=np.asarray(rewards),
                    costs=np.asarray(cost_rows),
                )
                traces[enabled] = trace
                name = f"seed{seed}_{'on' if enabled else 'off'}"
                np.savez_compressed(args.output / (name + ".npz"), **trace)
                cases.append(
                    dict(
                        name=name,
                        steps=args.steps,
                        physical_substeps=args.steps * 3,
                        mean_control_ms=float(np.mean(times) * 1000),
                        max_control_ms=float(np.max(times) * 1000),
                        max_cost_fields=np.max(trace["costs"], axis=0).tolist(),
                        status="readiness_prefix_only",
                    )
                )
            finally:
                env._backend._pool.close()
                env.close()
        for k in ("physics", "obs", "action"):
            np.testing.assert_array_equal(traces[False][k], traces[True][k])
        assert np.all(traces[True]["reward"] <= traces[False]["reward"])
    assert not any(
        opt.state for opt in (learner.actor_optimizer, learner.q_optimizer, learner.alpha_optimizer)
    )
    assert (
        learner.update_count == 0
        and parent_sha == hashlib.sha256(args.parent.read_bytes()).hexdigest()
    )
    result = dict(
        scope="CPU readiness only; no optimizer/runner training, no new weights",
        cases=cases,
        actor_only=metadata,
        official_teacher_action_error=zero_start_error,
        native_off_on_exact=True,
        replay_initial_rows=0,
        replay_final_rows=int(replay.size[0]),
        cold_optimizer_states=True,
        learner_update_count=learner.update_count,
        proposed_formal_memory=estimate_offpolicy_bytes(
            num_envs=cfg.algo.num_envs,
            replay_buffer_n=cfg.algo.replay_buffer_n,
            obs_dim=99,
            action_dim=29,
            critic_dim=102,
            batch_size=cfg.algo.batch_size,
            updates_per_step=cfg.algo.updates_per_step,
        ),
        note="Measured force normalization and soft margins are research settings, not physical limits or acceptance thresholds.",
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
