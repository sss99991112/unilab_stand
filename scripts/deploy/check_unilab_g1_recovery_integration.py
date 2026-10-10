"""Bounded recovery-to-standing live route, with explicit reference/student action source."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.distill import (
    DistillationTeacherSpec,
    load_distillation_student_policy,
    load_sac_teacher_policy,
)
from unilab.algos.torch.distill.g1_observation_adapter import RecoveryTeacherObservationAdapter
from unilab.algos.torch.distill.recovery_integration import (
    RecoveryRoutedPolicy,
    recovery_expert_indices,
    validate_recovery_env,
    validate_recovery_student,
)
from unilab.training import BackendAdapter, create_env, ensure_registries


def run_probe(*, student_checkpoint, recovery_teacher=None, steps=750, seed=1):
    root = Path(__file__).resolve().parents[2]
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = OmegaConf.load(root / "conf/distill/config.yaml")
    del cfg["defaults"]
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.load(root / "conf/distill/task/g1_recovery_combined/mujoco.yaml"),
        {
            "env": {
                "recovery_reset_supine": True,
                "commands": {
                    "vel_limit": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    "rel_standing_envs": 1.0,
                },
            }
        },
    )
    loaded = load_distillation_student_policy(student_checkpoint)
    if recovery_teacher is None:
        validate_recovery_student(loaded.policy, loaded.distill_runtime_cfg)
        actor = loaded.policy
        source = "unified_three_expert_student"
    else:
        if loaded.obs_dim != 99 or loaded.action_dim != 29 or loaded.policy.num_experts != 2:
            raise ValueError("reference requires the existing 99/29 two-expert student")
        recovery = load_sac_teacher_policy(
            recovery_teacher,
            DistillationTeacherSpec(
                obs_dim=99,
                action_dim=29,
                actor_hidden_dim=512,
                use_layer_norm=True,
                obs_normalization=True,
            ),
        )
        actor = torch.nn.ModuleDict(
            {
                "recovery": RecoveryTeacherObservationAdapter(recovery),
                "stand_height": loaded.policy.experts[1],
            }
        )
        source = "reference_recovery_teacher_then_existing_stand_expert"
    ensure_registries()
    env = create_env(
        cfg,
        num_envs=1,
        env_cfg_override=BackendAdapter(
            cfg, root_dir=root, algo_name="distill"
        ).build_task_env_cfg_override(),
        sim_backend="mujoco",
        task_name="G1RecoveryCombined",
    )
    routes = []
    first_handover = None
    heights = []
    try:
        validate_recovery_env(env)
        env.init_state()
        env.set_autoreset(False)
        obs, info = env.reset(np.arange(1, dtype=np.int32))
        if not info["recovery_active"][0]:
            raise AssertionError("supine reset must select recovery before the first action")
        policy = RecoveryRoutedPolicy(actor, env)
        with torch.inference_mode():
            for step in range(steps):
                route = int(recovery_expert_indices(env.state.info)[0])
                routes.append(route)
                if route == 1 and first_handover is None:
                    first_handover = step
                action = policy(torch.as_tensor(obs["obs"], dtype=torch.float32)).numpy()
                state = env.step(action)
                obs = state.obs
                heights.append(float(env._backend.get_base_pos()[0, 2]))
                if state.terminated[0] or state.truncated[0]:
                    break
        return {
            "action_source": source,
            "seed": seed,
            "steps": len(routes),
            "expert_step_counts": {str(i): routes.count(i) for i in (0, 1, 2)},
            "first_handover_seconds": None
            if first_handover is None
            else first_handover * float(env.cfg.ctrl_dt),
            "final_base_height_m": heights[-1],
            "max_base_height_m": max(heights),
            "terminated": bool(state.terminated[0]),
            "truncated": bool(state.truncated[0]),
            "actor_obs_dim": loaded.obs_dim,
            "action_dim": loaded.action_dim,
            "policy_quality_accepted": False,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-teacher", type=Path)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_probe(
        student_checkpoint=args.student_checkpoint,
        recovery_teacher=args.recovery_teacher,
        steps=args.steps,
        seed=args.seed,
    )
    if args.output:
        if args.output.exists():
            raise FileExistsError("probe evidence must use a fresh output path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
