"""Assembly-independent recovery-to-standing DAgger scenario owner."""

import os
from pathlib import Path

from omegaconf import OmegaConf

from unilab.training import BackendAdapter, create_env, ensure_registries

from .data import save_distillation_dataset
from .performance import DISTILLATION_METRICS_SCHEMA_VERSION, DistillationStageObservation
from .playback import load_distillation_student_policy
from .recovery_integration import collect_recovery_handover, validate_recovery_student
from .teacher import DistillationTeacherSpec, load_sac_teacher_policy
from .workflow import WorkflowScenarioCollectionResult


def collect_recovery_workflow_scenario(
    cfg, role_cfgs, checkpoint_path, output_path, *, performance_clock
):
    request_start = performance_clock()
    root = Path(__file__).resolve().parents[5]
    scenario_cfg = OmegaConf.merge(
        # Hydra locks the source task schema; isolate the different task's overlay.
        OmegaConf.to_container(cfg, resolve=False),
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
    device = OmegaConf.select(cfg, "training.device") or "cpu"

    def load(role):
        role_cfg = role_cfgs[role]
        spec = DistillationTeacherSpec(
            obs_dim=99,
            action_dim=29,
            actor_hidden_dim=int(role_cfg.teacher.actor_hidden_dim),
            use_layer_norm=bool(role_cfg.teacher.use_layer_norm),
            obs_normalization=bool(role_cfg.teacher.obs_normalization),
        )
        return load_sac_teacher_policy(role_cfg.teacher.checkpoint_path, spec, device=device)

    recovery_teacher, standing_teacher = load("recovery"), load("stand_height")
    loaded = load_distillation_student_policy(checkpoint_path, device=device)
    validate_recovery_student(loaded.policy, loaded.distill_runtime_cfg)
    ensure_registries()
    env = create_env(
        scenario_cfg,
        num_envs=int(cfg.training.workflow.collect_num_envs),
        env_cfg_override=BackendAdapter(
            scenario_cfg, root_dir=root, algo_name="distill"
        ).build_task_env_cfg_override(),
        sim_backend="mujoco",
        task_name="G1RecoveryCombined",
    )
    cold_start_seconds = performance_clock() - request_start
    try:
        dataset = collect_recovery_handover(
            env,
            recovery_teacher=recovery_teacher,
            standing_teacher=standing_teacher,
            student=loaded.policy,
            num_samples=int(cfg.training.workflow.dagger_samples_per_role),
            metadata={
                "workflow_scenario": "recovery_to_stand",
                "rollout_checkpoint": str(checkpoint_path),
            },
            performance_clock=performance_clock,
        )
        # Transition schema is scenario-specific; the generic walk transition validator must not
        # interpret recovery rows as commanded walking rows.
        write_start = performance_clock()
        save_distillation_dataset(output_path, dataset)
        write_seconds = performance_clock() - write_start
    finally:
        env.close()
    observations = tuple(
        DistillationStageObservation.from_dict(item)
        for item in dataset.metadata["performance_stage_observations"]
    )

    def observation(stage, duration, rows=0, steps=0, cleanup="not_applicable"):
        return DistillationStageObservation(
            stage=stage,
            duration_seconds=duration,
            row_count=rows,
            env_step_count=steps,
            success=True,
            error=None,
            cleanup_state=cleanup,
        )

    return WorkflowScenarioCollectionResult(
        num_samples=dataset.num_samples,
        worker_pid=os.getpid(),
        performance_metrics_schema_version=DISTILLATION_METRICS_SCHEMA_VERSION,
        performance_stage_observations=(
            observation("cold_start", cold_start_seconds),
            *observations,
            observation("artifact_write", write_seconds, rows=dataset.num_samples),
            observation(
                "total_elapsed",
                performance_clock() - request_start,
                rows=dataset.num_samples,
                steps=dataset.metadata["env_steps"],
                cleanup="pending",
            ),
        ),
    )
