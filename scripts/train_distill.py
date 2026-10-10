"""Generic behavior distillation entrypoint assembly.

This module keeps live environment sampling in distill-owned helpers and only
assembles the configured entrypoint routes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill import (
    COLLECTOR_REQUEST_STAGE_NAMES,
    DISTILLATION_METRICS_SCHEMA_VERSION,
    LEGACY_REQUEST_STAGE_NAMES,
    BehaviorDistillationTrainer,
    DistillationPerformanceRunContext,
    DistillationStageObservation,
    DistillationTeacherSpec,
    MLPStudentPolicy,
    MoEStudentPolicy,
    RoleArtifactSpec,
    WorkflowDatasetSource,
    WorkflowScenarioCollectionResult,
    WorkflowScenarioSpec,
    WorkflowStudentUpdateResult,
    adopt_legacy_role_artifact,
    build_multitask_distillation_dataset,
    collect_distillation_dataset_from_env,
    collect_transition_distillation_dataset_from_env,
    config_fingerprint,
    file_sha256,
    finalize_workflow_performance,
    fork_workflow_run,
    load_distillation_checkpoint,
    load_distillation_dataset,
    load_distillation_student_policy,
    load_sac_teacher_policy,
    make_fake_distillation_dataset,
    required_balanced_replay_updates,
    resolve_command_intent_rollout_policies,
    run_bootstrap_workflow,
    run_iterative_dagger_updates,
    run_multirole_dagger_workflow,
    run_offline_distillation_updates,
    save_distillation_dataset,
    validate_sac_teacher_checkpoint_contract,
)
from unilab.algos.torch.distill.g1_persistent_worker import (
    build_persistent_g1_distillation_runtime,
)
from unilab.logging import OffPolicyLogger
from unilab.training import BackendAdapter, ExperimentTracker, create_env, ensure_registries
from unilab.training.run import resolve_task_checkpoint_path

ROOT_DIR = Path(__file__).resolve().parents[1]
_OWNER_COMMAND_SAMPLE_FILTERS = {
    "G1WalkFlat": "active",
    "G1StandStill": "inactive",
}
_HEIGHT_OWNER_COMMAND_SAMPLE_FILTERS = {
    "G1WalkHeight": "active",
    "G1StandHeight": "inactive",
    "G1Recovery": "inactive",
}
_DISTILL_TASK_NAME_HINTS = frozenset(
    {*_OWNER_COMMAND_SAMPLE_FILTERS, *_HEIGHT_OWNER_COMMAND_SAMPLE_FILTERS}
)
_CLI_SEQUENCE_SUMMARY_LIMIT = 16


def _probe_torch_serialization_runtime(stage: str) -> dict[str, Any]:
    """Fail fast when the parent process torch serialization identity is corrupted."""

    is_storage = torch.is_storage
    snapshot = {
        "stage": str(stage),
        "pid": os.getpid(),
        "is_storage_type": type(is_storage).__name__,
        "is_storage_callable": callable(is_storage),
        "is_storage_module": getattr(is_storage, "__module__", None),
    }
    print(
        f"[distill-runtime-sentinel] {json.dumps(snapshot, sort_keys=True)}",
        flush=True,
    )
    if not snapshot["is_storage_callable"]:
        raise RuntimeError(
            "torch serialization runtime identity corrupted: "
            f"stage={snapshot['stage']} pid={snapshot['pid']} "
            f"type={snapshot['is_storage_type']} "
            f"callable={snapshot['is_storage_callable']}"
        )
    return snapshot


def _workflow_role_entries(cfg: DictConfig) -> list[dict[str, Any]]:
    entries = OmegaConf.to_container(
        OmegaConf.select(cfg, "training.workflow.roles", default=[]),
        resolve=True,
    )
    if not isinstance(entries, list) or not entries:
        raise ValueError("training.workflow.roles must be a non-empty list")
    return [dict(cast(dict[str, Any], entry)) for entry in entries]


def _workflow_scenario_specs(
    cfg: DictConfig,
    role_names: set[str],
) -> tuple[WorkflowScenarioSpec, ...] | None:
    entries = OmegaConf.to_container(
        OmegaConf.select(cfg, "training.workflow.scenarios", default=[]),
        resolve=True,
    )
    if entries in (None, []):
        return None
    if not isinstance(entries, list):
        raise ValueError("training.workflow.scenarios must be a list")
    specs: list[WorkflowScenarioSpec] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("every workflow scenario entry must be a mapping")
        name = str(entry.get("name", entry.get("scenario", "")))
        source_roles = entry.get("source_roles")
        if source_roles is None and entry.get("role") not in (None, ""):
            source_roles = [entry["role"]]
        specs.append(
            WorkflowScenarioSpec(
                name=name,
                kind=str(entry.get("kind", "role")),
                source_roles=tuple(str(role) for role in (source_roles or ())),
                quota=float(entry.get("quota", 1.0)),
            )
        )
    missing = sorted({role for spec in specs for role in spec.source_roles} - set(role_names))
    if missing:
        raise ValueError(f"workflow scenarios reference unknown roles: {missing}")
    return tuple(specs)


def _workflow_path(value: Any, *, root: Path = ROOT_DIR) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _workflow_role_cfg(cfg: DictConfig, entry: dict[str, Any]) -> DictConfig:
    task = str(entry.get("task", ""))
    task_path = Path(task)
    if not task or task_path.is_absolute() or ".." in task_path.parts:
        raise ValueError(f"workflow role task must be a relative owner selector, got {task!r}")
    owner_path = ROOT_DIR / "conf" / "distill" / "task" / task_path.with_suffix(".yaml")
    if not owner_path.is_file():
        raise FileNotFoundError(f"workflow role task owner does not exist: {owner_path}")

    base = OmegaConf.load(ROOT_DIR / "conf" / "distill" / "config.yaml")
    if "defaults" in base:
        del base["defaults"]
    role_cfg = OmegaConf.merge(
        base,
        OmegaConf.load(owner_path),
        {
            "algo": OmegaConf.to_container(cfg.algo, resolve=True),
            "student": OmegaConf.to_container(cfg.student, resolve=True),
            "training": {
                "device": OmegaConf.select(cfg, "training.device"),
                "recovery_integration": bool(
                    OmegaConf.select(cfg, "training.recovery_integration", default=False)
                ),
                "collect_num_samples": int(
                    entry.get(
                        "collect_num_samples",
                        OmegaConf.select(
                            cfg,
                            "training.workflow.collect_num_samples",
                            default=262144,
                        ),
                    )
                ),
                "collect_num_envs": int(
                    entry.get(
                        "collect_num_envs",
                        OmegaConf.select(cfg, "training.workflow.collect_num_envs", default=64),
                    )
                ),
                "collect_action_mode": "teacher_policy",
            },
            "teacher": {"checkpoint_path": str(entry.get("teacher_checkpoint_path", ""))},
        },
    )
    if not str(OmegaConf.select(role_cfg, "teacher.checkpoint_path", default="")):
        raise ValueError(f"workflow role {entry.get('role')!r} requires teacher_checkpoint_path")
    role_cfg.teacher.checkpoint_path = str(
        _workflow_path(OmegaConf.select(role_cfg, "teacher.checkpoint_path"))
    )
    if "command_sample_filter" in entry:
        role_cfg.training.collect_command_sample_filter = str(entry["command_sample_filter"])
    return cast(DictConfig, role_cfg)


def _workflow_owner_fingerprint_cfg(role_cfg: DictConfig) -> dict[str, Any]:
    return {
        "training": {
            "task_name": str(role_cfg.training.task_name),
            "sim_backend": str(role_cfg.training.sim_backend),
            "collect_action_mode": str(role_cfg.training.collect_action_mode),
            "collect_command_sample_filter": str(role_cfg.training.collect_command_sample_filter),
            "collect_target_height_info_key": OmegaConf.select(
                role_cfg, "training.collect_target_height_info_key"
            ),
            "collect_command_xy_threshold": float(role_cfg.training.collect_command_xy_threshold),
            "collect_command_yaw_threshold": float(role_cfg.training.collect_command_yaw_threshold),
        },
        "env": OmegaConf.to_container(role_cfg.env, resolve=True),
        "teacher": {
            "algo_type": str(role_cfg.teacher.algo_type),
            "obs_dim": int(role_cfg.teacher.obs_dim),
            "action_dim": int(role_cfg.teacher.action_dim),
            "actor_hidden_dim": int(role_cfg.teacher.actor_hidden_dim),
            "use_layer_norm": bool(role_cfg.teacher.use_layer_norm),
            "obs_normalization": bool(role_cfg.teacher.obs_normalization),
        },
        "student": {
            "obs_dim": int(role_cfg.student.obs_dim),
            "action_dim": int(role_cfg.student.action_dim),
        },
    }


def _int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in values)


def _student_model_type(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "student.model_type", default="mlp"))


def _compact_cli_result(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(child_key): _compact_cli_result(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) <= _CLI_SEQUENCE_SUMMARY_LIMIT:
            return [_compact_cli_result(item) for item in value]
        summary: dict[str, Any] = {
            "count": len(value),
            "head": [_compact_cli_result(item) for item in value[:4]],
            "tail": [_compact_cli_result(item) for item in value[-4:]],
        }
        if key in {"role_labels", "command_intents", "offline_balanced_labels"} and all(
            isinstance(item, str) for item in value
        ):
            summary["counts"] = dict(Counter(str(item) for item in value))
        return summary
    return value


def _format_cli_result(result: dict[str, Any]) -> str:
    return json.dumps(_compact_cli_result(result), ensure_ascii=False, sort_keys=True)


def build_teacher_spec(cfg: DictConfig) -> DistillationTeacherSpec:
    """Build the frozen teacher load contract from owner config."""

    return DistillationTeacherSpec(
        obs_dim=int(cfg.teacher.obs_dim),
        action_dim=int(cfg.teacher.action_dim),
        algo_type=str(cfg.teacher.algo_type),
        actor_hidden_dim=int(cfg.teacher.actor_hidden_dim),
        use_layer_norm=bool(cfg.teacher.use_layer_norm),
        obs_normalization=bool(cfg.teacher.obs_normalization),
    )


def build_student_policy(
    cfg: DictConfig,
    *,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Build the student actor from owner config."""

    model_type = _student_model_type(cfg)
    if model_type == "mlp":
        student = MLPStudentPolicy(
            obs_dim=int(cfg.student.obs_dim),
            action_dim=int(cfg.student.action_dim),
            hidden_dims=_int_tuple(cfg.student.hidden_dims),
            activation=str(cfg.student.activation),
            squash_action=bool(cfg.student.squash_action),
        )
    elif model_type == "moe":
        student = MoEStudentPolicy(
            obs_dim=int(cfg.student.obs_dim),
            action_dim=int(cfg.student.action_dim),
            num_experts=int(cfg.student.num_experts),
            expert_hidden_dims=_int_tuple(cfg.student.expert_hidden_dims),
            router_hidden_dims=_int_tuple(cfg.student.router_hidden_dims),
            activation=str(cfg.student.activation),
            squash_action=bool(cfg.student.squash_action),
            routing_mode=str(cfg.student.routing_mode),
            router_temperature=float(cfg.student.router_temperature),
        )
    else:
        raise ValueError(f"Unsupported distillation student.model_type: {model_type!r}")
    return cast(torch.nn.Module, student.to(device))


def _student_runtime_cfg(cfg: DictConfig) -> dict[str, Any]:
    model_type = _student_model_type(cfg)
    payload: dict[str, Any] = {
        "student_model_type": model_type,
        "student_obs_dim": int(cfg.student.obs_dim),
        "student_action_dim": int(cfg.student.action_dim),
        "student_activation": str(cfg.student.activation),
        "student_squash_action": bool(cfg.student.squash_action),
    }
    if model_type == "mlp":
        payload["student_hidden_dims"] = [int(dim) for dim in cfg.student.hidden_dims]
    elif model_type == "moe":
        payload.update(
            {
                "student_num_experts": int(cfg.student.num_experts),
                "student_expert_hidden_dims": [int(dim) for dim in cfg.student.expert_hidden_dims],
                "student_router_hidden_dims": [int(dim) for dim in cfg.student.router_hidden_dims],
                "student_routing_mode": str(cfg.student.routing_mode),
                "student_router_temperature": float(cfg.student.router_temperature),
            }
        )
    else:
        raise ValueError(f"Unsupported distillation student.model_type: {model_type!r}")
    return payload


def _resolve_optional_checkpoint_path(
    checkpoint_path: str | Path | None,
    *,
    root_dir: str | Path = ROOT_DIR,
    field_name: str,
) -> Path | None:
    if checkpoint_path in (None, ""):
        return None
    path = Path(str(checkpoint_path))
    resolved_path = path if path.is_absolute() else Path(root_dir) / path
    if not resolved_path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {resolved_path}")
    return resolved_path


def _runtime_cfg_subset_for_student(cfg: DictConfig) -> dict[str, Any]:
    runtime_cfg = _student_runtime_cfg(cfg)
    if runtime_cfg["student_model_type"] == "moe":
        return {
            key: runtime_cfg[key]
            for key in (
                "student_model_type",
                "student_obs_dim",
                "student_action_dim",
                "student_activation",
                "student_squash_action",
                "student_num_experts",
                "student_expert_hidden_dims",
                "student_router_hidden_dims",
                "student_routing_mode",
            )
        }
    return {
        key: runtime_cfg[key]
        for key in (
            "student_model_type",
            "student_obs_dim",
            "student_action_dim",
            "student_activation",
            "student_squash_action",
            "student_hidden_dims",
        )
    }


def _validate_student_init_runtime_cfg(
    cfg: DictConfig,
    *,
    runtime_cfg: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    expected = _runtime_cfg_subset_for_student(cfg)
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = runtime_cfg.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
    if mismatches:
        raise ValueError(
            "training.offline_init_checkpoint student runtime config mismatch for "
            f"{checkpoint_path}: " + "; ".join(mismatches)
        )


def _load_student_init_checkpoint(
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    *,
    cfg: DictConfig,
    device: str | torch.device,
    resume_optimizer: bool,
) -> dict[str, Any]:
    loaded_student = load_distillation_student_policy(checkpoint_path, device=device)
    runtime_cfg = dict(loaded_student.distill_runtime_cfg)
    _validate_student_init_runtime_cfg(
        cfg,
        runtime_cfg=runtime_cfg,
        checkpoint_path=checkpoint_path,
    )
    checkpoint = load_distillation_checkpoint(
        student,
        checkpoint_path,
        optimizer=optimizer if resume_optimizer else None,
        device=device,
    )
    return {
        "path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "recovery_migration": runtime_cfg.get("recovery_migration"),
        "agent_steps": int(checkpoint.get("agent_steps", loaded_student.agent_steps)),
        "optimizer_requested": bool(resume_optimizer),
        "optimizer_loaded": bool(resume_optimizer)
        and checkpoint.get("optimizer_state_dict") is not None,
        "student_model_type": runtime_cfg.get("student_model_type"),
        "student_obs_dim": runtime_cfg.get("student_obs_dim"),
        "student_action_dim": runtime_cfg.get("student_action_dim"),
    }


def _teacher_metadata(cfg: DictConfig, teacher_checkpoint: str | Path) -> dict[str, Any]:
    metadata = {
        "algo_family": str(cfg.teacher.algo_family),
        "algo_type": str(cfg.teacher.algo_type),
        "task": str(cfg.teacher.task),
        "task_name": str(cfg.teacher.task_name),
        "checkpoint_path": str(teacher_checkpoint),
    }
    info = validate_sac_teacher_checkpoint_contract(
        teacher_checkpoint,
        build_teacher_spec(cfg),
        device="cpu",
    )
    metadata.update(
        {
            "checkpoint_actor_input_dim": info.actor_input_dim,
            "checkpoint_first_weight_key": info.first_weight_key,
        }
    )
    return metadata


def _recovery_routing_contract(cfg):
    if not bool(OmegaConf.select(cfg, "training.recovery_integration", default=False)):
        return None
    from unilab.algos.torch.distill.recovery_integration import RECOVERY_ROUTING_CONTRACT

    return dict(RECOVERY_ROUTING_CONTRACT)


def _distill_runtime_cfg(
    cfg: DictConfig,
    *,
    distill_source: str,
    dataset_path: str | Path | None = None,
    student_init_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "distill_source": str(distill_source),
        "loss_type": str(cfg.algo.loss_type),
        "learning_rate": float(cfg.algo.learning_rate),
        "aux_loss_coef": float(OmegaConf.select(cfg, "algo.aux_loss_coef", default=0.0)),
        "role_loss_coef": float(OmegaConf.select(cfg, "algo.role_loss_coef", default=0.0)),
        "role_expert_targets": dict(
            OmegaConf.to_container(
                OmegaConf.select(cfg, "algo.role_expert_targets", default={}),
                resolve=True,
            )
        ),
        "command_intent_loss_coef": float(
            OmegaConf.select(cfg, "algo.command_intent_loss_coef", default=0.0)
        ),
        "command_intent_expert_targets": dict(
            OmegaConf.to_container(
                OmegaConf.select(cfg, "algo.command_intent_expert_targets", default={}),
                resolve=True,
            )
        ),
        "expert_behavior_loss_source": str(
            OmegaConf.select(cfg, "algo.expert_behavior_loss_source", default="auto")
        ),
        "offline_repeat_dataset": bool(
            OmegaConf.select(cfg, "training.offline_repeat_dataset", default=False)
        ),
        "offline_shuffle": bool(OmegaConf.select(cfg, "training.offline_shuffle", default=False)),
        "offline_balance_key": str(
            OmegaConf.select(cfg, "training.offline_balance_key", default="none")
        ),
        "offline_balanced_labels": list(
            OmegaConf.select(cfg, "training.offline_balanced_labels", default=[])
        ),
        "offline_balance_quotas": dict(
            OmegaConf.to_container(
                OmegaConf.select(cfg, "training.offline_balance_quotas", default={}),
                resolve=True,
            )
        ),
        "offline_min_balanced_replay_passes": int(
            OmegaConf.select(
                cfg,
                "training.offline_min_balanced_replay_passes",
                default=0,
            )
        ),
        "offline_min_balanced_replay_labels": list(
            OmegaConf.select(
                cfg,
                "training.offline_min_balanced_replay_labels",
                default=[],
            )
        ),
        **_student_runtime_cfg(cfg),
        "recovery_integration": bool(
            OmegaConf.select(cfg, "training.recovery_integration", default=False)
        ),
        "recovery_routing_contract": _recovery_routing_contract(cfg),
        "recovery_teacher_sha256": OmegaConf.select(cfg, "training.recovery_teacher_sha256"),
        "teacher_obs_dim": int(cfg.teacher.obs_dim),
    }
    if dataset_path is not None:
        payload["dataset_path"] = str(dataset_path)
    if student_init_metadata:
        payload["student_init_checkpoint_path"] = str(student_init_metadata["path"])
        payload["student_init_checkpoint_sha256"] = student_init_metadata.get("checkpoint_sha256")
        if student_init_metadata.get("recovery_migration"):
            payload["recovery_migration"] = student_init_metadata["recovery_migration"]
        payload["student_init_agent_steps"] = int(student_init_metadata["agent_steps"])
        payload["student_init_optimizer_requested"] = bool(
            student_init_metadata.get("optimizer_requested", False)
        )
        payload["student_init_optimizer_loaded"] = bool(student_init_metadata["optimizer_loaded"])
    return payload


def _probe_result(
    cfg: DictConfig,
    *,
    dataset: Any,
    result: Any,
    distill_source: str,
    dataset_path: str | Path | None = None,
    student_init_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probe = {
        "distill_source": str(distill_source),
        "student_model_type": _student_model_type(cfg),
        "student_obs_shape": result.student_obs_shape,
        "teacher_obs_shape": result.teacher_obs_shape,
        "dataset_num_samples": dataset.num_samples,
        "dataset_student_obs_dim": dataset.student_obs_dim,
        "dataset_teacher_obs_dim": dataset.teacher_obs_dim,
        "dataset_metadata": dict(getattr(dataset, "metadata", {})),
        "student_action_shape": result.student_action_shape,
        "teacher_action_shape": result.teacher_action_shape,
        "teacher_action_requires_grad": result.teacher_action_requires_grad,
        "teacher_action_source": result.last_teacher_action_source,
        "student_grad_norm": result.last_student_grad_norm,
        "loss": result.last_loss,
        "behavior_loss": result.last_behavior_loss,
        "behavior_action_shape": result.last_behavior_action_shape,
        "behavior_action_source": result.last_behavior_action_source,
        "behavior_target_count": result.last_behavior_target_count,
        "aux_loss": result.last_aux_loss,
        "role_loss": result.last_role_loss,
        "role_target_count": result.last_role_target_count,
        "command_intent_loss": result.last_command_intent_loss,
        "command_intent_target_count": result.last_command_intent_target_count,
        "expert_usage": result.last_expert_usage,
        "route_entropy": result.last_route_entropy,
        "offline_balance_key": result.balance_key,
        "offline_batch_label_counts": result.batch_label_counts,
        "offline_last_balance_label_counts": result.last_balance_label_counts,
        "update_count": result.update_count,
        "samples_seen": result.samples_seen,
        "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else None,
        "performance_stage_observations": [
            observation.as_dict() for observation in result.performance_stage_observations
        ],
    }
    if isinstance(student_init_metadata, dict):
        probe["student_init_checkpoint_path"] = student_init_metadata.get("path")
        probe["student_init_agent_steps"] = student_init_metadata.get("agent_steps")
        probe["student_init_optimizer_requested"] = student_init_metadata.get("optimizer_requested")
        probe["student_init_optimizer_loaded"] = student_init_metadata.get("optimizer_loaded")
    if dataset_path is not None:
        probe["dataset_path"] = str(dataset_path)
    return probe


def _normalize_checkpoint_selector(selector: Any) -> str | None:
    if selector in (None, "", -1, "-1"):
        return None
    return str(selector)


def resolve_teacher_checkpoint(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
) -> tuple[Path | None, Path | None]:
    """Resolve the teacher checkpoint through shared training path semantics."""

    explicit_checkpoint_path = OmegaConf.select(cfg, "teacher.checkpoint_path")
    if explicit_checkpoint_path not in (None, ""):
        path = Path(str(explicit_checkpoint_path))
        resolved_path = path if path.is_absolute() else Path(root_dir) / path
        if not resolved_path.is_file():
            raise FileNotFoundError(f"teacher.checkpoint_path does not exist: {resolved_path}")
        return resolved_path, resolved_path.parent

    log_root = OmegaConf.select(
        cfg,
        "teacher.log_root",
        default=OmegaConf.select(cfg, "training.log_root"),
    )
    return resolve_task_checkpoint_path(
        root_dir,
        task_name=str(cfg.teacher.task_name),
        load_run=str(cfg.teacher.load_run),
        algo_log_name=str(cfg.teacher.algo_log_name),
        checkpoint=_normalize_checkpoint_selector(cfg.teacher.checkpoint),
        suffix=".pt",
        log_root=log_root,
    )


def build_distillation_trainer(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path,
    student_init_checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> BehaviorDistillationTrainer:
    """Load teacher, build student, and assemble one behavior distillation trainer."""

    validate_sac_teacher_checkpoint_contract(
        teacher_checkpoint,
        build_teacher_spec(cfg),
        device=device,
    )
    teacher = load_sac_teacher_policy(
        teacher_checkpoint,
        build_teacher_spec(cfg),
        device=device,
    )
    student = build_student_policy(cfg, device=device)
    optimizer = torch.optim.Adam(student.parameters(), lr=float(cfg.algo.learning_rate))
    student_init_metadata: dict[str, Any] = {}
    resolved_student_init_checkpoint = _resolve_optional_checkpoint_path(
        student_init_checkpoint,
        field_name="training.offline_init_checkpoint",
    )
    if resolved_student_init_checkpoint is not None:
        student_init_metadata = _load_student_init_checkpoint(
            student,
            optimizer,
            resolved_student_init_checkpoint,
            cfg=cfg,
            device=device,
            resume_optimizer=bool(
                OmegaConf.select(cfg, "training.offline_resume_optimizer", default=True)
            ),
        )
    return BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        loss_type=str(cfg.algo.loss_type),
        max_grad_norm=float(cfg.algo.max_grad_norm),
        aux_loss_coef=float(OmegaConf.select(cfg, "algo.aux_loss_coef", default=0.0)),
        role_loss_coef=float(OmegaConf.select(cfg, "algo.role_loss_coef", default=0.0)),
        role_expert_targets=dict(
            OmegaConf.to_container(
                OmegaConf.select(cfg, "algo.role_expert_targets", default={}),
                resolve=True,
            )
        ),
        command_intent_loss_coef=float(
            OmegaConf.select(cfg, "algo.command_intent_loss_coef", default=0.0)
        ),
        command_intent_expert_targets=dict(
            OmegaConf.to_container(
                OmegaConf.select(cfg, "algo.command_intent_expert_targets", default={}),
                resolve=True,
            )
        ),
        student_init_metadata=student_init_metadata,
        expert_behavior_loss_source=str(
            OmegaConf.select(cfg, "algo.expert_behavior_loss_source", default="auto")
        ),
    )


def run_fake_batch_update(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path,
    batch_size: int = 8,
    max_updates: int = 1,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run a bounded shape-valid offline distillation probe for the entrypoint."""

    torch.manual_seed(int(cfg.algo.seed))
    trainer = build_distillation_trainer(
        cfg,
        teacher_checkpoint=teacher_checkpoint,
        device=device,
    )
    dataset = make_fake_distillation_dataset(
        num_samples=int(batch_size) * int(max_updates),
        student_obs_dim=int(cfg.student.obs_dim),
        teacher_obs_dim=int(cfg.teacher.obs_dim),
        seed=int(cfg.algo.seed),
        device=device,
    )
    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=int(batch_size),
        max_updates=int(max_updates),
        checkpoint_path=checkpoint_path,
        teacher_metadata=_teacher_metadata(cfg, teacher_checkpoint),
        distill_runtime_cfg=_distill_runtime_cfg(cfg, distill_source="fake_probe"),
    )

    return _probe_result(cfg, dataset=dataset, result=result, distill_source="fake_probe")


def run_offline_dataset_update(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path,
    dataset_path: str | Path,
    batch_size: int | None = None,
    max_updates: int | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    auto_expand_replay_budget: bool = False,
    progress_callback: Callable[[int, int, Any], None] | None = None,
    performance_clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Run bounded offline updates from a saved distillation tensor dataset."""

    torch.manual_seed(int(cfg.algo.seed))
    resolved_batch_size = int(
        batch_size
        if batch_size is not None
        else OmegaConf.select(cfg, "training.offline_batch_size", default=256)
    )
    resolved_max_updates = int(
        max_updates
        if max_updates is not None
        else OmegaConf.select(cfg, "training.offline_max_updates", default=1)
    )
    trainer = build_distillation_trainer(
        cfg,
        teacher_checkpoint=teacher_checkpoint,
        student_init_checkpoint=OmegaConf.select(
            cfg,
            "training.offline_init_checkpoint",
            default=None,
        ),
        device=device,
    )
    student_init_metadata = dict(getattr(trainer, "student_init_metadata", {}))
    dataset = load_distillation_dataset(
        dataset_path,
        expected_student_obs_dim=int(cfg.student.obs_dim),
        expected_teacher_obs_dim=int(cfg.teacher.obs_dim),
        expected_teacher_action_dim=int(cfg.teacher.action_dim),
        device=device,
    )
    offline_balance_key = str(OmegaConf.select(cfg, "training.offline_balance_key", default="none"))
    offline_balanced_labels = list(
        OmegaConf.select(cfg, "training.offline_balanced_labels", default=[])
    )
    offline_balance_quotas = dict(
        OmegaConf.to_container(
            OmegaConf.select(cfg, "training.offline_balance_quotas", default={}),
            resolve=True,
        )
    )
    offline_replay_passes = int(
        OmegaConf.select(cfg, "training.offline_min_balanced_replay_passes", default=0)
    )
    offline_replay_labels = list(
        OmegaConf.select(cfg, "training.offline_min_balanced_replay_labels", default=[])
    )
    if auto_expand_replay_budget:
        resolved_max_updates = max(
            resolved_max_updates,
            required_balanced_replay_updates(
                dataset,
                balance_key=offline_balance_key,
                batch_size=resolved_batch_size,
                balanced_labels=offline_balanced_labels,
                balance_quotas=offline_balance_quotas,
                replay_labels=offline_replay_labels,
                replay_passes=offline_replay_passes,
            ),
        )
    progress_enabled = os.environ.get("UNILAB_DISTILL_PROGRESS", "0").lower() not in {
        "",
        "0",
        "false",
        "off",
    }
    progress_interval = int(os.environ.get("UNILAB_DISTILL_PROGRESS_INTERVAL", "0") or 0)
    if progress_callback is not None and progress_interval <= 0:
        progress_interval = max(1, resolved_max_updates // 20)
    if progress_enabled:
        print(
            "[distill-progress] "
            f"dataset={dataset_path} samples={dataset.num_samples} "
            f"updates={resolved_max_updates} batch_size={resolved_batch_size}",
            flush=True,
        )
    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=resolved_batch_size,
        max_updates=resolved_max_updates,
        checkpoint_path=checkpoint_path,
        teacher_metadata=_teacher_metadata(cfg, teacher_checkpoint),
        distill_runtime_cfg=_distill_runtime_cfg(
            cfg,
            distill_source="offline_dataset",
            dataset_path=dataset_path,
            student_init_metadata=student_init_metadata,
        ),
        repeat_dataset=bool(
            OmegaConf.select(cfg, "training.offline_repeat_dataset", default=False)
        ),
        shuffle=bool(OmegaConf.select(cfg, "training.offline_shuffle", default=False)),
        seed=int(cfg.algo.seed),
        balance_key=offline_balance_key,
        balanced_labels=offline_balanced_labels,
        balance_quotas=offline_balance_quotas,
        min_balanced_replay_passes=offline_replay_passes,
        min_balanced_replay_labels=offline_replay_labels,
        save_optimizer_state=bool(
            OmegaConf.select(cfg, "training.offline_save_optimizer", default=True)
        ),
        progress_interval=(
            progress_interval if progress_enabled or progress_callback is not None else 0
        ),
        progress_callback=progress_callback,
        performance_clock=performance_clock,
    )
    return _probe_result(
        cfg,
        dataset=dataset,
        result=result,
        distill_source="offline_dataset",
        dataset_path=dataset_path,
        student_init_metadata=student_init_metadata,
    )


def _multitask_sources(cfg: DictConfig) -> list[dict[str, Any]]:
    sources = OmegaConf.to_container(
        OmegaConf.select(cfg, "training.multitask_sources", default=[]),
        resolve=True,
    )
    if not isinstance(sources, list):
        raise ValueError("training.multitask_sources must be a list")
    return [dict(cast(dict[str, Any], source)) for source in sources]


def _optional_int_cfg(cfg: DictConfig, path: str) -> int | None:
    value = OmegaConf.select(cfg, path)
    if value in (None, ""):
        return None
    return int(value)


_ROLE_DATA_ASSEMBLY_DEVICE = "cpu"


def run_multitask_dataset_assembly(
    cfg: DictConfig,
    *,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Merge saved role-specific datasets into one CPU-owned cached dataset."""

    resolved_dataset_path = dataset_path or OmegaConf.select(
        cfg,
        "training.multitask_dataset_path",
    )
    if resolved_dataset_path in (None, ""):
        raise ValueError("training.multitask_dataset_path must be set")
    dataset = build_multitask_distillation_dataset(
        _multitask_sources(cfg),
        expected_student_obs_dim=_optional_int_cfg(
            cfg,
            "training.multitask_expected_student_obs_dim",
        ),
        expected_teacher_obs_dim=_optional_int_cfg(
            cfg,
            "training.multitask_expected_teacher_obs_dim",
        ),
        expected_teacher_action_dim=_optional_int_cfg(
            cfg,
            "training.multitask_expected_teacher_action_dim",
        ),
        device=_ROLE_DATA_ASSEMBLY_DEVICE,
    )
    save_distillation_dataset(resolved_dataset_path, dataset)
    return {
        "distill_source": "multitask_adapter",
        "dataset_path": str(resolved_dataset_path),
        "aggregate_assembly_device": _ROLE_DATA_ASSEMBLY_DEVICE,
        "dataset_num_samples": dataset.num_samples,
        "dataset_student_obs_dim": dataset.student_obs_dim,
        "dataset_teacher_obs_dim": dataset.teacher_obs_dim,
        "dataset_teacher_action_dim": dataset.teacher_action_dim,
        "dataset_metadata": dict(dataset.metadata),
        "student_obs_shape": tuple(dataset.student_obs.shape),
        "teacher_obs_shape": tuple(dataset.teacher_obs.shape),
        "teacher_actions_shape": (
            None if dataset.teacher_actions is None else tuple(dataset.teacher_actions.shape)
        ),
        "source_roles": list(dataset.metadata["source_roles"]),
        "source_sample_counts": list(dataset.metadata["source_sample_counts"]),
    }


def _resolve_formal_run_dir(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
) -> Path:
    explicit = OmegaConf.select(cfg, "training.formal_run_dir")
    if explicit not in (None, ""):
        path = Path(str(explicit))
        return path if path.is_absolute() else Path(root_dir) / path

    log_dir = OmegaConf.select(cfg, "training.log_dir")
    if log_dir not in (None, ""):
        path = Path(str(log_dir))
        return path if path.is_absolute() else Path(root_dir) / path

    log_root = OmegaConf.select(cfg, "training.log_root")
    root = Path(str(log_root)) if log_root not in (None, "") else Path(root_dir) / "logs"
    if not root.is_absolute():
        root = Path(root_dir) / root
    run_name = OmegaConf.select(cfg, "training.formal_run_name")
    if run_name in (None, ""):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{OmegaConf.select(cfg, 'training.sim_backend', default='mujoco')}"
    return root / str(cfg.algo.algo_log_name) / str(cfg.training.task_name) / str(run_name)


def _expected_samples_seen_for_offline_run(
    cfg: DictConfig,
    *,
    dataset_path: str | Path,
    batch_size: int,
    max_updates: int,
    device: str | torch.device = "cpu",
) -> int:
    dataset = load_distillation_dataset(
        dataset_path,
        expected_student_obs_dim=int(cfg.student.obs_dim),
        expected_teacher_obs_dim=int(cfg.teacher.obs_dim),
        expected_teacher_action_dim=int(cfg.teacher.action_dim),
        device=device,
    )
    if str(OmegaConf.select(cfg, "training.offline_balance_key", default="none")) != "none":
        return int(batch_size) * int(max_updates)
    if bool(OmegaConf.select(cfg, "training.offline_repeat_dataset", default=False)):
        samples_seen = 0
        cursor = 0
        for _ in range(int(max_updates)):
            if cursor >= dataset.num_samples:
                cursor = 0
            end = min(dataset.num_samples, cursor + int(batch_size))
            samples_seen += end - cursor
            cursor = end
        return samples_seen
    return min(int(dataset.num_samples), int(batch_size) * int(max_updates))


def run_formal_offline_dataset_update(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path,
    dataset_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    batch_size: int | None = None,
    max_updates: int | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run the saved-dataset distillation path with run metadata and checkpoint layout."""

    resolved_dataset_path = dataset_path or OmegaConf.select(cfg, "training.offline_dataset_path")
    if resolved_dataset_path in (None, ""):
        raise ValueError("training.offline_dataset_path must be set for formal distill runs")
    resolved_batch_size = int(
        batch_size
        if batch_size is not None
        else OmegaConf.select(cfg, "training.offline_batch_size", default=256)
    )
    resolved_max_updates = int(
        max_updates
        if max_updates is not None
        else OmegaConf.select(cfg, "training.offline_max_updates", default=1)
    )
    resolved_run_dir = (
        Path(run_dir) if run_dir is not None else _resolve_formal_run_dir(cfg, root_dir=ROOT_DIR)
    )
    samples_seen = _expected_samples_seen_for_offline_run(
        cfg,
        dataset_path=resolved_dataset_path,
        batch_size=resolved_batch_size,
        max_updates=resolved_max_updates,
        device=device,
    )
    checkpoint_path = resolved_run_dir / f"model_{samples_seen}.pt"

    tracker = ExperimentTracker(
        root_dir=ROOT_DIR,
        log_dir=resolved_run_dir,
        algo_name=str(cfg.algo.algo_log_name),
        task_name=str(cfg.training.task_name),
        sim_backend=str(OmegaConf.select(cfg, "training.sim_backend", default="mujoco")),
        training_cfg=cfg.training,
        full_cfg=cfg,
        device=str(device),
    )
    tracker.start()
    try:
        probe = run_offline_dataset_update(
            cfg,
            teacher_checkpoint=teacher_checkpoint,
            dataset_path=resolved_dataset_path,
            batch_size=resolved_batch_size,
            max_updates=resolved_max_updates,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        tracker.update_summary(
            {
                "status": "completed",
                "distill_source": "formal_offline_dataset",
                "dataset_path": str(resolved_dataset_path),
                "checkpoint_path": str(checkpoint_path),
                "update_count": int(probe["update_count"]),
                "samples_seen": int(probe["samples_seen"]),
                "loss": float(probe["loss"]),
                "behavior_loss": float(probe["behavior_loss"]),
                "aux_loss": float(probe["aux_loss"]),
                "student_grad_norm": float(probe["student_grad_norm"]),
            }
        )
    finally:
        tracker.finish()

    probe.update(
        {
            "distill_source": "formal_offline_dataset",
            "run_dir": str(resolved_run_dir),
            "run_config_path": str(resolved_run_dir / "run_config.json"),
            "run_summary_path": str(resolved_run_dir / "run_summary.json"),
            "checkpoint_path": str(checkpoint_path),
        }
    )
    return probe


def _collect_action_mode(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "training.collect_action_mode", default="zero"))


def _distill_device(cfg: DictConfig) -> str:
    device = OmegaConf.select(cfg, "training.device", default="cpu")
    return "cpu" if device in (None, "") else str(device)


def _teacher_task_name_for_collection(cfg: DictConfig) -> str:
    teacher_task_name = str(
        OmegaConf.select(
            cfg,
            "teacher.task_name",
            default=str(OmegaConf.select(cfg, "training.task_name")),
        )
    )
    checkpoint_path = OmegaConf.select(cfg, "teacher.checkpoint_path")
    if checkpoint_path in (None, ""):
        return teacher_task_name
    checkpoint_parts = set(Path(str(checkpoint_path)).parts)
    hinted_task_names = sorted(_DISTILL_TASK_NAME_HINTS & checkpoint_parts)
    if not hinted_task_names:
        return teacher_task_name
    if len(hinted_task_names) > 1:
        raise ValueError(
            f"teacher.checkpoint_path contains multiple distill task hints: {hinted_task_names}"
        )
    hinted_task_name = hinted_task_names[0]
    if teacher_task_name != hinted_task_name:
        default_task_name = str(OmegaConf.select(cfg, "training.task_name"))
        if teacher_task_name != default_task_name:
            raise ValueError(
                "teacher.task_name conflicts with teacher.checkpoint_path task hint: "
                f"teacher.task_name={teacher_task_name!r}, checkpoint_hint={hinted_task_name!r}"
            )
        teacher_task_name = hinted_task_name
    return teacher_task_name


def _expected_owner_command_sample_filter(cfg: DictConfig) -> str | None:
    task_name = str(OmegaConf.select(cfg, "training.task_name"))
    teacher_task_name = _teacher_task_name_for_collection(cfg)
    if task_name == "G1WalkFlat" and teacher_task_name == "G1StandStill":
        return "inactive"
    target_height_info_key = OmegaConf.select(cfg, "training.collect_target_height_info_key")
    if target_height_info_key not in (None, ""):
        return _HEIGHT_OWNER_COMMAND_SAMPLE_FILTERS.get(task_name)
    return _OWNER_COMMAND_SAMPLE_FILTERS.get(task_name)


def _require_owner_command_sample_filter(cfg: DictConfig) -> None:
    expected_filter = _expected_owner_command_sample_filter(cfg)
    if expected_filter is None:
        return
    actual_filter = str(
        OmegaConf.select(cfg, "training.collect_command_sample_filter", default="none")
    )
    if actual_filter != expected_filter:
        task_name = str(OmegaConf.select(cfg, "training.task_name"))
        raise ValueError(
            f"{task_name} requires training.collect_command_sample_filter={expected_filter} "
            f"for command-intent distillation collection; got {actual_filter!r}"
        )


def _require_collected_command_intent_contract(cfg: DictConfig, dataset: Any) -> None:
    expected_filter = _expected_owner_command_sample_filter(cfg)
    if expected_filter is None:
        return
    expected_intent = "active" if expected_filter == "active" else "inactive"
    actual_filter = str(dataset.metadata.get("command_sample_filter", "none"))
    if actual_filter != expected_filter:
        raise ValueError(
            "collected dataset command filter mismatch: "
            f"expected {expected_filter!r}, got {actual_filter!r}"
        )
    if dataset.commands is None:
        raise ValueError("owner command-intent collection must persist dataset.commands")
    if dataset.command_intents is None:
        raise ValueError("owner command-intent collection must persist dataset.command_intents")
    intent_counts = dict(dataset.metadata.get("command_intent_counts") or {})
    expected_counts = {expected_intent: int(dataset.num_samples)}
    if intent_counts != expected_counts:
        raise ValueError(
            "collected dataset command intent mismatch: "
            f"expected {expected_counts}, got {intent_counts}"
        )
    seen_samples = dataset.metadata.get("command_seen_samples")
    selected_samples = dataset.metadata.get("command_selected_samples")
    if seen_samples is None or selected_samples is None:
        raise ValueError(
            "owner command-intent collection must record command_seen/selected samples"
        )
    if int(selected_samples) < int(dataset.num_samples):
        raise ValueError(
            "owner command-intent collection selected too few samples: "
            f"selected={selected_samples}, dataset_num_samples={dataset.num_samples}"
        )


def _require_collected_target_height_contract(cfg: DictConfig, dataset: Any) -> None:
    info_key = OmegaConf.select(cfg, "training.collect_target_height_info_key")
    if info_key in (None, ""):
        return
    if str(info_key) != "height_commands":
        raise ValueError(
            "99-D height collection requires "
            "training.collect_target_height_info_key=height_commands"
        )
    target_height = dataset.target_height
    if target_height is None:
        raise ValueError("99-D height collection must persist dataset.target_height")
    if tuple(target_height.shape) != (int(dataset.num_samples), 1):
        raise ValueError(
            "collected target_height shape mismatch: "
            f"expected={(int(dataset.num_samples), 1)} got={tuple(target_height.shape)}"
        )
    if int(dataset.student_obs_dim) != 99 or int(dataset.teacher_obs_dim) != 99:
        raise ValueError("height-aware role data requires 99-D student and teacher observations")
    if not torch.equal(dataset.student_obs[:, 96:97], target_height):
        raise ValueError(
            "student observation target-height column does not match dataset.target_height"
        )
    if not torch.equal(dataset.teacher_obs[:, 96:97], target_height):
        raise ValueError(
            "teacher observation target-height column does not match dataset.target_height"
        )
    if str(OmegaConf.select(cfg, "training.task_name")) == "G1WalkHeight":
        if bool(OmegaConf.select(cfg, "env.commands.random_height_during_walking")):
            raise ValueError("Walk role must use fixed nominal target height")
        nominal_height = float(OmegaConf.select(cfg, "env.commands.default_height"))
        if not torch.equal(target_height, torch.full_like(target_height, nominal_height)):
            raise ValueError("Walk role target height must stay at its nominal owner-config value")


def _collect_command_distribution_overrides(cfg: DictConfig) -> dict[str, Any]:
    expected_filter = _expected_owner_command_sample_filter(cfg)
    actual_filter = str(
        OmegaConf.select(cfg, "training.collect_command_sample_filter", default="none")
    )
    task_name = str(OmegaConf.select(cfg, "training.task_name"))
    teacher_task_name = _teacher_task_name_for_collection(cfg)
    if (
        task_name == "G1WalkFlat"
        and teacher_task_name == "G1StandStill"
        and expected_filter == "inactive"
        and actual_filter == "inactive"
    ):
        return {
            "env.commands.vel_limit": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "env.commands.transition_vel_limit": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "env.commands.rel_standing_envs": 1.0,
            "env.commands.rel_transition_envs": 0.0,
            "env.commands.small_xy_threshold": 0.0,
        }
    return {}


def _apply_collect_command_distribution_overrides(cfg: DictConfig) -> dict[str, Any]:
    overrides = _collect_command_distribution_overrides(cfg)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    return overrides


def _require_teacher_policy_collection_route(cfg: DictConfig) -> None:
    """Require an explicit legacy-98-D or height-aware-99-D owner route."""

    task_name = str(OmegaConf.select(cfg, "training.task_name"))
    teacher_task_name = _teacher_task_name_for_collection(cfg)
    legacy_tasks = {"G1WalkFlat", "G1StandStill"}
    height_tasks = {"G1WalkHeight", "G1StandHeight", "G1Recovery"}
    if task_name not in legacy_tasks | height_tasks:
        raise ValueError(
            "teacher target collection only supports explicit G1 flat/stand or "
            "height-aware owner routes"
        )
    cross_stand_teacher = task_name == "G1WalkFlat" and teacher_task_name == "G1StandStill"
    if teacher_task_name != task_name and not cross_stand_teacher:
        raise ValueError(
            "teacher target collection requires teacher.task_name to match training.task_name, "
            "except G1WalkFlat inactive collection may use a G1StandStill teacher"
        )
    if cross_stand_teacher:
        actual_filter = str(
            OmegaConf.select(cfg, "training.collect_command_sample_filter", default="none")
        )
        if actual_filter != "inactive":
            raise ValueError(
                "G1WalkFlat collection with a G1StandStill teacher requires "
                "training.collect_command_sample_filter=inactive"
            )
    expected_obs_dim = 98 if task_name in legacy_tasks else 99
    if int(cfg.teacher.obs_dim) != expected_obs_dim or int(cfg.student.obs_dim) != expected_obs_dim:
        raise ValueError(
            f"teacher target collection requires {expected_obs_dim}-D teacher and student obs"
        )
    if str(OmegaConf.select(cfg, "training.collect_teacher_obs_key", default="obs")) != "obs":
        raise ValueError("teacher target collection requires training.collect_teacher_obs_key=obs")
    if (
        str(OmegaConf.select(cfg, "training.collect_teacher_projection", default="identity"))
        != "identity"
    ):
        raise ValueError("teacher target collection requires identity teacher projection")
    required_student_projection = (
        "g1_recovery_to_walk_99_v1" if task_name == "G1Recovery" else "identity"
    )
    if (
        str(OmegaConf.select(cfg, "training.collect_student_projection", default="identity"))
        != required_student_projection
    ):
        raise ValueError(
            f"teacher target collection requires {required_student_projection} student projection"
        )
    if OmegaConf.select(cfg, "training.collect_student_drop_index") is not None:
        raise ValueError("teacher target collection does not support collect_student_drop_index")
    if OmegaConf.select(cfg, "training.collect_action_seed") is not None:
        raise ValueError("teacher target collection does not use training.collect_action_seed")
    observes_height = bool(
        OmegaConf.select(cfg, "env.commands.observe_height_command", default=False)
    )
    target_height_info_key = OmegaConf.select(cfg, "training.collect_target_height_info_key")
    if task_name in legacy_tasks:
        if observes_height or target_height_info_key not in (None, ""):
            raise ValueError("98-D teacher target collection must not use height commands")
    elif not observes_height or str(target_height_info_key) != "height_commands":
        raise ValueError(
            "99-D teacher target collection requires observed height_commands and "
            "training.collect_target_height_info_key=height_commands"
        )


def _resolve_collect_rollout_checkpoint(cfg: DictConfig) -> Path:
    checkpoint_path = OmegaConf.select(cfg, "training.collect_rollout_checkpoint_path")
    if checkpoint_path in (None, ""):
        raise ValueError(
            "training.collect_rollout_checkpoint_path must be set when "
            "training.collect_action_mode=student_policy"
        )
    path = Path(str(checkpoint_path))
    resolved_path = path if path.is_absolute() else ROOT_DIR / path
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"training.collect_rollout_checkpoint_path does not exist: {resolved_path}"
        )
    return resolved_path


def run_collect_dataset(
    cfg: DictConfig,
    *,
    dataset_path: str | Path | None = None,
    create_env_fn: Any | None = None,
    env_cfg_override_fn: Any | None = None,
    performance_clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Collect and save a small live-env distillation observation dataset."""

    request_start = None if performance_clock is None else float(performance_clock())
    resolved_dataset_path = dataset_path or OmegaConf.select(cfg, "training.collect_dataset_path")
    if resolved_dataset_path in (None, ""):
        raise ValueError("training.collect_dataset_path must be set for live dataset collection")

    action_mode = _collect_action_mode(cfg)
    _require_owner_command_sample_filter(cfg)
    command_distribution_overrides = _apply_collect_command_distribution_overrides(cfg)
    teacher_policy = None
    rollout_policy = None
    teacher_policy_checkpoint_path: Path | None = None
    rollout_policy_checkpoint_path: Path | None = None
    if action_mode in {"teacher_policy", "student_policy"}:
        _require_teacher_policy_collection_route(cfg)
        teacher_policy_checkpoint_path, _run_dir = resolve_teacher_checkpoint(
            cfg, root_dir=ROOT_DIR
        )
        if teacher_policy_checkpoint_path is None:
            raise FileNotFoundError(
                "No SAC teacher checkpoint resolved for teacher target collection. "
                "Set teacher.load_run/teacher.checkpoint or training.log_root."
            )
        teacher_policy = load_sac_teacher_policy(
            teacher_policy_checkpoint_path,
            build_teacher_spec(cfg),
            device=_distill_device(cfg),
        )
    if action_mode == "student_policy":
        rollout_policy_checkpoint_path = _resolve_collect_rollout_checkpoint(cfg)
        loaded_rollout_policy = load_distillation_student_policy(
            rollout_policy_checkpoint_path,
            device=_distill_device(cfg),
        )
        if int(loaded_rollout_policy.obs_dim) != int(cfg.student.obs_dim):
            raise ValueError(
                "student_policy rollout obs dim mismatch: "
                f"checkpoint={loaded_rollout_policy.obs_dim} cfg.student.obs_dim={int(cfg.student.obs_dim)}"
            )
        if int(loaded_rollout_policy.action_dim) != int(cfg.student.action_dim):
            raise ValueError(
                "student_policy rollout action dim mismatch: "
                f"checkpoint={loaded_rollout_policy.action_dim} "
                f"cfg.student.action_dim={int(cfg.student.action_dim)}"
            )
        rollout_policy = loaded_rollout_policy.policy
        if bool(OmegaConf.select(cfg, "training.recovery_integration", default=False)):
            from unilab.algos.torch.distill.recovery_integration import (
                RoleExpertPolicy,
                validate_recovery_student,
            )

            validate_recovery_student(rollout_policy, loaded_rollout_policy.distill_runtime_cfg)
            role = str(OmegaConf.select(cfg, "training.collect_role_label"))
            rollout_policy = RoleExpertPolicy(rollout_policy, role)

    if create_env_fn is None:
        ensure_registries()
        create_env_fn = create_env
    if env_cfg_override_fn is None:
        env_cfg_override_fn = lambda cfg: BackendAdapter(  # noqa: E731
            cfg,
            root_dir=ROOT_DIR,
            algo_name="distill",
        ).build_task_env_cfg_override()

    env = create_env_fn(
        cfg,
        num_envs=int(OmegaConf.select(cfg, "training.collect_num_envs", default=1)),
        env_cfg_override=env_cfg_override_fn(cfg),
        sim_backend=str(OmegaConf.select(cfg, "training.sim_backend", default="mujoco")),
        task_name=str(OmegaConf.select(cfg, "training.task_name")),
    )
    cold_start_seconds = (
        None
        if performance_clock is None or request_start is None
        else float(performance_clock()) - request_start
    )
    request_observations: tuple[DistillationStageObservation, ...] | None = None
    try:
        drop_index = OmegaConf.select(cfg, "training.collect_student_drop_index")
        collect_max_env_steps = OmegaConf.select(cfg, "training.collect_max_env_steps")
        metadata = {
            "task_name": str(OmegaConf.select(cfg, "training.task_name")),
            "sim_backend": str(OmegaConf.select(cfg, "training.sim_backend", default="mujoco")),
        }
        workflow_scenario = OmegaConf.select(cfg, "training.collect_workflow_scenario")
        if workflow_scenario not in (None, ""):
            metadata["workflow_scenario"] = str(workflow_scenario)
        if teacher_policy_checkpoint_path is not None:
            metadata["teacher_policy_checkpoint_path"] = str(teacher_policy_checkpoint_path)
        if rollout_policy_checkpoint_path is not None:
            metadata["rollout_policy_checkpoint_path"] = str(rollout_policy_checkpoint_path)
        if command_distribution_overrides:
            metadata["command_distribution_overrides"] = command_distribution_overrides
        dataset = collect_distillation_dataset_from_env(
            env,
            num_samples=int(OmegaConf.select(cfg, "training.collect_num_samples", default=1024)),
            expected_student_obs_dim=int(cfg.student.obs_dim),
            expected_teacher_obs_dim=int(cfg.teacher.obs_dim),
            teacher_obs_key=str(
                OmegaConf.select(cfg, "training.collect_teacher_obs_key", default="obs")
            ),
            teacher_projection=str(
                OmegaConf.select(cfg, "training.collect_teacher_projection", default="identity")
            ),
            student_projection=str(
                OmegaConf.select(cfg, "training.collect_student_projection", default="identity")
            ),
            student_drop_index=None if drop_index is None else int(drop_index),
            action_mode=action_mode,
            action_seed=OmegaConf.select(cfg, "training.collect_action_seed"),
            teacher_policy=teacher_policy,
            rollout_policy=rollout_policy,
            command_sample_filter=str(
                OmegaConf.select(cfg, "training.collect_command_sample_filter", default="none")
            ),
            command_info_key=str(
                OmegaConf.select(cfg, "training.collect_command_info_key", default="commands")
            ),
            target_height_info_key=OmegaConf.select(cfg, "training.collect_target_height_info_key"),
            command_xy_threshold=float(
                OmegaConf.select(cfg, "training.collect_command_xy_threshold", default=0.05)
            ),
            command_yaw_threshold=float(
                OmegaConf.select(cfg, "training.collect_command_yaw_threshold", default=0.05)
            ),
            max_env_steps=None if collect_max_env_steps is None else int(collect_max_env_steps),
            role_label=OmegaConf.select(cfg, "training.collect_role_label"),
            metadata=metadata,
            performance_clock=performance_clock,
        )
        _require_collected_command_intent_contract(cfg, dataset)
        _require_collected_target_height_contract(cfg, dataset)
        write_start = None if performance_clock is None else float(performance_clock())
        save_distillation_dataset(resolved_dataset_path, dataset)
        if performance_clock is not None:
            assert request_start is not None
            assert cold_start_seconds is not None
            assert write_start is not None
            artifact_write_seconds = float(performance_clock()) - write_start
            collector_payloads = dataset.metadata.get("performance_stage_observations")
            if not isinstance(collector_payloads, list):
                raise ValueError("legacy collector performance observations are missing")
            collector_observations = tuple(
                DistillationStageObservation.from_dict(payload) for payload in collector_payloads
            )
            collector_stages = tuple(item.stage for item in collector_observations)
            if collector_stages != COLLECTOR_REQUEST_STAGE_NAMES:
                raise ValueError(
                    "legacy collector performance stage order mismatch: "
                    f"expected={COLLECTOR_REQUEST_STAGE_NAMES} "
                    f"observed={collector_stages}"
                )
            total_elapsed_seconds = float(performance_clock()) - request_start
            env_steps = int(dataset.metadata.get("env_steps", 0))
            request_observations = (
                DistillationStageObservation(
                    stage="cold_start",
                    duration_seconds=cold_start_seconds,
                    row_count=0,
                    env_step_count=0,
                    success=True,
                    error=None,
                    cleanup_state="not_applicable",
                ),
                *collector_observations,
                DistillationStageObservation(
                    stage="artifact_write",
                    duration_seconds=artifact_write_seconds,
                    row_count=dataset.num_samples,
                    env_step_count=0,
                    success=True,
                    error=None,
                    cleanup_state="not_applicable",
                ),
                DistillationStageObservation(
                    stage="total_elapsed",
                    duration_seconds=total_elapsed_seconds,
                    row_count=dataset.num_samples,
                    env_step_count=env_steps,
                    success=True,
                    error=None,
                    cleanup_state="pending",
                ),
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    result = {
        "distill_source": "live_env_rollout",
        "dataset_path": str(resolved_dataset_path),
        "dataset_num_samples": dataset.num_samples,
        "dataset_student_obs_dim": dataset.student_obs_dim,
        "dataset_teacher_obs_dim": dataset.teacher_obs_dim,
        "dataset_metadata": dict(dataset.metadata),
        "student_obs_shape": tuple(dataset.student_obs.shape),
        "teacher_obs_shape": tuple(dataset.teacher_obs.shape),
        "collect_num_envs": int(OmegaConf.select(cfg, "training.collect_num_envs", default=1)),
        "collect_action_mode": action_mode,
        "collect_action_seed": OmegaConf.select(cfg, "training.collect_action_seed"),
        "collect_action_abs_max": float(dataset.metadata.get("action_abs_max", 0.0)),
        "teacher_policy_checkpoint_path": (
            str(teacher_policy_checkpoint_path)
            if teacher_policy_checkpoint_path is not None
            else None
        ),
        "rollout_policy_checkpoint_path": (
            str(rollout_policy_checkpoint_path)
            if rollout_policy_checkpoint_path is not None
            else None
        ),
        "teacher_obs_key": str(
            OmegaConf.select(cfg, "training.collect_teacher_obs_key", default="obs")
        ),
        "teacher_projection": str(
            OmegaConf.select(cfg, "training.collect_teacher_projection", default="identity")
        ),
        "student_projection": str(
            OmegaConf.select(cfg, "training.collect_student_projection", default="identity")
        ),
        "student_drop_index": None if drop_index is None else int(drop_index),
        "collect_command_sample_filter": str(dataset.metadata.get("command_sample_filter", "none")),
        "collect_command_seen_samples": dataset.metadata.get("command_seen_samples"),
        "collect_command_selected_samples": dataset.metadata.get("command_selected_samples"),
        "collect_command_intent_counts": dataset.metadata.get("command_intent_counts"),
        "collect_target_height_shape": (
            None if dataset.target_height is None else tuple(dataset.target_height.shape)
        ),
        "collect_command_distribution_overrides": dataset.metadata.get(
            "command_distribution_overrides"
        ),
    }
    if request_observations is not None:
        result["performance_metrics_schema_version"] = DISTILLATION_METRICS_SCHEMA_VERSION
        result["performance_stage_observations"] = [
            observation.as_dict() for observation in request_observations
        ]
    return result


def run_online_dagger_update(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path,
    create_env_fn: Any | None = None,
    env_cfg_override_fn: Any | None = None,
) -> dict[str, Any]:
    """Assemble the iterative student-rollout DAgger owner loop."""

    _require_owner_command_sample_filter(cfg)
    _require_teacher_policy_collection_route(cfg)
    command_distribution_overrides = _apply_collect_command_distribution_overrides(cfg)
    init_checkpoint = OmegaConf.select(cfg, "training.offline_init_checkpoint")
    if init_checkpoint in (None, ""):
        raise ValueError("training.offline_init_checkpoint must be set for online DAgger")
    output_checkpoint = OmegaConf.select(cfg, "training.dagger_checkpoint")
    if output_checkpoint in (None, ""):
        raise ValueError("training.dagger_checkpoint must be set for online DAgger")
    output_checkpoint = Path(str(output_checkpoint))

    role_label = OmegaConf.select(cfg, "training.dagger_role_label")
    if float(OmegaConf.select(cfg, "algo.role_loss_coef", default=0.0)) > 0.0 and role_label in (
        None,
        "",
    ):
        raise ValueError("training.dagger_role_label is required when algo.role_loss_coef > 0")

    device = _distill_device(cfg)
    trainer = build_distillation_trainer(
        cfg,
        teacher_checkpoint=teacher_checkpoint,
        student_init_checkpoint=init_checkpoint,
        device=device,
    )
    if create_env_fn is None:
        ensure_registries()
        create_env_fn = create_env
    if env_cfg_override_fn is None:
        env_cfg_override_fn = lambda cfg: BackendAdapter(  # noqa: E731
            cfg,
            root_dir=ROOT_DIR,
            algo_name="distill",
        ).build_task_env_cfg_override()

    env = create_env_fn(
        cfg,
        num_envs=int(OmegaConf.select(cfg, "training.collect_num_envs", default=1)),
        env_cfg_override=env_cfg_override_fn(cfg),
        sim_backend=str(OmegaConf.select(cfg, "training.sim_backend", default="mujoco")),
        task_name=str(OmegaConf.select(cfg, "training.task_name")),
    )
    try:
        drop_index = OmegaConf.select(cfg, "training.collect_student_drop_index")
        max_env_steps = OmegaConf.select(cfg, "training.collect_max_env_steps")
        result = run_iterative_dagger_updates(
            env,
            trainer=trainer,
            num_iterations=int(OmegaConf.select(cfg, "training.dagger_iterations", default=8)),
            samples_per_iteration=int(
                OmegaConf.select(cfg, "training.dagger_samples_per_iteration", default=65536)
            ),
            batch_size=int(OmegaConf.select(cfg, "training.dagger_batch_size", default=512)),
            updates_per_iteration=int(
                OmegaConf.select(cfg, "training.dagger_updates_per_iteration", default=128)
            ),
            expected_student_obs_dim=int(cfg.student.obs_dim),
            expected_teacher_obs_dim=int(cfg.teacher.obs_dim),
            teacher_obs_key=str(
                OmegaConf.select(cfg, "training.collect_teacher_obs_key", default="obs")
            ),
            teacher_projection=str(
                OmegaConf.select(cfg, "training.collect_teacher_projection", default="identity")
            ),
            student_projection=str(
                OmegaConf.select(cfg, "training.collect_student_projection", default="identity")
            ),
            student_drop_index=None if drop_index is None else int(drop_index),
            command_sample_filter=str(
                OmegaConf.select(cfg, "training.collect_command_sample_filter", default="none")
            ),
            command_info_key=str(
                OmegaConf.select(cfg, "training.collect_command_info_key", default="commands")
            ),
            target_height_info_key=OmegaConf.select(cfg, "training.collect_target_height_info_key"),
            command_xy_threshold=float(
                OmegaConf.select(cfg, "training.collect_command_xy_threshold", default=0.05)
            ),
            command_yaw_threshold=float(
                OmegaConf.select(cfg, "training.collect_command_yaw_threshold", default=0.05)
            ),
            max_env_steps=None if max_env_steps is None else int(max_env_steps),
            role_label=None if role_label in (None, "") else str(role_label),
            shuffle=bool(OmegaConf.select(cfg, "training.dagger_shuffle", default=True)),
            seed=int(cfg.algo.seed),
            balance_key=str(OmegaConf.select(cfg, "training.dagger_balance_key", default="none")),
            balanced_labels=list(
                OmegaConf.select(cfg, "training.dagger_balanced_labels", default=[])
            ),
            checkpoint_path=output_checkpoint,
            teacher_metadata=_teacher_metadata(cfg, teacher_checkpoint),
            distill_runtime_cfg=_distill_runtime_cfg(
                cfg,
                distill_source="iterative_dagger",
                student_init_metadata=dict(getattr(trainer, "student_init_metadata", {})),
            ),
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    last_result = result.iteration_results[-1]
    return {
        "distill_source": "iterative_dagger",
        "iteration_count": result.iteration_count,
        "update_count": result.update_count,
        "samples_collected": result.samples_collected,
        "samples_seen": result.samples_seen,
        "loss": last_result.last_loss,
        "checkpoint_path": str(result.checkpoint_path),
        "role_label": role_label,
        "command_sample_filter": str(
            OmegaConf.select(cfg, "training.collect_command_sample_filter", default="none")
        ),
        "command_distribution_overrides": command_distribution_overrides,
    }


def run_single_entry_workflow(
    cfg: DictConfig,
    *,
    persistent_scenario_collector_factory: Callable[..., Any] | None = None,
    performance_clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Adapt role owner configs to the distillation workflow stage owner."""

    execution_mode = str(
        OmegaConf.select(
            cfg,
            "training.workflow.execution_mode",
            default="legacy",
        )
    )
    if execution_mode not in {"legacy", "persistent_async"}:
        raise ValueError(
            "training.workflow.execution_mode must be 'legacy' or "
            f"'persistent_async', got {execution_mode!r}"
        )
    if execution_mode == "legacy" and persistent_scenario_collector_factory is not None:
        raise ValueError("legacy execution_mode forbids persistent_scenario_collector_factory")

    if (
        bool(OmegaConf.select(cfg, "training.recovery_integration", default=False))
        and execution_mode != "legacy"
    ):
        raise ValueError("recovery integration currently requires execution_mode=legacy")
    entries = _workflow_role_entries(cfg)
    recovery_enabled = bool(OmegaConf.select(cfg, "training.recovery_integration", default=False))
    has_recovery = any(entry.get("role") == "recovery" for entry in entries)
    if has_recovery != recovery_enabled:
        raise ValueError("recovery role and recovery_integration must be activated together")
    if recovery_enabled:
        from unilab.algos.torch.distill.recovery_integration import validate_recovery_student

        init_path = _resolve_optional_checkpoint_path(
            OmegaConf.select(cfg, "training.offline_init_checkpoint"),
            field_name="training.offline_init_checkpoint",
        )
        if init_path is None:
            raise ValueError(
                "recovery integration requires an explicit migrated student checkpoint"
            )
        loaded_init = load_distillation_student_policy(init_path, device="cpu")
        validate_recovery_student(loaded_init.policy, loaded_init.distill_runtime_cfg)
        _validate_student_init_runtime_cfg(
            cfg, runtime_cfg=loaded_init.distill_runtime_cfg, checkpoint_path=init_path
        )
        if (
            cfg.student.num_experts != 3
            or str(cfg.algo.expert_behavior_loss_source) != "role"
            or float(cfg.algo.command_intent_loss_coef) != 0.0
            or dict(cfg.algo.role_expert_targets) != {"walk": 0, "stand_height": 1, "recovery": 2}
        ):
            raise ValueError("recovery integration requires the three-role supervision contract")
        recovery_entry = next(entry for entry in entries if entry["role"] == "recovery")
        recovery_teacher = _workflow_path(recovery_entry["teacher_checkpoint_path"])
        expected_hash = OmegaConf.select(cfg, "training.recovery_teacher_sha256")
        if expected_hash is None or file_sha256(recovery_teacher) != str(expected_hash):
            raise ValueError("recovery teacher hash does not match the selected checkpoint")
    configured_run_dir = OmegaConf.select(cfg, "training.workflow.run_dir")
    if configured_run_dir in (None, ""):
        run_dir = (
            ROOT_DIR / "logs" / "distill_workflow" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
    else:
        run_dir = _workflow_path(configured_run_dir)
    artifact_dir = _workflow_path(
        OmegaConf.select(
            cfg,
            "training.workflow.artifact_dir",
            default="logs/distill_role_artifacts",
        )
    )

    role_cfgs: dict[str, DictConfig] = {}
    specs: list[RoleArtifactSpec] = []
    for entry in entries:
        role = str(entry.get("role", ""))
        if not role:
            raise ValueError("every training.workflow.roles entry requires a role")
        role_cfg = _workflow_role_cfg(cfg, entry)
        dataset_value = entry.get("dataset_path")
        dataset_path = (
            _workflow_path(dataset_value)
            if dataset_value not in (None, "")
            else artifact_dir / f"{role}.pt"
        )
        role_cfgs[role] = role_cfg
        specs.append(
            RoleArtifactSpec(
                role=role,
                task=str(entry["task"]),
                teacher_checkpoint_path=Path(str(role_cfg.teacher.checkpoint_path)),
                dataset_path=dataset_path,
                schema_version=int(
                    OmegaConf.select(cfg, "training.workflow.schema_version", default=1)
                ),
                student_obs_dim=int(role_cfg.student.obs_dim),
                teacher_obs_dim=int(role_cfg.teacher.obs_dim),
                teacher_action_dim=int(role_cfg.teacher.action_dim),
                teacher_obs_key=str(role_cfg.training.collect_teacher_obs_key),
                teacher_projection=str(role_cfg.training.collect_teacher_projection),
                student_projection=str(role_cfg.training.collect_student_projection),
                student_drop_index=(
                    None
                    if OmegaConf.select(role_cfg, "training.collect_student_drop_index") is None
                    else int(role_cfg.training.collect_student_drop_index)
                ),
                command_sample_filter=str(role_cfg.training.collect_command_sample_filter),
                command_info_key=str(role_cfg.training.collect_command_info_key),
                command_xy_threshold=float(role_cfg.training.collect_command_xy_threshold),
                command_yaw_threshold=float(role_cfg.training.collect_command_yaw_threshold),
                owner_config=_workflow_owner_fingerprint_cfg(role_cfg),
                target_height_info_key=OmegaConf.select(
                    role_cfg, "training.collect_target_height_info_key"
                ),
            )
        )

    scenario_specs = _workflow_scenario_specs(cfg, {spec.role for spec in specs})
    if execution_mode == "persistent_async":
        if scenario_specs is None:
            raise ValueError("persistent_async execution_mode requires training.workflow.scenarios")
        if persistent_scenario_collector_factory is None:
            persistent_scenario_collector_factory = build_persistent_g1_distillation_runtime
    if bool(OmegaConf.select(cfg, "training.workflow.adopt_legacy_artifacts", default=False)):
        for spec in specs:
            if spec.dataset_path.is_file():
                adopt_legacy_role_artifact(spec)

    def collect_role(spec: RoleArtifactSpec) -> int:
        spec.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        role_cfg = OmegaConf.create(OmegaConf.to_container(role_cfgs[spec.role], resolve=True))
        role_cfg.training.collect_role_label = spec.role
        result = run_collect_dataset(role_cfg, dataset_path=spec.dataset_path)
        return int(result["dataset_num_samples"])

    def assemble_roles(dataset_paths: tuple[Path, ...], output_path: Path) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        assembly_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        role_scenarios = {
            scenario.source_roles[0]: scenario.name
            for scenario in (scenario_specs or ())
            if scenario.kind == "role"
        }
        assembly_cfg.training.multitask_sources = [
            {
                "path": str(path),
                "role": spec.role,
                **(
                    {
                        "scenario": role_scenarios[spec.role],
                        "preserve_row_role_labels": True,
                    }
                    if spec.role in role_scenarios
                    else {}
                ),
            }
            for path, spec in zip(dataset_paths, specs, strict=True)
        ]
        assembly_cfg.training.multitask_expected_student_obs_dim = specs[0].student_obs_dim
        assembly_cfg.training.multitask_expected_teacher_obs_dim = specs[0].teacher_obs_dim
        assembly_cfg.training.multitask_expected_teacher_action_dim = specs[0].teacher_action_dim
        result = run_multitask_dataset_assembly(assembly_cfg, dataset_path=output_path)
        return int(result["dataset_num_samples"])

    def update_student(dataset_path: Path, checkpoint_path: Path) -> int:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        updates = int(OmegaConf.select(cfg, "training.workflow.bootstrap_updates", default=20000))
        run_offline_dataset_update(
            cfg,
            teacher_checkpoint=specs[0].teacher_checkpoint_path,
            dataset_path=dataset_path,
            batch_size=int(
                OmegaConf.select(cfg, "training.workflow.bootstrap_batch_size", default=512)
            ),
            max_updates=updates,
            checkpoint_path=checkpoint_path,
            device=_distill_device(cfg),
        )
        return updates

    mode = str(OmegaConf.select(cfg, "training.workflow.mode", default="auto"))
    if mode not in {"auto", "fresh", "resume", "fork"}:
        raise ValueError(f"Unsupported training.workflow.mode: {mode!r}")
    manifest_path = run_dir / "run_manifest.json"
    if mode == "fork":
        parent_run_dir = OmegaConf.select(cfg, "training.workflow.parent_run_dir")
        if parent_run_dir in (None, ""):
            raise ValueError("training.workflow.mode=fork requires parent_run_dir")
        fork_workflow_run(
            parent_run_dir=_workflow_path(parent_run_dir),
            run_dir=run_dir,
        )
        bootstrap_result = None
    elif mode == "resume" or (mode == "auto" and manifest_path.is_file()):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"workflow resume manifest does not exist: {manifest_path}")
        bootstrap_result = None
    else:
        bootstrap_result = run_bootstrap_workflow(
            run_dir=run_dir,
            role_specs=tuple(specs),
            collect_role=collect_role,
            assemble_roles=assemble_roles,
            update_student=update_student,
            scenario_specs=scenario_specs,
        )

    runtime_sentinel = (
        _probe_torch_serialization_runtime if execution_mode == "persistent_async" else None
    )
    if runtime_sentinel is not None:
        runtime_sentinel("workflow/after_bootstrap")

    target_iterations = int(OmegaConf.select(cfg, "training.workflow.dagger_iterations", default=8))
    current_iteration = 0
    dagger_logger = OffPolicyLogger(
        algo_name="distill",
        max_iterations=target_iterations,
        num_envs=int(OmegaConf.select(cfg, "training.workflow.collect_num_envs", default=64)),
        env_name=str(OmegaConf.select(cfg, "training.task_name", default="G1WalkStand")),
        obs_dim=specs[0].student_obs_dim,
        action_dim=specs[0].teacher_action_dim,
        log_dir=str(run_dir),
        log_backend=str(OmegaConf.select(cfg, "training.logger", default="tensorboard")),
        display_title="UniLab Distillation / DAgger",
    )
    if runtime_sentinel is not None:
        runtime_sentinel("workflow/after_logger_construct")
    dagger_logger.start(status="Preparing DAgger workflow...")
    if runtime_sentinel is not None:
        runtime_sentinel("workflow/after_logger_start")

    def on_iteration(iteration: int, total: int) -> None:
        nonlocal current_iteration
        current_iteration = int(iteration)
        if runtime_sentinel is not None:
            runtime_sentinel(f"workflow/iteration_{iteration}/logger_iteration_entry")
        dagger_logger.log_step(current_iteration)
        dagger_logger.log_status(f"Iteration {iteration}/{total}: collecting scenarios", force=True)
        if runtime_sentinel is not None:
            runtime_sentinel(f"workflow/iteration_{iteration}/logger_iteration_exit")

    def on_status(status: str) -> None:
        if runtime_sentinel is not None:
            runtime_sentinel(f"workflow/iteration_{current_iteration}/status_callback_entry")
        dagger_logger.log_status(status, force=True)
        if runtime_sentinel is not None:
            runtime_sentinel(f"workflow/iteration_{current_iteration}/status_callback_exit")

    def on_update_progress(update: int, total: int, stats: Any) -> None:
        metrics = {
            "loss/total": float(stats.loss),
            "loss/behavior": float(stats.behavior_loss),
            "loss/aux": float(stats.aux_loss),
            "loss/role": float(stats.role_loss),
            "loss/command_intent": float(stats.command_intent_loss),
            "train/grad_norm": float(stats.student_grad_norm),
            "train/update": float(update),
        }
        if stats.route_entropy is not None:
            metrics["router/route_entropy"] = float(stats.route_entropy)
        dagger_logger.log_step(current_iteration, metrics=metrics)
        dagger_logger.log_status(
            f"Iteration {current_iteration}: update {update:,}/{total:,}",
            force=True,
        )

    def collect_dagger_role(
        output_spec: RoleArtifactSpec,
        checkpoint_path: Path,
        _iteration: int,
        output_path: Path,
        *,
        workflow_scenario: str | None = None,
    ) -> int | WorkflowScenarioCollectionResult:
        role_cfg = OmegaConf.create(
            OmegaConf.to_container(role_cfgs[output_spec.role], resolve=True)
        )
        role_cfg.training.collect_action_mode = "student_policy"
        role_cfg.training.collect_rollout_checkpoint_path = str(checkpoint_path)
        role_cfg.training.collect_role_label = output_spec.role
        if workflow_scenario is not None:
            role_cfg.training.collect_workflow_scenario = workflow_scenario
        role_cfg.training.collect_num_samples = int(
            OmegaConf.select(
                cfg,
                "training.workflow.dagger_samples_per_role",
                default=65536,
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        collected = run_collect_dataset(
            role_cfg,
            dataset_path=output_path,
            performance_clock=(performance_clock if execution_mode == "legacy" else None),
        )
        if execution_mode != "legacy":
            return int(collected["dataset_num_samples"])
        payloads = collected.get("performance_stage_observations")
        if not isinstance(payloads, list):
            raise ValueError("legacy role request performance observations are missing")
        return WorkflowScenarioCollectionResult(
            num_samples=int(collected["dataset_num_samples"]),
            worker_pid=os.getpid(),
            performance_metrics_schema_version=int(collected["performance_metrics_schema_version"]),
            performance_stage_observations=tuple(
                DistillationStageObservation.from_dict(payload) for payload in payloads
            ),
        )

    def collect_dagger_scenario(
        scenario: WorkflowScenarioSpec,
        checkpoint_path: Path,
        _iteration: int,
        output_path: Path,
    ) -> int | WorkflowScenarioCollectionResult:
        if scenario.kind == "role":
            role_spec = next(spec for spec in specs if spec.role == scenario.source_roles[0])
            return collect_dagger_role(
                replace(role_spec, dataset_path=output_path),
                checkpoint_path,
                _iteration,
                output_path,
                workflow_scenario=scenario.name,
            )
        if scenario.name == "recovery_to_stand":
            from unilab.algos.torch.distill.recovery_workflow import (
                collect_recovery_workflow_scenario,
            )

            return collect_recovery_workflow_scenario(
                cfg,
                role_cfgs,
                checkpoint_path,
                output_path,
                performance_clock=performance_clock,
            )
        if scenario.name != "walk_to_stop":
            raise ValueError(f"unsupported transition workflow scenario: {scenario.name!r}")
        scenario_role_cfgs = {
            role: role_cfgs[role] for role in scenario.source_roles if role in role_cfgs
        }
        if len(scenario_role_cfgs) != 2:
            raise ValueError("walk_to_stop scenario requires exactly two configured source roles")
        roles_by_filter: dict[str, list[str]] = {"active": [], "inactive": []}
        for role, role_cfg in scenario_role_cfgs.items():
            command_filter = str(role_cfg.training.collect_command_sample_filter)
            if command_filter in roles_by_filter:
                roles_by_filter[command_filter].append(role)
        if any(len(roles) != 1 for roles in roles_by_filter.values()):
            raise ValueError(
                "walk_to_stop scenario requires one active Walk role and one inactive "
                f"Stand role, got {roles_by_filter}"
            )
        walk_role = roles_by_filter["active"][0]
        stand_role = roles_by_filter["inactive"][0]
        request_start = float(performance_clock())
        device = _distill_device(cfg)
        walk_cfg = role_cfgs[walk_role]
        stand_cfg = role_cfgs[stand_role]
        walk_target_height_info_key = OmegaConf.select(
            walk_cfg, "training.collect_target_height_info_key"
        )
        stand_target_height_info_key = OmegaConf.select(
            stand_cfg, "training.collect_target_height_info_key"
        )
        if walk_target_height_info_key != stand_target_height_info_key:
            raise ValueError(
                "transition roles must agree on collect_target_height_info_key: "
                f"walk={walk_target_height_info_key!r} stand={stand_target_height_info_key!r}"
            )
        loaded_student = load_distillation_student_policy(checkpoint_path, device=device)
        student = loaded_student.policy
        rollout_policy: torch.nn.Module | None = student
        rollout_policies_by_intent: dict[str, torch.nn.Module] | None = None
        rollout_policy_metadata: dict[str, Any] = {}
        if isinstance(student, MoEStudentPolicy):
            rollout_policies_by_intent, expert_targets = resolve_command_intent_rollout_policies(
                student,
                loaded_student.distill_runtime_cfg,
            )
            rollout_policy = None
            rollout_policy_metadata = {
                "rollout_policy_expert_targets": expert_targets,
                "rollout_policy_source": "checkpoint_command_intent_experts",
            }
        walking_teacher = load_sac_teacher_policy(
            walk_cfg.teacher.checkpoint_path,
            build_teacher_spec(walk_cfg),
            device=device,
        )
        standing_teacher = load_sac_teacher_policy(
            stand_cfg.teacher.checkpoint_path,
            build_teacher_spec(stand_cfg),
            device=device,
        )
        ensure_registries()
        env = create_env(
            walk_cfg,
            num_envs=int(OmegaConf.select(cfg, "training.workflow.collect_num_envs", default=64)),
            env_cfg_override=BackendAdapter(
                walk_cfg,
                root_dir=ROOT_DIR,
                algo_name="distill",
            ).build_task_env_cfg_override(),
            sim_backend=str(walk_cfg.training.sim_backend),
            task_name=str(walk_cfg.training.task_name),
        )
        cold_start_seconds = float(performance_clock()) - request_start
        try:
            transition_max_env_steps = OmegaConf.select(
                cfg,
                "training.workflow.transition_max_env_steps",
                default=0,
            )
            transition_walk_commands = OmegaConf.select(
                cfg,
                "training.workflow.transition_walk_commands",
                default=[],
            )
            transition_walk_target_height = OmegaConf.select(
                cfg,
                "training.workflow.transition_walk_target_height",
                default=None,
            )
            transition_post_switch_target_heights = OmegaConf.select(
                cfg,
                "training.workflow.transition_post_switch_target_heights",
                default=[],
            )
            dataset = collect_transition_distillation_dataset_from_env(
                env,
                num_samples=int(
                    OmegaConf.select(
                        cfg,
                        "training.workflow.dagger_samples_per_role",
                        default=65536,
                    )
                ),
                expected_student_obs_dim=int(walk_cfg.student.obs_dim),
                expected_teacher_obs_dim=int(walk_cfg.teacher.obs_dim),
                walking_teacher_policy=walking_teacher,
                standing_teacher_policy=standing_teacher,
                rollout_policy=rollout_policy,
                rollout_policies_by_intent=rollout_policies_by_intent,
                pre_switch_steps=int(
                    OmegaConf.select(
                        cfg,
                        "training.workflow.transition_pre_switch_steps",
                        default=8,
                    )
                ),
                nominal_settle_steps=int(
                    OmegaConf.select(
                        cfg,
                        "training.workflow.transition_nominal_settle_steps",
                        default=0,
                    )
                ),
                min_post_switch_steps=int(
                    OmegaConf.select(
                        cfg,
                        "training.workflow.transition_min_post_switch_steps",
                        default=0,
                    )
                ),
                walk_command=tuple(
                    float(value)
                    for value in OmegaConf.select(
                        cfg,
                        "training.workflow.transition_walk_command",
                        default=[0.4, 0.0, 0.0],
                    )
                ),
                walk_commands=[
                    [float(value) for value in command] for command in transition_walk_commands
                ],
                nominal_walk_target_height=(
                    None
                    if transition_walk_target_height in (None, "")
                    else float(transition_walk_target_height)
                ),
                post_switch_target_heights=[
                    float(value) for value in transition_post_switch_target_heights
                ],
                teacher_obs_key=str(walk_cfg.training.collect_teacher_obs_key),
                teacher_projection=str(walk_cfg.training.collect_teacher_projection),
                student_projection=str(walk_cfg.training.collect_student_projection),
                student_drop_index=OmegaConf.select(
                    walk_cfg,
                    "training.collect_student_drop_index",
                ),
                command_info_key=str(walk_cfg.training.collect_command_info_key),
                target_height_info_key=walk_target_height_info_key,
                walking_role_label=walk_role,
                standing_role_label=stand_role,
                scenario_label=scenario.name,
                max_env_steps=(
                    None
                    if transition_max_env_steps in (None, "")
                    else int(transition_max_env_steps)
                ),
                metadata={
                    "workflow_scenario": scenario.name,
                    **rollout_policy_metadata,
                },
                performance_clock=performance_clock,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_start = float(performance_clock())
            save_distillation_dataset(output_path, dataset)
            artifact_write_seconds = float(performance_clock()) - write_start
            collector_payloads = dataset.metadata.get("performance_stage_observations")
            if not isinstance(collector_payloads, list):
                raise ValueError("legacy transition performance observations are missing")
            collector_observations = tuple(
                DistillationStageObservation.from_dict(payload) for payload in collector_payloads
            )
            collector_stages = tuple(item.stage for item in collector_observations)
            if collector_stages != COLLECTOR_REQUEST_STAGE_NAMES:
                raise ValueError(
                    "legacy transition performance stage order mismatch: "
                    f"expected={COLLECTOR_REQUEST_STAGE_NAMES} "
                    f"observed={collector_stages}"
                )
            return WorkflowScenarioCollectionResult(
                num_samples=dataset.num_samples,
                worker_pid=os.getpid(),
                performance_metrics_schema_version=(DISTILLATION_METRICS_SCHEMA_VERSION),
                performance_stage_observations=(
                    DistillationStageObservation(
                        stage="cold_start",
                        duration_seconds=cold_start_seconds,
                        row_count=0,
                        env_step_count=0,
                        success=True,
                        error=None,
                        cleanup_state="not_applicable",
                    ),
                    *collector_observations,
                    DistillationStageObservation(
                        stage="artifact_write",
                        duration_seconds=artifact_write_seconds,
                        row_count=dataset.num_samples,
                        env_step_count=0,
                        success=True,
                        error=None,
                        cleanup_state="not_applicable",
                    ),
                    DistillationStageObservation(
                        stage="total_elapsed",
                        duration_seconds=float(performance_clock()) - request_start,
                        row_count=dataset.num_samples,
                        env_step_count=int(dataset.metadata.get("env_steps", 0)),
                        success=True,
                        error=None,
                        cleanup_state="pending",
                    ),
                ),
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    def aggregate_dagger_sources(
        sources: tuple[WorkflowDatasetSource, ...],
        output_path: Path,
    ) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        assembly_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        source_records = []
        for source_index, source in enumerate(sources):
            source_record = {
                "source_index": source_index,
                "path": str(source.path),
                "role": source.role,
            }
            if source.scenario is not None:
                source_record["scenario"] = source.scenario
                source_record["preserve_row_role_labels"] = source.preserve_row_role_labels
            source_records.append(source_record)
        source_snapshot_path = output_path.parent / f"{output_path.name}.sources.json"
        source_snapshot_path.write_text(
            json.dumps(
                {
                    "schema": "unilab.distill.workflow.aggregate_sources.v1",
                    "aggregate_path": str(output_path),
                    "source_count": len(source_records),
                    "sources": source_records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        assembly_cfg.training.multitask_sources = source_records
        assembly_cfg.training.multitask_expected_student_obs_dim = specs[0].student_obs_dim
        assembly_cfg.training.multitask_expected_teacher_obs_dim = specs[0].teacher_obs_dim
        assembly_cfg.training.multitask_expected_teacher_action_dim = specs[0].teacher_action_dim
        assembled = run_multitask_dataset_assembly(assembly_cfg, dataset_path=output_path)
        return int(assembled["dataset_num_samples"])

    def update_dagger_student(
        dataset_path: Path,
        input_checkpoint_path: Path,
        output_checkpoint_path: Path,
    ) -> WorkflowStudentUpdateResult:
        update_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        update_cfg.training.offline_init_checkpoint = str(input_checkpoint_path)
        update_cfg.training.offline_resume_optimizer = False
        update_cfg.training.offline_save_optimizer = False
        update_cfg.training.offline_repeat_dataset = True
        update_cfg.training.offline_shuffle = True
        update_cfg.training.offline_balance_key = str(
            OmegaConf.select(cfg, "training.workflow.dagger_balance_key", default="role")
        )
        update_cfg.training.offline_balanced_labels = list(
            (
                [scenario.name for scenario in scenario_specs]
                if scenario_specs is not None
                else OmegaConf.select(cfg, "training.workflow.dagger_balanced_labels", default=[])
            )
        )
        if scenario_specs is not None:
            update_cfg.training.offline_balance_key = "scenario"
            update_cfg.training.offline_balance_quotas = {
                scenario.name: scenario.quota for scenario in scenario_specs
            }
            update_cfg.training.offline_min_balanced_replay_passes = int(
                OmegaConf.select(
                    cfg,
                    "training.workflow.dagger_min_transition_replay_passes",
                    default=0,
                )
            )
            update_cfg.training.offline_min_balanced_replay_labels = list(
                OmegaConf.select(
                    cfg,
                    "training.workflow.dagger_min_transition_replay_labels",
                    default=["walk_to_stop"],
                )
            )
        updates = int(
            OmegaConf.select(
                cfg,
                "training.workflow.dagger_updates_per_iteration",
                default=128,
            )
        )
        output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        result = run_offline_dataset_update(
            update_cfg,
            teacher_checkpoint=specs[0].teacher_checkpoint_path,
            dataset_path=dataset_path,
            batch_size=int(
                OmegaConf.select(cfg, "training.workflow.dagger_batch_size", default=512)
            ),
            max_updates=updates,
            checkpoint_path=output_checkpoint_path,
            device=_distill_device(cfg),
            auto_expand_replay_budget=True,
            progress_callback=on_update_progress,
            performance_clock=performance_clock,
        )
        return WorkflowStudentUpdateResult(
            updates=int(result["update_count"]),
            performance_stage_observations=tuple(
                DistillationStageObservation.from_dict(observation)
                for observation in result["performance_stage_observations"]
            ),
        )

    scenario_collector = None
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_config, dict):
        raise TypeError("resolved distillation config must be a mapping")
    performance_context = DistillationPerformanceRunContext(
        execution_mode=execution_mode,
        teacher_checkpoint_sha256=tuple(
            sorted({file_sha256(spec.teacher_checkpoint_path) for spec in specs})
        ),
        config_sha256=config_fingerprint(resolved_config),
        seed=int(cfg.algo.seed),
        device=_distill_device(cfg),
        num_envs=int(
            OmegaConf.select(
                cfg,
                "training.workflow.collect_num_envs",
                default=64,
            )
        ),
    )
    if execution_mode == "persistent_async":
        assert persistent_scenario_collector_factory is not None
        scenario_collector = persistent_scenario_collector_factory(
            cfg=cfg,
            role_cfgs=role_cfgs,
            role_specs=tuple(specs),
            scenario_specs=scenario_specs,
        )
        if not callable(getattr(scenario_collector, "close", None)):
            raise TypeError("persistent runtime factory result must provide close()")
        assert runtime_sentinel is not None
        runtime_sentinel("workflow/after_persistent_runtime_factory")
    cleanup_duration_seconds = 0.0
    cleanup_report: Mapping[str, Any] = {
        "execution_mode": "legacy",
        "resource_scope": "per_request",
    }
    try:
        dagger_result = run_multirole_dagger_workflow(
            run_dir=run_dir,
            role_specs=tuple(specs),
            target_iterations=target_iterations,
            collect_role=collect_dagger_role,
            aggregate_datasets=aggregate_dagger_sources,
            update_student=update_dagger_student,
            scenario_specs=scenario_specs,
            collect_scenario=(
                collect_dagger_scenario
                if execution_mode == "legacy" and scenario_specs is not None
                else None
            ),
            execution_mode=execution_mode,
            scenario_collector=scenario_collector,
            performance_context=performance_context,
            performance_clock=performance_clock,
            status_callback=on_status,
            iteration_callback=on_iteration,
            runtime_sentinel=runtime_sentinel,
        )
    except BaseException:
        dagger_logger.close()
        raise
    finally:
        if scenario_collector is not None:
            cleanup_start = float(performance_clock())
            scenario_collector.close()
            cleanup_duration_seconds = float(performance_clock()) - cleanup_start
            close_report = getattr(scenario_collector, "close_report", None)
            if not isinstance(close_report, Mapping):
                raise ValueError("persistent runtime close() must publish close_report mapping")
            cleanup_report = close_report
    if dagger_result.completed_iterations > 0:
        finalize_workflow_performance(
            run_dir=run_dir,
            performance_context=performance_context,
            cleanup_duration_seconds=cleanup_duration_seconds,
            cleanup_report=cleanup_report,
        )
    dagger_logger.log_save(str(dagger_result.checkpoint_path))
    dagger_logger.finish(
        title="Distillation Summary",
        extra_summary=(
            f"  DAgger iterations: [yellow]{dagger_result.completed_iterations}[/]/{target_iterations}\n"
            f"  Cumulative samples: [yellow]{dagger_result.cumulative_num_samples:,}[/]"
        ),
    )
    return {
        "distill_source": "single_entry_workflow",
        "stage": (
            "BOOTSTRAP_COMPLETE"
            if dagger_result.completed_iterations == 0
            else f"DAGGER_ITERATION_{dagger_result.completed_iterations}_COMPLETE"
        ),
        "mode": mode,
        "execution_mode": execution_mode,
        "run_dir": str(dagger_result.run_dir),
        "manifest_path": str(dagger_result.manifest_path),
        "role_decisions": (None if bootstrap_result is None else bootstrap_result.role_decisions),
        "bootstrap_dataset_path": (
            None if bootstrap_result is None else str(bootstrap_result.bootstrap_dataset_path)
        ),
        "bootstrap_num_samples": (
            None if bootstrap_result is None else bootstrap_result.bootstrap_num_samples
        ),
        "checkpoint_path": str(dagger_result.checkpoint_path),
        "completed_dagger_iterations": dagger_result.completed_iterations,
        "cumulative_num_samples": dagger_result.cumulative_num_samples,
    }


@hydra.main(config_path="../conf/distill", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Assemble offline, collection, or iterative online DAgger distillation."""

    if bool(OmegaConf.select(cfg, "training.workflow.enabled", default=False)):
        print(_format_cli_result(run_single_entry_workflow(cfg)))
        return

    if bool(OmegaConf.select(cfg, "training.online_dagger", default=False)):
        checkpoint_path, _run_dir = resolve_teacher_checkpoint(cfg, root_dir=ROOT_DIR)
        if checkpoint_path is None:
            raise FileNotFoundError(
                "No SAC teacher checkpoint resolved for online DAgger. "
                "Set teacher.checkpoint_path or teacher.load_run/teacher.checkpoint."
            )
        print(
            _format_cli_result(
                run_online_dagger_update(
                    cfg,
                    teacher_checkpoint=checkpoint_path,
                )
            )
        )
        return

    multitask_dataset_path = OmegaConf.select(cfg, "training.multitask_dataset_path")
    if multitask_dataset_path not in (None, ""):
        print(
            _format_cli_result(
                run_multitask_dataset_assembly(cfg, dataset_path=multitask_dataset_path)
            )
        )
        return

    collect_dataset_path = OmegaConf.select(cfg, "training.collect_dataset_path")
    if collect_dataset_path not in (None, ""):
        print(_format_cli_result(run_collect_dataset(cfg, dataset_path=collect_dataset_path)))
        return

    checkpoint_path, _run_dir = resolve_teacher_checkpoint(cfg, root_dir=ROOT_DIR)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No SAC teacher checkpoint resolved for distillation. "
            "Set teacher.load_run/teacher.checkpoint or training.log_root."
        )

    if bool(OmegaConf.select(cfg, "training.dry_run", default=False)):
        print(
            _format_cli_result(
                run_fake_batch_update(
                    cfg,
                    teacher_checkpoint=checkpoint_path,
                    batch_size=int(OmegaConf.select(cfg, "training.dry_run_batch_size", default=8)),
                    max_updates=int(OmegaConf.select(cfg, "training.dry_run_updates", default=1)),
                    checkpoint_path=OmegaConf.select(cfg, "training.dry_run_checkpoint"),
                )
            )
        )
        return

    offline_dataset_path = OmegaConf.select(cfg, "training.offline_dataset_path")
    if offline_dataset_path not in (None, ""):
        if bool(OmegaConf.select(cfg, "training.formal_run", default=False)):
            print(
                _format_cli_result(
                    run_formal_offline_dataset_update(
                        cfg,
                        teacher_checkpoint=checkpoint_path,
                        dataset_path=offline_dataset_path,
                        batch_size=int(
                            OmegaConf.select(cfg, "training.offline_batch_size", default=256)
                        ),
                        max_updates=int(
                            OmegaConf.select(cfg, "training.offline_max_updates", default=1)
                        ),
                        device=_distill_device(cfg),
                    )
                )
            )
            return
        print(
            _format_cli_result(
                run_offline_dataset_update(
                    cfg,
                    teacher_checkpoint=checkpoint_path,
                    dataset_path=offline_dataset_path,
                    batch_size=int(
                        OmegaConf.select(cfg, "training.offline_batch_size", default=256)
                    ),
                    max_updates=int(
                        OmegaConf.select(cfg, "training.offline_max_updates", default=1)
                    ),
                    checkpoint_path=OmegaConf.select(cfg, "training.offline_checkpoint"),
                    device=_distill_device(cfg),
                )
            )
        )
        return

    raise NotImplementedError(
        "No distillation route selected. Use training.online_dagger=true for the live "
        "student-rollout loop, training.collect_dataset_path for dataset collection, "
        "training.dry_run=true "
        "for the fake-batch probe, or set training.offline_dataset_path for saved-dataset "
        "offline updates."
    )


def _run_main_with_native_fail_stop() -> None:
    """Run Hydra and preserve any unhandled diagnostic failure in a core."""

    try:
        main()
    except BaseException:
        if os.environ.get("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "0") == "1":
            sys.stdout.flush()
            sys.stderr.flush()
            os.abort()
        raise


if __name__ == "__main__":
    _run_main_with_native_fail_stop()
