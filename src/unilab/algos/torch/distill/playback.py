from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn

from .checkpoint import restore_distillation_checkpoint
from .models import MLPStudentPolicy
from .moe_student import MoEStudentPolicy


@dataclass(frozen=True)
class LoadedDistillationStudentPolicy:
    """Student-only policy loaded from a distillation checkpoint for playback."""

    policy: nn.Module
    obs_dim: int
    action_dim: int
    agent_steps: int
    teacher_metadata: dict[str, Any]
    distill_runtime_cfg: dict[str, Any]


def _required_int(cfg: dict[str, Any], key: str) -> int:
    value = cfg.get(key)
    if value is None:
        raise ValueError(f"distillation runtime config missing {key}")
    return int(value)


def _required_int_tuple(cfg: dict[str, Any], key: str) -> tuple[int, ...]:
    value = cfg.get(key)
    if value is None:
        raise ValueError(f"distillation runtime config missing {key}")
    return tuple(int(dim) for dim in value)


def load_distillation_student_policy(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedDistillationStudentPolicy:
    """Load the deployable student policy without teacher or privileged observations."""

    raw = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    runtime_cfg = dict(raw.get("distill_runtime_cfg") or {})
    obs_dim = _required_int(runtime_cfg, "student_obs_dim")
    action_dim = _required_int(runtime_cfg, "student_action_dim")
    activation = str(runtime_cfg.get("student_activation", "elu"))
    squash_action = bool(runtime_cfg.get("student_squash_action", True))
    model_type = str(runtime_cfg.get("student_model_type", "mlp"))

    if model_type == "mlp":
        hidden_dims = tuple(
            int(dim) for dim in runtime_cfg.get("student_hidden_dims", (256, 256, 256))
        )
        policy = MLPStudentPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            squash_action=squash_action,
        ).to(device)
    elif model_type == "moe":
        routing_mode = str(runtime_cfg.get("student_routing_mode", "soft"))
        if routing_mode not in {"soft", "hard"}:
            raise ValueError(f"student_routing_mode must be 'soft' or 'hard', got {routing_mode!r}")
        validated_routing_mode = cast(Literal["soft", "hard"], routing_mode)
        policy = MoEStudentPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_experts=_required_int(runtime_cfg, "student_num_experts"),
            expert_hidden_dims=_required_int_tuple(
                runtime_cfg,
                "student_expert_hidden_dims",
            ),
            router_hidden_dims=_required_int_tuple(
                runtime_cfg,
                "student_router_hidden_dims",
            ),
            activation=activation,
            squash_action=squash_action,
            routing_mode=validated_routing_mode,
            router_temperature=float(runtime_cfg.get("student_router_temperature", 1.0)),
        ).to(device)
    else:
        raise ValueError(f"Unsupported distillation student_model_type: {model_type!r}")
    checkpoint = restore_distillation_checkpoint(policy, raw)
    policy.eval()
    for param in policy.parameters():
        param.requires_grad_(False)

    return LoadedDistillationStudentPolicy(
        policy=policy,
        obs_dim=obs_dim,
        action_dim=action_dim,
        agent_steps=int(checkpoint.get("agent_steps", 0)),
        teacher_metadata=dict(checkpoint.get("teacher_metadata") or {}),
        distill_runtime_cfg=runtime_cfg,
    )
