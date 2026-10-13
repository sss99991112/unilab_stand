from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from torch import nn


def _cpu_checkpoint_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_checkpoint_value(item) for item in value)
    if isinstance(value, list):
        return [_cpu_checkpoint_value(item) for item in value]
    return value


def save_distillation_checkpoint(
    path: str | Path,
    *,
    student: nn.Module,
    agent_steps: int,
    teacher_metadata: Mapping[str, Any] | None = None,
    distill_runtime_cfg: Mapping[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    obs_normalizer: nn.Module | None = None,
) -> None:
    """Save the deployable student and distillation provenance."""

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "student_state_dict": _cpu_checkpoint_value(student.state_dict()),
        "agent_steps": int(agent_steps),
        "teacher_metadata": dict(teacher_metadata or {}),
        "distill_runtime_cfg": dict(distill_runtime_cfg or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = _cpu_checkpoint_value(optimizer.state_dict())
    if obs_normalizer is not None:
        payload["obs_normalizer"] = _cpu_checkpoint_value(obs_normalizer.state_dict())
    tmp_path = resolved_path.with_name(
        f".{resolved_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(resolved_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"distillation checkpoint was not saved: {resolved_path}")


def load_distillation_checkpoint(
    student: nn.Module,
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    obs_normalizer: nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a student-only distillation checkpoint."""

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    return restore_distillation_checkpoint(
        student, checkpoint, optimizer=optimizer, obs_normalizer=obs_normalizer
    )


def restore_distillation_checkpoint(
    student: nn.Module,
    checkpoint: Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    obs_normalizer: nn.Module | None = None,
) -> dict[str, Any]:
    """Restore weights and optional state from one already-loaded snapshot."""
    student_state = checkpoint.get("student_state_dict")
    if student_state is None:
        raise ValueError("Checkpoint does not contain student_state_dict")
    student.load_state_dict(student_state, strict=True)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    normalizer_state = checkpoint.get("obs_normalizer")
    if obs_normalizer is not None and normalizer_state is not None:
        obs_normalizer.load_state_dict(normalizer_state)

    return cast(dict[str, Any], checkpoint)
