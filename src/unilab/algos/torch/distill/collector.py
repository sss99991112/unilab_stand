"""Distillation role/transition collection and opt-in stage observations.

状态: active collector owner, HP-4a2b observations wired behind an explicit clock.
上游: legacy collection and persistent G1 worker.
下游: DistillationTensorDataset rows, metadata, and worker pass-through.
证据: S1/S2 semantic collector, differential, and fake-clock tests.
缺口: reset/resource live timing and live A/B.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from .data import DistillationTensorDataset, build_distillation_dataset
from .performance import (
    DISTILLATION_METRICS_SCHEMA_VERSION,
    DistillationStageObservationAccumulator,
)


def _performance_span(
    accumulator: DistillationStageObservationAccumulator | None,
    stage: str,
):
    return nullcontext() if accumulator is None else accumulator.measure(stage)


def _attach_collector_performance(
    dataset: DistillationTensorDataset,
    *,
    accumulator: DistillationStageObservationAccumulator | None,
    teacher_inference_rows: int,
    student_inference_rows: int,
    env_steps: int,
) -> DistillationTensorDataset:
    if accumulator is None:
        return dataset
    observations = (
        accumulator.observation(
            stage="teacher_inference",
            row_count=teacher_inference_rows,
            env_step_count=0,
        ),
        accumulator.observation(
            stage="student_inference",
            row_count=student_inference_rows,
            env_step_count=0,
        ),
        accumulator.observation(
            stage="env_step",
            row_count=0,
            env_step_count=env_steps,
        ),
        accumulator.observation(
            stage="tensor_pack",
            row_count=dataset.num_samples,
            env_step_count=0,
        ),
    )
    return replace(
        dataset,
        metadata={
            **dataset.metadata,
            "performance_metrics_schema_version": DISTILLATION_METRICS_SCHEMA_VERSION,
            "performance_stage_observations": [
                observation.as_dict() for observation in observations
            ],
        },
    )


def _obs_array(obs: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"Observation key {key!r} not found; available keys={sorted(obs.keys())}")
    arr = np.asarray(obs[key], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Observation key {key!r} must be rank-2, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Observation key {key!r} must contain only finite values")
    return arr


def command_active_mask(
    commands: np.ndarray,
    *,
    xy_threshold: float,
    yaw_threshold: float,
) -> np.ndarray:
    """Return rows whose velocity command should use walking teacher samples."""

    commands_np = np.asarray(commands, dtype=np.float32)
    if commands_np.ndim != 2 or commands_np.shape[1] != 3:
        raise ValueError(f"commands must have shape (N, 3), got {commands_np.shape}")
    if not np.all(np.isfinite(commands_np)):
        raise ValueError("commands must contain only finite values")

    xy_threshold = float(xy_threshold)
    yaw_threshold = float(yaw_threshold)
    if not np.isfinite(xy_threshold) or xy_threshold < 0.0:
        raise ValueError(f"xy_threshold must be finite and non-negative, got {xy_threshold}")
    if not np.isfinite(yaw_threshold) or yaw_threshold < 0.0:
        raise ValueError(f"yaw_threshold must be finite and non-negative, got {yaw_threshold}")

    xy_norm = np.linalg.norm(commands_np[:, :2], axis=1)
    yaw_abs = np.abs(commands_np[:, 2])
    return np.asarray((xy_norm > xy_threshold) | (yaw_abs > yaw_threshold), dtype=np.bool_)


def _info_array(
    info: Mapping[str, Any],
    key: str,
    *,
    expected_rows: int,
) -> np.ndarray:
    if key not in info:
        raise KeyError(f"Info key {key!r} not found; available keys={sorted(info.keys())}")
    arr = np.asarray(info[key], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Info key {key!r} must be rank-2, got shape {arr.shape}")
    if arr.shape[0] != int(expected_rows):
        raise ValueError(
            f"Info key {key!r} row mismatch: expected {int(expected_rows)}, got {arr.shape[0]}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Info key {key!r} must contain only finite values")
    return arr


def _target_height_array(
    info: Mapping[str, Any],
    key: str,
    *,
    expected_rows: int,
) -> np.ndarray:
    target_height = _info_array(info, key, expected_rows=expected_rows)
    if target_height.shape[1] != 1:
        raise ValueError(f"Info key {key!r} must have shape (N, 1), got {target_height.shape}")
    return target_height


def _command_sample_mask(
    info: Mapping[str, Any],
    *,
    sample_filter: str,
    command_info_key: str,
    expected_rows: int,
    xy_threshold: float,
    yaw_threshold: float,
) -> np.ndarray:
    if sample_filter == "none":
        return np.ones((int(expected_rows),), dtype=np.bool_)

    commands = _info_array(info, command_info_key, expected_rows=int(expected_rows))
    active = command_active_mask(
        commands,
        xy_threshold=xy_threshold,
        yaw_threshold=yaw_threshold,
    )
    if sample_filter == "active":
        return active
    if sample_filter == "inactive":
        return np.asarray(~active, dtype=np.bool_)
    raise ValueError(f"Unsupported command_sample_filter: {sample_filter!r}")


def project_student_obs(
    source_obs: np.ndarray,
    *,
    projection: str,
    expected_student_obs_dim: int,
    student_drop_index: int | None = None,
) -> np.ndarray:
    """Project teacher/env observations into deployable student observations."""

    source_obs = np.asarray(source_obs, dtype=np.float32)
    if projection == "identity":
        student_obs = source_obs
    elif projection == "g1_recovery_to_walk_99_v1":
        from .g1_observation_adapter import recovery_to_walk_obs

        student_obs = recovery_to_walk_obs(source_obs)
    elif projection == "drop_index":
        if student_drop_index is None:
            raise ValueError("student_drop_index is required when student_projection='drop_index'")
        drop_index = int(student_drop_index)
        if drop_index < 0 or drop_index >= source_obs.shape[1]:
            raise ValueError(
                f"student_drop_index must be in [0, {source_obs.shape[1]}), got {drop_index}"
            )
        student_obs = np.concatenate(
            [source_obs[:, :drop_index], source_obs[:, drop_index + 1 :]],
            axis=1,
        )
    else:
        raise ValueError(f"Unsupported student_projection: {projection!r}")

    if student_obs.shape[1] != int(expected_student_obs_dim):
        raise ValueError(
            "student projection dim mismatch: "
            f"expected {int(expected_student_obs_dim)}, got {student_obs.shape[1]}"
        )
    return np.asarray(student_obs, dtype=np.float32)


def project_teacher_obs(
    source_obs: np.ndarray,
    *,
    projection: str,
    expected_teacher_obs_dim: int,
) -> tuple[np.ndarray, bool]:
    """Project live env observations into the frozen teacher checkpoint input."""

    source_obs = np.asarray(source_obs, dtype=np.float32)
    if projection == "identity":
        teacher_obs = source_obs
        synthetic_tail = False
    elif projection == "pad_zeros":
        pad_dim = int(expected_teacher_obs_dim) - int(source_obs.shape[1])
        if pad_dim < 0:
            raise ValueError(
                "teacher pad_zeros projection cannot shrink observations: "
                f"source={source_obs.shape[1]} expected={int(expected_teacher_obs_dim)}"
            )
        teacher_obs = np.concatenate(
            [source_obs, np.zeros((source_obs.shape[0], pad_dim), dtype=np.float32)],
            axis=1,
        )
        synthetic_tail = pad_dim > 0
    else:
        raise ValueError(f"Unsupported teacher_projection: {projection!r}")

    if teacher_obs.shape[1] != int(expected_teacher_obs_dim):
        raise ValueError(
            "teacher projection dim mismatch: "
            f"expected {int(expected_teacher_obs_dim)}, got {teacher_obs.shape[1]}"
        )
    return np.asarray(teacher_obs, dtype=np.float32), synthetic_tail


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _policy_actions(
    policy: torch.nn.Module,
    obs: np.ndarray,
    *,
    action_dim: int,
    policy_name: str,
) -> np.ndarray:
    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
        device=_module_device(policy),
    )
    with torch.inference_mode():
        action_tensor = policy(obs_tensor)
    if isinstance(action_tensor, tuple):
        action_tensor = action_tensor[0]
    action_tensor = torch.as_tensor(action_tensor).detach()
    if action_tensor.ndim != 2:
        raise ValueError(
            f"{policy_name} action must be rank-2, got shape {tuple(action_tensor.shape)}"
        )
    if action_tensor.shape[0] != obs.shape[0] or action_tensor.shape[1] != int(action_dim):
        raise ValueError(
            f"{policy_name} action shape mismatch: "
            f"expected ({obs.shape[0]}, {int(action_dim)}), "
            f"got {tuple(action_tensor.shape)}"
        )
    return action_tensor.cpu().numpy().astype(np.float32)


def _state_done_mask(state: Any, *, expected_rows: int) -> np.ndarray:
    terminated = getattr(state, "terminated", None)
    truncated = getattr(state, "truncated", None)
    if terminated is None and truncated is None:
        return np.zeros((int(expected_rows),), dtype=np.bool_)

    done = np.zeros((int(expected_rows),), dtype=np.bool_)
    for name, value in (("terminated", terminated), ("truncated", truncated)):
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.bool_).reshape(-1)
        if arr.shape != done.shape:
            raise ValueError(
                f"collector state.{name} shape mismatch: expected {done.shape}, got {arr.shape}"
            )
        np.logical_or(done, arr, out=done)
    return done


def _state_has_autoreset_final_observation(state: Any, done: np.ndarray) -> bool:
    if not np.any(done):
        return False
    final_observation = getattr(state, "final_observation", None)
    if isinstance(final_observation, Mapping):
        return True
    info = getattr(state, "info", None)
    if isinstance(info, Mapping) and isinstance(info.get("final_observation"), Mapping):
        terminal_mask = info.get("_final_observation")
        if terminal_mask is None:
            return True
        mask = np.asarray(terminal_mask, dtype=np.bool_).reshape(-1)
        return mask.shape == done.shape and bool(np.any(mask[done]))
    return False


def _reset_done_rows_after_step(
    env: Any,
    state: Any,
    *,
    num_envs: int,
) -> tuple[dict[str, Any], dict[str, Any], int, int, int]:
    """Return next obs/info after guarding student-policy rollouts from terminal drift.

    UniLab NpEnv autoreset keeps the terminal flag visible while replacing done
    rows with reset observations and recording final_observation. Fake or custom
    envs may return terminal rows without autoreset; those rows are reset here
    before the next collection iteration samples them.
    """

    obs = getattr(state, "obs")
    info = getattr(state, "info", {})
    done = _state_done_mask(state, expected_rows=int(num_envs))
    done_count = int(np.count_nonzero(done))
    if done_count == 0:
        return obs, info, 0, 0, 0
    if _state_has_autoreset_final_observation(state, done):
        return obs, info, done_count, done_count, 0
    if not callable(getattr(env, "reset", None)):
        raise ValueError("collector saw done rows but env.reset is not callable")

    done_indices = np.flatnonzero(done).astype(np.int32)
    reset_obs, reset_info = env.reset(done_indices)
    next_obs = {key: np.asarray(value).copy() for key, value in obs.items()}
    for key, value in reset_obs.items():
        if key not in next_obs:
            raise KeyError(f"Reset observation key {key!r} not found in step observation")
        next_obs[key][done_indices] = value

    next_info = dict(info)
    if reset_info:
        for key, value in reset_info.items():
            if isinstance(value, np.ndarray):
                if key not in next_info:
                    full_shape = (int(num_envs),) + value.shape[1:]
                    next_info[key] = np.zeros(full_shape, dtype=value.dtype)
                next_info[key][done_indices] = value
            else:
                next_info[key] = value
    return next_obs, next_info, done_count, 0, done_count


@dataclass(frozen=True)
class _TransitionCaseAssignment:
    walk_commands: np.ndarray
    post_switch_target_heights: np.ndarray | None
    case_commands: np.ndarray
    case_target_heights: np.ndarray | None
    env_case_indices: np.ndarray
    active_command_rows: np.ndarray
    nominal_target_rows: np.ndarray | None
    post_switch_target_rows: np.ndarray | None


def _build_transition_case_assignment(
    *,
    num_envs: int,
    walk_command: np.ndarray | tuple[float, float, float],
    walk_commands: Sequence[Sequence[float]] | np.ndarray | None,
    target_height_info_key: str | None,
    nominal_walk_target_height: float | None,
    post_switch_target_heights: Sequence[float] | np.ndarray | None,
) -> _TransitionCaseAssignment:
    if walk_commands is None or np.asarray(walk_commands).size == 0:
        fallback_command = np.asarray(walk_command, dtype=np.float32)
        if fallback_command.shape != (3,):
            raise ValueError(
                "walk_command must have shape (3,) when transition walk commands "
                f"are not configured, got {fallback_command.shape}"
            )
        configured_commands = fallback_command.reshape(1, 3)
    else:
        configured_commands = np.asarray(walk_commands, dtype=np.float32)
    if configured_commands.ndim != 2 or configured_commands.shape[1] != 3:
        raise ValueError(
            "transition walk commands must have shape (num_commands, 3), "
            f"got {configured_commands.shape}"
        )
    if not np.all(np.isfinite(configured_commands)):
        raise ValueError("transition walk commands must contain only finite values")
    active_commands = command_active_mask(
        configured_commands,
        xy_threshold=0.05,
        yaw_threshold=0.05,
    )
    if not bool(np.all(active_commands)):
        invalid = np.flatnonzero(~active_commands).tolist()
        raise ValueError(
            "every transition walk command must be active under the command thresholds; "
            f"inactive_indices={invalid}"
        )

    configured_targets = (
        np.asarray([], dtype=np.float32)
        if post_switch_target_heights is None
        else np.asarray(post_switch_target_heights, dtype=np.float32)
    )
    if configured_targets.ndim != 1:
        raise ValueError(
            "post-switch target heights must have shape (num_heights,), "
            f"got {configured_targets.shape}"
        )
    if not np.all(np.isfinite(configured_targets)):
        raise ValueError("post-switch target heights must contain only finite values")

    if configured_targets.size == 0:
        if nominal_walk_target_height is not None:
            raise ValueError("nominal_walk_target_height requires post_switch_target_heights")
        case_commands = configured_commands.copy()
        case_targets = None
        nominal_target_rows = None
        post_switch_target_rows = None
    else:
        if target_height_info_key in (None, ""):
            raise ValueError("post-switch target heights require target_height_info_key")
        if nominal_walk_target_height is None or not np.isfinite(float(nominal_walk_target_height)):
            raise ValueError(
                "post-switch target heights require a finite nominal_walk_target_height"
            )
        case_commands = np.repeat(
            configured_commands,
            repeats=int(configured_targets.shape[0]),
            axis=0,
        )
        case_targets = np.tile(configured_targets, int(configured_commands.shape[0]))
        nominal_target_rows = np.full(
            (int(num_envs), 1),
            float(nominal_walk_target_height),
            dtype=np.float32,
        )
        post_switch_target_rows = None

    case_count = int(case_commands.shape[0])
    if int(num_envs) < case_count:
        raise ValueError(
            "transition collection requires at least one env row per command-height case: "
            f"num_envs={int(num_envs)} case_count={case_count}"
        )
    env_case_indices = np.arange(int(num_envs), dtype=np.int64) % case_count
    active_command_rows = case_commands[env_case_indices].copy()
    if case_targets is not None:
        post_switch_target_rows = case_targets[env_case_indices, None].copy()

    return _TransitionCaseAssignment(
        walk_commands=configured_commands,
        post_switch_target_heights=(None if configured_targets.size == 0 else configured_targets),
        case_commands=case_commands,
        case_target_heights=case_targets,
        env_case_indices=env_case_indices,
        active_command_rows=active_command_rows,
        nominal_target_rows=nominal_target_rows,
        post_switch_target_rows=post_switch_target_rows,
    )


def _set_transition_input_rows(
    env: Any,
    *,
    command_info_key: str,
    command_rows: np.ndarray,
    target_height_info_key: str | None = None,
    target_height_rows: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Atomically update transition inputs before one observation refresh."""

    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    commands = info.get(command_info_key) if isinstance(info, Mapping) else None
    commands_np = np.asarray(commands) if commands is not None else None
    command_rows = np.asarray(command_rows, dtype=np.float32)
    if commands_np is None or commands_np.ndim != 2 or commands_np.shape[1] < 3:
        raise RuntimeError(
            "transition collection requires env.state.info[command_info_key] "
            "with shape (num_envs, >=3)"
        )
    if command_rows.shape != (commands_np.shape[0], 3):
        raise ValueError(
            "transition command rows shape mismatch: "
            f"expected {(commands_np.shape[0], 3)}, got {command_rows.shape}"
        )
    if not np.all(np.isfinite(command_rows)):
        raise ValueError("transition command rows must contain only finite values")

    normalized_target_key = (
        None if target_height_info_key in (None, "") else str(target_height_info_key)
    )
    if normalized_target_key is None and target_height_rows is not None:
        raise ValueError("transition target-height rows require target_height_info_key")
    target_height_np: np.ndarray | None = None
    target_rows_np: np.ndarray | None = None
    if normalized_target_key is not None and target_height_rows is not None:
        target_heights = info.get(normalized_target_key) if isinstance(info, Mapping) else None
        target_height_np = np.asarray(target_heights) if target_heights is not None else None
        target_rows_np = np.asarray(target_height_rows, dtype=np.float32)
        expected_target_shape = (commands_np.shape[0], 1)
        if target_height_np is None or target_height_np.shape != expected_target_shape:
            observed_shape = getattr(target_height_np, "shape", None)
            raise RuntimeError(
                "transition collection requires env.state.info[target_height_info_key] "
                f"with shape {expected_target_shape}, got {observed_shape}"
            )
        if target_rows_np.shape != expected_target_shape:
            raise ValueError(
                "transition target-height rows shape mismatch: "
                f"expected {expected_target_shape}, got {target_rows_np.shape}"
            )
        if not np.all(np.isfinite(target_rows_np)):
            raise ValueError("transition target-height rows must contain only finite values")

    # Validate both fields before mutating either input owner.
    commands_np[:, :3] = command_rows.astype(commands_np.dtype, copy=False)
    if target_height_np is not None and target_rows_np is not None:
        target_height_np[:, :] = target_rows_np.astype(target_height_np.dtype, copy=False)
    refresh_state = getattr(env, "refresh_state", None)
    if not callable(refresh_state):
        raise RuntimeError("transition collection requires env.refresh_state()")
    refreshed = refresh_state()
    refreshed_state = refreshed if refreshed is not None else getattr(env, "state", None)
    if refreshed_state is None:
        raise RuntimeError("env.refresh_state() did not return or retain an env state")
    refreshed_obs = getattr(refreshed_state, "obs", None)
    refreshed_info = getattr(refreshed_state, "info", None)
    if not isinstance(refreshed_obs, dict) or not isinstance(refreshed_info, Mapping):
        raise RuntimeError("transition input refresh must return dict obs and info")
    observed_commands = np.asarray(refreshed_info.get(command_info_key))
    if (
        observed_commands.ndim != 2
        or observed_commands.shape[0] != command_rows.shape[0]
        or observed_commands.shape[1] < 3
        or not np.allclose(observed_commands[:, :3], command_rows, atol=1.0e-6, rtol=0.0)
    ):
        raise RuntimeError("transition command rows changed during observation refresh")
    if normalized_target_key is not None and target_rows_np is not None:
        observed_targets = np.asarray(refreshed_info.get(normalized_target_key))
        if observed_targets.shape != target_rows_np.shape or not np.allclose(
            observed_targets,
            target_rows_np,
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise RuntimeError("transition target-height rows changed during observation refresh")
    return refreshed_obs, dict(refreshed_info)


def _resolve_collection_reset(
    env: Any,
    *,
    num_envs: int,
    initial_reset: tuple[Any, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if initial_reset is None:
        reset_output = env.reset(np.arange(num_envs, dtype=np.int32))
    else:
        reset_output = initial_reset
    if not isinstance(reset_output, tuple) or len(reset_output) != 2:
        raise ValueError("collection reset must be a two-item (obs_dict, info_dict) tuple")
    obs, info = reset_output
    if not isinstance(obs, dict) or not isinstance(info, Mapping):
        raise ValueError("collection reset must contain dict obs and mapping info")
    return obs, dict(info)


def collect_transition_distillation_dataset_from_env(
    env: Any,
    *,
    num_samples: int,
    expected_student_obs_dim: int,
    expected_teacher_obs_dim: int,
    walking_teacher_policy: torch.nn.Module,
    standing_teacher_policy: torch.nn.Module,
    rollout_policy: torch.nn.Module | None = None,
    rollout_policies_by_intent: Mapping[str, torch.nn.Module] | None = None,
    pre_switch_steps: int = 8,
    nominal_settle_steps: int = 0,
    min_post_switch_steps: int = 0,
    walk_command: np.ndarray | tuple[float, float, float] = (0.4, 0.0, 0.0),
    walk_commands: Sequence[Sequence[float]] | np.ndarray | None = None,
    nominal_walk_target_height: float | None = None,
    post_switch_target_heights: Sequence[float] | np.ndarray | None = None,
    teacher_obs_key: str = "obs",
    teacher_projection: str = "identity",
    student_projection: str = "identity",
    student_drop_index: int | None = None,
    command_info_key: str = "commands",
    target_height_info_key: str | None = None,
    walking_role_label: str = "walk_flat",
    standing_role_label: str = "stand",
    scenario_label: str = "walk_to_stop",
    max_env_steps: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    initial_reset: tuple[Any, Any] | None = None,
    performance_clock: Callable[[], float] | None = None,
) -> DistillationTensorDataset:
    """Collect one opt-in walk-to-stop student-state DAgger scenario.

    Each vectorized row starts with an active walking command, switches once to
    the zero command after ``pre_switch_steps``, retains the nominal walking
    height for ``nominal_settle_steps``, and only then applies the requested
    post-switch height. It resets its own scenario to walking if the environment
    terminates that row. When ``min_post_switch_steps`` is positive, the
    collector requires that many requested-height rows after the settling
    window and fails closed otherwise. The
    rollout is driven by either ``rollout_policy`` or the explicit
    ``rollout_policies_by_intent`` map. The latter uses the walking expert
    before the switch and the standing expert after it; teachers only label
    the observed state.
    """

    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if int(pre_switch_steps) <= 0:
        raise ValueError(f"pre_switch_steps must be positive, got {pre_switch_steps}")
    if int(nominal_settle_steps) < 0:
        raise ValueError(f"nominal_settle_steps must be non-negative, got {nominal_settle_steps}")
    if int(min_post_switch_steps) < 0:
        raise ValueError(f"min_post_switch_steps must be non-negative, got {min_post_switch_steps}")
    if not str(walking_role_label) or not str(standing_role_label):
        raise ValueError("transition role labels must be non-empty")
    if not str(scenario_label):
        raise ValueError("transition scenario_label must be non-empty")
    action_shape = getattr(getattr(env, "action_space", None), "shape", None)
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined for transition collection")
    num_envs = int(getattr(env, "num_envs"))
    minimum_samples = int(num_envs) * (
        int(pre_switch_steps) + int(nominal_settle_steps) + int(min_post_switch_steps)
    )
    if int(min_post_switch_steps) > 0 and int(num_samples) < minimum_samples:
        raise ValueError(
            "transition collection requires enough samples to cover the configured "
            f"post-switch horizon: num_samples={int(num_samples)} "
            f"minimum={minimum_samples} num_envs={int(num_envs)} "
            f"pre_switch_steps={int(pre_switch_steps)} "
            f"nominal_settle_steps={int(nominal_settle_steps)} "
            f"min_post_switch_steps={int(min_post_switch_steps)}"
        )
    if rollout_policy is None and rollout_policies_by_intent is None:
        raise ValueError(
            "transition collection requires rollout_policy or rollout_policies_by_intent"
        )
    if rollout_policy is not None and rollout_policies_by_intent is not None:
        raise ValueError("transition collection accepts only one rollout policy contract")
    if rollout_policies_by_intent is not None:
        missing_intents = {"active", "inactive"} - set(rollout_policies_by_intent)
        if missing_intents:
            raise ValueError(
                f"rollout_policies_by_intent is missing intents: {sorted(missing_intents)}"
            )
    action_dim = int(action_shape[0])
    transition_cases = _build_transition_case_assignment(
        num_envs=num_envs,
        walk_command=walk_command,
        walk_commands=walk_commands,
        target_height_info_key=target_height_info_key,
        nominal_walk_target_height=nominal_walk_target_height,
        post_switch_target_heights=post_switch_target_heights,
    )
    if int(nominal_settle_steps) > 0 and transition_cases.nominal_target_rows is None:
        raise ValueError(
            "nominal_settle_steps requires nominal_walk_target_height and "
            "post_switch_target_heights"
        )
    effective_max_env_steps = (
        int(max_env_steps)
        if max_env_steps is not None
        else max(
            int(np.ceil(int(num_samples) / max(num_envs, 1)))
            * (int(pre_switch_steps) + int(nominal_settle_steps) + 16),
            1,
        )
    )
    if effective_max_env_steps < 1:
        raise ValueError(f"max_env_steps must be positive, got {effective_max_env_steps}")

    if getattr(env, "state", None) is None and callable(getattr(env, "init_state", None)):
        env.init_state()
    obs, current_info = _resolve_collection_reset(
        env,
        num_envs=num_envs,
        initial_reset=initial_reset,
    )
    active_command_rows = transition_cases.active_command_rows
    zero_command_rows = np.zeros((num_envs, 3), dtype=np.float32)
    obs, current_info = _set_transition_input_rows(
        env,
        command_info_key=str(command_info_key),
        command_rows=active_command_rows,
        target_height_info_key=target_height_info_key,
        target_height_rows=transition_cases.nominal_target_rows,
    )

    student_chunks: list[torch.Tensor] = []
    teacher_chunks: list[torch.Tensor] = []
    teacher_action_chunks: list[torch.Tensor] = []
    command_chunks: list[torch.Tensor] = []
    target_height_chunks: list[torch.Tensor] = []
    command_before_chunks: list[torch.Tensor] = []
    command_after_chunks: list[torch.Tensor] = []
    role_labels: list[str] = []
    command_intents: list[str] = []
    scenario_labels: list[str] = []
    transition_age_chunks: list[torch.Tensor] = []
    post_switch = np.zeros((num_envs,), dtype=np.bool_)
    pre_age = np.zeros((num_envs,), dtype=np.int64)
    transition_ages = np.full((num_envs,), -1, dtype=np.int64)
    collected_count = 0
    env_steps = 0
    switch_count = 0
    post_switch_rows = 0
    done_seen_samples = 0
    action_abs_max = 0.0
    synthetic_teacher_tail = False
    case_sample_counts = np.zeros((transition_cases.case_commands.shape[0],), dtype=np.int64)
    case_post_switch_counts = np.zeros_like(case_sample_counts)
    case_max_post_switch_ages = np.full_like(case_sample_counts, -1)
    case_nominal_settle_counts = np.zeros_like(case_sample_counts)
    case_height_tracking_counts = np.zeros_like(case_sample_counts)
    case_max_height_tracking_ages = np.full_like(case_sample_counts, -1)
    nominal_settle_rows = 0
    height_tracking_rows = 0
    performance = (
        None
        if performance_clock is None
        else DistillationStageObservationAccumulator(clock=performance_clock)
    )
    teacher_inference_rows = 0
    student_inference_rows = 0

    while collected_count < int(num_samples):
        source_np = _obs_array(obs, teacher_obs_key)
        teacher_np, synthetic_tail = project_teacher_obs(
            source_np,
            projection=str(teacher_projection),
            expected_teacher_obs_dim=int(expected_teacher_obs_dim),
        )
        synthetic_teacher_tail = synthetic_teacher_tail or synthetic_tail
        student_np = project_student_obs(
            source_np,
            projection=str(student_projection),
            expected_student_obs_dim=int(expected_student_obs_dim),
            student_drop_index=student_drop_index,
        )
        current_commands = _info_array(
            current_info,
            str(command_info_key),
            expected_rows=num_envs,
        )[:, :3]
        current_target_height = (
            None
            if target_height_info_key in (None, "")
            else _target_height_array(
                current_info,
                str(target_height_info_key),
                expected_rows=num_envs,
            )
        )
        height_tracking = post_switch & (transition_ages >= int(nominal_settle_steps))
        nominal_settling = post_switch & ~height_tracking
        with _performance_span(performance, "teacher_inference"):
            walking_actions = _policy_actions(
                walking_teacher_policy,
                teacher_np,
                action_dim=action_dim,
                policy_name="walking_teacher_policy",
            )
            standing_actions = _policy_actions(
                standing_teacher_policy,
                teacher_np,
                action_dim=action_dim,
                policy_name="standing_teacher_policy",
            )
        teacher_inference_rows += 2 * int(teacher_np.shape[0])
        teacher_actions = np.where(post_switch[:, None], standing_actions, walking_actions)
        with _performance_span(performance, "student_inference"):
            if rollout_policies_by_intent is None:
                if rollout_policy is None:
                    raise RuntimeError("transition rollout policy contract was not materialized")
                rollout_actions = _policy_actions(
                    rollout_policy,
                    student_np,
                    action_dim=action_dim,
                    policy_name="rollout_policy",
                )
                student_inference_rows += int(student_np.shape[0])
            else:
                active_actions = _policy_actions(
                    rollout_policies_by_intent["active"],
                    student_np,
                    action_dim=action_dim,
                    policy_name="active_rollout_policy",
                )
                inactive_actions = _policy_actions(
                    rollout_policies_by_intent["inactive"],
                    student_np,
                    action_dim=action_dim,
                    policy_name="inactive_rollout_policy",
                )
                rollout_actions = np.where(post_switch[:, None], inactive_actions, active_actions)
                student_inference_rows += 2 * int(student_np.shape[0])
        if not np.all(np.isfinite(teacher_actions)) or not np.all(np.isfinite(rollout_actions)):
            raise ValueError("transition collection produced non-finite actions")

        with _performance_span(performance, "tensor_pack"):
            remaining = int(num_samples) - collected_count
            take = min(remaining, num_envs)
            student_chunks.append(torch.as_tensor(student_np[:take], dtype=torch.float32).clone())
            teacher_chunks.append(torch.as_tensor(teacher_np[:take], dtype=torch.float32).clone())
            teacher_action_chunks.append(
                torch.as_tensor(teacher_actions[:take], dtype=torch.float32).clone()
            )
            command_chunks.append(
                torch.as_tensor(current_commands[:take], dtype=torch.float32).clone()
            )
            if current_target_height is not None:
                target_height_chunks.append(
                    torch.as_tensor(current_target_height[:take], dtype=torch.float32).clone()
                )
            command_before_chunks.append(
                torch.as_tensor(active_command_rows[:take], dtype=torch.float32).clone()
            )
            command_after_chunks.append(
                torch.as_tensor(current_commands[:take], dtype=torch.float32).clone()
            )
            transition_age_chunks.append(
                torch.as_tensor(transition_ages[:take], dtype=torch.int64).clone()
            )
            taken_case_indices = transition_cases.env_case_indices[:take]
            np.add.at(case_sample_counts, taken_case_indices, 1)
            taken_post_switch = post_switch[:take]
            if bool(np.any(taken_post_switch)):
                post_case_indices = taken_case_indices[taken_post_switch]
                post_ages = transition_ages[:take][taken_post_switch]
                np.add.at(case_post_switch_counts, post_case_indices, 1)
                np.maximum.at(case_max_post_switch_ages, post_case_indices, post_ages)
            taken_nominal_settling = nominal_settling[:take]
            if bool(np.any(taken_nominal_settling)):
                np.add.at(
                    case_nominal_settle_counts,
                    taken_case_indices[taken_nominal_settling],
                    1,
                )
            taken_height_tracking = height_tracking[:take]
            if bool(np.any(taken_height_tracking)):
                tracking_case_indices = taken_case_indices[taken_height_tracking]
                tracking_ages = transition_ages[:take][taken_height_tracking] - int(
                    nominal_settle_steps
                )
                np.add.at(case_height_tracking_counts, tracking_case_indices, 1)
                np.maximum.at(
                    case_max_height_tracking_ages,
                    tracking_case_indices,
                    tracking_ages,
                )
            role_labels.extend(
                str(standing_role_label) if value else str(walking_role_label)
                for value in post_switch[:take]
            )
            command_intents.extend(
                "inactive" if value else "active" for value in post_switch[:take]
            )
            scenario_labels.extend(str(scenario_label) for _ in range(take))
            post_switch_rows += int(np.count_nonzero(post_switch[:take]))
            nominal_settle_rows += int(np.count_nonzero(taken_nominal_settling))
            height_tracking_rows += int(np.count_nonzero(taken_height_tracking))
            collected_count += take
            action_abs_max = max(action_abs_max, float(np.max(np.abs(rollout_actions))))
        if collected_count >= int(num_samples):
            break
        if env_steps >= effective_max_env_steps:
            raise RuntimeError(
                "transition collection exceeded max_env_steps before reaching "
                f"{num_samples} samples; collected={collected_count}"
            )

        with _performance_span(performance, "env_step"):
            state = env.step(rollout_actions)
        done_mask = _state_done_mask(state, expected_rows=num_envs)
        done_seen_samples += int(np.count_nonzero(done_mask))
        obs, current_info, done_count, _autoreset_count, _manual_reset_count = (
            _reset_done_rows_after_step(env, state, num_envs=num_envs)
        )
        env_steps += 1

        previous_post_switch = post_switch.copy()
        post_switch[done_mask] = False
        pre_age[done_mask] = 0
        transition_ages[done_mask] = -1
        pre_age[(~previous_post_switch) & ~done_mask] += 1
        transition_ages[previous_post_switch & ~done_mask] += 1
        switch_mask = (~post_switch) & ~done_mask & (pre_age >= int(pre_switch_steps))
        post_switch[switch_mask] = True
        transition_ages[switch_mask] = 0
        switch_count += int(np.count_nonzero(switch_mask))
        height_switch_mask = (
            post_switch
            & ~done_mask
            & (transition_ages == int(nominal_settle_steps))
            & (int(nominal_settle_steps) > 0)
        )

        command_rows = active_command_rows.copy()
        command_rows[post_switch] = zero_command_rows[post_switch]
        target_height_rows = None
        if transition_cases.nominal_target_rows is not None:
            if transition_cases.post_switch_target_rows is None:
                raise RuntimeError("transition post-switch target rows unexpectedly missing")
            target_height_rows = transition_cases.nominal_target_rows.copy()
            requested_height_rows = post_switch & (transition_ages >= int(nominal_settle_steps))
            target_height_rows[requested_height_rows] = transition_cases.post_switch_target_rows[
                requested_height_rows
            ]
        if done_count > 0 or bool(np.any(switch_mask)) or bool(np.any(height_switch_mask)):
            obs, current_info = _set_transition_input_rows(
                env,
                command_info_key=str(command_info_key),
                command_rows=command_rows,
                target_height_info_key=target_height_info_key,
                target_height_rows=target_height_rows,
            )

    if switch_count == 0 or post_switch_rows == 0:
        raise RuntimeError(
            "transition collection did not produce both pre-switch and post-switch rows"
        )
    missing_case_indices = np.flatnonzero(case_height_tracking_counts == 0).tolist()
    if missing_case_indices:
        raise RuntimeError(
            "transition collection did not produce requested-height rows for every case: "
            f"missing_case_indices={missing_case_indices}"
        )
    transition_ages_tensor = torch.cat(transition_age_chunks, dim=0)[: int(num_samples)]
    post_switch_ages = transition_ages_tensor[transition_ages_tensor >= 0]
    max_post_switch_age = int(post_switch_ages.max().item()) if post_switch_ages.numel() else -1
    max_height_tracking_age = max_post_switch_age - int(nominal_settle_steps)
    if int(min_post_switch_steps) > 0 and max_height_tracking_age < int(min_post_switch_steps) - 1:
        raise RuntimeError(
            "transition collection did not reach the configured requested-height horizon: "
            f"max_height_tracking_age={max_height_tracking_age} "
            f"required={int(min_post_switch_steps) - 1}"
        )
    if int(min_post_switch_steps) > 0:
        required_case_age = int(min_post_switch_steps) - 1
        short_case_indices = np.flatnonzero(
            case_max_height_tracking_ages < required_case_age
        ).tolist()
        if short_case_indices:
            raise RuntimeError(
                "transition collection did not reach the configured requested-height horizon "
                "for every case: "
                f"short_case_indices={short_case_indices} required={required_case_age}"
            )
    transition_case_metadata = []
    for case_index, case_command in enumerate(transition_cases.case_commands):
        case_target = (
            None
            if transition_cases.case_target_heights is None
            else float(transition_cases.case_target_heights[case_index])
        )
        transition_case_metadata.append(
            {
                "index": int(case_index),
                "walk_command": case_command.tolist(),
                "post_switch_target_height": case_target,
                "sample_count": int(case_sample_counts[case_index]),
                "post_switch_sample_count": int(case_post_switch_counts[case_index]),
                "max_post_switch_age": int(case_max_post_switch_ages[case_index]),
                "nominal_settle_sample_count": int(case_nominal_settle_counts[case_index]),
                "height_tracking_sample_count": int(case_height_tracking_counts[case_index]),
                "max_height_tracking_age": int(case_max_height_tracking_ages[case_index]),
            }
        )
    payload = dict(metadata or {})
    payload.update(
        {
            "source": "live_env_transition_rollout",
            "scenario": str(scenario_label),
            "teacher_obs_key": str(teacher_obs_key),
            "teacher_projection": str(teacher_projection),
            "student_projection": str(student_projection),
            "student_drop_index": student_drop_index,
            "rollout_policy": (
                "command_intent_experts"
                if rollout_policies_by_intent is not None
                else "distillation_student"
            ),
            "pre_switch_steps": int(pre_switch_steps),
            "nominal_settle_steps": int(nominal_settle_steps),
            "height_switch_age": int(nominal_settle_steps),
            "min_post_switch_steps": int(min_post_switch_steps),
            "max_post_switch_age": int(max_post_switch_age),
            "max_height_tracking_age": int(max_height_tracking_age),
            "walk_command": transition_cases.walk_commands[0].tolist(),
            "walk_commands": transition_cases.walk_commands.tolist(),
            "nominal_walk_target_height": (
                None if nominal_walk_target_height is None else float(nominal_walk_target_height)
            ),
            "post_switch_target_heights": (
                None
                if transition_cases.post_switch_target_heights is None
                else transition_cases.post_switch_target_heights.tolist()
            ),
            "transition_case_count": len(transition_case_metadata),
            "transition_cases": transition_case_metadata,
            "zero_command": zero_command_rows[0].tolist(),
            "command_info_key": str(command_info_key),
            "target_height_info_key": (
                None if target_height_info_key in (None, "") else str(target_height_info_key)
            ),
            "walking_role_label": str(walking_role_label),
            "standing_role_label": str(standing_role_label),
            "env_steps": int(env_steps),
            "switch_count": int(switch_count),
            "post_switch_rows": int(post_switch_rows),
            "nominal_settle_rows": int(nominal_settle_rows),
            "height_tracking_rows": int(height_tracking_rows),
            "done_seen_samples": int(done_seen_samples),
            "action_abs_max": float(action_abs_max),
            "synthetic_teacher_tail": bool(synthetic_teacher_tail),
        }
    )
    with _performance_span(performance, "tensor_pack"):
        dataset = build_distillation_dataset(
            torch.cat(student_chunks, dim=0)[: int(num_samples)],
            torch.cat(teacher_chunks, dim=0)[: int(num_samples)],
            expected_student_obs_dim=int(expected_student_obs_dim),
            expected_teacher_obs_dim=int(expected_teacher_obs_dim),
            expected_teacher_action_dim=action_dim,
            metadata=payload,
            role_labels=tuple(role_labels[: int(num_samples)]),
            teacher_actions=torch.cat(teacher_action_chunks, dim=0)[: int(num_samples)],
            commands=torch.cat(command_chunks, dim=0)[: int(num_samples)],
            target_height=(
                torch.cat(target_height_chunks, dim=0)[: int(num_samples)]
                if target_height_chunks
                else None
            ),
            command_intents=tuple(command_intents[: int(num_samples)]),
            scenario_labels=tuple(scenario_labels[: int(num_samples)]),
            transition_ages=transition_ages_tensor,
            command_before=torch.cat(command_before_chunks, dim=0)[: int(num_samples)],
            command_after=torch.cat(command_after_chunks, dim=0)[: int(num_samples)],
        )
    return _attach_collector_performance(
        dataset,
        accumulator=performance,
        teacher_inference_rows=teacher_inference_rows,
        student_inference_rows=student_inference_rows,
        env_steps=env_steps,
    )


def collect_distillation_dataset_from_env(
    env: Any,
    *,
    num_samples: int,
    expected_student_obs_dim: int,
    expected_teacher_obs_dim: int,
    teacher_obs_key: str = "obs",
    teacher_projection: str = "identity",
    student_projection: str = "identity",
    student_drop_index: int | None = None,
    action_mode: str = "zero",
    action_seed: int | None = None,
    teacher_policy: torch.nn.Module | None = None,
    rollout_policy: torch.nn.Module | None = None,
    command_sample_filter: str = "none",
    command_info_key: str = "commands",
    target_height_info_key: str | None = None,
    command_xy_threshold: float = 0.05,
    command_yaw_threshold: float = 0.05,
    max_env_steps: int | None = None,
    role_label: str | None = None,
    role_selector: Callable[[Mapping[str, Any]], tuple[str, ...]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    initial_reset: tuple[Any, Any] | None = None,
    performance_clock: Callable[[], float] | None = None,
) -> DistillationTensorDataset:
    """Collect a small observation-only distillation dataset from a UniLab env."""

    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    command_sample_filter = str(command_sample_filter)
    if command_sample_filter not in {"none", "active", "inactive"}:
        raise ValueError(f"Unsupported command_sample_filter: {command_sample_filter!r}")
    if action_mode not in {"zero", "random", "teacher_policy", "student_policy"}:
        raise ValueError(f"Unsupported collect action_mode: {action_mode!r}")
    if action_mode in {"teacher_policy", "student_policy"} and teacher_policy is None:
        raise ValueError(
            f"teacher_policy is required when action_mode={action_mode!r} "
            "to cache teacher target actions"
        )
    if action_mode not in {"teacher_policy", "student_policy"} and teacher_policy is not None:
        raise ValueError(
            "teacher_policy can only be set when action_mode='teacher_policy' "
            "or action_mode='student_policy'"
        )
    if action_mode == "student_policy" and rollout_policy is None:
        raise ValueError("rollout_policy is required when action_mode='student_policy'")
    if action_mode != "student_policy" and rollout_policy is not None:
        raise ValueError("rollout_policy can only be set when action_mode='student_policy'")
    if action_mode in {"teacher_policy", "student_policy"} and action_seed is not None:
        raise ValueError("action_seed is only supported when action_mode='random'")
    action_shape = getattr(getattr(env, "action_space", None), "shape", None)
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined for distillation collection")
    if max_env_steps is not None and int(max_env_steps) < 0:
        raise ValueError(f"max_env_steps must be non-negative when set, got {max_env_steps}")

    num_envs = int(getattr(env, "num_envs"))
    action_dim = int(action_shape[0])
    effective_max_env_steps = (
        int(max_env_steps)
        if max_env_steps is not None
        else (
            max(int(np.ceil(int(num_samples) / max(num_envs, 1))) * 100, 1)
            if command_sample_filter != "none"
            else None
        )
    )
    rng = np.random.default_rng(None if action_seed is None else int(action_seed))
    action_abs_max = 0.0
    if getattr(env, "state", None) is None and callable(getattr(env, "init_state", None)):
        env.init_state()
    obs, current_info = _resolve_collection_reset(
        env,
        num_envs=num_envs,
        initial_reset=initial_reset,
    )

    student_chunks: list[torch.Tensor] = []
    teacher_chunks: list[torch.Tensor] = []
    teacher_action_chunks: list[torch.Tensor] = []
    command_chunks: list[torch.Tensor] = []
    target_height_chunks: list[torch.Tensor] = []
    command_intent_chunks: list[str] = []
    row_role_labels: list[str] = []
    env_steps = 0
    collected_count = 0
    command_seen_samples = 0
    command_selected_samples = 0
    synthetic_teacher_tail = False
    done_seen_samples = 0
    autoreset_done_count = 0
    manual_done_reset_count = 0
    performance = (
        None
        if performance_clock is None
        else DistillationStageObservationAccumulator(clock=performance_clock)
    )
    teacher_inference_rows = 0
    student_inference_rows = 0

    while collected_count < int(num_samples):
        source_np = _obs_array(obs, teacher_obs_key)
        teacher_np, synthetic_teacher_tail = project_teacher_obs(
            source_np,
            projection=str(teacher_projection),
            expected_teacher_obs_dim=int(expected_teacher_obs_dim),
        )
        student_np = project_student_obs(
            source_np,
            projection=str(student_projection),
            expected_student_obs_dim=int(expected_student_obs_dim),
            student_drop_index=student_drop_index,
        )
        commands_np = (
            _info_array(
                current_info,
                str(command_info_key),
                expected_rows=teacher_np.shape[0],
            )
            if command_sample_filter != "none"
            else None
        )
        target_height_np = (
            None
            if target_height_info_key in (None, "")
            else _target_height_array(
                current_info,
                str(target_height_info_key),
                expected_rows=teacher_np.shape[0],
            )
        )
        command_active = (
            command_active_mask(
                commands_np,
                xy_threshold=float(command_xy_threshold),
                yaw_threshold=float(command_yaw_threshold),
            )
            if commands_np is not None
            else None
        )
        row_mask = _command_sample_mask(
            current_info,
            sample_filter=command_sample_filter,
            command_info_key=str(command_info_key),
            expected_rows=teacher_np.shape[0],
            xy_threshold=float(command_xy_threshold),
            yaw_threshold=float(command_yaw_threshold),
        )
        if row_mask.shape[0] != teacher_np.shape[0]:
            raise ValueError(
                "command sample mask row mismatch: "
                f"expected {teacher_np.shape[0]}, got {row_mask.shape[0]}"
            )
        if command_sample_filter != "none":
            command_seen_samples += int(row_mask.shape[0])
            command_selected_samples += int(np.count_nonzero(row_mask))
        current_roles = None if role_selector is None else role_selector(current_info)
        if current_roles is not None and len(current_roles) != teacher_np.shape[0]:
            raise ValueError("role_selector must return one label per pre-action row")
        label_actions = None
        if teacher_policy is not None:
            with _performance_span(performance, "teacher_inference"):
                label_actions = _policy_actions(
                    teacher_policy,
                    teacher_np,
                    action_dim=action_dim,
                    policy_name="teacher_policy",
                )
            teacher_inference_rows += int(teacher_np.shape[0])
            if not np.all(np.isfinite(label_actions)):
                raise ValueError("teacher_policy produced non-finite target actions")
        if action_mode == "teacher_policy":
            actions = label_actions
        elif action_mode == "student_policy":
            if rollout_policy is None:
                raise RuntimeError("student rollout policy contract was not materialized")
            with _performance_span(performance, "student_inference"):
                actions = _policy_actions(
                    rollout_policy,
                    student_np,
                    action_dim=action_dim,
                    policy_name="rollout_policy",
                )
            student_inference_rows += int(student_np.shape[0])
            if not np.all(np.isfinite(actions)):
                raise ValueError("rollout_policy produced non-finite rollout actions")
        else:
            actions = None
        with _performance_span(performance, "tensor_pack"):
            selected_teacher_np = teacher_np[row_mask]
            selected_student_np = student_np[row_mask]
            selected_actions = label_actions[row_mask] if label_actions is not None else None
            selected_commands = commands_np[row_mask] if commands_np is not None else None
            selected_target_height = (
                target_height_np[row_mask] if target_height_np is not None else None
            )
            selected_command_active = (
                command_active[row_mask] if command_active is not None else None
            )
            remaining = int(num_samples) - collected_count
            take = min(remaining, selected_teacher_np.shape[0])
            if take > 0:
                teacher_chunks.append(
                    torch.as_tensor(selected_teacher_np[:take], dtype=torch.float32)
                )
                student_chunks.append(
                    torch.as_tensor(selected_student_np[:take], dtype=torch.float32)
                )
                if selected_actions is not None:
                    teacher_action_chunks.append(
                        torch.as_tensor(selected_actions[:take], dtype=torch.float32)
                    )
                if selected_commands is not None and selected_command_active is not None:
                    command_chunks.append(
                        torch.as_tensor(selected_commands[:take], dtype=torch.float32)
                    )
                    command_intent_chunks.extend(
                        "active" if bool(value) else "inactive"
                        for value in selected_command_active[:take]
                    )
                if selected_target_height is not None:
                    target_height_chunks.append(
                        torch.as_tensor(selected_target_height[:take], dtype=torch.float32)
                    )
                if current_roles is not None:
                    row_role_labels.extend(np.asarray(current_roles)[row_mask][:take].tolist())
                collected_count += int(take)
            if actions is not None:
                action_abs_max = max(action_abs_max, float(np.max(np.abs(actions))))
        if collected_count >= int(num_samples):
            break
        if effective_max_env_steps is not None and env_steps >= effective_max_env_steps:
            raise RuntimeError(
                f"command_sample_filter={command_sample_filter!r} selected "
                f"{command_selected_samples}/{command_seen_samples} samples after "
                f"{env_steps} env steps; increase max_env_steps or relax command thresholds"
            )

        if action_mode == "zero":
            actions = np.zeros((num_envs, action_dim), dtype=np.float32)
        elif action_mode == "random":
            actions = rng.uniform(-1.0, 1.0, size=(num_envs, action_dim)).astype(np.float32)
        if actions is None:
            raise RuntimeError(f"collect action_mode={action_mode!r} did not materialize actions")
        if not np.all(np.isfinite(actions)):
            raise ValueError(f"collect action_mode={action_mode!r} produced non-finite actions")
        action_abs_max = max(action_abs_max, float(np.max(np.abs(actions))))
        with _performance_span(performance, "env_step"):
            state = env.step(actions)
        env_steps += 1
        obs, current_info, done_count, autoreset_count, manual_reset_count = (
            _reset_done_rows_after_step(env, state, num_envs=num_envs)
        )
        done_seen_samples += done_count
        autoreset_done_count += autoreset_count
        manual_done_reset_count += manual_reset_count

    payload = dict(metadata or {})
    normalized_role_label = None if role_label in (None, "") else str(role_label)
    if normalized_role_label is not None:
        payload["role_label"] = normalized_role_label
    payload.update(
        {
            "source": "live_env_rollout",
            "teacher_obs_key": str(teacher_obs_key),
            "teacher_projection": str(teacher_projection),
            "student_projection": str(student_projection),
            "student_drop_index": student_drop_index,
            "target_height_info_key": (
                None if target_height_info_key in (None, "") else str(target_height_info_key)
            ),
            "action_mode": str(action_mode),
            "action_seed": None if action_seed is None else int(action_seed),
            "action_abs_max": float(action_abs_max),
            "num_envs": num_envs,
            "env_steps": env_steps,
            "done_seen_samples": int(done_seen_samples),
            "autoreset_done_count": int(autoreset_done_count),
            "manual_done_reset_count": int(manual_done_reset_count),
            "synthetic_teacher_tail": bool(synthetic_teacher_tail),
        }
    )
    if command_sample_filter != "none":
        payload.update(
            {
                "command_sample_filter": command_sample_filter,
                "command_info_key": str(command_info_key),
                "command_xy_threshold": float(command_xy_threshold),
                "command_yaw_threshold": float(command_yaw_threshold),
                "command_seen_samples": int(command_seen_samples),
                "command_selected_samples": int(command_selected_samples),
                "max_env_steps": effective_max_env_steps,
            }
        )
    if action_mode == "student_policy":
        payload["rollout_policy"] = "distillation_student"
    with _performance_span(performance, "tensor_pack"):
        dataset = build_distillation_dataset(
            torch.cat(student_chunks, dim=0),
            torch.cat(teacher_chunks, dim=0),
            expected_student_obs_dim=int(expected_student_obs_dim),
            expected_teacher_obs_dim=int(expected_teacher_obs_dim),
            expected_teacher_action_dim=action_dim,
            metadata=payload,
            teacher_actions=(
                torch.cat(teacher_action_chunks, dim=0) if teacher_action_chunks else None
            ),
            commands=torch.cat(command_chunks, dim=0) if command_chunks else None,
            target_height=(
                torch.cat(target_height_chunks, dim=0) if target_height_chunks else None
            ),
            command_intents=(tuple(command_intent_chunks) if command_intent_chunks else None),
            role_labels=(
                tuple(row_role_labels)
                if role_selector is not None
                else (normalized_role_label,) * int(num_samples)
                if normalized_role_label is not None
                else None
            ),
        )
    return _attach_collector_performance(
        dataset,
        accumulator=performance,
        teacher_inference_rows=teacher_inference_rows,
        student_inference_rows=student_inference_rows,
        env_steps=env_steps,
    )
