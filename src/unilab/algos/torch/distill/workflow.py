from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .async_runtime import (
    DaggerCollectRequest,
    DaggerCollectResult,
    validate_dagger_collect_result,
)
from .data import load_distillation_dataset, save_distillation_dataset
from .performance import (
    LEGACY_REQUEST_STAGE_NAMES,
    PERSISTENT_REQUEST_STAGE_NAMES,
    WORKFLOW_ITERATION_STAGE_NAMES,
    DistillationMetricsRecorder,
    DistillationPerformanceRunContext,
    DistillationStageObservation,
    load_distillation_metrics,
)
from .playback import load_distillation_student_policy

ROLE_ARTIFACT_MANIFEST_VERSION = 1


def _progress(message: str) -> None:
    if os.environ.get("UNILAB_DISTILL_PROGRESS", "0").lower() not in {
        "",
        "0",
        "false",
        "off",
    }:
        print(f"[distill-workflow] {message}", flush=True)


class WorkflowScenarioCollector(Protocol):
    """Activate one student barrier, then collect scenarios against it."""

    def activate_checkpoint(self, checkpoint_path: Path) -> int: ...

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult: ...


@dataclass(frozen=True)
class WorkflowScenarioCollectionResult:
    """Return legacy scenario rows plus owner-local performance observations."""

    num_samples: int
    worker_pid: int
    performance_metrics_schema_version: int
    performance_stage_observations: tuple[DistillationStageObservation, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_samples, bool)
            or not isinstance(self.num_samples, int)
            or self.num_samples <= 0
        ):
            raise ValueError("num_samples must be a positive integer")
        if (
            isinstance(self.worker_pid, bool)
            or not isinstance(self.worker_pid, int)
            or self.worker_pid <= 0
        ):
            raise ValueError("worker_pid must be a positive integer")


@dataclass(frozen=True)
class WorkflowStudentUpdateResult:
    """Return update count plus learner-owner performance observations."""

    updates: int
    performance_stage_observations: tuple[DistillationStageObservation, ...]

    def __post_init__(self) -> None:
        if isinstance(self.updates, bool) or not isinstance(self.updates, int) or self.updates <= 0:
            raise ValueError("updates must be a positive integer")


class ArtifactDecision(str, Enum):
    REUSE = "REUSE"
    COLLECT = "COLLECT"
    STALE = "STALE"
    INCOMPATIBLE = "INCOMPATIBLE"


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _normalize_json(config),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RoleArtifactSpec:
    role: str
    task: str
    teacher_checkpoint_path: Path
    dataset_path: Path
    schema_version: int
    student_obs_dim: int
    teacher_obs_dim: int
    teacher_action_dim: int
    teacher_obs_key: str
    teacher_projection: str
    student_projection: str
    student_drop_index: int | None
    command_sample_filter: str
    command_info_key: str
    command_xy_threshold: float
    command_yaw_threshold: float
    owner_config: Mapping[str, Any]
    target_height_info_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "teacher_checkpoint_path", Path(self.teacher_checkpoint_path))
        object.__setattr__(self, "dataset_path", Path(self.dataset_path))
        if not self.role:
            raise ValueError("role must be non-empty")
        if not self.task:
            raise ValueError("task must be non-empty")

    @property
    def manifest_path(self) -> Path:
        return self.dataset_path.with_suffix(self.dataset_path.suffix + ".manifest.json")

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "task": self.task,
            "teacher_checkpoint_path": self.teacher_checkpoint_path,
            "dataset_path": self.dataset_path,
            "schema_version": self.schema_version,
            "student_obs_dim": self.student_obs_dim,
            "teacher_obs_dim": self.teacher_obs_dim,
            "teacher_action_dim": self.teacher_action_dim,
            "teacher_obs_key": self.teacher_obs_key,
            "teacher_projection": self.teacher_projection,
            "student_projection": self.student_projection,
            "student_drop_index": self.student_drop_index,
            "command_sample_filter": self.command_sample_filter,
            "command_info_key": self.command_info_key,
            "command_xy_threshold": self.command_xy_threshold,
            "command_yaw_threshold": self.command_yaw_threshold,
            "target_height_info_key": self.target_height_info_key,
            "owner_config": self.owner_config,
        }


@dataclass(frozen=True)
class RoleArtifactManifest:
    manifest_version: int
    role: str
    task: str
    teacher_checkpoint_path: str
    teacher_checkpoint_sha256: str
    dataset_path: str
    dataset_sha256: str
    schema_version: int
    student_obs_dim: int
    teacher_obs_dim: int
    teacher_action_dim: int
    teacher_obs_key: str
    teacher_projection: str
    student_projection: str
    student_drop_index: int | None
    command_sample_filter: str
    command_info_key: str
    command_xy_threshold: float
    command_yaw_threshold: float
    owner_config_sha256: str
    num_samples: int
    target_height_info_key: str | None = None


@dataclass(frozen=True)
class RoleArtifactPreflight:
    role: str
    decision: ArtifactDecision
    dataset_path: Path
    manifest_path: Path
    mismatches: tuple[str, ...] = ()


@dataclass(frozen=True)
class BootstrapWorkflowResult:
    run_dir: Path
    manifest_path: Path
    role_decisions: dict[str, str]
    bootstrap_dataset_path: Path
    bootstrap_num_samples: int
    checkpoint_path: Path
    bootstrap_updates: int


@dataclass(frozen=True)
class WorkflowDatasetSource:
    path: Path
    role: str
    scenario: str | None = None
    preserve_row_role_labels: bool = False


@dataclass(frozen=True)
class WorkflowScenarioSpec:
    name: str
    kind: str
    source_roles: tuple[str, ...]
    quota: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "source_roles", tuple(str(role) for role in self.source_roles))
        object.__setattr__(self, "quota", float(self.quota))
        if not self.name:
            raise ValueError("workflow scenario name must be non-empty")
        if self.kind not in {"role", "transition"}:
            raise ValueError(
                f"workflow scenario kind must be 'role' or 'transition', got {self.kind!r}"
            )
        if not self.source_roles:
            raise ValueError(f"workflow scenario {self.name!r} requires source_roles")
        if self.kind == "role" and len(self.source_roles) != 1:
            raise ValueError("role workflow scenarios require exactly one source role")
        if not math.isfinite(self.quota) or self.quota <= 0.0:
            raise ValueError("workflow scenario quota must be finite and positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "source_roles": list(self.source_roles),
            "quota": self.quota,
        }


@dataclass(frozen=True)
class DaggerWorkflowResult:
    run_dir: Path
    manifest_path: Path
    completed_iterations: int
    checkpoint_path: Path
    cumulative_num_samples: int


def create_role_artifact_manifest(
    spec: RoleArtifactSpec,
    *,
    num_samples: int,
) -> RoleArtifactManifest:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if not spec.teacher_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"teacher checkpoint does not exist: {spec.teacher_checkpoint_path}"
        )
    if not spec.dataset_path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {spec.dataset_path}")
    return RoleArtifactManifest(
        manifest_version=ROLE_ARTIFACT_MANIFEST_VERSION,
        role=spec.role,
        task=spec.task,
        teacher_checkpoint_path=str(spec.teacher_checkpoint_path.resolve()),
        teacher_checkpoint_sha256=file_sha256(spec.teacher_checkpoint_path),
        dataset_path=str(spec.dataset_path.resolve()),
        dataset_sha256=file_sha256(spec.dataset_path),
        schema_version=int(spec.schema_version),
        student_obs_dim=int(spec.student_obs_dim),
        teacher_obs_dim=int(spec.teacher_obs_dim),
        teacher_action_dim=int(spec.teacher_action_dim),
        teacher_obs_key=spec.teacher_obs_key,
        teacher_projection=spec.teacher_projection,
        student_projection=spec.student_projection,
        student_drop_index=spec.student_drop_index,
        command_sample_filter=spec.command_sample_filter,
        command_info_key=spec.command_info_key,
        command_xy_threshold=float(spec.command_xy_threshold),
        command_yaw_threshold=float(spec.command_yaw_threshold),
        owner_config_sha256=config_fingerprint(spec.owner_config),
        num_samples=int(num_samples),
        target_height_info_key=spec.target_height_info_key,
    )


def _validate_workflow_scenarios(
    scenario_specs: Sequence[WorkflowScenarioSpec],
    role_specs: Sequence[RoleArtifactSpec],
) -> tuple[WorkflowScenarioSpec, ...]:
    scenarios = tuple(scenario_specs)
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ValueError(f"workflow scenario names must be unique, got {names}")
    roles = {spec.role for spec in role_specs}
    for scenario in scenarios:
        missing = sorted(set(scenario.source_roles) - roles)
        if missing:
            raise ValueError(
                f"workflow scenario {scenario.name!r} references unknown roles: {missing}"
            )
    return scenarios


def write_role_artifact_manifest(
    path: str | Path,
    manifest: RoleArtifactManifest,
) -> None:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved_path.with_suffix(resolved_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, resolved_path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(_normalize_json(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"workflow manifest must contain a JSON object: {path}")
    return payload


def _load_role_artifact_manifest(path: Path) -> RoleArtifactManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RoleArtifactManifest(**payload)


_COMPATIBILITY_FIELDS = (
    "manifest_version",
    "schema_version",
    "student_obs_dim",
    "teacher_obs_dim",
    "teacher_action_dim",
    "teacher_obs_key",
    "teacher_projection",
    "student_projection",
    "student_drop_index",
    "command_info_key",
    "target_height_info_key",
)


def _expected_manifest_values(spec: RoleArtifactSpec) -> dict[str, Any]:
    return {
        "manifest_version": ROLE_ARTIFACT_MANIFEST_VERSION,
        "role": spec.role,
        "task": spec.task,
        "teacher_checkpoint_path": str(spec.teacher_checkpoint_path.resolve()),
        "dataset_path": str(spec.dataset_path.resolve()),
        "schema_version": int(spec.schema_version),
        "student_obs_dim": int(spec.student_obs_dim),
        "teacher_obs_dim": int(spec.teacher_obs_dim),
        "teacher_action_dim": int(spec.teacher_action_dim),
        "teacher_obs_key": spec.teacher_obs_key,
        "teacher_projection": spec.teacher_projection,
        "student_projection": spec.student_projection,
        "student_drop_index": spec.student_drop_index,
        "command_sample_filter": spec.command_sample_filter,
        "command_info_key": spec.command_info_key,
        "target_height_info_key": spec.target_height_info_key,
        "command_xy_threshold": float(spec.command_xy_threshold),
        "command_yaw_threshold": float(spec.command_yaw_threshold),
        "owner_config_sha256": config_fingerprint(spec.owner_config),
    }


def preflight_role_artifact(
    spec: RoleArtifactSpec,
    *,
    require_row_role_labels: bool = False,
) -> RoleArtifactPreflight:
    if not spec.dataset_path.is_file():
        return RoleArtifactPreflight(
            role=spec.role,
            decision=ArtifactDecision.COLLECT,
            dataset_path=spec.dataset_path,
            manifest_path=spec.manifest_path,
            mismatches=("dataset_missing",),
        )
    if not spec.manifest_path.is_file():
        return RoleArtifactPreflight(
            role=spec.role,
            decision=ArtifactDecision.STALE,
            dataset_path=spec.dataset_path,
            manifest_path=spec.manifest_path,
            mismatches=("manifest_missing",),
        )
    try:
        manifest = _load_role_artifact_manifest(spec.manifest_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return RoleArtifactPreflight(
            role=spec.role,
            decision=ArtifactDecision.STALE,
            dataset_path=spec.dataset_path,
            manifest_path=spec.manifest_path,
            mismatches=("manifest_invalid",),
        )

    expected = _expected_manifest_values(spec)
    mismatches = [
        field_name
        for field_name, expected_value in expected.items()
        if getattr(manifest, field_name) != expected_value
    ]
    incompatible = [field_name for field_name in mismatches if field_name in _COMPATIBILITY_FIELDS]
    if incompatible:
        return RoleArtifactPreflight(
            role=spec.role,
            decision=ArtifactDecision.INCOMPATIBLE,
            dataset_path=spec.dataset_path,
            manifest_path=spec.manifest_path,
            mismatches=tuple(incompatible),
        )

    if not spec.teacher_checkpoint_path.is_file():
        mismatches.append("teacher_checkpoint_missing")
    elif file_sha256(spec.teacher_checkpoint_path) != manifest.teacher_checkpoint_sha256:
        mismatches.append("teacher_checkpoint_sha256")
    if file_sha256(spec.dataset_path) != manifest.dataset_sha256:
        mismatches.append("dataset_sha256")
    if require_row_role_labels or spec.target_height_info_key is not None:
        try:
            dataset = load_distillation_dataset(
                spec.dataset_path,
                expected_student_obs_dim=spec.student_obs_dim,
                expected_teacher_obs_dim=spec.teacher_obs_dim,
                expected_teacher_action_dim=spec.teacher_action_dim,
            )
        except (OSError, TypeError, ValueError, KeyError, pickle.UnpicklingError):
            mismatches.append("dataset_schema")
        else:
            if dataset.role_labels is None:
                if require_row_role_labels:
                    mismatches.append("role_labels")
            if spec.target_height_info_key is not None and dataset.target_height is None:
                mismatches.append("target_height")
    if mismatches:
        return RoleArtifactPreflight(
            role=spec.role,
            decision=ArtifactDecision.STALE,
            dataset_path=spec.dataset_path,
            manifest_path=spec.manifest_path,
            mismatches=tuple(mismatches),
        )
    return RoleArtifactPreflight(
        role=spec.role,
        decision=ArtifactDecision.REUSE,
        dataset_path=spec.dataset_path,
        manifest_path=spec.manifest_path,
    )


def preflight_role_artifacts(
    specs: Sequence[RoleArtifactSpec],
    *,
    require_row_role_labels: bool = False,
) -> tuple[RoleArtifactPreflight, ...]:
    roles = [spec.role for spec in specs]
    if len(roles) != len(set(roles)):
        raise ValueError(f"workflow roles must be unique, got {roles}")
    return tuple(
        preflight_role_artifact(spec, require_row_role_labels=require_row_role_labels)
        for spec in specs
    )


def adopt_legacy_role_artifact(spec: RoleArtifactSpec) -> RoleArtifactManifest:
    """Adopt or repair a legacy role dataset under the current role contract."""

    existing_manifest = spec.manifest_path.is_file()
    if existing_manifest:
        preflight = preflight_role_artifact(spec)
        if preflight.decision is not ArtifactDecision.REUSE:
            raise ValueError(
                f"cannot adopt stale role artifact {spec.role!r}: {list(preflight.mismatches)}"
            )
    dataset = load_distillation_dataset(
        spec.dataset_path,
        expected_student_obs_dim=spec.student_obs_dim,
        expected_teacher_obs_dim=spec.teacher_obs_dim,
        expected_teacher_action_dim=spec.teacher_action_dim,
        device="cpu",
    )
    if dataset.teacher_actions is None:
        raise ValueError(f"legacy role dataset has no cached teacher actions: {spec.dataset_path}")
    normalized_legacy = False
    metadata = dataset.metadata
    expected_task_name = (
        spec.owner_config.get("training", {}).get("task_name")
        if isinstance(spec.owner_config.get("training"), Mapping)
        else None
    )
    checks: dict[str, Any] = {
        "teacher_projection": spec.teacher_projection,
        "student_projection": spec.student_projection,
        "teacher_obs_key": spec.teacher_obs_key,
        "student_drop_index": spec.student_drop_index,
        "command_sample_filter": spec.command_sample_filter,
        "command_info_key": spec.command_info_key,
        "command_xy_threshold": float(spec.command_xy_threshold),
        "command_yaw_threshold": float(spec.command_yaw_threshold),
    }
    if expected_task_name not in (None, ""):
        checks["task_name"] = expected_task_name
    mismatches = [key for key, expected in checks.items() if metadata.get(key) != expected]
    if spec.target_height_info_key is not None and dataset.target_height is None:
        mismatches.append("target_height")
    teacher_path = metadata.get("teacher_policy_checkpoint_path")
    if teacher_path in (None, ""):
        mismatches.append("teacher_policy_checkpoint_path")
    elif Path(str(teacher_path)).resolve() != spec.teacher_checkpoint_path.resolve():
        mismatches.append("teacher_policy_checkpoint_path")
    if mismatches:
        raise ValueError(
            f"legacy role dataset metadata is incompatible for {spec.role!r}: {mismatches}"
        )
    if dataset.role_labels is None:
        # Legacy collectors did not persist row labels. The workflow already
        # owns this file as a role-specific artifact, so this is an explicit,
        # lossless schema migration rather than an inferred router decision.
        normalized_metadata = dict(metadata)
        normalized_metadata["legacy_role_labels_adopted"] = True
        dataset = replace(
            dataset,
            metadata=normalized_metadata,
            role_labels=(spec.role,) * dataset.num_samples,
        )
        save_distillation_dataset(spec.dataset_path, dataset)
        normalized_legacy = True
    elif any(label != spec.role for label in dataset.role_labels):
        raise ValueError(
            f"legacy role dataset labels do not match role {spec.role!r}: "
            f"{sorted(set(dataset.role_labels))}"
        )
    if existing_manifest and not normalized_legacy:
        return _load_role_artifact_manifest(spec.manifest_path)
    manifest = create_role_artifact_manifest(spec, num_samples=dataset.num_samples)
    write_role_artifact_manifest(spec.manifest_path, manifest)
    return manifest


def run_bootstrap_workflow(
    *,
    run_dir: str | Path,
    role_specs: Sequence[RoleArtifactSpec],
    collect_role: Callable[[RoleArtifactSpec], int],
    assemble_roles: Callable[[tuple[Path, ...], Path], int],
    update_student: Callable[[Path, Path], int],
    scenario_specs: Sequence[WorkflowScenarioSpec] | None = None,
) -> BootstrapWorkflowResult:
    resolved_run_dir = Path(run_dir)
    manifest_path = resolved_run_dir / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"workflow run already exists; use resume or fork: {manifest_path}")
    if not role_specs:
        raise ValueError("bootstrap workflow requires at least one role")
    active_scenarios = (
        None if scenario_specs is None else _validate_workflow_scenarios(scenario_specs, role_specs)
    )

    require_row_role_labels = active_scenarios is not None and any(
        scenario.kind == "role" for scenario in active_scenarios
    )
    preflight = preflight_role_artifacts(
        role_specs,
        require_row_role_labels=require_row_role_labels,
    )
    blocked = [
        result
        for result in preflight
        if result.decision in (ArtifactDecision.STALE, ArtifactDecision.INCOMPATIBLE)
    ]
    if blocked:
        details = ", ".join(
            f"{result.role}={result.decision.value}:{list(result.mismatches)}" for result in blocked
        )
        raise ValueError(f"workflow artifact preflight failed closed: {details}")

    role_decisions = {result.role: result.decision.value for result in preflight}
    for spec, result in zip(role_specs, preflight, strict=True):
        if result.decision is not ArtifactDecision.COLLECT:
            continue
        num_samples = int(collect_role(spec))
        if num_samples <= 0:
            raise ValueError(
                f"collector for role {spec.role!r} returned invalid sample count {num_samples}"
            )
        manifest = create_role_artifact_manifest(spec, num_samples=num_samples)
        write_role_artifact_manifest(spec.manifest_path, manifest)

    verified = preflight_role_artifacts(role_specs)
    not_reusable = [result for result in verified if result.decision is not ArtifactDecision.REUSE]
    if not_reusable:
        details = ", ".join(
            f"{result.role}={result.decision.value}:{list(result.mismatches)}"
            for result in not_reusable
        )
        raise RuntimeError(f"workflow role artifact verification failed: {details}")

    datasets_dir = resolved_run_dir / "datasets"
    checkpoints_dir = resolved_run_dir / "checkpoints"
    bootstrap_dataset_path = datasets_dir / "bootstrap_merged.pt"
    bootstrap_num_samples = int(
        assemble_roles(tuple(spec.dataset_path for spec in role_specs), bootstrap_dataset_path)
    )
    if bootstrap_num_samples <= 0 or not bootstrap_dataset_path.is_file():
        raise RuntimeError(
            "bootstrap assembler must create the merged dataset and return a positive sample count"
        )

    checkpoint_path = checkpoints_dir / "bootstrap_student.pt"
    bootstrap_updates = int(update_student(bootstrap_dataset_path, checkpoint_path))
    if bootstrap_updates <= 0 or not checkpoint_path.is_file():
        raise RuntimeError(
            "bootstrap updater must create the checkpoint and return a positive update count"
        )

    role_artifacts = [
        asdict(_load_role_artifact_manifest(spec.manifest_path)) for spec in role_specs
    ]
    scenario_by_role = {
        scenario.source_roles[0]: scenario.name
        for scenario in (active_scenarios or ())
        if scenario.kind == "role"
    }
    bootstrap_sources: list[dict[str, Any]] = []
    for spec in role_specs:
        source: dict[str, Any] = {
            "path": str(spec.dataset_path.resolve()),
            "role": spec.role,
        }
        if spec.role in scenario_by_role:
            source["scenario"] = scenario_by_role[spec.role]
            source["preserve_row_role_labels"] = True
        bootstrap_sources.append(source)
    manifest_payload = {
        "manifest_version": 1,
        "run_id": resolved_run_dir.name,
        "mode": "fresh",
        "stage": "BOOTSTRAP_COMPLETE",
        "role_decisions": role_decisions,
        "role_artifacts": role_artifacts,
        "bootstrap_dataset_path": str(bootstrap_dataset_path.resolve()),
        "bootstrap_dataset_sha256": file_sha256(bootstrap_dataset_path),
        "bootstrap_num_samples": bootstrap_num_samples,
        "bootstrap_checkpoint_path": str(checkpoint_path.resolve()),
        "bootstrap_checkpoint_sha256": file_sha256(checkpoint_path),
        "bootstrap_updates": bootstrap_updates,
        "completed_dagger_iterations": 0,
        "dagger_iterations": [],
        "bootstrap_sources": bootstrap_sources,
    }
    if active_scenarios is not None:
        manifest_payload["scenario_specs"] = [scenario.as_dict() for scenario in active_scenarios]
    _write_json_atomic(
        manifest_path,
        manifest_payload,
    )
    return BootstrapWorkflowResult(
        run_dir=resolved_run_dir,
        manifest_path=manifest_path,
        role_decisions=role_decisions,
        bootstrap_dataset_path=bootstrap_dataset_path,
        bootstrap_num_samples=bootstrap_num_samples,
        checkpoint_path=checkpoint_path,
        bootstrap_updates=bootstrap_updates,
    )


def _verified_current_checkpoint(manifest: Mapping[str, Any]) -> Path:
    iterations = manifest.get("dagger_iterations", [])
    if iterations:
        latest = iterations[-1]
        path = Path(str(latest["checkpoint_path"]))
        expected_hash = str(latest["checkpoint_sha256"])
    else:
        path = Path(str(manifest["bootstrap_checkpoint_path"]))
        expected_hash = str(manifest["bootstrap_checkpoint_sha256"])
    if not path.is_file():
        raise FileNotFoundError(f"workflow checkpoint does not exist: {path}")
    if file_sha256(path) != expected_hash:
        raise ValueError(f"workflow checkpoint hash mismatch: {path}")
    return path


def _manifest_sources(manifest: Mapping[str, Any]) -> list[WorkflowDatasetSource]:
    sources = [
        WorkflowDatasetSource(
            path=Path(str(item["path"])),
            role=str(item["role"]),
            scenario=item.get("scenario"),
            preserve_row_role_labels=bool(item.get("preserve_row_role_labels", False)),
        )
        for item in manifest.get("bootstrap_sources", [])
    ]
    for iteration in manifest.get("dagger_iterations", []):
        if iteration.get("scenario_artifacts"):
            sources.extend(
                WorkflowDatasetSource(
                    path=Path(str(item["dataset_path"])),
                    role=str(item["source_roles"][0]),
                    scenario=str(item["scenario"]),
                    preserve_row_role_labels=True,
                )
                for item in iteration["scenario_artifacts"]
            )
            continue
        sources.extend(
            WorkflowDatasetSource(
                path=Path(str(item["dataset_path"])),
                role=str(item["role"]),
            )
            for item in iteration["role_artifacts"]
        )
    for source in sources:
        if not source.path.is_file():
            raise FileNotFoundError(f"workflow cumulative source does not exist: {source.path}")
    return sources


def run_multirole_dagger_workflow(
    *,
    run_dir: str | Path,
    role_specs: Sequence[RoleArtifactSpec],
    target_iterations: int,
    collect_role: Callable[[RoleArtifactSpec, Path, int, Path], int],
    aggregate_datasets: Callable[[tuple[WorkflowDatasetSource, ...], Path], int],
    update_student: Callable[[Path, Path, Path], int | WorkflowStudentUpdateResult],
    scenario_specs: Sequence[WorkflowScenarioSpec] | None = None,
    collect_scenario: Callable[
        [WorkflowScenarioSpec, Path, int, Path],
        int | WorkflowScenarioCollectionResult,
    ]
    | None = None,
    execution_mode: str = "legacy",
    scenario_collector: WorkflowScenarioCollector | None = None,
    performance_context: DistillationPerformanceRunContext | None = None,
    performance_clock: Callable[[], float] = time.perf_counter,
    status_callback: Callable[[str], None] | None = None,
    iteration_callback: Callable[[int, int], None] | None = None,
    runtime_sentinel: Callable[[str], None] | None = None,
) -> DaggerWorkflowResult:
    resolved_run_dir = Path(run_dir)
    manifest_path = resolved_run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"workflow manifest does not exist: {manifest_path}")
    if target_iterations < 0:
        raise ValueError(f"target_iterations must be non-negative, got {target_iterations}")
    manifest = _load_json(manifest_path)
    active_scenarios = (
        None if scenario_specs is None else _validate_workflow_scenarios(scenario_specs, role_specs)
    )
    if execution_mode not in {"legacy", "persistent_async"}:
        raise ValueError(
            f"execution_mode must be 'legacy' or 'persistent_async', got {execution_mode!r}"
        )
    if execution_mode == "legacy" and scenario_collector is not None:
        raise ValueError("legacy execution_mode forbids scenario_collector")
    if performance_context is not None and performance_context.execution_mode != execution_mode:
        raise ValueError("performance_context execution_mode mismatch")
    if execution_mode == "persistent_async":
        if active_scenarios is None:
            raise ValueError("persistent_async execution_mode requires scenario_specs")
        if collect_scenario is not None:
            raise ValueError("persistent_async execution_mode forbids collect_scenario")
        if scenario_collector is None:
            raise ValueError("persistent_async execution_mode requires scenario_collector")
        if performance_context is None:
            raise ValueError("persistent_async execution_mode requires performance_context")
    if active_scenarios is not None:
        expected_scenarios = [scenario.as_dict() for scenario in active_scenarios]
        if manifest.get("scenario_specs") != expected_scenarios:
            raise ValueError("workflow scenario specs do not match run manifest")
    completed = int(manifest.get("completed_dagger_iterations", 0))
    if target_iterations < completed:
        raise ValueError(
            f"target_iterations {target_iterations} is below completed iterations {completed}"
        )
    metrics_path = resolved_run_dir / "distillation_metrics.json"
    metrics_recorder: DistillationMetricsRecorder | None = None
    if performance_context is not None:
        metrics_recorder = DistillationMetricsRecorder()
        if metrics_path.is_file():
            recorded_path = manifest.get("distillation_metrics_path")
            if recorded_path != str(metrics_path.resolve()):
                raise ValueError("distillation metrics manifest path mismatch")
            recorded_hash = manifest.get("distillation_metrics_sha256")
            if recorded_hash != file_sha256(metrics_path):
                raise ValueError("distillation metrics manifest hash mismatch")
            loaded_metrics = load_distillation_metrics(metrics_path)
            if (
                loaded_metrics.records[0].identity.run_signature()
                != performance_context.run_signature()
            ):
                raise ValueError("distillation metric identity drift within one artifact")
            if manifest.get("distillation_metrics_record_count") != len(loaded_metrics.records):
                raise ValueError("distillation metrics manifest record count mismatch")
            for record in loaded_metrics.records:
                metrics_recorder.add(record)
        elif completed > 0:
            raise FileNotFoundError(
                "completed persistent workflow is missing distillation_metrics.json"
            )
    require_row_role_labels = active_scenarios is not None and any(
        scenario.kind == "role" for scenario in active_scenarios
    )
    role_preflight = preflight_role_artifacts(
        role_specs,
        require_row_role_labels=require_row_role_labels,
    )
    not_reusable = [item for item in role_preflight if item.decision is not ArtifactDecision.REUSE]
    if not_reusable:
        details = ", ".join(
            f"{item.role}={item.decision.value}:{list(item.mismatches)}" for item in not_reusable
        )
        raise ValueError(f"DAgger role artifacts no longer match the run contract: {details}")

    current_checkpoint = _verified_current_checkpoint(manifest)
    cumulative_sources = _manifest_sources(manifest)
    cumulative_num_samples = int(
        manifest.get("bootstrap_num_samples", sum(1 for _ in cumulative_sources))
    )

    def emit_status(message: str) -> None:
        _progress(message)
        if status_callback is not None:
            status_callback(message)

    for iteration in range(completed + 1, target_iterations + 1):
        input_checkpoint_path = str(current_checkpoint.resolve())
        input_checkpoint_sha256 = file_sha256(current_checkpoint)
        iteration_dir = resolved_run_dir / "datasets" / f"dagger_iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        role_artifacts: list[dict[str, Any]] = []
        scenario_artifacts: list[dict[str, Any]] = []
        if iteration_callback is not None:
            iteration_callback(iteration, target_iterations)
        emit_status(f"iteration={iteration}/{target_iterations} collecting datasets")
        input_weight_version: int | None = None
        if execution_mode == "persistent_async":
            assert scenario_collector is not None
            if runtime_sentinel is not None:
                runtime_sentinel(f"workflow/iteration_{iteration}/before_activate_checkpoint")
            input_weight_version = int(scenario_collector.activate_checkpoint(current_checkpoint))
            if runtime_sentinel is not None:
                runtime_sentinel(f"workflow/iteration_{iteration}/after_activate_checkpoint")
            if input_weight_version < 0:
                raise ValueError(
                    "persistent scenario collector returned a negative weight version: "
                    f"{input_weight_version}"
                )
        if active_scenarios is None:
            for spec in role_specs:
                output_path = iteration_dir / f"{spec.role}.pt"
                output_spec = replace(spec, dataset_path=output_path)
                num_samples = int(
                    collect_role(output_spec, current_checkpoint, iteration, output_path)
                )
                if num_samples <= 0:
                    raise ValueError(
                        f"DAgger collector for role {spec.role!r} returned {num_samples} samples"
                    )
                emit_status(
                    f"iteration={iteration} collected role={spec.role} samples={num_samples}"
                )
                artifact_manifest = create_role_artifact_manifest(
                    output_spec,
                    num_samples=num_samples,
                )
                write_role_artifact_manifest(output_spec.manifest_path, artifact_manifest)
                role_artifacts.append(asdict(artifact_manifest))
                cumulative_sources.append(WorkflowDatasetSource(output_path, spec.role))
        else:
            role_specs_by_name = {spec.role: spec for spec in role_specs}
            if execution_mode == "legacy" and collect_scenario is None:
                raise ValueError("scenario workflow requires collect_scenario callback")
            for scenario in active_scenarios:
                output_path = iteration_dir / f"{scenario.name}.pt"
                persistent_result: DaggerCollectResult | None = None
                legacy_result: WorkflowScenarioCollectionResult | None = None
                if execution_mode == "legacy":
                    assert collect_scenario is not None
                    raw_result = collect_scenario(
                        scenario, current_checkpoint, iteration, output_path
                    )
                    if isinstance(raw_result, WorkflowScenarioCollectionResult):
                        legacy_result = raw_result
                        num_samples = raw_result.num_samples
                    else:
                        num_samples = int(raw_result)
                    if performance_context is not None and legacy_result is None:
                        raise ValueError("legacy performance_context requires rich scenario result")
                else:
                    assert scenario_collector is not None
                    assert input_weight_version is not None
                    request = DaggerCollectRequest(
                        request_id=f"dagger-{iteration}-{scenario.name}",
                        scenario=scenario.name,
                        iteration=iteration,
                        checkpoint_path=input_checkpoint_path,
                        output_path=str(output_path.resolve()),
                        expected_weight_version=input_weight_version,
                    )
                    if runtime_sentinel is not None:
                        runtime_sentinel(
                            f"workflow/iteration_{iteration}/scenario_{scenario.name}/before_collect"
                        )
                    persistent_result = scenario_collector.collect(request)
                    if runtime_sentinel is not None:
                        runtime_sentinel(
                            f"workflow/iteration_{iteration}/scenario_{scenario.name}/after_collect"
                        )
                    validate_dagger_collect_result(request, persistent_result)
                    num_samples = int(persistent_result.num_samples)
                if num_samples <= 0:
                    raise ValueError(
                        f"DAgger collector for scenario {scenario.name!r} returned {num_samples} samples"
                    )
                emit_status(
                    f"iteration={iteration} collected scenario={scenario.name} "
                    f"samples={num_samples}"
                )
                if persistent_result is not None or legacy_result is not None:
                    assert performance_context is not None
                    assert metrics_recorder is not None
                    if persistent_result is not None:
                        observation_payloads = persistent_result.metadata.get(
                            "performance_stage_observations"
                        )
                        if not isinstance(observation_payloads, list):
                            raise ValueError(
                                "persistent result performance_stage_observations are missing"
                            )
                        observations = tuple(
                            DistillationStageObservation.from_dict(payload)
                            for payload in observation_payloads
                        )
                        worker_pid = persistent_result.worker_pid
                        request_id = persistent_result.request_id
                        weight_version = persistent_result.observed_weight_version
                        schema_version = persistent_result.metadata.get(
                            "performance_metrics_schema_version"
                        )
                    else:
                        assert legacy_result is not None
                        observations = legacy_result.performance_stage_observations
                        worker_pid = legacy_result.worker_pid
                        request_id = f"dagger-{iteration}-{scenario.name}"
                        weight_version = None
                        schema_version = legacy_result.performance_metrics_schema_version
                    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
                        raise ValueError(
                            "collector performance_metrics_schema_version must be an integer"
                        )
                    records = performance_context.enrich_request(
                        outer_iteration=iteration,
                        scenario=scenario.name,
                        worker_pid=worker_pid,
                        request_id=request_id,
                        checkpoint_path=input_checkpoint_path,
                        checkpoint_sha256=input_checkpoint_sha256,
                        weight_version=weight_version,
                        schema_version=schema_version,
                        observations=observations,
                    )
                    for record in records:
                        metrics_recorder.add(record)
                    metrics_recorder.write(
                        metrics_path,
                        required_stages=(
                            LEGACY_REQUEST_STAGE_NAMES
                            if execution_mode == "legacy"
                            else PERSISTENT_REQUEST_STAGE_NAMES
                        ),
                    )
                    persisted_metrics = load_distillation_metrics(metrics_path)
                    if persisted_metrics.records != metrics_recorder.records:
                        raise RuntimeError(
                            "distillation metrics reload differs from recorder state"
                        )
                if scenario.kind == "role":
                    source_role = scenario.source_roles[0]
                    output_spec = replace(
                        role_specs_by_name[source_role],
                        dataset_path=output_path,
                    )
                    artifact_manifest = create_role_artifact_manifest(
                        output_spec,
                        num_samples=num_samples,
                    )
                    write_role_artifact_manifest(output_spec.manifest_path, artifact_manifest)
                    role_artifacts.append(asdict(artifact_manifest))
                scenario_artifact = {
                    "scenario": scenario.name,
                    "kind": scenario.kind,
                    "source_roles": list(scenario.source_roles),
                    "quota": scenario.quota,
                    "dataset_path": str(output_path.resolve()),
                    "dataset_sha256": file_sha256(output_path),
                    "num_samples": num_samples,
                    "input_checkpoint_path": input_checkpoint_path,
                    "input_checkpoint_sha256": input_checkpoint_sha256,
                }
                if persistent_result is not None:
                    scenario_artifact.update(
                        {
                            "input_weight_version": persistent_result.observed_weight_version,
                            "collector_worker_pid": persistent_result.worker_pid,
                            "collector_metrics": dict(persistent_result.metrics),
                            "collector_metadata": dict(persistent_result.metadata),
                        }
                    )
                scenario_artifacts.append(scenario_artifact)
                cumulative_sources.append(
                    WorkflowDatasetSource(
                        output_path,
                        scenario.source_roles[0],
                        scenario=scenario.name,
                        preserve_row_role_labels=True,
                    )
                )

        aggregate_path = (
            resolved_run_dir / "datasets" / f"dagger_iteration_{iteration}_aggregate.pt"
        )
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_start = float(performance_clock())
        if runtime_sentinel is not None:
            runtime_sentinel(f"workflow/iteration_{iteration}/before_aggregate")
        cumulative_num_samples = int(aggregate_datasets(tuple(cumulative_sources), aggregate_path))
        if runtime_sentinel is not None:
            runtime_sentinel(f"workflow/iteration_{iteration}/after_aggregate")
        aggregate_seconds = float(performance_clock()) - aggregate_start
        emit_status(
            f"iteration={iteration} aggregated samples={cumulative_num_samples} "
            f"path={aggregate_path}"
        )
        if cumulative_num_samples <= 0 or not aggregate_path.is_file():
            raise RuntimeError(
                "DAgger aggregator must create the cumulative dataset and return a positive count"
            )
        output_checkpoint = resolved_run_dir / "checkpoints" / f"dagger_iteration_{iteration}.pt"
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        emit_status(f"iteration={iteration} updating student checkpoint={output_checkpoint}")
        raw_update_result = update_student(aggregate_path, current_checkpoint, output_checkpoint)
        if isinstance(raw_update_result, WorkflowStudentUpdateResult):
            learner_result = raw_update_result
            updates = learner_result.updates
        else:
            learner_result = None
            updates = int(raw_update_result)
        if updates <= 0 or not output_checkpoint.is_file():
            raise RuntimeError(
                "DAgger updater must create the next checkpoint and return a positive count"
            )
        emit_status(
            f"iteration={iteration} update complete updates={updates} "
            f"checkpoint={output_checkpoint}"
        )
        if learner_result is not None:
            assert performance_context is not None
            assert metrics_recorder is not None
            workflow_observations = (
                DistillationStageObservation(
                    stage="cumulative_aggregation",
                    duration_seconds=aggregate_seconds,
                    row_count=cumulative_num_samples,
                    env_step_count=0,
                    success=True,
                    error=None,
                    cleanup_state="not_applicable",
                ),
                *learner_result.performance_stage_observations,
            )
            workflow_records = performance_context.enrich_workflow_iteration(
                outer_iteration=iteration,
                worker_pid=os.getpid(),
                checkpoint_path=input_checkpoint_path,
                checkpoint_sha256=input_checkpoint_sha256,
                weight_version=input_weight_version,
                observations=workflow_observations,
            )
            for record in workflow_records:
                metrics_recorder.add(record)
            metrics_recorder.write(
                metrics_path,
                required_stages=WORKFLOW_ITERATION_STAGE_NAMES,
            )
            persisted_metrics = load_distillation_metrics(metrics_path)
            if persisted_metrics.records != metrics_recorder.records:
                raise RuntimeError("workflow metrics reload differs from recorder state")
        iteration_record = {
            "iteration": iteration,
            "input_checkpoint_path": input_checkpoint_path,
            "input_checkpoint_sha256": input_checkpoint_sha256,
            "role_artifacts": role_artifacts,
            "aggregate_dataset_path": str(aggregate_path.resolve()),
            "aggregate_dataset_sha256": file_sha256(aggregate_path),
            "aggregate_num_samples": cumulative_num_samples,
            "checkpoint_path": str(output_checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(output_checkpoint),
            "updates": updates,
        }
        if active_scenarios is not None:
            iteration_record["scenario_artifacts"] = scenario_artifacts
        if input_weight_version is not None:
            iteration_record["collection_execution_mode"] = execution_mode
            iteration_record["input_weight_version"] = input_weight_version
        if metrics_recorder is not None:
            manifest["distillation_metrics_path"] = str(metrics_path.resolve())
            manifest["distillation_metrics_sha256"] = file_sha256(metrics_path)
            manifest["distillation_metrics_record_count"] = len(metrics_recorder.records)
        manifest.setdefault("dagger_iterations", []).append(iteration_record)
        manifest["completed_dagger_iterations"] = iteration
        manifest["stage"] = f"DAGGER_ITERATION_{iteration}_COMPLETE"
        _write_json_atomic(manifest_path, manifest)
        current_checkpoint = output_checkpoint

    return DaggerWorkflowResult(
        run_dir=resolved_run_dir,
        manifest_path=manifest_path,
        completed_iterations=target_iterations,
        checkpoint_path=current_checkpoint,
        cumulative_num_samples=cumulative_num_samples,
    )


def finalize_workflow_performance(
    *,
    run_dir: str | Path,
    performance_context: DistillationPerformanceRunContext,
    cleanup_duration_seconds: float,
    cleanup_report: Mapping[str, Any],
) -> Path:
    """Persist the post-close cleanup metric and lifecycle report atomically."""

    if not math.isfinite(cleanup_duration_seconds) or cleanup_duration_seconds < 0:
        raise ValueError("cleanup_duration_seconds must be finite and non-negative")
    if not isinstance(cleanup_report, Mapping):
        raise TypeError("cleanup_report must be a mapping")
    report = dict(cleanup_report)
    if performance_context.execution_mode == "persistent_async":
        worker_pid = report.get("worker_pid")
        resource_counters = report.get("resource_counters")
        if isinstance(worker_pid, bool) or not isinstance(worker_pid, int) or worker_pid <= 0:
            raise ValueError("persistent cleanup_report requires positive worker_pid")
        if not isinstance(resource_counters, Mapping):
            raise ValueError("persistent cleanup_report requires resource_counters mapping")
    else:
        worker_pid = os.getpid()

    resolved_run_dir = Path(run_dir).resolve()
    manifest_path = resolved_run_dir / "run_manifest.json"
    metrics_path = resolved_run_dir / "distillation_metrics.json"
    manifest = _load_json(manifest_path)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"workflow cleanup requires distillation metrics: {metrics_path}")
    existing_cleanup = manifest.get("performance_cleanup")
    if existing_cleanup is not None:
        if not isinstance(existing_cleanup, Mapping) or existing_cleanup.get("state") != "complete":
            raise ValueError("existing performance_cleanup must be complete")
        if manifest.get("distillation_metrics_path") != str(metrics_path.resolve()):
            raise ValueError("cleanup metrics manifest path mismatch")
        if manifest.get("distillation_metrics_sha256") != file_sha256(metrics_path):
            raise ValueError("cleanup metrics manifest hash mismatch")
        existing_metrics = load_distillation_metrics(metrics_path)
        if manifest.get("distillation_metrics_record_count") != len(existing_metrics.records):
            raise ValueError("cleanup metrics manifest record count mismatch")
        cleanup_records = [
            record for record in existing_metrics.records if record.stage == "cleanup"
        ]
        if len(cleanup_records) != 1:
            raise ValueError("completed cleanup requires exactly one cleanup record")
        if cleanup_records[0].identity.run_signature() != performance_context.run_signature():
            raise ValueError("existing cleanup performance context identity drift")
        if cleanup_records[0].cleanup_state != "complete":
            raise ValueError("existing cleanup record must be complete")
        return metrics_path
    iterations = manifest.get("dagger_iterations", [])
    if not isinstance(iterations, list) or not iterations:
        raise ValueError("workflow cleanup requires a completed DAgger iteration")
    last_iteration = iterations[-1]
    if not isinstance(last_iteration, Mapping):
        raise ValueError("last DAgger iteration manifest entry must be a mapping")

    loaded_metrics = load_distillation_metrics(metrics_path)
    if not loaded_metrics.records:
        raise ValueError("workflow cleanup requires non-empty distillation metrics")
    if loaded_metrics.records[0].identity.run_signature() != performance_context.run_signature():
        raise ValueError("cleanup performance context identity drift")
    recorder = DistillationMetricsRecorder()
    for record in loaded_metrics.records:
        recorder.add(record)

    input_weight_version = last_iteration.get("input_weight_version")
    cleanup_observation = DistillationStageObservation(
        stage="cleanup",
        duration_seconds=cleanup_duration_seconds,
        row_count=0,
        env_step_count=0,
        success=True,
        error=None,
        cleanup_state="complete",
    )
    recorder.add(
        performance_context.enrich_cleanup(
            outer_iteration=int(last_iteration["iteration"]),
            worker_pid=worker_pid,
            checkpoint_path=str(last_iteration["input_checkpoint_path"]),
            checkpoint_sha256=str(last_iteration["input_checkpoint_sha256"]),
            weight_version=(None if input_weight_version is None else int(input_weight_version)),
            observation=cleanup_observation,
        )
    )
    recorder.write(metrics_path, required_stages=("cleanup",))
    persisted_metrics = load_distillation_metrics(metrics_path)
    if persisted_metrics.records != recorder.records:
        raise RuntimeError("cleanup metrics reload differs from recorder state")

    manifest["performance_cleanup"] = {
        "state": "complete",
        "duration_seconds": cleanup_duration_seconds,
        "report": report,
    }
    manifest["distillation_metrics_path"] = str(metrics_path.resolve())
    manifest["distillation_metrics_sha256"] = file_sha256(metrics_path)
    manifest["distillation_metrics_record_count"] = len(recorder.records)
    _write_json_atomic(manifest_path, manifest)
    return metrics_path


def fork_workflow_run(
    *,
    parent_run_dir: str | Path,
    run_dir: str | Path,
    checkpoint_override: str | Path | None = None,
    dataset_override: str | Path | None = None,
) -> Path:
    parent_manifest_path = Path(parent_run_dir) / "run_manifest.json"
    if not parent_manifest_path.is_file():
        raise FileNotFoundError(f"parent workflow manifest does not exist: {parent_manifest_path}")
    resolved_run_dir = Path(run_dir)
    manifest_path = resolved_run_dir / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"fork workflow run already exists: {manifest_path}")
    if (checkpoint_override is None) != (dataset_override is None):
        raise ValueError("fork checkpoint and dataset overrides must be provided together")
    parent = _load_json(parent_manifest_path)
    checkpoint = _verified_current_checkpoint(parent)
    parent_iterations = parent.get("dagger_iterations", [])
    if parent_iterations:
        latest = parent_iterations[-1]
        bootstrap_dataset_path = Path(str(latest["aggregate_dataset_path"]))
        bootstrap_dataset_sha256 = str(latest["aggregate_dataset_sha256"])
        bootstrap_num_samples = int(latest["aggregate_num_samples"])
        bootstrap_sources = [
            {
                "path": str(source.path.resolve()),
                "role": source.role,
                **({"scenario": source.scenario} if source.scenario is not None else {}),
                "preserve_row_role_labels": source.preserve_row_role_labels,
            }
            for source in _manifest_sources(parent)
        ]
    else:
        bootstrap_dataset_path = Path(str(parent["bootstrap_dataset_path"]))
        bootstrap_dataset_sha256 = str(parent["bootstrap_dataset_sha256"])
        bootstrap_num_samples = int(parent["bootstrap_num_samples"])
        bootstrap_sources = list(parent.get("bootstrap_sources", []))
    if not bootstrap_dataset_path.is_file():
        raise FileNotFoundError(f"parent workflow dataset does not exist: {bootstrap_dataset_path}")
    if file_sha256(bootstrap_dataset_path) != bootstrap_dataset_sha256:
        raise ValueError(f"parent workflow dataset hash mismatch: {bootstrap_dataset_path}")
    payload = {
        "manifest_version": 1,
        "run_id": resolved_run_dir.name,
        "mode": "fork",
        "parent_run_manifest": str(parent_manifest_path.resolve()),
        "parent_run_manifest_sha256": file_sha256(parent_manifest_path),
        "stage": "BOOTSTRAP_COMPLETE",
        "role_decisions": {item["role"]: "REUSE" for item in parent["role_artifacts"]},
        "role_artifacts": list(parent["role_artifacts"]),
        "bootstrap_sources": bootstrap_sources,
        "bootstrap_dataset_path": str(bootstrap_dataset_path.resolve()),
        "bootstrap_dataset_sha256": bootstrap_dataset_sha256,
        "bootstrap_num_samples": bootstrap_num_samples,
        "bootstrap_checkpoint_path": str(checkpoint.resolve()),
        "bootstrap_checkpoint_sha256": file_sha256(checkpoint),
        "bootstrap_updates": 0,
        "completed_dagger_iterations": 0,
        "dagger_iterations": [],
        "scenario_specs": list(parent.get("scenario_specs", [])),
    }
    if checkpoint_override is not None and dataset_override is not None:
        seed_checkpoint = Path(checkpoint_override).resolve()
        seed_dataset = Path(dataset_override).resolve()
        loaded = load_distillation_student_policy(seed_checkpoint, device="cpu")
        runtime = loaded.distill_runtime_cfg
        if Path(str(runtime.get("dataset_path", ""))).resolve() != seed_dataset:
            raise ValueError("fork seed checkpoint does not name the supplied dataset")
        if runtime.get("student_init_checkpoint_sha256") != file_sha256(checkpoint):
            raise ValueError("fork seed checkpoint does not descend from the parent checkpoint")
        role_contract = parent["role_artifacts"][0]
        dataset = load_distillation_dataset(
            seed_dataset,
            expected_student_obs_dim=int(role_contract["student_obs_dim"]),
            expected_teacher_obs_dim=int(role_contract["teacher_obs_dim"]),
            expected_teacher_action_dim=int(role_contract["teacher_action_dim"]),
            device="cpu",
        )
        if dataset.teacher_actions is None or dataset.role_labels is None:
            raise ValueError("fork seed dataset requires cached targets and row role labels")
        roles = {item["role"] for item in parent["role_artifacts"]}
        if set(dataset.role_labels) - roles:
            raise ValueError("fork seed dataset contains unknown role labels")
        if (
            loaded.obs_dim != dataset.student_obs_dim
            or loaded.action_dim != dataset.teacher_action_dim
        ):
            raise ValueError("fork seed checkpoint and dataset dimensions differ")
        payload.update(
            {
                "bootstrap_checkpoint_path": str(seed_checkpoint),
                "bootstrap_checkpoint_sha256": file_sha256(seed_checkpoint),
                "bootstrap_dataset_path": str(seed_dataset),
                "bootstrap_dataset_sha256": file_sha256(seed_dataset),
                "bootstrap_num_samples": dataset.num_samples,
                "bootstrap_sources": [
                    {
                        "path": str(seed_dataset),
                        "role": role_contract["role"],
                        "preserve_row_role_labels": True,
                    }
                ],
                "seed_kind": "saved_student_update",
            }
        )
    _write_json_atomic(manifest_path, payload)
    return manifest_path
