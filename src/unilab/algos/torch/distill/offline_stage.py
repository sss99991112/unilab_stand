"""Fresh-process boundaries for saved-data workflow aggregation and learning."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OfflineStageProcessError(subprocess.CalledProcessError):
    """Keep the worker traceback available after the workflow display closes."""

    def __init__(self, error: subprocess.CalledProcessError, stderr_path: Path):
        with stderr_path.open("rb") as stream:
            stream.seek(max(0, stderr_path.stat().st_size - 16384))
            tail = stream.read().decode("utf-8", errors="replace")
        super().__init__(error.returncode, error.cmd, output=error.output, stderr=tail)
        self.stderr_path = stderr_path

    def __str__(self) -> str:
        return f"{super().__str__()}\nWorker stderr: {self.stderr_path}\n{self.stderr}"


def read_offline_stage_result(
    *, request_path: Path, result_path: Path, output_path: Path, operation: str
) -> dict[str, Any]:
    """Verify a persisted offline acknowledgement against its exact input/output."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("request_sha256") != _sha256(request_path):
        raise ValueError("offline worker request hash mismatch")
    if result.get("operation") != operation:
        raise ValueError("offline worker operation mismatch")
    if result.get("output_path") != str(output_path.resolve()):
        raise ValueError("offline worker output path mismatch")
    if result.get("output_sha256") != _sha256(output_path):
        raise ValueError("offline worker output hash mismatch")
    if type(result.get("worker_pid")) is not int or result["worker_pid"] <= 0:
        raise ValueError("offline worker PID must be a positive integer")
    return result


def run_offline_stage_process(
    *,
    entrypoint: Path,
    config: Mapping[str, Any],
    operation: str,
    arguments: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Invoke the existing offline owner once, then verify its saved output."""
    if operation not in {"aggregate", "update"}:
        raise ValueError(f"unsupported offline operation: {operation}")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request_path = output_path.with_name(output_path.name + ".offline-request.json")
    result_path = output_path.with_name(output_path.name + ".offline-result.json")
    stderr_path = output_path.with_name(output_path.name + ".offline-stderr.log")
    if any(path.exists() for path in (output_path, request_path, result_path, stderr_path)):
        raise FileExistsError(f"offline stage requires a fresh output: {output_path}")
    with request_path.open("x", encoding="utf-8") as stream:
        json.dump(
            {"operation": operation, "config": dict(config), "arguments": dict(arguments)},
            stream,
            sort_keys=True,
        )
    try:
        with stderr_path.open("x", encoding="utf-8") as stderr_stream:
            subprocess.run(
                [
                    sys.executable,
                    str(entrypoint.resolve()),
                    "--offline-stage-request",
                    str(request_path),
                    "--offline-stage-result",
                    str(result_path),
                ],
                cwd=entrypoint.resolve().parents[1],
                stderr=stderr_stream,
                check=True,
            )
    except subprocess.CalledProcessError as error:
        raise OfflineStageProcessError(error, stderr_path) from error
    result = read_offline_stage_result(
        request_path=request_path,
        result_path=result_path,
        output_path=output_path,
        operation=operation,
    )
    if result.get("worker_pid") == os.getpid():
        raise ValueError("offline stage did not run in a separate process")
    return dict(result["result"])
