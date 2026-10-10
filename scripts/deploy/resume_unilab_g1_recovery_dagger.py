"""Finish an interrupted saved-data update, then train only the remaining rounds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf
from scripts.train_distill import run_single_entry_workflow

from unilab.algos.torch.distill.workflow import recover_workflow_student_update


def continue_after_recovered_update(receipt: dict, recovery_dir: Path) -> dict:
    """Assemble the existing fork connector from the verified saved-update seed."""
    if receipt["remaining_iterations"] == 0:
        return {"checkpoint_path": receipt["checkpoint_path"], "remaining_iterations": 0}
    request = json.loads(Path(receipt["original_request_path"]).read_text())
    cfg = OmegaConf.create(request["config"])
    cfg.training.offline_init_checkpoint = receipt["checkpoint_path"]
    cfg.training.workflow.mode = "fork"
    cfg.training.workflow.parent_run_dir = receipt["parent_run_dir"]
    cfg.training.workflow.fork_checkpoint_path = receipt["checkpoint_path"]
    cfg.training.workflow.fork_dataset_path = receipt["dataset_path"]
    cfg.training.workflow.run_dir = str(recovery_dir / "continued")
    cfg.training.workflow.dagger_iterations = receipt["remaining_iterations"]
    cfg.training.workflow.isolate_offline_stages = True
    print(
        f"[distill-recovery] completed original round {receipt['recovered_original_iteration']}; "
        f"training {receipt['remaining_iterations']} remaining rounds in {cfg.training.workflow.run_dir}",
        flush=True,
    )
    return run_single_entry_workflow(cfg)


def main() -> None:
    os.environ.setdefault("UNILAB_DISTILL_PROGRESS", "1")
    os.environ.setdefault("UNILAB_DISTILL_PROGRESS_INTERVAL", "500")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path)
    args = parser.parse_args()
    parent = args.run_dir.resolve()
    recovery = args.recovery_dir or parent.with_name(
        parent.name + "_recovered_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    recovery = recovery.resolve()
    print(f"[distill-recovery] recovery output: {recovery}", flush=True)
    receipt = recover_workflow_student_update(
        parent_run_dir=parent,
        recovery_run_dir=recovery,
        entrypoint=ROOT / "scripts/train_distill.py",
    )
    result = continue_after_recovered_update(receipt, recovery)
    print(
        json.dumps({"recovered_update": receipt, "continuation": result}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
