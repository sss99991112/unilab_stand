#!/usr/bin/env bash
set -euo pipefail

UNILAB_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$UNILAB_REPO_ROOT"
export UNILAB_G1_WALK_HEIGHT_TEACHER="${UNILAB_G1_WALK_HEIGHT_TEACHER:-$UNILAB_REPO_ROOT/logs/G1WalkHeight/20260724-020039_g1_walk_height_nominal_0754/model_5000.pt}"
export UNILAB_G1_STAND_HEIGHT_TEACHER="${UNILAB_G1_STAND_HEIGHT_TEACHER:-$UNILAB_REPO_ROOT/logs/G1StandHeight/20260724-013445_g1_stand_height_stage2_065_0754/model_5000.pt}"
export UNILAB_G1_RECOVERY_TEACHER="${UNILAB_G1_RECOVERY_TEACHER:-$UNILAB_REPO_ROOT/logs/fast_sac/G1Recovery/2026-10-09_19-37-31_mujoco/model_12000.pt}"
UNILAB_SEED_RUN="$UNILAB_REPO_ROOT/logs/distill_workflow/20261010-182545_recovery_dagger1_saved_data"
export UNILAB_G1_RECOVERY_MOE_INIT="$UNILAB_SEED_RUN/dagger_iteration_1.pt"
UNILAB_SEED_DATA="$UNILAB_REPO_ROOT/logs/distill_debug/20261010-175817_recovery_aggregate_check/cycle-000001.pt"
UNILAB_PARENT_RUN="$UNILAB_REPO_ROOT/logs/distill_workflow/20261010-171936_stand_height_walk_recovery"
UNILAB_DAGGER_RUN="${1:-$UNILAB_REPO_ROOT/logs/distill_workflow/$(date +%Y%m%d-%H%M%S)_recovery_dagger8}"

for UNILAB_REQUIRED_FILE in "$UNILAB_G1_WALK_HEIGHT_TEACHER" "$UNILAB_G1_STAND_HEIGHT_TEACHER" "$UNILAB_G1_RECOVERY_TEACHER" "$UNILAB_G1_RECOVERY_MOE_INIT" "$UNILAB_SEED_DATA" "$UNILAB_PARENT_RUN/run_manifest.json"; do
  test -f "$UNILAB_REQUIRED_FILE" || { printf 'Missing required artifact: %s\n' "$UNILAB_REQUIRED_FILE" >&2; exit 1; }
done
printf 'Student: %s\nRun: %s\nDAgger rounds: 8 additional rounds\n' "$UNILAB_G1_RECOVERY_MOE_INIT" "$UNILAB_DAGGER_RUN"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" HYDRA_FULL_ERROR=1 \
  UNILAB_DISTILL_PROGRESS=1 UNILAB_DISTILL_PROGRESS_INTERVAL=500 \
  uv run --no-sync python scripts/train_distill.py \
  task=g1_walk_height_nominal/mujoco workflow=g1_stand_height_walk_recovery \
  training.workflow.enabled=true training.device=cuda:0 \
  training.workflow.mode=fork training.workflow.dagger_iterations=8 \
  training.workflow.isolate_offline_stages=true \
  "training.workflow.parent_run_dir=$UNILAB_PARENT_RUN" \
  "training.workflow.fork_checkpoint_path=$UNILAB_G1_RECOVERY_MOE_INIT" \
  "training.workflow.fork_dataset_path=$UNILAB_SEED_DATA" \
  "training.workflow.run_dir=$UNILAB_DAGGER_RUN" \
  "training.workflow.artifact_dir=$UNILAB_PARENT_RUN/role_artifacts"
