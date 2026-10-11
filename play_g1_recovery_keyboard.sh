#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CHECKPOINT_PATH="${HOME}/dagger_iteration_2.pt"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_PATH="$(dirname "$PROJECT_DIR")/dagger_iteration_2.pt"
fi
SUPINE=false
EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --checkpoint-path)
            if [ "$#" -lt 2 ]; then
                echo "--checkpoint-path requires a file path" >&2
                exit 2
            fi
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --supine)
            SUPINE=true
            shift
            ;;
        --help|-h)
            echo "Usage: bash play_g1_recovery_keyboard.sh [--checkpoint-path FILE] [--supine] [--dry-run] [Hydra overrides]"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Student checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 2
fi

cd "$PROJECT_DIR"
export UV_NO_SYNC="${UV_NO_SYNC:-true}"
exec bash "$PROJECT_DIR/start.sh" \
    --algo distill --task g1_recovery_combined --sim mujoco \
    --checkpoint-path "$CHECKPOINT_PATH" \
    training.device=cpu \
    interactive.start_paused=true \
    "env.recovery_reset_supine=$SUPINE" \
    "env.commands.height_range=[0.65,0.754]" \
    "++env.commands.vel_limit=[[-0.8,-0.4,-0.8],[0.8,0.4,0.8]]" \
    ++env.commands.rel_standing_envs=1.0 \
    ++env.commands.resampling_time=0.0 \
    ++env.commands.heading_command=false \
    env.max_episode_seconds=3600.0 \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
