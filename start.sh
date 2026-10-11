#!/bin/bash

# 获取当前操作系统名称
OS="$(uname -s)"

ALGO="sac"
TASK="g1_walk_flat"
SIM="mujoco"
DRY_RUN=0
CHECKPOINT_PATH=""
CHECKPOINT_SELECTOR=""
LOG_FILE="${UNILAB_LOG_FILE:-}"
POSITIONAL_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --algo)
            if [ "$#" -lt 2 ]; then
                echo "Usage error: --algo requires a value"
                exit 2
            fi
            ALGO="$2"
            shift 2
            ;;
        --task)
            if [ "$#" -lt 2 ]; then
                echo "Usage error: --task requires a value"
                exit 2
            fi
            TASK="$2"
            shift 2
            ;;
        --sim)
            if [ "$#" -lt 2 ]; then
                echo "Usage error: --sim requires a value"
                exit 2
            fi
            SIM="$2"
            shift 2
            ;;
        --checkpoint-path|--ckpt-path)
            if [ "$#" -lt 2 ]; then
                echo "Usage error: $1 requires a value"
                exit 2
            fi
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --checkpoint|--ckpt)
            if [ "$#" -lt 2 ]; then
                echo "Usage error: $1 requires a value"
                exit 2
            fi
            CHECKPOINT_SELECTOR="$2"
            shift 2
            ;;
        --log-file)
            if [ "$#" -lt 2 ]; then
                echo "Usage error: --log-file requires a value"
                exit 2
            fi
            LOG_FILE="$2"
            shift 2
            ;;
        --log-file=*)
            LOG_FILE="${1#--log-file=}"
            shift
            ;;
        g1_walk_flat|g1_stand_still|g1_walk_height)
            TASK="$1"
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

KEYBOARD="true"
if [ "$TASK" = "g1_stand_still" ]; then
    KEYBOARD="false"
fi

if [ -z "${UNILAB_G1_ACTION_TRACE_INTERVAL+x}" ]; then
    export UNILAB_G1_ACTION_TRACE_INTERVAL=20
fi

if [ -n "$LOG_FILE" ]; then
    LOG_DIR="$(dirname "$LOG_FILE")"
    if [ "$LOG_DIR" != "." ]; then
        mkdir -p "$LOG_DIR"
    fi
    exec > >(tee "$LOG_FILE") 2>&1
fi

# 将那些两边都一样的长参数提取出来，方便以后修改
ARGS=(
    scripts/play_interactive.py
    --algo "$ALGO"
    --task "$TASK"
    --sim "$SIM"
    interactive.action_mode=policy
    "interactive.keyboard=${KEYBOARD}"
)

# 便捷用法：
#   ./start.sh 2026-06-12_15-46-01_mujoco
#   ./start.sh --task g1_stand_still 2026-07-09_00-00-00_mujoco
# 等价于追加：
#   algo.load_run=<RUN_NAME>
# 并由播放入口自动选择该 run 目录下最新的 model_*.pt。
if [ "${#POSITIONAL_ARGS[@]}" -gt 0 ] && [[ "${POSITIONAL_ARGS[0]}" != *=* ]] && [[ "${POSITIONAL_ARGS[0]}" != -* ]]; then
    RUN_NAME="${POSITIONAL_ARGS[0]}"
    POSITIONAL_ARGS=("${POSITIONAL_ARGS[@]:1}")
    ARGS+=(
        "algo.load_run=${RUN_NAME}"
    )
fi

if [ -n "$CHECKPOINT_SELECTOR" ]; then
    ARGS+=("algo.checkpoint=${CHECKPOINT_SELECTOR}")
fi

if [ -n "$CHECKPOINT_PATH" ]; then
    if [ "$ALGO" != "distill" ]; then
        echo "Usage error: --checkpoint-path is currently supported for --algo distill playback"
        exit 2
    fi
    ARGS+=("training.play_checkpoint_path=${CHECKPOINT_PATH}")
fi

ARGS+=("${POSITIONAL_ARGS[@]}")

echo "[start.sh] task=${TASK} sim=${SIM} algo=${ALGO} keyboard=${KEYBOARD}"
if [ -n "$LOG_FILE" ]; then
    echo "[start.sh] log_file=${LOG_FILE}"
fi
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "[start.sh] checkpoint_path=${CHECKPOINT_PATH}"
elif [ -n "$CHECKPOINT_SELECTOR" ]; then
    echo "[start.sh] checkpoint=${CHECKPOINT_SELECTOR}"
else
    echo "[start.sh] checkpoint=latest-by-run-resolver"
fi
if [ -n "${UNILAB_G1_ACTION_TRACE+x}" ]; then
    echo "[start.sh] action_trace=${UNILAB_G1_ACTION_TRACE} interval=${UNILAB_G1_ACTION_TRACE_INTERVAL}"
fi
if [ -n "${UNILAB_G1_STANDING_TEACHER_CHECKPOINT+x}" ]; then
    echo "[start.sh] standing_teacher_checkpoint=${UNILAB_G1_STANDING_TEACHER_CHECKPOINT}"
elif [ -n "${UNILAB_G1_DISTILL_STANDING_TEACHER_CHECKPOINT+x}" ]; then
    echo "[start.sh] standing_teacher_checkpoint=${UNILAB_G1_DISTILL_STANDING_TEACHER_CHECKPOINT}"
fi
printf '[start.sh] command:'
printf ' %q' "${ARGS[@]}"
printf '\n'

if [ "$DRY_RUN" = "1" ]; then
    exit 0
fi

# 根据操作系统执行不同的命令
if [ "$OS" = "Darwin" ]; then
    # Mac OS 系统 (Darwin) 必须使用 mjpython
    echo "🍏 检测到 macOS，正在使用 mjpython 启动仿真..."
    # Invoke the trampoline with the current interpreter: a moved virtualenv
    # may retain an old absolute path in mjpython's shebang.
    MJPYTHON_WRAPPER="$(uv run python -c 'from pathlib import Path; import sys; print(Path(sys.executable).with_name("mjpython"))')"
    uv run python "$MJPYTHON_WRAPPER" "${ARGS[@]}"
elif [ "$OS" = "Linux" ]; then
    # 4090 服务器 (Linux) 直接使用默认的 python 即可
    echo "🐧 检测到 Linux，正在使用常规 python 启动仿真..."
    uv run python "${ARGS[@]}"
else
    echo "⚠️ 未知的操作系统: $OS"
    exit 1
fi
