# G1 三专家 MoE/DAgger 接入与运行

当前分支：`codex/g1-recovery-moe-integration`。
已有行走/站立权重保留；第三个起身专家已经接线，但尚未训练。
独立起身教师继续使用选定的 `model_12000.pt`，不修改 SAC 奖励。

## 本地已生成

- 三专家初始化：`.local-build/g1_recovery_moe/initial_3expert_canonical.pt`。
- 原学生 SHA-256：`4d9eed26b39a875a2475289f84a40949a95c3a7d5a618b7110f90df051fae6fe`。
- 起身教师 SHA-256：`9f9fa902e6228c93e26e319505bfec584632b38129b35471d04038ad64047bbd`。
- 接入探针：`.local-build/g1_recovery_moe/reference_handover_guard_seed1.json`。

起身教师和旧 MoE 都是 99 维，但角速度/关节速度的缩放不同。
显式适配已接入采样和交接，统一学生继续使用旧 MoE 的输入单位。
独立教师输出只作为监督标签；正式训练之后才能说第三专家学会了起身。

## 同步代码

本次按用户选择，通过 GitHub 分支同步代码、测试和文档。模型和本地探针
产物不随代码上传；继续使用服务器已有的三个教师和原两专家学生。

在训练服务器执行：

```bash
cd /ssd1/cyx/liujun/UniLab
# 单分支抓取配置需要先登记新分支；在首次同步时执行一次。
git config --add remote.origin.fetch '+refs/heads/codex/g1-recovery-moe-integration:refs/remotes/origin/codex/g1-recovery-moe-integration'
git fetch origin
git switch codex/g1-recovery-moe-integration
git pull --ff-only origin codex/g1-recovery-moe-integration
```

首次切换时，Git 会从同名远端分支创建本地跟踪分支。若服务器现有改动与
切换冲突，先核对并保留这些改动。采用分支同步后，不再应用同一份补丁。
本地曾导出的补丁只作为初始准备阶段的证据保留。

## 服务器正式训练

以下为已准备的命令；本会话没有连接服务器或执行它。先同步本分支代码，
并确认三个教师与原学生路径仍存在。两个旧教师路径来自历史运行记录，
在服务器执行前需要核对。输出采用新目录，不覆盖原 run。

```bash
cd /ssd1/cyx/liujun/UniLab
export UNILAB_G1_WALK_HEIGHT_TEACHER=/ssd1/cyx/liujun/UniLab/logs/G1WalkHeight/20260724-020039_g1_walk_height_nominal_0754/model_5000.pt
export UNILAB_G1_STAND_HEIGHT_TEACHER=/ssd1/cyx/liujun/UniLab/logs/G1StandHeight/20260724-013445_g1_stand_height_stage2_065_0754/model_5000.pt
export UNILAB_G1_RECOVERY_TEACHER=/ssd1/cyx/liujun/UniLab/logs/fast_sac/G1Recovery/2026-10-09_19-37-31_mujoco/model_12000.pt
UNILAB_RECOVERY_RUN="/ssd1/cyx/liujun/UniLab/logs/distill_workflow/$(date +%Y%m%d-%H%M%S)_stand_height_walk_recovery"
mkdir -p "$UNILAB_RECOVERY_RUN"
export UNILAB_G1_RECOVERY_MOE_INIT="$UNILAB_RECOVERY_RUN/initial_3expert.pt"
uv run --no-sync python scripts/deploy/extend_unilab_g1_recovery_moe.py   --source /ssd1/cyx/liujun/UniLab/logs/distill_workflow/20260727-151700_stand_height_walk_ordered_b_r4/checkpoints/dagger_iteration_1.pt   --output "$UNILAB_G1_RECOVERY_MOE_INIT"

# 只展开配置，确认入口；不进入训练。
uv run --no-sync train --algo distill --task g1_walk_height_nominal --sim mujoco   workflow=g1_stand_height_walk_recovery training.device=cuda:0 --cfg job --resolve

# 正式蒸馏：继承前两个专家，bootstrap 起身专家，然后累计 DAgger。
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 uv run --no-sync train   --algo distill --task g1_walk_height_nominal --sim mujoco   workflow=g1_stand_height_walk_recovery training.device=cuda:0   training.workflow.mode=fresh   "training.workflow.run_dir=$UNILAB_RECOVERY_RUN"   "training.workflow.artifact_dir=$UNILAB_RECOVERY_RUN/role_artifacts"
```

默认保留原训练预算（bootstrap 20000 updates，8 轮 DAgger）。Bootstrap 按三角色均衡重复采样，DAgger 场景配额为
行走 35%、静止站立 20%、行走后停稳 15%、仰躺起身 20%、起身交接 10%。
这是一组初始实验预算和配额，不能保证收敛。训练入口校验初始化模型、
第三专家映射、观测契约及起身教师哈希；本阶段不恢复旧两专家优化器。

## 已完成 Bootstrap 后中断

2026-10-10 的服务器 run `20261010-171936_stand_height_walk_recovery`
已进入 DAgger，但恢复交接场景在合并 Hydra 锁定配置时失败。修复仅在
场景 owner 内复制配置再覆盖新任务字段，保留原配置的锁定状态和内容。

先拉取修复，不再重新生成初始化文件或执行 bootstrap。中断轮可能已有
尚未提交到 run manifest 的性能记录，因此使用已有 fork 流程在新目录继续，
原失败目录保留。父 checkpoint 和合并数据会按 manifest 校验哈希。

```bash
# 三个教师环境变量继续指向原文件。
UNILAB_RECOVERY_PARENT="$PWD/logs/distill_workflow/20261010-171936_stand_height_walk_recovery"
export UNILAB_G1_RECOVERY_MOE_INIT="$UNILAB_RECOVERY_PARENT/initial_3expert.pt"
UNILAB_RECOVERY_RUN="${UNILAB_RECOVERY_PARENT}_continued_$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 uv run --no-sync train \
  --algo distill --task g1_walk_height_nominal --sim mujoco \
  workflow=g1_stand_height_walk_recovery training.device=cuda:0 \
  training.workflow.mode=fork \
  "training.workflow.parent_run_dir=$UNILAB_RECOVERY_PARENT" \
  "training.workflow.run_dir=$UNILAB_RECOVERY_RUN" \
  "training.workflow.artifact_dir=$UNILAB_RECOVERY_PARENT/role_artifacts"
```

新目录从父 run 的最后一个已验证 checkpoint 开始；本次预计为
`bootstrap_student.pt`。Bootstrap 不再优化，未完成的 DAgger 轮重新采样。
本地小采样只验证场景连通性，不代表统一学生起身或交接质量验收。

## 训练后验收

先用单个新 checkpoint 跑接入探针：

```bash
uv run --no-sync python scripts/deploy/check_unilab_g1_recovery_integration.py   --student-checkpoint "$UNILAB_RECOVERY_RUN/checkpoints/dagger_iteration_8.pt"   --steps 750 --seed 1 --output "$UNILAB_RECOVERY_RUN/student_handover_seed1.json"
```

不传 `--recovery-teacher` 时，动作来自统一三专家学生。传该参数只测试
“原起身教师→原站立专家”的参考上界，不能算统一学生验收。
之后检查多初态起身、交接后的持续站立、回到行走与旧能力回归。
当前探针和 loss 下降都不能替代这些行为验收；正式晋级阈值尚未约定。

播放统一学生必须选择 `g1_recovery_combined` 任务，使用 checkpoint 中
记录的恢复路由契约。直接导出自由路由神经网络尚不能代表完整恢复控制器。
