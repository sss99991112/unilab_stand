# G1 三专家 MoE/DAgger 接入与运行

当前分支：`codex/g1-recovery-moe-integration`。
初始化保留了行走/站立权重；服务器已完成 bootstrap 和首轮已采样数据的
DAgger 学生更新。新 checkpoint 的实际起身与旧能力保持尚未验收。
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

## 第四轮中断后的当前续训入口

服务器 run `20261010-193512_recovery_dagger8` 已完成前三轮，第四轮在学生
更新子进程的数据加载处退出。用户独立复现同一路径已成功读取 2424832 行；
原始失败未复现，错误类型仍未知。现有数据可供这一条恢复流程尝试继续，
不把读入成功当成根因修复或物理质量验收。

```bash
cd /ssd1/cyx/liujun/UniLab
git pull --ff-only origin codex/g1-recovery-moe-integration
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/deploy/resume_unilab_g1_recovery_dagger.py \
  --run-dir /ssd1/cyx/liujun/UniLab/logs/distill_workflow/20261010-193512_recovery_dagger8
```

入口先校验第三轮权重、教师哈希、第四轮聚合文件及其成功回执、累计源路径。
然后使用已保存的第四轮请求，在新进程、新目录中完成该轮更新；不重新采样，
不更新原失败 run 的 manifest 或 partial metrics。学生保存后验证父权重哈希和
原始数据路径，再通过已有 fork 入口继续剩余四轮。采样配额、缓存监督、batch
和重放预算自动扩展都沿用第四轮请求的配置。

输出目录在启动时打印，格式为原 run 名加 `_recovered_<恢复时间>`：

- `checkpoints/recovered_iteration_4.pt`：补完的原第四轮学生。
- `continued/checkpoints/dagger_iteration_1.pt` 至 `dagger_iteration_4.pt`：
  对应原第五至第八轮，末个文件为最终学生。
- `recovered_update.json`：真实更新次数及原始 checkpoint/data/request 身份。

每个离线子进程的 stderr 保存到输出文件旁的 `.offline-stderr.log`，异常消息
带日志路径及末尾内容，避免终端看板夹断真正错误。如果再次失败就停止，保留
原错误和所有已完成产物；不静默替换坏标签或自动无限重试。此入口已做本地
契约验证，尚未在本会话中远程执行。不要重跑下面的旧八轮启动入口。

## 首次从已更新学生追加 8 轮的历史入口

用户于 2026-10-10 明确要求直接开始 8 轮训练，跳过拟议的人工预检查。
当前学生为
`logs/distill_workflow/20261010-182545_recovery_dagger1_saved_data/dagger_iteration_1.pt`，
累计数据为
`logs/distill_debug/20261010-175817_recovery_aggregate_check/cycle-000001.pt`。
此入口从它们开始追加 8 轮；不再初始化第三专家、重做 bootstrap 或训练 SAC 教师。

```bash
cd /ssd1/cyx/liujun/UniLab
git pull --ff-only origin codex/g1-recovery-moe-integration
bash scripts/deploy/train_unilab_g1_recovery_dagger8.sh
```

启动脚本自带本次服务器教师与学生路径，创建带时间戳的新输出目录。
每轮用上一轮学生采集五场景，累计合并并由缓存的教师动作监督更新。
合并与更新分别通过现有离线 owner 在新进程中执行，避免复用之前出错的
长驻采样进程上下文；该选项只影响 DAgger 的离线阶段，默认关闭。
场景配额与两种交接场景的八次预期重放预算保持原配置，更新次数按累计数据
自动增加，不把每轮优化步数固定为 128。

新增 fork 参数校验学生声明的数据路径、父 checkpoint 哈希、维度与角色。
累计数据作为一个保留逐行角色和场景的 seed，原失败 run 与已更新学生不被覆盖，
也不伪造原 manifest 中的首轮完成状态。新 run 从第 1 轮计数，最终 checkpoint
为其 `checkpoints/dagger_iteration_8.pt`。失败会停止，保留阶段请求和原始错误。
原生异常的写入者仍未确认；进程隔离是执行上的隔离措施，不代表根因已修复。
本机未连接训练服务器，正式启动与行为质量需要服务器后续输出才能确认。

## 首次 bootstrap 的历史启动命令

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


## 本地键盘检查（训练保持暂停）

`dagger_iteration_2.pt` 为原训练第 6 轮。启动脚本先检查主目录，
没有该文件时检查项目上一级目录。运行实际项目根目录的
`bash play_g1_recovery_keyboard.sh`；脚本自行切换项目目录，并在 macOS 用
当前 Python 运行官方 mjpython trampoline，兼容已搬迁虚拟环境。
默认站姿、初始暂停、CPU；可加 `--supine` 从仰躺开始，或用
`--checkpoint-path /完整路径/学生.pt` 指定其他三专家恢复学生。
默认只使用现有项目环境，不安装或更新依赖。

点击仿真窗口后，空格开始/暂停，方向上/下增减前后速度，左/右增减转向，
A/D 增减横向速度，Enter 将所有速度归零，[/] 每次降低/升高目标 0.01 m
（范围 0.650–0.754 m）。速度指令会保持，松开按键不会自动归零。
R 重置为仰躺并自动恢复；Backspace 回到启动姿态；N 单步，Esc 退出。
默认速度范围 vx ±0.8 m/s、vy ±0.4 m/s、yaw ±0.8 rad/s。

恢复和恢复后的稳定等待期仍由原物理路由屏蔽速度、将有效高度保持
0.754 m；最新人工请求保留，待原稳定窗口与 100 步等待期完成后生效。
终端显示 control=walk/stand_height/recovery。人工播放关闭自动重置，
不会每 15 秒重新开始。键盘调试不改变模型参数、reward 或稳定门槛；
手工观察不替代 E132 的多初态定量评测。


R 是仰躺重置快捷键。正常行走中的跌倒检测同样会自动触发 Recovery，
触发过程不重置当前位置、关节、速度或回合。行走中受推后的实际恢复、
侧倒和趴倒成功率尚未专项验证；仰躺重置演示不能代替这些验收。
