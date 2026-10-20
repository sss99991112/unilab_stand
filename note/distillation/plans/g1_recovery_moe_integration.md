# G1 三专家恢复：当前播放与验证

当前分支：`codex/g1-standing-assist-sim`。正式播放接入已提交为
`bfdae68a` 并推送到同名 GitHub 分支。独立起身教师继续使用选定的
`model_12000.pt`；三专家学生为项目上一级的 `dagger_iteration_2.pt`
（原训练第6轮），当前训练暂停。

这是现有20帧协助和负荷回退的默认关闭接入，未改模型、奖励、PD和原50帧
稳定条件。历史8/20可靠性结果不被本批有界仿真替代，真机尚未验收。

旧 bootstrap、追加八轮、第四轮中断恢复及当时服务器路径已统一移至
[历史训练命令](../history/plans/g1_recovery_moe_training_commands.md)。
当前任务状态只以 [任务看板](../task_canvas.md) 为准。

## 同步当前播放代码

```bash
git fetch origin refs/heads/codex/g1-standing-assist-sim:refs/remotes/origin/codex/g1-standing-assist-sim
git switch codex/g1-standing-assist-sim
git pull --ff-only origin codex/g1-standing-assist-sim
```

切换前保留服务器未提交修改；模型与本地轨迹不随 Git 同步。

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


## 无重置受推倒地检查（2026-10-11）

键盘启动脚本默认启用物理推力控制：F 开启持续推力，C 撤掉这股推力。
默认作用于躯干 torso_link，世界坐标 +X 方向 150 N。重复按 F 不叠加
力；推力持续到 C，松开 F 不会自动撤力。空格暂停时不推进物理，也不
累积冲量。R / Backspace 重置会同时关闭手动推力。

先启动仿真、空格开始，控制机器人行走，按 F 推倒后按 C 撤力，观察
自动 Recovery 接管及后续站稳交接。F/C 不修改关节姿态、位置、速度、
回合步数或行走请求；需要取消原行走请求时另外按 Enter。
实际推力进入环境物理后端，而非只改用于显示的 MjData。

可通过 Hydra 改大小/方向，例如侧推 100 N：

```bash
bash play_g1_recovery_keyboard.sh 'interactive.push_force_world=[0,100,0]'
```

不使用推力时加 `interactive.push_force_enabled=false`。训练、reward、
稳定门槛和模型参数保持原样。E135 的单回合受推检查验证物理连接，
不代表多姿态恢复成功率或可靠性验收。


## 正式播放的默认关闭渐进协助（2026-10-20）

源正式播放现可显式开启已有20帧协助和逐帧负荷回退，保留原入口：

```bash
# 原入口，默认关闭协助
bash play_g1_recovery_keyboard.sh --supine
# 显式开启；仍默认暂停，空格开始，原键盘/F-C推力/R重置继续可用
bash play_g1_recovery_keyboard.sh --standing-assist --supine
# 仅检查启动参数
bash play_g1_recovery_keyboard.sh --standing-assist --supine --dry-run
```

配置为 `interactive.standing_assist: false`。ON 使用 CPU、单环境和原
G1RecoveryCombined 三专家恢复学生；原99维观测、29动作、PD和50帧稳定
门槛保持。选择器仅生成原始动作，原 env.step 独占 apply_action，避免重复
默认角偏置、历史写入或相位推进。原独立实验 decide() API 保留。

新播放默认噪声/随机化设置不变。定向94测试和原预算正式 native 回归
5650控制步通过；七个ON场景包括三个仰躺种子、普通站立、低站高、行走、
无重置受推。低站高原偏差仍在，行走不按静站条件判成功。seed2 1.78s
开始协助、2.02s负荷回退，仅实际执行到alpha .60，随后3.82s释放、5.82s
等待结束；并非完整走完20帧。150Hz记录仍显示腕饱和增加和自碰撞代价。

完整结果及复现命令见
`.local-build/g1_standing_assist/formal_verified_20261020/REPORT.md`。
这是小预算、nominal无噪声的接入验收；不取代E132可靠性评测，未做新ON
图形窗口手工验收、广泛噪声/DR或真机测试。训练保持暂停；接入代码已发布
到上述分支，权重不上传、不修改。去掉 --standing-assist 或设置false可回原路径。
