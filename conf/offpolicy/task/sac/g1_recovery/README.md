# G1Recovery: independent supine recovery task baseline

This owner selects `G1Recovery` with MuJoCo and the existing SAC runtime.
It is a task baseline inspired by HoST with optional fixed-per-run assistance
stages, not a reproduction of HoST's multi-critic PPO or dynamic action scale.
No trained recovery checkpoint or policy-quality claim accompanies this task.

## Contracts

- Actor observations retain the existing 99-D height-aware G1 layout; critic
  observations are 102-D. Velocity commands are zero, target height is fixed
  at 0.754 m, and gait phase is fixed at zero.
- Actions are 29-D reference-angle offsets with a fixed scale. Supine reset
  changes the floating-base orientation and initial joint perturbations,
  while preserving the standing action reference.
- The initial base height is 0.30 m; the initial falling/contact transient is
  part of the task. There is no hidden settling controller. The default final stage has zero
  auxiliary force; assisted profiles are explicit opt-ins.
- Low height, large tilt, and ordinary ground contacts do not end an episode.
  Excessive joint/base velocity terminates; the normal 10-second timeout
  truncates through the existing NpEnv lifecycle.
- Success requires at least one continuous second of height >= 0.65 m,
  torso tilt <= 10 degrees, full linear speed <= 0.20 m/s, angular speed <=
  0.30 rad/s, and contact on both feet. It is recorded in
  `recovery_succeeded`; success does not end the episode, allowing continued
  stability training. `recovery_stable` reports current stability, so a
  later second fall can be distinguished from ever having succeeded.
- Reward weights are initial task-design values. They have not been tuned
  or validated by training. Assistance withdrawal uses explicit stages described below; teacher/MoE
  integration and policy-quality verification remain future work.

## Bounded task check

Run from a UniLab checkout with its normal dependencies installed:

```bash
uv run scripts/deploy/check_unilab_g1_recovery_task.py
```

The headless probe uses seed 1, two environments, 100 zero-action steps,
checks finite 99/102-D observations and rewards, preserves the action
reference, verifies refresh calls do not duplicate stability time, and checks
that resetting one robot leaves the other robot's physics state unchanged.
A PASS proves lifecycle connectivity, not autonomous getting-up capability.

## Training route

Once the task and planned training curriculum have been reviewed on the
training server, the existing entrypoint can select this owner:

```bash
uv run train --algo sac --task g1_recovery --sim mujoco training.no_play=true
```

The owner starts with 128 environments. Choose the actual training scale and
budget after measuring the server, and judge the teacher with unassisted
closed-loop recovery tests rather than training reward alone.


## Fixed-stage assistance and withdrawal

The initial schedule is 40% -> 20% -> 0% of nominal robot weight, applied
upward to `torso_link`. These are untrained initial settings. Body IDs, robot
subtree mass, and gravity are resolved once at construction. During stepping,
only cached metadata and public sensor/force interfaces are used.

Force is gated off during the first 0.6 seconds and while the torso's world-Z
up-vector component is <= 0.8. The force covers one control period and is
cleared by the backend afterward. Action scale remains 1.0 at every stage.
Stage identity and gating settings are recorded by the existing run_config.json.

Each stage is a separate SAC run. Do not change assistance within a live replay
buffer. A continuation may stay at its stage or move to the next stage, keeping
the schedule and actor interface fixed. It transfers only actor weights and
optional observation normalization; Q networks, optimizer state, and replay are
fresh. Parent checkpoint and run_config.json must remain together. `load_run`
is not admitted for recovery training; use explicit actor-only continuation.
This also supports restarting an interrupted stage with fresh replay.

The following commands describe later training; the implementation/probes do
not execute them:

```bash
# Stage 0: start a new assisted run.
uv run train --algo sac --task g1_recovery --sim mujoco --profile assist40 training.no_play=true

# Stage 1: separate run, starting from the preceding actor.
uv run train --algo sac --task g1_recovery --sim mujoco --profile assist20 training.no_play=true \
  algo.actor_warm_start_checkpoint=/absolute/path/to/stage0/model_500.pt \
  algo.actor_warm_start_adapter=g1_height_actor_obs_99_to_99_v1

# Stage 2: train without assistance, again with fresh replay and critics.
uv run train --algo sac --task g1_recovery --sim mujoco training.no_play=true \
  algo.actor_warm_start_checkpoint=/absolute/path/to/stage1/model_500.pt \
  algo.actor_warm_start_adapter=g1_height_actor_obs_99_to_99_v1
```

No automatic promotion is performed merely because a training run finishes.
Choose stage changes after inspecting policy rollouts. Assisted success is not
an unassisted teacher qualification. The standard playback adapters force
assistance off, including automatic playback after an assisted training run.

Verify the assistance mechanism without training:

```bash
uv run scripts/deploy/check_unilab_g1_recovery_assistance.py
```

This runs four two-step, same-state airborne-upright physics branches to isolate
the applied force from contacts: 40%, 20%, zero, and evaluation of a 40% profile.
It checks upward-force effect, force consumption, initial-fall gating, and exact
zero-force equivalence for evaluation. It does not assess getting-up quality.


## Unassisted v2 preparation after the failed first teacher

`mujoco_unassisted_v2.yaml` selects a **fresh**, zero-assistance experiment via
`--profile unassisted_v2`. It keeps UniLab's FastSAC / DoubleBufferOffPolicyRunner,
99/102/29 observation/action dimensions, reference-angle actions, all reward
weights, and SAC optimizer settings. No parent actor or checkpoint is selected.

The new `env.recovery.height_gated_upright=true` multiplies posture reward by
normalized height progress. The progress floor is 0.06 m, below the measured
0.07–0.08 m grounded baseline, so height shaping remains gradual. Stable-standing
criteria and standing rewards are unchanged. This addresses the observed posture
reward dominating failed rollouts; it does not prove recovery convergence.

The gate defaults to false in the dataclass. Existing owners and saved configs
that omit it reproduce the original reward, including checkpoint diagnostics.
Select the v2 profile explicitly for both new training and its playback.

Training budget: 50,000 iterations, saving every 2,500 (approximately 6.4 million
collected transitions with 128 environments plus initial replay filling). The
budget is a candidate experiment, not a convergence threshold. If successive
checkpoints do not improve unassisted recovery, inspect the trajectories rather
than treating iteration count or mean reward as acceptance. Use_compile is off;
other inherited SAC settings are unchanged.

Training preparation, **without learning or GPU execution**:

```bash
uv run python scripts/deploy/check_unilab_g1_recovery_task.py \
  --profile unassisted_v2 --check-runner
```

This runs a 100-step, two-environment lifecycle probe and constructs the actual
CPU runner/learner to check actor/critic connectivity. It never calls `learn`,
starts a collector process, or writes policy checkpoints. CPU success does not
confirm the server's CUDA training path.

Ready-to-run training command (preparation does not execute it):

```bash
uv run train --algo sac --task g1_recovery --sim mujoco \
  --profile unassisted_v2 training.device=cuda:0 training.no_play=true
```

New-run evaluation selects the same profile and forces zero assistance:

```bash
uv run eval --algo sac --task g1_recovery --sim mujoco \
  --profile unassisted_v2 --load-run YOUR_NEW_RUN_ID --render-mode record \
  training.device=cuda:0 training.play_env_num=1 training.export_onnx=false
```


## Same-task learner-state continuation

`algo.resume_checkpoint` restores all existing standard FastSAC checkpoint
components: actor, critic, target critic, actor/critic/alpha optimizer states,
log-alpha and learner update count. The saved checkpoint does **not** contain
replay storage, environment state or random-generator state. Those start fresh;
this is learner-state continuation, not exact restoration of a running process.

The original `run_config.json` must remain next to the source checkpoint. The
resume validator requires the same task, backend, environment, rewards, network
shape and learner mathematics. Switching assistance stages or reward versions
uses the separate actor-only path; it cannot be mixed with learner resume.
The source directory is preserved and the continuation uses a new timestamped
run directory. Resume origin and restored counter are recorded in run_summary.

The iteration budget is **additional iterations in the new run**. For a source
with update_count=15000, another 15000 reaches learner update_count=30000.
Checkpoint filenames retain the existing run-local numbering: the new run's
model_15000.pt is its local final checkpoint, in a different directory.

Example, after code synchronization (preparation does not execute training):

```bash
uv run train --algo sac --task g1_recovery --sim mujoco \
  --profile unassisted_v2 training.device=cuda:0 training.no_play=true \
  algo.resume_checkpoint=/absolute/path/to/original/model_15000.pt \
  algo.max_iterations=15000 algo.save_interval=2500 algo.learning_starts=500
```

With 128 environments, learning_starts=500 refills 64000 transitions before
updates, below the 65536-transition replay capacity. The collector receives
restored actor weights before collecting; critic and optimizer states remain
restored during replay filling. All work stays within the existing UniLab
DoubleBufferOffPolicyRunner / AsyncRunner lifecycle.
