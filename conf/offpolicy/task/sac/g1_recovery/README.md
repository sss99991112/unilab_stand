# G1Recovery: independent supine recovery task baseline

This owner selects `G1Recovery` with MuJoCo and the existing SAC runtime.
It is a task baseline inspired by HoST with optional fixed-per-run assistance
stages, not a reproduction of HoST's multi-critic PPO or dynamic action scale.
The baseline sections below retain the original task-development context.
The selected frozen parent is now `2026-10-09_19-37-31_mujoco/model_12000.pt`;
its load problems remain. The refinement section describes preparation only,
not a newly trained or qualified teacher.

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


## Constrained teacher refinement v1 (prepared; do not start yet)

The user explicitly says formal training belongs on their server and cannot
start now. No server was contacted, no training runner was started, and no new
teacher/student weights were saved. GPU model, server paths and budget still
require confirmation. Use branch `codex/g1-recovery-teacher-refine` for this
preparation; publishing code does not authorize starting training. Synchronizing
this version to the server is a separate future step. Old runs, model files
and rewards are retained.

Select `task=sac/g1_recovery/mujoco_constrained_v1` in the existing entrypoint.
Its single owner flag `env.recovery_constraints.enabled` defaults to false.
OFF performs no load replay and all four new terms are zero; the original
task profile and full-learner resume path remain unchanged. ON requires all
four negative cost weights, an unassisted task, the original environment/PD
configuration and a versioned actor-only initialization. No action clipping,
new action mapping, observation fields or physical range changes are added.

| Cost | Measurement | Initial weight |
| --- | --- | ---: |
| recovery_joint_boundary | measured q inside a .05rad soft boundary zone or beyond the actual model range | -1.0 |
| recovery_outward_speed | outward dq inside that zone, normalized by 5rad/s | -0.2 |
| recovery_pd_excess | raw PD request excess before actuator clipping, normalized by the model force range | -0.2 |
| recovery_self_collision | normal force squared for two explicitly identified self-body pairs, normalized by 100N | -0.2 |

The .05rad/5rad/s/100N values are research normalization settings, not new
physical limits or verified hardware ratings. Original seven reward weights
are unchanged. These are soft penalties, not a proof or enforcement of safety.
Self-pairs are right hip-yaw/right wrist-pitch and left elbow/torso; world/ground
is excluded, including hand/elbow ground support. This does not cover every
possible anomalous self-contact; full acceptance must inspect other pairs too.

The backend captures loads at150Hz through an independent exact replay beside
the unchanged native3-substep step. Native terminal float32 state must match
exactly or execution fails. This avoids splitting native integration. Actual
reset-randomized model values are refreshed in owned snapshots; metadata is
cached at initialization. There is additional CPU and model-memory overhead.

Adapter `g1_recovery_load_actor_99_v1` loads only the original99-D actor. Critic,
target critic, optimizers and alpha start fresh; replay is created empty by the
existing double-buffer runner. The source has no external normalization stats;
v1 preserves that observed identity path and rejects fabricated or incompatible
stats. Action semantics remain `g1_joint_offset_default_angles_scale1_v1`;
other scale/bias values need a different explicit adapter. Checkpoint warm-start
metadata preserves these versions. New reward training cannot use old full
learner resume, and a nonempty/existing output path is rejected before tracking.

Local preparation command, **no optimization or runner.learn**:

```bash
uv run --no-sync python -m scripts.deploy.check_unilab_g1_recovery_refinement \
  --parent ../2026-10-09_19-37-31_mujoco/model_12000.pt \
  --steps 150 --output /private/tmp/g1-refinement-readiness-fresh
```

The two3s prefixes in each OFF/ON mode check native physics/observations,
official teacher action parity and real reward storage in fresh replay. They
are not complete recovery/settle acceptance. Formal GPU collection/update and
the full15s/20s/50-frame/100-settle evaluation remain unverified.

### Proposed server run (not approved or executed)

- Machine: user's server; exact host/root/GPU unknown. Propose one CUDA GPU
  with at least12GiB VRAM and16GiB host RAM; these are requested reservations,
  not measured requirements. MuJoCo may use up to32 native workers for32envs;
  CPU allowance must be confirmed. No automatic MPS/GPU substitution.
- Pilot cap:32envs,5000 learner iterations,one critic/actor update per iteration,
  batch1024, replay512 steps/env, warmup64 steps/env, actor LR1e-5.
  Warmup is2048 rows below16384-row capacity. Replay plus two transfer slots
  estimate30.52MiB, excluding simulator/observer/CUDA/driver memory. Allow2GiB
  for this new run's checkpoints/logs; actual machine usage needs measurement.
- Time cap:2hours or5000 iterations, whichever first, plus up to30s cleanup.
  Save every500 iterations. Budgets are proposals only; no convergence claim.
- Immediate failures: non-finite physics/reward, native/replay discrepancy,
  wrong parent/observation/action/PD contract, collector failure or OOM. Stop
  on a non-finite learner metric as well; learner health requires server-side
  observation, not inference from this CPU readiness check.
- Promotion stop: original recovery/settle or joint/PD/self-contact acceptance
  fails in any case: retain the candidate, do not update expert2 or migrate it.
  Never select by reward alone. Do not silently resume or extend the run.

After the user confirms host/GPU/paths/budget, the exact candidate command is
(requires GNU timeout on the Linux server; all variables are deliberately unset):

```bash
cd "${UNILAB_SERVER_ROOT:?confirm the server project root}"
CUDA_VISIBLE_DEVICES="${G1_REFINE_GPU:?confirm the GPU index}" OMP_NUM_THREADS=4 \
timeout --signal=INT --kill-after=30s 2h \
uv run --no-sync python scripts/train_offpolicy.py \
  algo=sac task=sac/g1_recovery/mujoco_constrained_v1 \
  env.recovery_constraints.enabled=true \
  "algo.actor_warm_start_checkpoint=${G1_REFINE_PARENT:?confirm parent plus run_config.json}" \
  "training.log_dir=${G1_REFINE_RUN:?choose a nonexistent new directory}" \
  training.device=cuda:0 training.num_gpus=1 \
  training.no_play=true training.logger=tensorboard
```

Teacher joint acceptance must finish before any student update. Experts0/1 and
their teachers remain untouched. The constrained profile alone is not evidence
of a better teacher. Current results:
`.local-build/g1_recovery_refinement/readiness_20261021_v2/REPORT.md`.

Publication validation (2026-10-22): 108 targeted tests passed with six existing
Gymnasium warnings; all 11 changed Python files passed Ruff and format checks.
`make check` was attempted in an isolated source snapshot to preserve unrelated
work; its format/lint stages completed, but type checking could not start because
`mypy` is absent. Full type checking and `make test-all` have not passed.
