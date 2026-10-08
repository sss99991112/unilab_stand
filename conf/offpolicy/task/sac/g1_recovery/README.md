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
