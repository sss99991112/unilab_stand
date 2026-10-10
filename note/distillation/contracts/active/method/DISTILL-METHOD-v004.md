---
contract_id: DISTILL-METHOD-v004
status: active
effective_date: 2026-10-10
updated_date: 2026-10-10
supersedes: DISTILL-METHOD-v003
scope: G1 Walk, StandHeight and unassisted supine Recovery three-expert integration
concept_figure: note/architecture/concept/03_g1_multiteacher_distillation_method.data.json
---

# G1 Walk, StandHeight And Recovery Integration

## Accepted Decision

On 2026-10-10 the user stopped standalone stability refinement and explicitly
selected adding a recovery expert through MoE/DAgger into one unified student.
The existing two-expert workflow and its checkpoint loading remain supported.
`workflow=g1_stand_height_walk_recovery` is the sole new activation group.
Standalone SAC recovery reward and the selected source checkpoint are unchanged.
Execution progress is tracked in the task canvas and evidence ledger.
Policy promotion and real-robot deployment remain unaccepted.

## Teacher Policies

`DISTILL-DP-01` / `DT-M-01`: frozen SAC Walk, StandHeight and G1Recovery actors
produce detached 29-D targets. The recovery source is
`2026-10-09_19-37-31_mujoco/model_12000.pt`, SHA-256
`9f9fa902e6228c93e26e319505bfec584632b38129b35471d04038ad64047bbd`.
Its 14/20 success result is selection evidence, not final reliability acceptance.

All actor observations have 99 columns, but their velocity units differ.
The canonical student layout retains the existing Walk/StandHeight layout:
gyro columns 0:3 have scale 0.25, joint-speed columns 35:64 have scale 0.05.
Recovery collection converts raw teacher observations into this student layout.
Recovery teacher queries in the combined environment invert these two scales;
all other columns, including target height at index 96, remain unchanged.
The adapter is `g1_recovery_to_walk_99_v1`; no padding or tolerant loading exists.
Cached teacher observations retain their source teacher's own units.

## Command Intent

`DISTILL-DP-02` / `DT-M-02`: recovery overrides velocity intent. The opt-in
`G1RecoveryCombined` environment latches recovery when measured height is below
0.40 m or torso tilt exceeds 60 degrees. Release requires the existing full
one-second stable window: height >= 0.65 m, tilt <= 10 degrees, linear speed
<= 0.20 m/s, angular speed <= 0.30 rad/s, and both feet contacting the floor.
Each physics frame counts once; reset clears only affected rows.

Recovery commands are zero, target height is 0.754 m, and gait phase is zero.
After release, StandHeight receives a 100-step nominal settling window; requested
commands/heights then become eligible again. Ordinary active velocity selects
Walk; inactive velocity selects StandHeight. The existing ordered
walk -> nominal settle -> requested height transition remains in the workflow.

The routing guard reads public MuJoCo physical sensors (height, IMU and foot
contact); these auxiliary signals never enter the 99-D learned actor. Collection
and playback consume the same env-owned latch and persisted routing contract.
This is a simulation deployment contract. Real-robot height/support estimation
and exporting the complete guard with the neural policy remain unverified.

## Role Data

`DISTILL-DP-03` / `DT-M-03`: schema 3 preserves student observations, original
teacher observations, cached actions, commands, target height, velocity intent,
role, scenario and transition ages. Recovery remains a role even though its
velocity intent is inactive. The five scenarios are `walk_flat`, `static_stand`,
`walk_to_stop`, `supine_recovery`, and `recovery_to_stand`.

In the handover scenario, labels and teacher targets use the same pre-action
routing state. Recovery frames query only the recovery teacher; post-release
frames query only StandHeight. Source identity, adapter and scenario fields
survive aggregation and save/reload. Mixing old schema artifacts fails closed.

## MoE Student

`DISTILL-DP-04` / `DT-M-04` and `DT-X-01`: the single checkpoint contains
expert 0 = Walk, expert 1 = StandHeight, expert 2 = Recovery. An explicit
`g1_moe_2_to_3_v1` migration copies the existing first two experts exactly.
The third expert is newly initialized and must be trained; it is not a copy of
the differently shaped SAC actor. Router rows 0/1 are retained, optimizer state
is fresh, source bytes are immutable and parent identity is persisted.

Behavior loss selects experts by role. Velocity-only router loss is disabled
because inactive recovery and inactive standing must not receive conflicting
expert targets. Updates leave non-selected experts and their optimizer state
unchanged. Deployment uses recovery-first hard selection; free-router output
alone is not the complete deployed controller.

## Student-State DAgger

`DISTILL-DP-05` / `DT-M-05`: inherit `DISTILL-TRAIN-v003`:
student_k rollout -> matching frozen teacher relabel -> cumulative aggregate
-> expert/router update -> student_(k+1). All scenarios use the same outer-loop
checkpoint. Role scenarios roll out their named expert; the handover scenario
uses the shared physical latch.

The new profile uses `execution_mode=legacy`; persistent recovery collection is
not implemented and is rejected before artifact writes. Old profiles retain
existing legacy and persistent behavior. No new runner or collector protocol is
introduced.

## Owners And Acceptance

- Profile and assembly: `conf/distill/workflow/g1_stand_height_walk_recovery.yaml`
  and `scripts/train_distill.py`.
- Units: `g1_observation_adapter.py`.
- Migration/selection: `recovery_integration.py`.
- Handover scenario: `recovery_workflow.py` and generic `collector.py`/`data.py`.
- Physical latch/reset: `envs/locomotion/g1/recovery_combined.py`.
- Deployment: `visualization/interactive_playback.py`.

Deterministic acceptance requires old-expert output equality, explicit unit
roundtrip and teacher-action parity, selected-expert gradient isolation,
pre-action handover labels, schema roundtrip, negative configuration/checkpoint
rejection, and public CLI compose. A bounded reference-policy MuJoCo handover
is connectivity evidence. Unified-student recovery, repeated falls, return to
walking and preservation of old behavior require trained checkpoints and live
acceptance; neither training reward nor startup tests qualify them.
