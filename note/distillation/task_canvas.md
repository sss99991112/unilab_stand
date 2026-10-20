# Distillation Task Canvas

## Objective And Accepted Method

Use one 99-D, 29-action MoE student: Walk=0, StandHeight=1, Recovery=2.
Accepted route: existing teachers retained, student learned by MoE/DAgger.
Authority: `contracts/active/method/DISTILL-METHOD-v004.md` and inherited
`contracts/active/training/DISTILL-TRAIN-v003.md`.

## Current Cursor — 2026-10-20

Completed the user's source-library formal simulation/keyboard playback
integration, default OFF. Source branch `codex/g1-standing-assist-sim`, parent
`e1717b94`; the user authorized a local Git commit on 2026-10-20.
No source GitHub publication is authorized. Reused the existing independent
20-frame assist and per-frame load fallback, preserving the old `decide()` API
and all prior diagnostic work. No training, reward/PD/50-frame gate change,
hardware connection, weight modification or source GitHub publication.

`bash play_g1_recovery_keyboard.sh --standing-assist --supine` opts in;
without `--standing-assist`, original playback is retained. The session selects
raw actions; the original `env.step` alone applies default-angle bias, phase and
action history. Reset/pause/preview/dryrun boundaries and actual reset-randomized
gains are covered by 94 focused passing tests. These do not establish a full
repository pass or new GUI manual acceptance.

Fresh formal-factory/native-step evidence: 12 nominal/no-noise episodes,
5650 control steps / 16950 150-Hz physics substeps. Missing flag and explicit
OFF match exactly; ordinary stand/low-height/walk match OFF with ON. All ON
same-state decisions, phases/history and replayed physics match their reference.
Seven ON scenarios use the original budgets. Seed2 releases at 3.82 s and ends
settle at 5.82 s, after a 2.02 s load fallback; applied alpha reaches only 0.60,
not a completed 20-frame blend. Left wrist pitch/yaw each add two saturated
150-Hz samples; late self-contact increases. Recovery-load problems remain.

Report and raw evidence:
`.local-build/g1_standing_assist/formal_verified_20261020/REPORT.md`.
The independent evaluator's rerun remains exactly equal to its original seed2
trajectory. Native float32-between-controls trajectories differ from standalone
float64 trajectories; compare them by same-state decisions, not by replacing
historical closed-loop numbers. RoboJudo's published `6e57327` is unchanged.

## Earlier E136 Audit — retained

The user considers the simulation task completed and explicitly targets real
deployment on a Unitree G1, 29 DoF. They report having successfully deployed
policies previously; do not ask again whether low-level access exists.
That experience does not establish safe full-body recovery behavior.

E136 completes the authorized simulation load audit: 20 matched seeds x three
controllers (recovery teacher, teacher then old stand expert, unified student),
15 s each. State/obs/physics pairing passes. Independent 150-Hz force sampling
replays all 45,000 native control steps exactly. Teacher and student both show
ankle targets beyond model range, wrist saturation and body self-contact;
major peaks are in Recovery, rather than only at expert handover. Model limits
and contact forces are simulation parameters, not verified hardware ratings.

Evidence/report:
`.local-build/g1_recovery_moe/hardware_load_audit_20261015/REPORT.md`.
A necessary diagnostic snapshot getter fix plus two regression tests are local,
uncommitted; 10 focused tests and Ruff/diff checks pass. No new training,
reward/routing change, hardware command, checkpoint rewrite or GitHub push.

## Candidate And Preserved History

- Current student: `/Users/sss9999/enbodied_intelligence/unilab/dagger_iteration_2.pt`.
  SHA `27d06b8d73734fb2a17c730aa63ff70b2ba4ffbef7d3d71373df43974ab9bf18`.
  This is continuation round 2 / original round 6. Server completion lineage,
  interrupted update and fresh-process load results remain E127-E132. The
  original child failure was not reproduced; do not claim its cause is proven.
- Training was explicitly paused (E131). Server process stop is unconfirmed
  because this session has no remote execution channel. No restart authorized.
- E132: original local reliability evaluation 8/20 stable handovers, all briefly
  upright; tested old capabilities show no clear regression. E136 uses another
  reset/noise protocol; its 10/20 student stable count does not replace E132.
- E133/E134: keyboard playback and moved model identity. E135: F/C physical
  push/clear without reset, one true walking-push-fall-recovery trial. R is a
  diagnostic supine reset; repeated falls and return-to-walk remain unaccepted.

## Next Boundary

This authorized formal-entry integration and bounded regression are complete.
Next investigate recovery ankle targets, wrist saturation and self-contact,
then broader initial-state/noise/DR and repeated-fall behavior, with a separately
agreed scope. The current opt-in simulation result is not hardware acceptance.
Training remains paused; no new training or hardware action is authorized.

## Future Hardware Boundary — unchanged

Review the previously successful hardware deployment program: joint ordering,
offsets, action scaling, PD gains, real limits/torque-speed capability, and
permitted support contacts. Then propose a bounded recovery-teacher correction
for ankle targets, wrist loads and self-contact, preserving old walk/stand
experts. Any controller/reward edits, retraining, new DAgger or hardware trial
await the user's next instruction. No arbitrary safe-load threshold or
simulation-success-to-hardware-safety promotion is accepted.

## Current Files

- Workflow: `conf/distill/workflow/g1_stand_height_walk_recovery.yaml`.
- Owners: `recovery_integration.py`, `g1_observation_adapter.py`,
  `recovery_workflow.py`, `recovery_combined.py`, `scripts/train_distill.py`.
- Keyboard inspection: `bash play_g1_recovery_keyboard.sh`; opt-in assist:
  `bash play_g1_recovery_keyboard.sh --standing-assist --supine`.
- Formal adapter: `src/unilab/visualization/standing_assist_playback.py`.
- Bounded verifier: `scripts/verify_g1_standing_assist_playback.py`.
- Plan: `plans/g1_recovery_moe_integration.md`.
- Checklist: `checklists/g1_recovery_moe_integration.md`.
- Evidence: E122-E136 in `evidence/current.md` and `.local-build/g1_recovery_moe/`.
