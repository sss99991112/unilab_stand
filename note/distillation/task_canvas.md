# Distillation Task Canvas

## Objective And Accepted Method

Train one 99-D, 29-action MoE student with Walk=0, StandHeight=1, Recovery=2.
The user selected this route on 2026-10-10 and stopped standalone reward refinement.
Authority: `contracts/active/method/DISTILL-METHOD-v004.md` and inherited
`contracts/active/training/DISTILL-TRAIN-v003.md`.

## Current Cursor

Server run `20261010-171936_stand_height_walk_recovery` reached DAgger after bootstrap,
then failed while merging the recovery-to-standing task into Hydra's locked source config.
The isolated scenario-overlay repair has 21 focused passing tests and a 1-env, 16-row
MuJoCo owner probe. Server continuation and unified-student physical acceptance remain pending.

| Design point | Status | Evidence / remaining boundary |
| --- | --- | --- |
| DISTILL-DP-01 teachers | selected sources | Recovery hash verified; Walk/StandHeight server source paths recorded, bytes unavailable locally |
| DISTILL-DP-02 routing | deterministic + bounded live | Physical latch, stable window, reset and nominal settling; seed-1 reference handover at 9.0 s |
| DISTILL-DP-03 role data | deterministic | Exact unit adapter, detached labels, five scenarios, pre-action handover schema roundtrip |
| DISTILL-DP-04 student | server bootstrap reached | Existing two experts copied exactly at initialization; server bootstrap checkpoint has not been evaluated locally |
| DISTILL-DP-05 DAgger | runtime failure repaired locally | Locked scenario overlay fixed; interrupted-round fork verified; server continuation pending |

## Current Files

- Workflow: `conf/distill/workflow/g1_stand_height_walk_recovery.yaml`.
- Owners: `recovery_integration.py`, `g1_observation_adapter.py`,
  `recovery_workflow.py`, `recovery_combined.py`, and `scripts/train_distill.py`.
- Run instructions: `plans/g1_recovery_moe_integration.md`.
- Checklist: `checklists/g1_recovery_moe_integration.md`.
- Evidence: E122/E124 in `evidence/current.md` and local `.local-build/g1_recovery_moe/`.

## Next Boundary

Pull the recovery scenario config repair onto the user's server, verify the failed
run's manifest and bootstrap checkpoint, then fork continuation into a new run
while reusing the parent's role artifacts. Keep the parent run and its incomplete
metrics unchanged; no bootstrap rerun is needed. No automatic server access is
configured. Verify actual recovery-to-standing collection before claiming formal
continuation or policy quality. The old two-expert profile remains available.
