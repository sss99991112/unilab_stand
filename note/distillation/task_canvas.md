# Distillation Task Canvas

## Objective And Accepted Method

Train one 99-D, 29-action MoE student with Walk=0, StandHeight=1, Recovery=2.
The user selected this route on 2026-10-10 and stopped standalone reward refinement.
Authority: `contracts/active/method/DISTILL-METHOD-v004.md` and inherited
`contracts/active/training/DISTILL-TRAIN-v003.md`.

## Current Cursor

Implementation, 171 focused regressions and bounded reference handover verified; formal three-expert
training and unified-student physical acceptance remain pending.

| Design point | Status | Evidence / remaining boundary |
| --- | --- | --- |
| DISTILL-DP-01 teachers | selected sources | Recovery hash verified; Walk/StandHeight server source paths recorded, bytes unavailable locally |
| DISTILL-DP-02 routing | deterministic + bounded live | Physical latch, stable window, reset and nominal settling; seed-1 reference handover at 9.0 s |
| DISTILL-DP-03 role data | deterministic | Exact unit adapter, detached labels, five scenarios, pre-action handover schema roundtrip |
| DISTILL-DP-04 student | initialization ready | Existing two experts copied exactly; new expert remains untrained |
| DISTILL-DP-05 DAgger | code + contracts | Existing cumulative workflow and CLI compose; no formal three-expert run |

## Current Files

- Workflow: `conf/distill/workflow/g1_stand_height_walk_recovery.yaml`.
- Owners: `recovery_integration.py`, `g1_observation_adapter.py`,
  `recovery_workflow.py`, `recovery_combined.py`, and `scripts/train_distill.py`.
- Run instructions: `plans/g1_recovery_moe_integration.md`.
- Checklist: `checklists/g1_recovery_moe_integration.md`.
- Evidence: E122 in `evidence/current.md` and local `.local-build/g1_recovery_moe/`.

## Next Boundary

Synchronize this branch onto the user's server, check exact teacher paths and
migrate the old student into a fresh three-expert initialization. Then run the
profile's formal bootstrap and DAgger command. No automatic server access is
configured in this session. Preserve source runs, failed probe evidence and old
student files. The legacy two-expert profile remains available for regression.
