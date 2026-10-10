# Distillation Task Canvas

## Objective And Accepted Method

Train one 99-D, 29-action MoE student with Walk=0, StandHeight=1, Recovery=2.
The user selected this route on 2026-10-10 and stopped standalone reward refinement.
Authority: `contracts/active/method/DISTILL-METHOD-v004.md` and inherited
`contracts/active/training/DISTILL-TRAIN-v003.md`.

## Current Cursor

The server completed the first saved-data DAgger student update from the
original bootstrap: 10281 updates, 5263872 sampled rows, frozen cached teacher
targets and a separate three-expert checkpoint at
`20261010-182545_recovery_dagger1_saved_data/dagger_iteration_1.pt`.
The effective role/intent summaries are valid, but CLI nested Walk source metadata
reports one `<class 'frame'>` label. Persisted corruption versus result-formatting
corruption remains unconfirmed; the native writer is unknown. Later outer rounds
and unified-student physical acceptance remain pending.
The user explicitly authorized proceeding directly with eight additional rounds
from this saved student, without the proposed manual pre-evaluation. The opt-in
continuation and fresh-process offline path are contract-verified (E128). Server
run `20261010-193512_recovery_dagger8` reached round 4, then the update child exited
with code 1 during load_distillation_dataset (E129). The user then replayed the
exact preparation/load path in a fresh process: LOAD_OK rows=2424832. The original
inner exception remains missing and the failure is not reproduced (E130).
A verified saved-update recovery entrypoint completes original round 4 from its
existing aggregate, then forks to train only the four remaining rounds; local
contracts pass, server replay/update and continuation remain pending.

| Design point | Status | Evidence / remaining boundary |
| --- | --- | --- |
| DISTILL-DP-01 teachers | selected sources | Recovery hash verified; Walk/StandHeight server source paths recorded, bytes unavailable locally |
| DISTILL-DP-02 routing | deterministic + bounded live | Physical latch, stable window, reset and nominal settling; seed-1 reference handover at 9.0 s |
| DISTILL-DP-03 role data | server fresh round-4 load PASS | E130: 2424832 rows load after trainer preparation; original failed exception and physical efficacy unconfirmed |
| DISTILL-DP-04 student | server reached new round 4 | Prior three new-round updates completed by workflow order; checkpoint bytes and physical quality unavailable locally |
| DISTILL-DP-05 DAgger | recovery continuation prepared | E130: current aggregate/hash/parent checks, real isolated repair update and four-round remainder proof; server execution pending |

## Current Files

- Workflow: `conf/distill/workflow/g1_stand_height_walk_recovery.yaml`.
- Owners: `recovery_integration.py`, `g1_observation_adapter.py`,
  `recovery_workflow.py`, `recovery_combined.py`, and `scripts/train_distill.py`.
- Run instructions: `plans/g1_recovery_moe_integration.md`.
- Checklist: `checklists/g1_recovery_moe_integration.md`.
- Evidence: E122/E124/E125/E126/E127/E128/E129/E130 in `evidence/current.md` and local `.local-build/g1_recovery_moe/`.

## Next Boundary

Pull the integration branch and run
`scripts/deploy/resume_unilab_g1_recovery_dagger.py --run-dir
/ssd1/cyx/liujun/UniLab/logs/distill_workflow/20261010-193512_recovery_dagger8`.
The owner verifies the third-round parent checkpoint, teacher hashes, successful
fourth aggregation acknowledgement and cumulative source identities; completes
only the saved fourth update in a fresh directory; then uses the existing fork
connector for four remaining rounds. Old files and partial metrics stay unchanged.
New continuation checkpoints 1..4 map to original rounds 5..8. Each child failure
now persists stderr and includes its tail in the raised error; failure stops the
run. This is recovery/observability, not a repair of an identified native or CUDA
root cause. Do not restart the original eight-round launcher from its older seed.
No direct server access exists; full completion and physical acceptance remain
pending server evidence.
