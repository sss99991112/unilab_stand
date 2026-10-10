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
continuation and fresh-process offline path are contract-verified (E128); server
launch remains pending.

| Design point | Status | Evidence / remaining boundary |
| --- | --- | --- |
| DISTILL-DP-01 teachers | selected sources | Recovery hash verified; Walk/StandHeight server source paths recorded, bytes unavailable locally |
| DISTILL-DP-02 routing | deterministic + bounded live | Physical latch, stable window, reset and nominal settling; seed-1 reference handover at 9.0 s |
| DISTILL-DP-03 role data | server saved-source replay PASS | 1114112 rows build/save/reload in a fresh CPU process; exact three-role/five-scenario fingerprints |
| DISTILL-DP-04 student | server first update completed | Bootstrap plus 10281 cached-target updates; new trained checkpoint awaits physical evaluation |
| DISTILL-DP-05 DAgger | eight-round continuation ready | E128: saved-update seed, cumulative mixed roles and eight fresh-process toy rounds pass; formal server run pending |

## Current Files

- Workflow: `conf/distill/workflow/g1_stand_height_walk_recovery.yaml`.
- Owners: `recovery_integration.py`, `g1_observation_adapter.py`,
  `recovery_workflow.py`, `recovery_combined.py`, and `scripts/train_distill.py`.
- Run instructions: `plans/g1_recovery_moe_integration.md`.
- Checklist: `checklists/g1_recovery_moe_integration.md`.
- Evidence: E122/E124/E125/E126/E127/E128 in `evidence/current.md` and local `.local-build/g1_recovery_moe/`.

## Next Boundary

Pull the integration branch and run
`scripts/deploy/train_unilab_g1_recovery_dagger8.sh` on the server. It forks from
`20261010-182545_recovery_dagger1_saved_data/dagger_iteration_1.pt` and the verified
1114112-row cumulative aggregate, then performs eight additional DAgger rounds.
Each round collects five scenarios with the previous student, aggregates all
past data in a fresh CPU process, and updates the student in a fresh learner
process. Cached SAC targets, quotas and replay-budget expansion are unchanged;
bootstrap and SAC teacher training are skipped. The original manifests remain
immutable. The native root cause and nested provenance anomaly are still open;
process isolation is containment, not a root-cause repair. Trained-student
physical reliability and old-capability regression remain unaccepted. No direct
server access exists, so do not report the remote run as started or completed.
