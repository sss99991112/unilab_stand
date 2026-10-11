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
The user reports recovery run `20261010-193512_recovery_dagger8_recovered_20261010-201041`
contains recovered_update.json and continued/checkpoints with complete
`dagger_iteration_1.pt` and `dagger_iteration_2.pt` plus successful offline results.
These map to original rounds 5 and 6; local round 3 has request/stderr only.
On 2026-10-10 the user explicitly paused training and chose to download the latest
completed student to inspect behavior. Latest candidate is the continuation's
`dagger_iteration_2.pt`. Server process termination is unconfirmed; this session
has no remote execution channel. Do not restart or continue training without a
new user instruction. The user then explicitly authorized checking the downloaded checkpoint. E132
verifies local bytes and bounded physical behavior: 8/20 successful stable
handovers, while all 20 briefly reached upright posture. Training remains paused;
general physical reliability is not accepted. The subsequent user request adds
keyboard inspection: E133 verifies the real input/action/reset chain and opens
the native MuJoCo viewer; actual manual key callbacks are observed. Training
remains paused.

| Design point | Status | Evidence / remaining boundary |
| --- | --- | --- |
| DISTILL-DP-01 teachers | selected sources | Recovery hash verified; Walk/StandHeight server source paths recorded, bytes unavailable locally |
| DISTILL-DP-02 routing | deterministic + bounded live | Physical latch, stable window, reset and nominal settling; seed-1 reference handover at 9.0 s |
| DISTILL-DP-03 role data | server fresh round-4 load PASS | E130: 2424832 rows load after trainer preparation; original failed exception and physical efficacy unconfirmed |
| DISTILL-DP-04 student | original round 6, bounded local evaluation | E132: downloaded bytes verified; 8/20 stable handovers, 11 finish upright without handover, 1 falls again; old capability short tests preserved |
| DISTILL-DP-05 DAgger | PAUSED BY USER | Original round 6 complete by reported files; local round 3 lacks checkpoint/result; no further training authorized |

## Current Files

- Workflow: `conf/distill/workflow/g1_stand_height_walk_recovery.yaml`.
- Owners: `recovery_integration.py`, `g1_observation_adapter.py`,
  `recovery_workflow.py`, `recovery_combined.py`, and `scripts/train_distill.py`.
- Run instructions: `plans/g1_recovery_moe_integration.md`.
- Checklist: `checklists/g1_recovery_moe_integration.md`.
- Evidence: E122/E124/E125/E126/E127/E128/E129/E130/E131/E132/E133/E134/E135 in `evidence/current.md` and local `.local-build/g1_recovery_moe/`.

## Next Boundary

Training is explicitly paused. The downloaded candidate was moved and is now verified at
`/Users/sss9999/enbodied_intelligence/unilab/dagger_iteration_2.pt` (original round 6), SHA-256
`27d06b8d73734fb2a17c730aa63ff70b2ba4ffbef7d3d71373df43974ab9bf18`.
The user's effect-check request authorized local simulation. E132 and
`.local-build/g1_recovery_moe/round6_local_eval_20261010/REPORT.md` contain the
completed bounded evaluation and metric-annotated clips. Existing capabilities
show no clear regression in the tested commands, but the student's recovery
stable-window handover remains unreliable. Do not infer the optimization/data
root cause from this result, lower thresholds, edit reward, or resume training
without the next user instruction. Repeated falls, return to walk, wider commands
and real-robot deployment remain untested.


For the current manual inspection, use `bash play_g1_recovery_keyboard.sh`.
The native viewer has been opened and user key events are recorded (E133).
R uses the existing supine reset provider, Backspace uses the startup pose,
and existing recovery/settling gates remain unchanged. Keep the viewer under
human control; do not resume training or change acceptance thresholds.


The user now authorizes pushing the keyboard playback updates to the existing
GitHub repository/branch. The launcher also recognizes the model's new project-
parent location; E134 records hash equality. Publication excludes model weights,
local metrics/video artifacts and the active viewer logs. Training stays paused.


The latest user request adds real disturbance keys: F holds a physical 150 N
horizontal push, C clears it while preserving state and episode. E135 verifies
100 relevant tests, a real rigid-body force/clear differential, and one
continuous G1 walking -> pushed fall -> no-force recovery episode. The model
regains upright posture in that trial but remains in Recovery without stable
handover. Manual verification is available; policy-quality promotion and
training remain unauthorized.
