# G1 Recovery MoE Integration Checklist

Active method: `DISTILL-METHOD-v004`; runtime: `DISTILL-TRAIN-v003`.

| Acceptance | Owner | Status | Evidence |
| --- | --- | --- | --- |
| Opt-in profile and legacy profile preservation | Hydra / script | PASS | New profile composes 3 roles/experts; old profile remains 2 |
| Exact old-expert preservation | checkpoint migration | PASS | Tensor equality before training and immutable source bytes |
| Recovery/Walk observation units | unit adapter | PASS | gyro 0.25 and dof speed 0.05 roundtrip; teacher action parity |
| Recovery frames select/update expert 2 | trainer / role collector | PASS | Role-supervised loss, role flag propagation, inactive expert isolation |
| Stable-window latch and reset | combined env | PASS | Frame freshness and latch tests; supine reset selects recovery before first action |
| Handover labels and schema | collector / workflow | PASS | Pre-action labels, cached actions, transition ages and save/reload |
| Formal CLI configuration | train CLI | PASS | Resolved 3-expert workflow with full activation; no learning |
| Existing policy reference handover | MuJoCo probe | PASS (bounded) | seed 1: handover at 9.0 s; 6 more seconds of stand expert; no early termination |
| Locked Hydra scenario config | recovery scenario owner | PASS (bounded) | Real CLI config regression plus 1-env, 16-row MuJoCo owner probe; source lock/content preserved |
| Unified student recovery | formal training / live eval | PENDING physical acceptance | Bootstrap and first saved-data update completed (E127); trained policy not evaluated |
| Repeated falls and return to walk | live eval | PENDING | No physical student acceptance yet |
| Server formal workflow | user server | FRESH LOAD PASS, UPDATE RECOVERY PENDING | E130: exact load replay succeeds for 2424832 rows; original failure cause unknown |
| Standalone saved-data learner | offline update owner | SERVER UPDATE COMPLETE | E127: 10281 updates, 5263872 sampled rows, cached targets, separate checkpoint saved; quality pending |
| Eight-round saved-student continuation | workflow / offline owners | LOCAL PASS, SERVER PARTIAL | E128 local eight-round proof; E129 server reached new round 4, preserve completed round-3 model |
| Interrupted update recovery | workflow / offline owners | LOCAL PASS, SERVER PENDING | E130: acknowledged round-4 data reused, input lineage verified, old files unchanged, exactly four rounds remain |
| Child error preservation | offline stage owner | LOCAL PASS | E130: stderr log persists the inner exception and appears in the raised error |
| Nested source-label provenance | raw artifact / CLI formatter | UNCONFIRMED | E127: output reports one frame-class Walk label; raw persisted metadata audit pending |
| Persistent recovery scenario | runtime | UNSUPPORTED | Rejected; new profile explicitly uses legacy |
| Real-robot guard and export | deployment | PENDING | Height/support state guard currently uses MuJoCo sensors |

Baseline-only diagnostic tests remain failed: nine runtime trace tests reproduce
on unmodified `cd9f895c`; do not report the entire repository test suite as passing.
