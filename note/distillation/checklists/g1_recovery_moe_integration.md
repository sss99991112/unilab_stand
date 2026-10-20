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
| Unified student recovery | formal training / live eval | BOUNDED: 8/20; acceptance pending | E132: original round 6 bytes verified; all 20 reach upright, only 8 stabilize and hand over; 1 falls again |
| Old walking/height capability preservation | matched MuJoCo evaluation | PASS (bounded) | E132: each of stand nominal, stand low, forward walk, walk-stop-height survives 3/3 matched 15 s trials for both old/new; wider commands untested |
| Repeated falls and return to walk | live eval | PENDING | E135 adds one true walking-push-fall trial; broad repeated-fall and return-to-walk acceptance pending |
| Server formal workflow | user server | PAUSED BY USER | E131: continuation round 2 (original round 6) complete; process stop must be confirmed on server |
| Standalone saved-data learner | offline update owner | SERVER UPDATE COMPLETE | E127: 10281 updates, 5263872 sampled rows, cached targets, separate checkpoint saved; quality pending |
| Eight-round saved-student continuation | workflow / offline owners | LOCAL PASS, SERVER PARTIAL | E128 local eight-round proof; E129 server reached new round 4, preserve completed round-3 model |
| Interrupted update recovery | workflow / offline owners | SERVER PARTIAL; latest bytes verified | E132: continuation round-2 student downloaded and load/route verified; server receipt/other bytes remain user-reported |
| Child error preservation | offline stage owner | LOCAL PASS | E130: stderr log persists the inner exception and appears in the raised error |
| Nested source-label provenance | raw artifact / CLI formatter | UNCONFIRMED | E127: output reports one frame-class Walk label; raw persisted metadata audit pending |
| Persistent recovery scenario | runtime | UNSUPPORTED | Rejected; new profile explicitly uses legacy |
| Manual keyboard recovery playback | playback / native launcher | PASS (bounded + native UI) | E133: 58 targeted checks; real forward/stop/height/recovery reset action parity; native key callbacks observed; unrelated tee-output check fails |
| Physical push / force clear without reset | NpEnv / playback commander | PASS (bounded) | E135: 100 tests, actual rigid-body acceleration/clear proof, continuous G1 walking-push-fall-recovery episode with no reset; stable handover not accepted |
| Default-OFF formal standing assist playback | playback / combined env / controller | PASS (bounded integration only) | 2026-10-20: 94 tests; 5650 native control steps / 16950 physical substeps; OFF exact; one action commit; seven original-budget ON cases; seed2 partial assist then load fallback, 3.82/5.82 s handover/settle; wrist/self-contact costs remain. `.local-build/g1_standing_assist/formal_verified_20261020/REPORT.md` |
| Matched simulation load audit | MuJoCo diagnostic snapshot / recovery actors | PASS (diagnosis only) | E136: 60 episodes, exact replay, ankle target violations / wrist saturation / self-contact; no hardware safety acceptance |
| Real-robot guard and export | deployment | PENDING | E136 identifies recovery-load issues; prior deployment experience reported, actual limits/mapping and real-robot recovery remain unverified |

Baseline-only diagnostic tests remain failed: nine runtime trace tests reproduce
on unmodified `cd9f895c`; do not report the entire repository test suite as passing.
