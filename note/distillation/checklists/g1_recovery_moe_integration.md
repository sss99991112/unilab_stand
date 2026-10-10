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
| Unified student recovery | formal training / live eval | PENDING | Server trace reaches DAgger after bootstrap; trained policy quality not evaluated |
| Repeated falls and return to walk | live eval | PENDING | No physical student acceptance yet |
| Server formal workflow | user server | BLOCKED, fix prepared | Run 20261010-171936 fails on locked recovery config overlay; fork continuation reuses verified bootstrap; no direct server access |
| Persistent recovery scenario | runtime | UNSUPPORTED | Rejected; new profile explicitly uses legacy |
| Real-robot guard and export | deployment | PENDING | Height/support state guard currently uses MuJoCo sensors |

Baseline-only diagnostic tests remain failed: nine runtime trace tests reproduce
on unmodified `cd9f895c`; do not report the entire repository test suite as passing.
