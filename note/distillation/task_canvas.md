# Current G1 recovery task

Goal: prepare isolated constrained refinement of the recovery teacher first;
only a jointly accepted teacher may update student expert2. Walk/stand remain frozen.

Do not: start formal/server training now, access the server, change PD/physical
ranges/99-D inputs/29-D action semantics/50-frame+100-settle acceptance, overwrite
old weights/runs, touch RoboJudo's in-flight work or upload weights.

## Current identity

- Source UniLab0.1.0: `/Users/sss9999/enbodied_intelligence/unilab/unilab_stand`.
- Local isolation branch `codex/g1-recovery-teacher-refine`, parent78add63e.
  Preparation uses this existing branch for the authorized GitHub publication.
  Three earlier untracked guard/prototype/test files
  remain unchanged. No student, export or target control code was modified.
- Frozen actor parent: project-parent `2026-10-09_19-37-31_mujoco/model_12000.pt`,
  SHA9f9fa902e6228c93e26e319505bfec584632b38129b35471d04038ad64047bbd.
- Student remains original round6, `../dagger_iteration_2.pt`, SHA27d06b8d73734fb2a17c730aa63ff70b2ba4ffbef7d3d71373df43974ab9bf18.
- This existing source canvas is the current board after the user's explicit
  transition back to training preparation. The target docs/g1_recovery_sim.md
  retains the previous diagnosis; it has not been overwritten or duplicated.

## Confirmed decisions

- User authorized publishing the preparation to GitHub, then requested clear
  branches without proliferation. Reuse `codex/g1-recovery-teacher-refine`; do not
  create another branch or remove unrelated branches.
- User: formal training belongs on their server and cannot begin now.
  Host/GPU/path/time/iteration budget remain to be confirmed. No remote work.
- Teacher itself exhibits actual-q and raw-PD violations; another distillation
  pass from the same frozen teacher is not an established repair.
- Keep original7 reward weights and action mapping. Add default-OFF privileged
 150Hz soft cost measurements, preserving useful ground support.
- New reward -> versioned actor-only load, fresh critic/optimizers/alpha and cold
  replay. Old full learner resume validation stays intact. No invented stats.

## Progress

- [x] Read new handoff sequence4 and verify all81 snapshot files,5 weights and
  protected source/target files; identity matched. CPU probes selected explicitly.
- [x] Add mujoco_constrained_v1 and the recovery_constraints owner flag(defaultOFF),
  four registered penalties and exact native-side load replay. Original physical
  step/control/success path unchanged; ground cannot be a monitored self-pair.
- [x] Add actor adapter g1_recovery_load_actor_99_v1 and fresh output/SHA/baseline
  environment guards. Replay warmup64 steps/env is reachable within512/env.
- [x] 108 targeted tests passed, including prior reward/task/resume/actor/prestep
  behavior, changed physical measurements, sparse randomized reset and cold
  replay/update fixtures. These are not policy-quality or full repository claims.
- [x] Bounded no-training readiness: seeds0/2, OFF/ON each150 control steps,
  600native controls /1800physical substeps plus separate measurement replay.
  Actor matches official teacher exactly, ON physics/obs/action match OFF exactly,
  actual changed rewards reach cold replay, learner_update_count remains0.
  Evidence: `.local-build/g1_recovery_refinement/readiness_20261021_v2/REPORT.md`.
- [x] Pre-publication verification: 108 related tests passed again; the 11 changed
  Python files passed Ruff and format checks. `make check` was attempted in an
  isolated snapshot; format/lint completed but absent `mypy` blocked type checking.
  No full type-check or `make test-all` success is claimed. Publish only this
  preparation; retain the three original untracked experimental files locally.
- [ ] Confirm server resource allocation before any formal run. Commands, proposed
 5000iteration/2hour pilot and stop conditions are in the task-owner README.

## Remaining boundary

No improved teacher or student checkpoint exists. Soft penalties do not enforce
hard limits. New measured3s prefixes still show actual-q excess .1309/.1435rad,
raw PD excess and specified self-contact. No15s/20s joint acceptance,32-env GPU
run, broad noise/DR/repeated-fall or hardware acceptance has been claimed.

## Next

Wait for the user's permission to run on the server and confirm GPU/path/budget.
Prepared command: `conf/offpolicy/task/sac/g1_recovery/README.md`.
