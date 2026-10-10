# G1 Multi-Teacher Distillation Control Room

This directory is the default human/LLM entrypoint for the G1 multi-teacher
distillation work. The historical migration compendium is evidence and history,
not the current design authority.

## Default Read Path

1. [Concept Figure](../architecture/concept/03_g1_multiteacher_distillation_method.data.json)
   - Human-controlled method intent.
2. [Active Method Contract](contracts/active/method/DISTILL-METHOD-v004.md)
   - Accepted semantics and design-point details.
3. [Current Task Canvas](task_canvas.md)
   - Current problem, verified facts, active proposal, and next decision.

Stop after these three unless a concrete question requires deeper evidence.

## Drill Down By Question

- Current three-expert integration and server commands:
  [Recovery MoE Integration](plans/g1_recovery_moe_integration.md)
- Current three-expert acceptance:
  [Recovery Checklist](checklists/g1_recovery_moe_integration.md)

- Current non-nominal transition repair plan:
  [DAgger Distribution Repair](plans/non_nominal_transition_dagger_repair.md)
- Completed StandHeight/Walk pre-training plan:
  [Five-Step Implementation Plan](plans/stand_height_walk_two_teacher_implementation.md)
- Inherited single-entry training workflow:
  [Single-Entry Training Workflow](plans/single_entry_training_workflow_proposal.md)
- Current StandHeight/Walk acceptance state:
  [Checklist](checklists/stand_height_walk_two_teacher.md)
- Legacy workflow acceptance state: [Checklist](checklists/current.md)
- Verified facts: [Current Evidence](evidence/current.md)
- Archived reset/stop diagnosis:
  [Playback Reset And Stop-Transition Root Causes](evidence/2026-07-15-playback-reset-and-stop-transition-root-causes.md)
- Code owners: [Method-to-Code Atlas](../architecture/architecture/02_g1_distillation_method_to_code.data.json)
- Repository runtime: [UniLab Runtime Atlas](../architecture/runtime/01_unilab_runtime_atlas.data.json)
- Superseded/completed plans and migration history: [History](history/README.md)

## Human/LLM Interaction Loop

```text
Human edits or confirms Concept Figure intent
  -> LLM proposes/versions the matching contract detail
  -> Human confirms semantic changes
  -> LLM refreshes Concept Figure + Method-to-Code + Runtime Atlas
  -> Engineering plan and checklist define bounded work
  -> Code/tests/live probes produce evidence
  -> Checklist and atlas are refreshed before another training run
```

The Concept Figure is the human method control surface. Markdown contracts are
the detailed machine-readable source of accepted semantics. Architecture maps
are derived current-state views. Plans and checklists are replaceable. Evidence
is retained. Raw logs are restricted inputs and must not become design truth by
accumulation.

## Current Method In One Sentence

Use recovery-first physical routing followed by velocity intent to select a
Walk, StandHeight or Recovery expert in one 99-D MoE student. Explicit unit
adapters preserve each teacher's input semantics. The original two-expert route
remains available; the three-expert student has not been formally trained.

## Current Boundary

The current cursor is recorded in `task_canvas.md` and the recovery checklist.
The entries below are retained two-expert baseline boundaries.

- Existing 98-D standing and walking teachers are legacy sources.
- The `G1StandHeight` task, explicit 98-D to 99-D actor adapter, and unified
  two-expert 99-D workflow are complete at their deterministic boundaries.
- E114 preserves the retained Step 2 result (`108 passed, 24 warnings in
  19.46s`) and Step 3 result (`8 passed in 6.77s`); E113 records Step 4 Ruff
  PASS and `27 passed in 20.56s`. These are contract and connector results, not
  policy-quality evidence.
- E115 confirms one real `G1StandHeight` MuJoCo environment for one step,
  including 99-D input, target index 96, zero command, support/tilt snapshot,
  no termination, and a no-training SSH-command compose preflight.
- E116 confirms that StandHeight SAC dispatches through the existing
  async/double-buffer runner and that the 99-D two-role workflow can explicitly
  opt into the persistent DAgger runtime; its focused suite is `3 passed in
  3.83s` with Ruff PASS. This is connector evidence, not a speedup result.
- E117 records a real 99-D round-2 student parent that failed four of nine
  non-nominal Walk-to-StandHeight cases, plus the local command x recovery-
  height collection repair. The repair has `10 + 7` focused tests and Ruff
  PASS, but has not been trained or evaluated.
- Remote StandHeight/Walk teacher identities, role datasets, and student runs
  now exist. This checkout did not read their bytes; exact paths and runtime
  claims remain conversation-backed evidence in E117.
- Existing student checkpoints, including the round-2 parent, are candidate
  artifacts only. None is a promoted or accepted policy.
- Playback reset/command ordering is repaired and contract/live-reset tested;
  transition policy quality remains a separate unresolved boundary.
- Stop-transition collection now includes forward/lateral/yaw crossed with
  `0.650/0.702/0.754 m` recovery targets while active walking stays at
  `0.754 m`; the repaired SSH fork and physical gate remain unexecuted.
- The seven-command training workflow is intended only for diagnostics, but
  its explicit diagnostic-only labeling remains an open checklist item. The
  accepted public path is the single-entry workflow owned by
`DISTILL-TRAIN-v003`.
- The persistent runtime is integrated but not promoted. It remains explicit
  opt-in and the default execution mode remains `legacy` because E67 found no
  stable end-to-end speedup.
- Iterative DAgger means repeated outer student rollouts followed by relabel,
  cumulative aggregation, and update. Repeating optimizer updates on one fixed
  dataset is not an additional DAgger iteration.

## Formal Walk/Stand Command

The same command handles fresh collection, reuse, Bootstrap, iterative DAgger,
and resume. Existing legacy datasets can be adopted once only after their
tensor and metadata contracts pass; the flag does not permit filename-only
reuse.

```bash
CUDA_VISIBLE_DEVICES=0 \
UNILAB_G1_WALK_TEACHER=/ssd1/cyx/UniLab/model/G1WalkFlat/model_5000.pt \
UNILAB_G1_STAND_TEACHER=/ssd1/cyx/UniLab/model/G1StandStill/model_5000.pt \
UNILAB_G1_WALK_DATASET=/ssd1/cyx/UniLab/model/teacher/walk_flat_teacher_policy.pt \
UNILAB_G1_STAND_DATASET=/ssd1/cyx/UniLab/model/teacher/stand_teacher_policy.pt \
HYDRA_FULL_ERROR=1 PYTHONWARNINGS="ignore" \
uv run train \
  --algo distill \
  --task g1_walk_flat \
  --sim mujoco \
  workflow=g1_walk_stand \
  training.workflow.run_dir=/ssd1/cyx/UniLab/logs/distill_workflow/g1_walk_stand \
  training.workflow.adopt_legacy_artifacts=true
```

On the first run, each role reports `REUSE` or `COLLECT` independently. Reusing
the same `run_dir` resumes only unfinished DAgger iterations. Increase
`training.workflow.dagger_iterations=N` to continue the same run to a larger
outer-loop target. Use `training.workflow.mode=fork` plus
`training.workflow.parent_run_dir=...` when downstream logic changes but
compatible role artifacts and the parent checkpoint should be preserved.

The resulting candidate is
`<run_dir>/checkpoints/dagger_iteration_<N>.pt`; exact lineage and hashes are in
`<run_dir>/run_manifest.json`. Physical acceptance is still a separate gate.
