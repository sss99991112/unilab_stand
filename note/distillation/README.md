# G1 Multi-Teacher Distillation

Start with the [current task canvas](task_canvas.md). It is the single current
status source. Older plans, evidence and architecture figures retain their
historical scope; they cannot override measured current behavior.

## Current scope

The 99-D, 29-action student has Walk=0, StandHeight=1 and Recovery=2 experts.
Teachers were retained and the student was trained using MoE/DAgger. The current
local candidate is `../dagger_iteration_2.pt` relative to the project root
(original training round 6); training is paused. The original two-expert path
remains available. This is not a hardware-accepted recovery policy.

Formal simulation playback includes opt-in standing assistance, default OFF.
It reuses frozen experts and the original stable-window gate. Use the
[recovery playback guide](plans/g1_recovery_moe_integration.md) for current
commands and [recovery acceptance checklist](checklists/g1_recovery_moe_integration.md)
for proof boundaries. Models and generated evidence remain local, outside Git.

## Read by purpose

- Current stage, remaining issue and next action: [Task canvas](task_canvas.md).
- Accepted method: [DISTILL-METHOD-v004](contracts/active/method/DISTILL-METHOD-v004.md).
- Accepted training lifecycle: [DISTILL-TRAIN-v003](contracts/active/training/DISTILL-TRAIN-v003.md).
- Verified historical observations E122–E136: [Evidence](evidence/current.md).
- Earlier server commands: [Recovery training history](history/plans/g1_recovery_moe_training_commands.md).
- Earlier plans and diagnoses: [History index](history/README.md).
- Inherited two-expert plan: [StandHeight/Walk implementation](plans/stand_height_walk_two_teacher_implementation.md).
- Inherited transition proposal: [Distribution repair](plans/non_nominal_transition_dagger_repair.md).
- Concept and architecture figures are explanatory views:
  [Concept](../architecture/concept/03_g1_multiteacher_distillation_method.data.json),
  [Method/code](../architecture/architecture/02_g1_distillation_method_to_code.data.json),
  [Runtime](../architecture/runtime/01_unilab_runtime_atlas.data.json).

The formal training interface remains the single-entry workflow. Repeated
optimizer updates on fixed data are not new DAgger outer iterations. The
persistent runtime remains opt-in; its inherited diagnostics do not establish
a current speedup or promote a policy. No training command in historical notes
is an instruction to restart training.
