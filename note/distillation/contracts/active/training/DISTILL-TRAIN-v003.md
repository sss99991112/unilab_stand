contract_id: DISTILL-TRAIN-v003
status: active
effective_date: 2026-07-17
updated_date: 2026-10-10
supersedes: DISTILL-TRAIN-v002
method_contract: DISTILL-METHOD-v004
concept_figure: note/architecture/concept/03_g1_multiteacher_distillation_method.data.json
integration_status: complete
promotion_status: deferred
default_execution_mode: legacy

# Single-Entry DAgger With Optional Persistent Runtime

## Decision

The persistent DAgger runtime is integrated and contract-verified, but it is
not promoted. The public workflow remains single-entry and the default
execution mode remains `legacy`. Persistent execution is an explicit opt-in
implementation of the same DAgger semantics, not a new method and not the
default production path.

This decision is supported by E34-E67 and E75-E86: ownership, barrier,
lifecycle, schema, lineage, metrics, differential behavior, and repository
gates pass, while E67 reports `NO_STABLE_SPEEDUP`. No HP-5 optimization owner
or default-on promotion is authorized.

## Inherited Semantic Contract

`DISTILL-TRAIN-v002` remains semantically inherited:

`student_k rollout -> teacher relabel -> aggregate_1..k -> update -> student_(k+1)`.

`walk_to_stop` remains a scenario, not a third role or expert. Transition rows
retain scenario, age, command-before/after, role, intent, and teacher-action
identity. Missing or mixed transition schemas fail closed. Physical acceptance
remains separate from schema, training completion, and repository tests.

## Runtime Ownership

- UniLab `AsyncRunner` owns the persistent collector process lifecycle.
- UniLab `SharedWeightSync` owns versioned student publication.
- The distillation workflow owns request/result semantics, teacher selection,
  dataset schema, artifact identity, cumulative aggregation, and the outer
  update barrier.
- `performance.py` owns structured request/workflow/learner/cleanup evidence.
- Scripts and Hydra select and assemble owners; they do not copy runner/IPC or
  implement long-term DAgger business rules.

Each scenario request names one checkpoint and expected weight version. All
scenarios in an outer iteration use the same `student_k`; publication of
`student_(k+1)` occurs only after aggregation and update complete. Result or
checkpoint identity mismatch fails closed.

## Promotion And Default Contract

- `training.workflow.execution_mode=legacy` is the default.
- Persistent execution requires explicit opt-in.
- `NO_STABLE_SPEEDUP` forbids a speedup claim and default-on promotion.
- Persistent artifacts may be used for bounded diagnostics and comparisons;
  they are not evidence of policy quality.
- Promotion requires a new human decision backed by new stable end-to-end
  evidence; it is not implied by repository tests or lifecycle correctness.

## Forbidden Behavior

- Do not copy UniLab runner, weight-sync, collector-error, replay, or ring-buffer
  implementations into distillation.
- Do not publish student weights between scenarios of one outer iteration.
- Do not switch backend through `training.sim_backend` alone.
- Do not claim speedup, physical recovery, or policy promotion from static,
  contract, or bounded lifecycle evidence.
- Do not silently enable persistent execution or begin HP-5 optimization.

## Required Current Evidence

- E61/E65/E67: formal timing, amortization, repeated comparison, and
  `NO_STABLE_SPEEDUP` verdict.
- E70/E71: production-readiness and Architecture alignment.
- E86: repository `make test-all` PASS.
- E116: v002 StandHeight/Walk persistent connector and cleanup PASS with
  synthetic fixtures; the E67 speedup verdict and explicit-opt-in decision are
  unchanged.
- Separate live gates: RT-10 physical acceptance and optional Motrix runtime
  remain outside this contract's completed integration claim.

## Recovery Integration Binding (2026-10-10)

The active method is v004. The outer-loop, immutable sources, synchronization
barrier and legacy default above are unchanged. The new three-role recovery
profile currently supports legacy execution only; its persistent mode is
rejected before writes. Existing two-role profiles retain their prior runtime
choices. This binding does not authorize or prove a formal training run.


Recovery continuation may explicitly fork from a saved offline student update
and its named cumulative dataset, with the student's parent-checkpoint hash
verified. This seeds a new run without recording uncommitted old-round completion.
Opt-in `training.workflow.isolate_offline_stages=true` executes DAgger aggregation
and learning in fresh processes through the existing offline entrypoint owners;
collection, quotas, cached targets and the cumulative outer barrier are unchanged.
The default remains false and persistent recovery remains unsupported. This is
runtime containment with local contract evidence, not a native root-cause or
physical-quality acceptance claim (E128).
