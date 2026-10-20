# Current G1 recovery task

Goal: retain working three-expert simulation recovery while reducing recovery
loads before any real-robot acceptance.

Do not: restart training, change weights/reward/PD/the original 50-frame stable
gate, connect hardware, or equate a video/bounded simulation with reliability.

## Current identity

- UniLab 0.1.0: `/Users/sss9999/enbodied_intelligence/unilab/unilab_stand`.
- Active branch: `codex/g1-standing-assist-sim`; feature commit `bfdae68a`
  is published on the same GitHub branch. Subsequent cleanup is documentation
  and local housekeeping only; use Git for the exact current tip.
- Frozen 99-D/29-action student: project-parent `dagger_iteration_2.pt`, original
  round 6, SHA `27d06b8d73734fb2a17c730aa63ff70b2ba4ffbef7d3d71373df43974ab9bf18`.
- RoboJudo published reference: `6e57327`; unchanged by this source work.

## Confirmed decisions

- Walk=0, StandHeight=1, Recovery=2; retain teachers and inherited MoE/DAgger.
- Standing assistance is opt-in simulation only. Old playback stays default.
  `bash play_g1_recovery_keyboard.sh --standing-assist --supine` opts in.
- Training remains paused. The user reports earlier hardware policy deployment;
  this does not establish safe full-body recovery.
- Original E132 student acceptance is 8/20 stable handovers. The different E136
  protocol and the current small regression do not replace it.

## Progress

- [x] Native playback integration — one env.step action commit per frame;
  preserved independent decide API and reset/pause/preview behavior.
  Evidence: 94 focused tests plus 2 backend snapshot regressions; all 11 changed
  Python files pass Ruff check/format, shell/diff checks pass. `make check`
  attempted on an isolated copy; type checks cannot start without mypy.
- [x] Original-budget OFF/ON regression — 12 nominal/no-noise episodes,
  5650 native control steps / 16950 physical substeps. OFF exact; ordinary
  stand/low-height/walk exact with ON; same-state ON obs/action/phase/route and
  independent physical replay exact.
  Evidence: `.local-build/g1_standing_assist/formal_verified_20261020/REPORT.md`.
- [x] Preserve original diagnostics — E136's 60 paired audit episodes identify
  recovery ankle/wrist/self-contact loads. Getter fix and tests are now included
  in feature commit bfdae68a. Historical evidence stays in `evidence/current.md`.
- [x] Repository cleanup — keep one active local branch, preserve the remote
  default/baseline refs; archive intermediate formal runs outside the repo;
  remove duplicate ignore entries and separate historical training commands
  from the current playback guide. No model or unique evidence is deleted.

## Remaining boundary

Seed2 starts assist at 1.78 s, falls back for load at 2.02 s (max applied alpha
0.60), releases at 3.82 s and ends settle at 5.82 s. It does not complete the
20-frame blend. Left wrist pitch/yaw each add 2 saturated 150-Hz samples;
4–15 s self-contact peak rises 15.36 to 20.21 N. Earlier recovery ankle target
excess and large force/contact peaks remain. GUI ON manual checks, broad
noise/DR/initial-state/repeated-fall reliability and hardware are unverified.

## Next

Propose the smallest bounded ankle/wrist/self-contact correction, using the
existing 150-Hz evidence. Implementation, training and hardware scope require
the user's next instruction. Current use: [playback guide](plans/g1_recovery_moe_integration.md).
