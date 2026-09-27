# Explicit opening after uncertain manipulation

The shared PnP executor can retain a confirmed-held bookkeeping flag after a later
visual observation finds the object far from the tool. Its next `pick` rejects
while that flag remains set, and another `place` can repeat the same near-tool
failure. The existing public session has no explicit opening operation that also
clears its own bookkeeping. Direct low-level gripper commands do not clear it.
This is a control-interface limitation, not evidence that the geometry is wrong.

A separate development prototype supplies `open_gripper()` equally to Visual Base
and Full. It issues the existing `api.gripper(.8, seconds=.6)` command and clears
the executor's held record only after acknowledgment. Its return value is
`{'status': 'opened_unverified', 'held': False}`. Here `held` describes controller
bookkeeping. The command certifies no physical release, contact, stable support,
object pose or task completion. It does not choose when opening is appropriate,
retreat the arm, sense objects or perform task-specific recovery.

If the command raises, the method does not report successful opening or erase the
previous record. Old visual evidence remains stale after the action; a subsequent
pick must obtain fresh visual evidence. Predictions and desired states are never
promoted into observations.

## Evidence

Three tests pass: both public interfaces issue identical commands without sensing;
an unacknowledged command preserves the previous record; and a subsequent pick can
reach fresh perception after explicit opening instead of stopping at the old held
flag. The tests do not certify task completion.

One live probe uses previously inspected development layout 990202 and the frozen
5 mm guard counterfactual. A manually inserted opening follows its placement
exception, after which the original policy still stops. The first 280 API calls and
all 48 camera archives match the parent. The first changed call is exactly the
explicit opening. Its checkpoint says `opened_unverified` and keeps the object
pose unavailable. This verifies the command in the actual simulated failure state;
it is not a learned policy or a success-rate improvement experiment.

No reserved layout or model call is used. The prototype and its human insertion
are never actor feedback. The constraint-feedback learning study remains frozen
and does not include this operation.

## Artifacts

- `scripts/prototype_open_gripper_session.py`: standalone library extension.
- `tests/test_explicit_open_gripper.py`: three behavioral tests.
- `scripts/run_explicit_open_probe.py`: bounded live probe.
- [Probe evidence](evaluations/explicit_open_development.json), including source and
  artifact hashes; full files under `outputs/explicit_open_development_01`.

A future comparison could expose the primitive identically to both arms. Such a
comparison would require a new frozen protocol and fresh reserved layouts. These
development checks alone establish no abstraction treatment effect.

## Bounded reacquisition counterfactual

A second, separately registered development probe adds one human-authored rule to
the same guard-counterfactual planner: after a placement exception, explicitly
open, visually pick the same target again, and retry its unchanged placement goal.
The rule runs at most once per invocation of the original placement wrapper; it
never calls itself recursively. Original downstream planning and all perception
acceptance gates remain intact. There is no newly supplied object pose or contact
signal. The executor reports its ordinary visual lift/release statuses.

All five previously inspected guard-development layouts were rerun:

| Seed | Original guard counterfactual | With bounded recovery | Explicit openings |
| --- | --- | --- | --- |
| 990201 | Failure | Failure | 0 |
| 990202 | Failure | Success | 1 |
| 990203 | Success | Success | 0 |
| 990204 | Failure | Failure | 0 |
| 990205 | Failure | Failure | 1 |

The development total changes from 1/5 to 2/5. In 990202, the first 280 API calls
and 48 camera archives match before the inserted opening. Fresh visual reacquisition
reports `lift_visually_confirmed`, the retry reports `released`, and the unchanged
remaining planner completes the task. In 990205, the first 257 calls and 44 images
match; reacquisition also confirms a lift, but placement again lacks an accepted
current pose. Opening itself leaves the prior pose unavailable in both cases.
The three untreated cases retain the same nonlogging API calls, camera archives
and binary outcomes (terminal CPU-profile metadata is excluded from API equality).

This is evidence that the missing explicit recovery operation contributes to one
observed failure. It is **not** a learned-policy result, a Base/Full comparison,
or a generalization estimate: the layouts were previously inspected and the rule
was authored manually. Final success is read only by the offline evaluator after
the runtime policy ends. No reserved layouts, model calls, or frozen-study actor
feedback are involved. The successful retry does not remove the perception gate
that still blocks the other case.

Driver: `scripts/run_explicit_open_retry.py`. Full protocol, source snapshots and
rollouts: `outputs/explicit_open_retry_development_01`.
[Counterfactual evidence](evaluations/explicit_open_retry_development.json) retains
all five outcomes and prefix checks. The running constraint-feedback study does
not include this operation or the recovery rule.
