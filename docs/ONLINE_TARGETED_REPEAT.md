# Targeted online repeat after runtime quality flags

The user requested recovery of verifiable interrupted token usage, followed by
rerunning problematic episodes. The narrow repeat scope is **9 complete layout
pairs (18 episodes)** with recorded timing violations or failed GPU dispatch
queries in the original 50-pair expansion. Selection does not use task success.
Both Base and Full run once on each selected seed, with fresh actor contexts.

| Task | Selected seeds | Pairs / episodes |
| --- | --- | ---: |
| Bridge-tower | 2035101007, 2035101008 | 2 / 4 |
| Ring-on-peg | 2035102006, 2035102007, 2035102008, 2035102027 | 4 / 8 |
| Tower relocation | 2035103006, 2035103007, 2035103008 | 3 / 6 |

The already independently verified terminal-image failure on Ring seed
2035102028 is outside this scope. Normal success/deadline cancellations are not
runtime defects and do not select episodes for repetition. Those cancellations
may also occur in the repeat; zero unknown token usage is not a success criterion.

## Conditions and validation

The repeat uses the original episode functions, task goals, prompts, geometric
estimators, skills, actuators, evaluators, model and medium reasoning setting.
The pinned 0.159.2 model client is reused. Budgets remain 200 seconds, with one
initial reset, continuous physics during model waits, and no within-episode
resets. Initial-state hashes must match both the original seed and the paired arm.

The loaded NVIDIA module is now 580.178.04 and matches the system userspace
libraries. The old private 580.173.02 runtime is incompatible with that loaded
module and is not reused. The repeat records and checks the current module and
library hashes without changing the system installation. Four concurrent
development-only sensing/rendering/physics probes passed with no model requests.
The previously validated Ring camera-pacing v2 hook applies symmetrically to
both arms; terminal image capture does not re-enter live physics.

The coordinator permits four workers globally and dispatches complete pairs.
GPU checks precede each batch. Independent interface, token, prompt, physical
success and two-second timing-gate audits must pass before further dispatch.
Infrastructure or audit failures stop the run. Existing attempts cannot be
overwritten; there are no automatic additional repeats, new seeds, or time grants.

## Results and accounting

All 18 repeats completed and passed the strict audits; maximum simulation lag
was 0.988 seconds. At the user's subsequent request, the primary 50-pair report
now uses both repeated arms for every selected pair, regardless of outcome.
This reporting amendment was made after viewing the repeats. There are still
300 selected episodes on 150 paired layouts; the 18 additional attempts do not
add independent layouts. Tower Full increases from 27/50 to 28/50; the other
success totals are unchanged. Relocation Base seed 2035103007 changes from
success to timeout and is retained, just like improvements.

Raw originals and repeat evidence remain separate and unchanged. The previous
report is archived under
`outputs/online_targeted_repeat_01/report_integration_01/previous_export/`.
The integration registration pins both result sets and the final repeat audits.
The current export includes `replacements.csv`, `original_attempts.csv`, and
`superseded_attempts.csv`, plus an original-attempt aggregate and the unchanged
sensitivity excluding all nine affected pairs.

Selected episodes record **41,603,662 tokens**. Superseded originals account for
another **2,745,300**, and startup overhead for **98,036**: **44,446,998** recorded
campaign tokens in total. Those additional costs are excluded from the selected
per-episode means but remain visible in `summary.json`. The selected episodes
have 77 interruptions: four recovered usages and **73 unknown**, including three
from the repeats. Unknown usage is not zero and does not itself invalidate an
episode's outcome.

Separately, the read-only terminal usage audit recovered **90,796 tokens from
four of the original 76 interrupted requests**. CLI totals agree with the
append-only actor session counters and last-turn deltas. The original-attempt export includes those tokens; 72 original usages remain
unknown. Replacement changes the selected set to 73 unknown usages. Original outcomes, accepted
decisions, clocks, images, receipts and raw logs are unchanged. The correction
does not itself require a physical or model rerun.

- [Primary expanded report](evaluations/online_50/README.md)
- [Token recovery evidence](evaluations/online_50/interrupted_usage_audit.json)
- [Repeat progress](../outputs/online_targeted_repeat_01/progress.json)
- [Repeat registration](../outputs/online_targeted_repeat_01/registration.json)
- [Development GPU preflight](../outputs/online_repeat_preflight_01/scenes/report.json)

The revised primary comparison uses the audited repeats for the selected pairs.
This runtime repair is not a geometry/feedback component ablation or a uniform
repeated-run evaluation across all layouts.
