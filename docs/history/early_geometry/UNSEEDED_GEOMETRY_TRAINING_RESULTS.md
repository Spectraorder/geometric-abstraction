# Completed three-round training without a supplied task-solving policy

Training selected **Base R3 at 2/5 (40%)** and **Full R2 at 3/5 (60%)**. Both began
with an observation-only R0. All three registered actor turns per arm completed;
no reserved test was run during training. The subsequently authorized
[test evaluation](UNSEEDED_GEOMETRY_TEST_RESULTS.md) scored Base R3 at 6/20 and
Full R2 at 12/20. This is one training chain per arm, not evidence of
held-out generalization or an isolated causal effect of geometric abstraction.

| Revision | Base training | Full training | Main observation |
| --- | --- | --- | --- |
| R0 | 0/5 | 0/5 | Both collect images without manipulation. |
| R1 | 0/5, invalid candidate | 0/5 | Base fails static validation; Full fails at its first action checkpoint. |
| R2 | 0/5 | 3/5 | Both repair one interface issue; Base encounters another. |
| R3 | 2/5 | 3/5 | Base completes its interface repair; Full recovery adds no successes. |

Base's selected successes are seeds 2026092603 and 2026092604. Full also succeeds
on 2026092601. Both still fail 2026092602 and 2026092605. The retention guard
accepted Base R3 and Full R2; Full R3 tied its incumbent and was rejected.

## Interface failures and interpretation

Base R1 placed a hypothesis constant assignment at module scope, which the
existing validator forbids. All five registered slots are retained as invalid
candidate failures and did not execute simulation. Base R2 replaced that assignment
with a function, but then passed an entire action-result dictionary as the
checkpoint's `action_status`. It caught the resulting type error and stopped on
all five layouts. Base R3 changed that argument to the returned status string and
achieved two successes.

Full R1 had the same checkpoint argument error. Full R2 corrected it and achieved
three successes. Full R3 added a sensor-triggered tabletop staging and regrasp
attempt for a tilted thin prism. Its command sequence changed on both failed
layouts, but neither was repaired. Policy diagnostics report no usable nearby
staging patch for 2602 and an object/tool association exception during staging on
2605; these are recorded policy observations, not verified physical causes.

Thus the observed learning curves include generated-code integration errors and
the actors' ability to repair them. They do not cleanly isolate physical planning
quality or the effect of supplied geometry. No candidate was manually corrected,
no free generation retry was inserted, and all consumed slots and tokens remain
in the accounting. R0's deliberate inactivity also means improvement from R0
alone is not a meaningful measure of model zero-shot capability.

## Model resources

| Resource, all three turns | Base | Full |
| --- | ---: | ---: |
| Input tokens, including cached input | 97,950 | 92,536 |
| Cached input, subset of input | 52,096 | 60,672 |
| Output tokens, including reasoning | 27,998 | 20,310 |
| Input + output tokens | 125,948 | 112,846 |
| Sum of generation wall time | 856.4 s | 622.2 s |
| New images at R1, R2, R3 | 4, 0, 2 | 4, 0, 2 |

Both arms resumed their own isolated sessions. R2 needed no supplemental images
under the registered error trigger; R3 attached two. Session history still incurs
input usage. The verified resource report subtracts consecutive cumulative counters
and checks the new token events, avoiding the earlier double-counting problem.
Cached input and reasoning tokens must not be added again to the totals.

Full used approximately 10.4% fewer total tokens in this run. Generation wall times
are summed per arm; calls ran concurrently and these sums are not whole-experiment
elapsed time. Different generated code, interface failures and cache hits prevent
attributing this observation solely to geometry abstraction or inferring monetary
cost directly from total tokens.

## Validation and execution accounting

The registered training budget comprises 40 slots: ten R0 rollouts, 25 additional
executed rollouts and five invalid candidate slots. All six model generations,
exact actor contexts, six hypothesis reviews, shared baseline parity, public-goal
contracts, frozen sources and training-only final selection passed audit.

The training subprocess completed and froze both selected policies. The subsequent
coordinator import failed because its launcher resolved the virtualenv interpreter
symlink to the system interpreter, where NumPy was unavailable. Audit and resource
reporting were then run with the project virtual environment. This postprocessing
repair changed no policy or trial and added no model call or rollout. The original
failure log and receipt were retained, and final lifecycle status is completed.

Evidence lives in `outputs/geometry_unseeded_01/`:

- `training_summary.json`, `all_candidates_frozen.json`: completed curves and selected policies.
- `audit_training.json`: 35 valid rollouts, five invalid slots and provenance checks.
- `resource_accounting.json`, `training_review.json`: corrected usage and retention decisions.
- `training_lifecycle.json`, `training.log`, `postprocessing_failure.json`: execution and repair history.
- `basic/frozen.py`, `symbolic/frozen.py`: selected policy artifacts.

The 20 registered reserved layouts were subsequently evaluated; see the
[test results](UNSEEDED_GEOMETRY_TEST_RESULTS.md). See the
[experiment configuration](../../UNSEEDED_GEOMETRY_EXPERIMENT.md) for the task-goal
contract, retained capabilities and evaluation budget.
