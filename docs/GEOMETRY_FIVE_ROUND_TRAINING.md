# Five-round continuation of the diagnostic experiment

The completed five-round chain selects **Visual Base R5 and Full R2**, both at
**4/5 training success** on the same five layouts. Full reached that result at R2;
Base caught up at R5. Neither arm repaired seed 2026092602. No new reserved
rollouts were executed during training. A subsequently authorized
[fresh test evaluation](GEOMETRY_FRESH_TEST_RESULTS.md) scored Base R5 at 15/20
and Full R2 at 14/20 on 20 shared new layouts.

| Revision | Base training | Full training |
| --- | --- | --- |
| R0 | 2/5 | 2/5 |
| R1 | 2/5 | 2/5 |
| R2 | 2/5 | 4/5 |
| R3 | 2/5 | 4/5 |
| R4 | 2/5 | 4/5 |
| R5 | 4/5 | 4/5 |

The strict success-set guard retained each incumbent until a candidate repaired
at least one training layout without losing a protected success. Full R3–R5 tied
R2 and were rejected. Base R5 repaired 2026092604 and 2026092605 while preserving
2026092601 and 2026092603. Thus the selected arms have identical training outcomes.

## What the extra rounds tested

- Base R3 triggered an extra placement correction on 2604, but the residual
  changed by only about 0.03 mm. Base R4 relocated the worksite and completed more
  actions, but construction still failed. R5 moved worksite selection before
  construction and repaired both remaining non-beam failures.
- Full R3's recovery branch did not trigger. R4 detected the marginal beam grasp
  but could not establish a safe staging surface. R5 planned a waypoint before
  grasping, but the subsequent current-view fit remained unresolved. Stored
  targets did not replace fresh observation checks.
- At R5, Base's 2602 notes still report the object/tool association mismatch;
  Full's notes report conflicting visible extent and insufficient hidden-axis
  constraints during staging. Both rollouts returned failure without an uncaught
  policy exception. Catching an observation error did not repair the task.

These are training diagnostics and candidate hypotheses, not verified simulator
truth about the physical cause of failure. No manual diagnosis was inserted into
an actor prompt during the continuation.

## Resources and interpretation

| Model resource, R1–R5 | Base | Full |
| --- | ---: | ---: |
| Input tokens, including cached input | 244,536 | 275,406 |
| Cached input tokens, subset of input | 143,872 | 195,968 |
| Output tokens, including reasoning | 35,808 | 40,249 |
| Input + output tokens | 280,344 | 315,655 |
| Sum of generation wall time | 1,093.7 s | 1,230.3 s |
| New images by revision | 8, 2, 2, 2, 2 | 8, 2, 2, 2, 2 |
| Tokens through first 4/5 training result | 280,344 (R5) | 75,773 (R2) |

The added R3–R5 calls used 208,021 tokens for Base and 239,882 for Full. Both arms
made three additional actor turns and 15 additional training rollouts. The whole
extension took approximately 17 minutes 26 seconds, with the two actor calls
running concurrently at each revision. Summed generation times are not experiment
elapsed time.

Full reached the observed training threshold earlier and with fewer tokens in this
chain. This threshold comparison is retrospective and based on five training
layouts, not a preregistered efficiency endpoint. With both arms run through R5,
Full used about 12.6% more total tokens. There is no demonstrated total-token
saving from geometric abstraction in this run. Both arms used the same image
attachment schedule; persistent history still grows even when fewer new images
are attached. Monetary cost cannot be inferred directly from total tokens because
cached input, uncached input and output have different accounting categories.

### Accounting correction

The CLI's `turn.completed.usage` records in this study contain cumulative session
counters. Summing those records double-counts earlier turns. The corrected report
subtracts consecutive totals and checks each difference against the new token
events' `last_token_usage` fields in archived session logs. Cached input is already
included in input, and reasoning is already included in output.

The original two-round totals are therefore Base **72,323**, Full **75,773** tokens,
superseding the earlier report's 101,366 and 107,039. Raw generation records and
original summaries/reviews remain unchanged as historical evidence. The maintained
result documentation now explicitly records this correction.

## Validation and evidence

The final continuation audit passed: 30 additional valid training rollouts, zero
invalid slots, six additional actor turns, exact prompt/image reconstruction,
session lineage, candidate integrity and training-only selection. It also rechecked
the original completed study and all preserved evidence hashes. Across the original
study and continuation there are 60 training and 120 historical reserved rollouts.

Artifacts are under `outputs/geometry_diagnostic_01/extension_05/`:

- `amendment.json`: separate registration and original evidence hashes.
- `training_summary.json`, `training_review.json`: curves, retention and resources.
- `resource_accounting.json`: verified per-turn usage and separate original/extension totals.
- `audit.json`, `lifecycle.json`: successful validation and completion.
- `basic/frozen.py`, `symbolic/frozen.py`: selected policies.

The original [reserved results](GEOMETRY_DIAGNOSTIC_RESULTS.md) remain Base R0
6/20 and Full R2 9/20. They do not measure the new Base R5. The continuation was
authorized after those results were inspected, although neither actor received
reserved outcomes. The subsequent comparison used a separately registered fresh reserved split;
see the [fresh test results](GEOMETRY_FRESH_TEST_RESULTS.md).
