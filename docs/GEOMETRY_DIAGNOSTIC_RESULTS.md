# Completed diagnostic geometry experiment

The frozen Full R2 policy achieved **9/20 (45%)** reserved success; the
training-selected Visual Base R0 achieved **6/20 (30%)**, a difference of
15 percentage points on the same 20 layouts. This is one policy-learning chain
per arm, not replicated evidence of a reliable method advantage.

## Complete curves

| Revision | Base training | Full training | Base reserved | Full reserved |
| --- | --- | --- | --- | --- |
| R0 | 2/5 | 2/5 | 6/20 (30%) | 6/20 (30%) |
| R1 | 2/5 | 2/5 | 6/20 (30%) | 6/20 (30%) |
| R2 | 2/5 | 4/5 | 6/20 (30%) | 9/20 (45%) |

Training selected Base R0 and Full R2 before any reserved rollout. No test-best
selection replaced these choices. Full's worksite change repaired two training
failures without losing a training success. Both R0 policies had identical binary
outcomes on every reserved layout.

## Paired reserved outcomes

Relative to Base R0 (also the shared R0 outcomes), Full R2 repaired five failed
layouts and lost two previously successful layouts. Four layouts succeeded in
both and nine failed in both. The exploratory exact paired McNemar p-value is
0.453125. The small single-chain result does not establish statistical significance
or isolate which protocol change caused the improvement. Training retention did
not prevent regressions on unseen layouts. Full's 80% training success versus
45% reserved success also limits the generalization claim.

## Budget and resources

All 120 registered reserved rollouts completed with four parallel workers in
approximately 193 seconds. Including training, the study contains 150 executed
rollouts and zero invalid slots. Evaluation made no model calls. The four actor
generations were completed and frozen before evaluation.

| Training generation resource | Base | Full |
| --- | ---: | ---: |
| Input tokens, including cached input | 57,337 | 62,764 |
| Cached input tokens (subset of input) | 22,144 | 35,328 |
| Output tokens, including reasoning | 14,986 | 13,009 |
| Input + output tokens | 72,323 | 75,773 |
| Sum of generation wall time | 456.6 s | 398.4 s |
| New image attachments by revision | 8, 2 | 8, 2 |

Accounting correction after extending the sessions: CLI `turn.completed.usage`
contains cumulative session totals. The earlier report summed those totals and
double-counted R1. The table now uses session-counter differences, checked against
new token events in each archived session. Original generation records and
`training_review.json` are preserved as historical evidence; their summed resource
fields are superseded by this correction.

Full used about 4.8% more total input-plus-output tokens, although its summed
observed generation time was about 12.7% lower. Different cache hits, outputs and
concurrent execution prevent interpreting this single run as an isolated latency
or monetary-cost benefit. Reasoning tokens are already included in output; cached
input is already included in input. Fewer new images do not erase session history.

## Audit and evidence

Strict final audit passed: all 150 trial records, four exact actor contexts,
four hypothesis reviews, starter parity, frozen sources and training-only final
selection were verified. The 20 reserved seeds are 2026102701 through 2026102720;
training seeds are 2026092601 through 2026092605. The closer camera and original
spawn distribution were shared by both arms.

Evidence lives in `outputs/geometry_diagnostic_01/`:

- `summary.json`: all training and reserved outcomes.
- `audit_final.json`: strict completed-study audit and paired statistics.
- `training_review.json`: retention and model-resource accounting.
- `reserved_paired_changes.json`: repaired and regressed reserved seeds.
- `evaluation_lifecycle.json` and `evaluation.log`: completion and execution log.

Reserved outcomes remain evaluation evidence; they were not returned to either
actor. Any future tuning based on these results must treat these layouts as
inspected development evidence and use a fresh reserved set for confirmation.

A later five-round continuation and separately registered fresh evaluation are
reported in [the fresh test results](GEOMETRY_FRESH_TEST_RESULTS.md). That test
compares Base R5 with Full R2 on different layouts; the original results above
remain unchanged.
