# Reserved evaluation without a supplied task-solving policy

The training-selected **Base R3 scored 6/20 (30%)**, while **Full R2 scored 12/20
(60%)** on the same 20 reserved layouts. Full minus Base is **30 percentage
points**. Both policies were selected and frozen before test execution; no
reserved result changed the selection.

## Complete curves

| Revision | Base training | Full training | Base reserved | Full reserved |
| --- | --- | --- | --- | --- |
| R0 | 0/5 | 0/5 | 0/20 | 0/20 |
| R1 | 0/5, invalid | 0/5 | 0/20, invalid | 0/20 |
| R2 | 0/5 | 3/5 | 0/20 | 12/20 |
| R3 | 2/5 | 3/5 | 6/20 | 10/20 |

R0 deliberately captures images without acting. Base R1 contains a forbidden
module-level assignment and consumed invalid candidate slots without simulation.
Full R1 and Base R2 encounter the checkpoint status-argument error described in
the [training report](UNSEEDED_GEOMETRY_TRAINING_RESULTS.md). These curves therefore
include interface debugging costs as well as manipulation-policy learning.

Full R3 tied R2 on training and was rejected by the registered strict-improvement
guard. On reserved layouts it lost successes on 2026122707 and 2026122715 and
repaired none. Retaining R2 happened before these outcomes were observed; the
report does not retrospectively choose the test-best revision.

## Paired comparison and limitations

The selected policies both succeeded on five layouts. Full alone succeeded on
seven, Base alone on one, and both failed on seven. The exploratory exact paired
McNemar p-value is **0.0703125**. Marginal Wilson 95% success intervals are
14.5–51.9% for Base and 38.7–78.1% for Full; these are not an interval for the
paired difference.

The observed difference favors Full in this run, but 20 paired layouts and one
learned chain per arm do not establish a reliable method advantage. The comparison
also includes the supplied perception/geometry package and differences in actor
interface debugging, not solely a text representation. Model-seed replication and
new untouched layouts would strengthen the conclusion. Do not compare these rates
directly to the older capable-starter studies, which used different policies and
test splits, to estimate the causal effect of removing the starter.

## Resources

The complete reserved phase took approximately **197.5 seconds**, including checks
and final audit, using four parallel rollout workers. It made zero model calls
and consumed zero model tokens.

| Selected-policy runtime metric, mean per reserved rollout | Base R3 | Full R2 |
| --- | ---: | ---: |
| Camera API calls | 90.30 | 63.35 |
| Policy compute CPU time | 1.379 s | 1.210 s |

These are simulator camera acquisitions, not image attachments to a model. Policy
CPU time includes worker compile/import/run and excludes RPC, simulator/rendering
and host IK. Different action paths and stopping conditions prevent interpreting
the averages as a controlled latency measurement of abstraction alone.

Training resources remain Base 125,948 and Full 112,846 input-plus-output tokens
across three turns, with new image attachments 4, 0, 2 in both arms. The
`resource_accounting.json` figures use verified per-turn differences of cumulative
session counters. Evaluation does not change those totals.

## Accounting and audit

All 160 registered reserved slots were accounted for: **140 executed rollouts**
and **20 invalid candidate slots** for Base R1. Together with training, the final
audit verifies **175 executed rollouts and 25 invalid slots**, totaling the
registered 200 slots. Invalid candidates count as failures in the full-budget
curves; they are not discarded or replaced.

Final audit verified all six actor contexts and hypothesis reviews, source and
policy hashes, paired baseline parity, public-goal contracts, complete reserved
curves and training-only selection. The coordinator additionally verified that
training summaries, freeze gates, resource accounting and live actor session state
remained unchanged during evaluation. Both selected policies had no uncaught
policy errors in the reserved records; this does not imply all rollouts succeeded
or that no caught observation failure occurred.

The shared reserved seeds are 2026122701 through 2026122720. Evidence lives in
`outputs/geometry_unseeded_01/`:

- `summary.json`: full training and reserved curves.
- `audit_final.json`: completed provenance audit and paired statistics.
- `reserved_review.json`: selected metrics and exact repaired/regressed seeds.
- `evaluation_lifecycle.json`, `evaluation.log`: completion, timing and execution log.
- `evaluation_preserved_files.json`: hashes of training evidence preserved during evaluation.

These reserved outcomes are now inspected evidence. Do not feed them into the
existing training sessions or describe a later tuned evaluation on the same
layouts as a fresh test.
