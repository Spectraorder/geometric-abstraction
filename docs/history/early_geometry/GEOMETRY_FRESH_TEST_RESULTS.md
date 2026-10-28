# Fresh test evaluation after five training iterations

On 20 fresh shared layouts, the training-selected **Visual Base R5 succeeded on
15/20 (75%)** and **Full geometry R2 on 14/20 (70%)**. Full minus Base is
**-5 percentage points**. This small paired sample does not establish a reliable
performance difference between the methods.

| Metric | Visual Base R5 | Full geometry R2 |
| --- | ---: | ---: |
| Training success | 4/5 | 4/5 |
| Fresh test success | 15/20 (75%) | 14/20 (70%) |
| Wilson 95% interval | 53.1–88.8% | 48.1–85.5% |
| Mean camera API calls per rollout | 53.30 | 54.05 |
| Mean policy compute CPU time per rollout | 0.947 s | 1.010 s |
| Median simulated policy duration | 92.725 s | 92.650 s |

Camera calls are simulator sensor acquisitions, not model image attachments.
Policy CPU time covers worker compilation, imports and execution, excluding
simulator/rendering, host IK and RPC time. Simulated duration is not wall latency.
These measurements do not demonstrate a runtime efficiency advantage for Full.
The deterministic evaluation made **zero model calls and consumed zero model
tokens**; generation resources are reported separately in the
[five-round training report](GEOMETRY_FIVE_ROUND_TRAINING.md).

## Paired outcomes

- Both succeeded: 13 layouts.
- Base only: 2026112704 and 2026112712.
- Full only: 2026112702.
- Both failed: 2026112703, 2026112709, 2026112713 and 2026112718.

The exploratory exact paired McNemar p-value is 1.0. There are only three
discordant pairs and one learned chain per arm; this is neither evidence of
superiority nor a formal equivalence result. Full reached 4/5 training earlier,
but this fresh test does not show better final generalization.

On the two Base-only successes, Full's notes report unconfirmed beam placement
(2704) and unconfirmed cap placement (2712). On the Full-only success, Base
reports unconfirmed foundation placement (2702). Full also encountered an
object/tool association exception on 2718. These are observed rollout diagnostics,
not isolated causal explanations of the method difference.

## Protocol and validation

The new seeds were 2026112701 through 2026112720, fixed before execution and
checked against prior registrations. Both arms used the same layout seeds, color
maps, recipe, object count, camera configuration, sensor settings and physics.
The selected policy hashes were registered before any fresh result. The budget
was exactly 40 rollouts, evaluated with four parallel workers in approximately
73.7 seconds including final audit. No model sessions advanced, policies changed,
or test-based policy selection occurred.

All 40 rollouts were valid, with zero invalid slots. Final checks verified frozen
source and policy hashes, runtime settings, paired scene metadata, policy sandbox
isolation, unchanged training evidence and the completed training audit. Three
focused runner tests passed: inspected/duplicate split rejection, evidence-tamper
rejection and exactly one rollout per selected policy per registered seed.

The original 20-layout results (Base R0 6/20 and Full R2 9/20) used different
layouts and a different Base policy. They must not be compared directly to these
rates as an improvement estimate. Full R2 itself scored differently across the
two splits, illustrating sampling variation. This fresh evaluation followed the
exploratory five-round continuation; it is a new frozen-policy comparison, not a
preregistered five-round learning-curve experiment.

Evidence is under
`outputs/geometry_diagnostic_01/extension_05/fresh_evaluation_20/`:

- `registration.json`: seed list, budget, selected policies and provenance hashes.
- `summary.json`, `audit.json`: metrics, paired statistics and verified result hashes.
- `paired_seeds.json`: all paired outcomes by seed.
- `progress.json`, `evaluation.log`: completed execution status and log.
- `basic/heldout/`, `symbolic/heldout/`: individual results, sensor data and traces.

To re-audit without executing rollouts:

```bash
.venv/bin/python scripts/evaluate_geometry_extension.py \
  --root outputs/geometry_diagnostic_01 \
  --output outputs/geometry_diagnostic_01/extension_05/fresh_evaluation_20 \
  --phase audit
```
