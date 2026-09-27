# Shared-interface selected-policy test results

The completed five-round study `outputs/geometry_interface_contract_01` selected
R4 for both arms using training outcomes only. Its separate selected-policy test
registration evaluated those frozen policies on the original 20 shared reserved
layouts, seeds 2027012701–2027012720. All 40 rollouts completed with four workers;
the evaluation and final audit took 84.08 seconds. No model calls or model tokens
were consumed by evaluation.

## Success and pairing

| Metric | Base R4 | Full geometry R4 |
| --- | ---: | ---: |
| Training successes | 3/5 | 4/5 |
| Reserved successes | 12/20 | 8/20 |
| Reserved success rate | 60% | 40% |
| Wilson 95% interval | 38.7–78.1% | 21.9–61.3% |
| Mean camera calls per rollout | 84.30 | 69.95 |
| Mean policy compute CPU seconds | 1.320 | 1.361 |
| Median policy simulation seconds | 124.55 | 102.60 |

The paired outcomes are eight shared successes, four Base-only successes, zero
Full-only successes and eight shared failures. Full minus Base is -20 percentage
points; the unadjusted exact McNemar p-value is 0.125. The four Base-only success
seeds are 2027012703, 2027012705, 2027012708 and 2027012717.

Full's higher training success did not translate into higher reserved success in
this run. Its lower average camera count does not establish greater efficiency
at matched task success: failed or prematurely terminated policies can issue
fewer calls. Policy CPU time excludes simulator rendering, host IK and RPC;
simulation time is not wall-clock latency. The comparison has one learned chain
per arm and only 20 paired layouts, so it does not establish a general advantage
for either method. Earlier studies used different learned policies and reserved
splits; their percentages are not paired measurements of this interface change.

## Failure accounting and provenance

All rollouts used valid frozen candidates; there were no invalid source slots or
infrastructure failures in this evaluation. Five of Base's eight task failures
reported policy exceptions: three unavailable/current-view perception failures
and two object/tool-neighborhood association failures. The other three Base
failures and all twelve Full failures had no top-level policy error. Absence of
a top-level exception does not establish correct behavior or successful execution
of every attempted action; detailed failure diagnosis requires trace review.

Audit verified candidate and observer hashes, runtime settings, sandbox isolation,
public goal receipts, identical paired initial scenes and images, and preserved
training history and persistent actor state. The original 30000-character study
and its invalid R5 records remain unchanged. The new 50000-character configuration
has not been trained or evaluated by this test run. These selected-policy results
do not complete the original full-revision reserved-evaluation protocol.

Artifacts are under `outputs/geometry_interface_contract_01/selected_evaluation_20/`:

- `registration.json`: fixed policies, seeds, rollout budget and provenance hashes.
- `summary.json` and `audit.json`: paired outcomes, metrics and result hashes.
- `progress.json` and `evaluation.log`: completion state and execution log.
- `basic/heldout/` and `symbolic/heldout/`: per-layout results and recorded traces.

Reproduce the read-only audit with:

```bash
.venv/bin/python scripts/evaluate_selected_interface_policies.py \
  --root outputs/geometry_interface_contract_01 \
  --output outputs/geometry_interface_contract_01/selected_evaluation_20 \
  --phase audit
```
