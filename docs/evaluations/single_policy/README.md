# Single-policy results and data

[Combined experiment index](../README.md) · [Online results and data](../online/README.md)

This is the consolidated export for the completed contract-only Bridge, contract-only
three-cube Stacking, and paired Ring studies. Start with the
[results report](../../SINGLE_POLICY_RESULTS.md),
[experimental setup](../../SINGLE_POLICY_EXPERIMENTS.md), or
[paper fragment](../../paper/single_policy_experiments.tex).

## Contents

| File | Contents |
| --- | --- |
| [summary.json](summary.json) | Mean curves, resource totals, study attribution, and validation scope |
| [test_outcomes.csv](test_outcomes.csv) | All 12,000 candidate/layout test slots, including invalid policies and failures |
| [training_outcomes.csv](training_outcomes.csv) | All 600 candidate/layout training slots |
| [chain_curves.csv](chain_curves.csv) | 120 chain/revision rows for the main studies plus 40 rows for Stacking's additional 80-layout subset |
| [mean_curves.csv](mean_curves.csv) | 24 mean-curve rows plus 8 additional-subset rows; sample SD across five chains |
| [model_requests.csv](model_requests.csv) | 90 completed requests and 2 interrupted requests with unknown usage |
| [resources.csv](resources.csv) | Six task/arm totals, summed over five chains |
| [provenance.json](provenance.json) | Source and export SHA-256 hashes, including all 12,000 original test result files |
| [learning_curves.pdf](learning_curves.pdf) / [PNG](learning_curves.png) | Raw and training-retained success against revision |
| [learning_curves_tokens.pdf](learning_curves_tokens.pdf) / [PNG](learning_curves_tokens.png) | The same outcomes against cumulative completed-turn tokens |

## Data conventions

- `basic` and `symbolic` in source paths map to `Base` and `Full` in these exports.
- `replica` identifies a paired training chain (1–5); `revision` is 0–3.
- Success rates are fractions in CSV/JSON and percentages in plots and tables.
- R0 is an observation-only scaffold. R1 is the first generated manipulation program.
- `retained_revision` is selected using strict inclusion of training success sets.
  Token curves include all completed proposals, including rejected candidates.
- `all_100` and `additional_80` are overlapping Stacking views. Filter to `all_100`
  for the main result; never count the subset as an additional experiment.
- Empty CSV cells and JSON nulls mean unavailable/not recorded, not zero. In
  particular, interrupted-request tokens are unknown. Original unchanged zero
  counter broadcasts are not measurements of zero consumption.
- Token columns count completed-turn deltas. Cached input is included in input;
  reasoning output is included in output. Total tokens are input plus output.
- Resource totals sum five chains; divide by five for mean per-chain acquisition
  cost. Request wall time includes waiting and is not model compute time or
  parallel campaign wall time. Interrupted wall time is reported separately.
- Test `wall_seconds`, `policy_sim_seconds`, and `policy_compute_cpu_seconds`
  have different meanings and recording coverage. Do not combine them. Missing
  execution/status fields for Bridge and Stacking are preserved as missing;
  their schema differs from Ring's explicit `rollout_executed` flag.
- Training rows for Bridge and Stacking come from frozen `training_summary.json`
  records; Ring rows come from individual trial results. A source SHA therefore
  sometimes identifies a multi-row training summary. It is not a per-row hash.
- The five chains share the same 100 test layouts per task. The 500 evaluations
  per arm/version are not 500 independent layouts. SD is descriptive chain
  variation, not a confidence interval or a standard error over 500 trials.

The bundle contains outcomes and resource data needed for the reported curves,
not a duplicate of all RGB-D archives, generated policy source, or private
physics traces. Original artifacts remain at the repository-relative `source`
paths; their contents have not been modified. Files here are derived exports.

## Reproduction

From the `aspire_so100` repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/export_single_policy_results.py
```

The script reads existing artifacts only. It verifies original test hashes,
complete paired evaluation matrices, train/test separation, strict training-only
retention, per-turn token accounting, and agreement with the frozen summaries.
It exports data and figures without model calls, policy execution, or test-based
selection. Hash agreement confirms provenance, not a new physical evaluation.
The results report and paper prose are maintained separately from the exporter.
