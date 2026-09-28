# Paper experiments: results, data, and figures

This is the common entry point for the three **single-policy acquisition** tasks
and three **continuous online** tasks. All six primary studies are complete.
Their experimental units and resource budgets differ, so they are reported
separately rather than pooled into a single success rate.

## Single-policy acquisition

[Results report](../SINGLE_POLICY_RESULTS.md) ·
[Setup and protocols](../SINGLE_POLICY_EXPERIMENTS.md) ·
[CSV/JSON and figure index](single_policy/README.md) ·
[Paper LaTeX](../paper/single_policy_experiments.tex)

Five paired training chains per task; R0–R3; five training layouts per pair;
100 shared held-out test layouts. Policies are retained using training success
sets only. Each study has 4,000 test slots; the three studies total 12,000 test
slots, 600 training slots, and 90 completed model generations, plus two interrupted
Ring requests with unknown tokens. Bridge uses the new contract-only study.

| Task | Base final retained SR | Full final retained SR | Difference |
| --- | ---: | ---: | ---: |
| Bridge | 56.0% | 65.4% | +9.4 pp |
| Three-cube stacking | 70.8% | 81.4% | +10.6 pp |
| Ring over peg | 0.0% | 38.0% | +38.0 pp |

Figures: [raw and retained learning curves](single_policy/learning_curves.pdf),
[cumulative-token curves](single_policy/learning_curves_tokens.pdf).

## Continuous online execution

[Three-task report](../ONLINE_EXPERIMENTS.md) ·
[CSV/JSON and figure index](online/README.md) ·
[Paper LaTeX](../paper/online_experiments.tex)

Five fresh paired layouts per task; one episode per arm/layout; 200 seconds per
episode with no reset after failure. The three separately registered studies
total 30 episodes, 262 completed model requests, and nine interruptions with
unknown token usage. Shared prior skills and action interfaces are matched within
each task. All final source/resource and interface audits passed.

| Task | Success, Base / Full | Mean capped time, Base / Full | Mean recorded tokens per episode, Base / Full |
| --- | ---: | ---: | ---: |
| Bridge-tower construction | 2/5 / 4/5 | 195.91 / 174.13 s | 101,396 / 107,811 |
| Ring over peg | 5/5 / 5/5 | 119.79 / 74.25 s | 233,831 / 93,023 |
| Tower relocation | 2/5 / 3/5 | 190.61 / 164.32 s | 181,377 / 151,163 |

Timeouts count as 200 seconds in the capped-time means. Token values are rounded
means from completed receipts and exclude unknown interrupted usage. Full's
recorded token use increases on Tower; improvements are task-dependent. Five
pairs are exploratory, not a precise population success-rate estimate.

Figures: [success versus elapsed time](online/success_vs_time.pdf),
[every paired completion time](online/paired_completion_times.pdf).

## Using the materials

The two subdirectories contain outcomes, resource tables, publication PDF/PNG
figures, data dictionaries, and source hashes. The reports provide study
attribution, protocol differences, all failures, and interpretation limits. The
LaTeX fragments contain editable main-text prose, result tables, and captions.
Figure paths in those fragments assume compilation from `docs/paper/`; adjust
paths when transferring to another paper project. A LaTeX compiler is not
installed in the export environment, so only structural checks were run here.

Reproduce data and figures from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/export_single_policy_results.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/export_online_results.py
```

These commands read frozen evidence and run no simulations or model requests.
Original RGB-D archives and private traces remain in the study outputs; this
bundle consolidates results and analysis data, not every raw sensor file.
