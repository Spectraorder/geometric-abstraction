# Per-iteration test curves and expanded rollout concurrency

The user requested reserved-test evaluation of every saved policy iteration to
measure whether Full reaches a given success rate earlier. The completed study
`geometry_parallel_01` already froze R0-R3 for five paired replicas. This extension
keeps every frozen policy, training session and original result unchanged.

All four revisions use the same 100 reserved layouts. The curve contains 4000
slots: 1000 selected-policy results are reused with audited provenance; 3000
additional rollouts are executed. No new model calls, training, test-based policy
selection or manual policy repairs are introduced. This extension was requested
after selected-policy results were inspected, so its analysis is exploratory.
For future studies, register this full-curve scope before running the experiment;
retain all revisions and evaluate them only after training is frozen.

## Curves and efficiency

The final report exports PNG, PDF and CSV versions of three plots:

1. Raw test success rate of each newly proposed policy at R0, R1, R2 and R3.
2. Test success rate of the training-retained policy at each proposal budget.
3. The retained curve against cumulative model tokens, including rejected turns.

Thin plot lines show individual replicas; thick lines show means across five
replicas. The token plot connects mean SR at each iteration to mean cumulative
cost; it is not an interpolated comparison of policies at exactly equal cost.
The primary success threshold is 80%; 60% and 70% are secondary. Each replica's
first crossing is recorded separately for raw and retained curves. Unreached
thresholds remain null. First crossings can be followed by regression; they are
not evidence of sustained performance. Training retention is never recomputed
using test scores, and the originally selected policy remains unchanged.

## Hardware and storage

Local inspection found an Intel Core i9-14900K with 24 physical cores and 32
logical CPUs, approximately 48 GB RAM (46 GiB reported by Linux), and an NVIDIA
GeForce RTX 3090 with 24 GiB VRAM. At registration approximately 37 GiB RAM was
available. Policy/perception runs on CPU; the GPU serves simulation rendering.
The model concurrency limit is unchanged because this extension makes no model
calls. Local core count is not a reason to increase remote model requests.

The system filesystem had only about 75 GB free. New rollouts therefore go to
`/mnt/ssd/agentic_policy/geometry_parallel_01_curves`, which had about 1 TB free.
Each replica has a `curve_rollouts` symlink to its new data directory. Registration,
progress, telemetry and charts remain under
`outputs/geometry_parallel_01/learning_curves`. Existing data is neither moved nor
deleted. The new data filesystem retains a 15 GiB free-space floor.

## Parallel pilot and monitoring

The extension uses one bounded queue and a global 16-slot rollout limit. The
first 32 pending action-taking rollouts constitute a hardware pilot; the cheap
observation-only R0 baseline is scheduled afterward. RAM availability and GPU
memory/utilization are sampled approximately every two seconds. If any pilot
sample has less than 6 GiB available RAM or over 90% VRAM use, subsequent dispatch
falls back to eight workers. The pilot's outcomes remain included; no performance
failures are discarded or retried. Infrastructure errors stop further dispatch
and preserve incomplete evidence for inspection.

The final report audits the actual concurrency, frozen candidate hashes, original
result hashes, simulator settings, goal receipts, sandbox configuration, and
identical initial images across layouts, replicas and revisions. Old selected
rollouts used four workers and new ones may use sixteen; their runtime metrics
are not a controlled measurement of speedup from concurrency.

```bash
.venv/bin/python scripts/evaluate_geometry_learning_curves.py \
  --root outputs/geometry_parallel_01 \
  --output outputs/geometry_parallel_01/learning_curves \
  --data-root /mnt/ssd/agentic_policy/geometry_parallel_01_curves \
  --workers 16 --phase prepare

.venv/bin/python -u scripts/evaluate_geometry_learning_curves.py \
  --root outputs/geometry_parallel_01 \
  --output outputs/geometry_parallel_01/learning_curves --phase run
```

Monitor `progress.json`, `campaign.log`, `hardware.jsonl` and `pilot.json` in the
extension directory. The original study's `progress.json` continues to report
its already completed selected-policy experiment. On completion, `summary.json`,
`learning_curves.png`, `learning_curves.pdf` and `learning_curves.csv` are written.
Use `--phase audit` for read-only result validation and chart regeneration.


## Retained-policy curve export

Recompute the focused curve from the frozen training decisions and audited test
results with no new rollouts or model calls:

```bash
.venv/bin/python scripts/plot_retained_geometry_curve.py \
  --root outputs/geometry_parallel_01 \
  --output outputs/geometry_parallel_01/learning_curves/retained_best
```

A candidate replaces the incumbent only when its training success set strictly
contains the incumbent's; ties retain the incumbent. Selection is per training
chain and never uses reserved-test scores. Each curve point uses the corresponding
selected revision's test results; cumulative tokens include rejected proposals.
This is not the cumulative maximum of test scores. The script verifies all 4000
result hashes and recomputes the selection independently of the summary's retained
values, then checks that the two calculations agree.

| Completed rounds | Base retained test SR | Full retained test SR | Full minus Base |
|---|---:|---:|---:|
| 0 | 0.0% | 0.0% | 0.0 pp |
| 1 | 32.0% | 62.4% | +30.4 pp |
| 2 | 53.2% | 62.6% | +9.4 pp |
| 3 | 54.6% | 62.6% | +8.0 pp |

PNG, PDF, mean/per-chain CSV and the verified JSON summary are exported under
`learning_curves/retained_best/`. The original raw and combined curve artifacts
remain unchanged. These are means over five chains on the same 100 test layouts;
test monotonicity is not guaranteed by training-only retention.
