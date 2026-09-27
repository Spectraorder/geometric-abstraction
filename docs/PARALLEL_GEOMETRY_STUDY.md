# Parallel replicated geometry study

The user authorized five independent training chains per arm, followed by tests
on 100 shared new layouts. The study uses the neutral starter, shared interface
contract, 50000-character source budget and three generated revisions per arm.
Each paired replica uses five shared training layouts; different replicas use
disjoint training sets. Both training-data variation and model-generation variation
are therefore included in the between-replica comparison.

The registered budget is 30 model calls, 200 training slots including R0, and
1000 selected-policy test rollouts. Invalid training submissions remain counted.
Only the training-selected policy from each arm/replica enters reserved testing.
All five paired replicas must finish and pass training audit before any test runs.
No previous policies, diagnoses or inspected test outcomes enter actor contexts.

## Global scheduling

`scripts/run_parallel_geometry_study.py` launches one isolated process per paired
replica. Each arm retains its own persistent model session, and successive turns
of that arm remain sequential. Model requests share four kernel-backed file-lock
slots across all replicas. Simulator rollouts share a separate four-slot pool,
including parallel training layouts. Neither limit is multiplied by the number
of replicas. These limits apply to this study; do not launch another independent
campaign concurrently without accounting for its resource use.

The test phase uses one bounded queue interleaving replicas, arms and layouts.
It submits at most four jobs at once rather than enqueuing all 1000 in advance.
Evaluation makes no model calls. Online policy/perception remains on CPU; GPU
rendering is allowed. Existing simulator and actor timeouts remain unchanged.
A 15 GiB free-space floor stops new rollouts before disk exhaustion.

Every acquired resource slot produces a receipt with process identity, job label,
queue time, start time and end time. The final scheduler audit checks observed
concurrency and the number of model attempts. Status files use atomic replacement;
`progress.json` has a five-second training heartbeat and per-rollout test updates.
Study and replica locks prevent duplicate coordinators writing the same outputs.

## Failures and resumption

An infrastructure failure writes `STOP.json`, prevents new resource acquisition
and drains already active work. Policy-level failures remain ordinary recorded
outcomes. Completed generations and rollout results are reused and validated on
resume. An incomplete model call or existing partial rollout is retained and
requires inspection; there is no automatic retry that could hide cost or change
a persistent session's history. A killed process releases its OS lock, but its
unfinished receipt prevents a clean final audit until explicitly resolved.
Do not simply delete `STOP.json` or incomplete artifacts to force a restart.
SIGINT/SIGTERM to the supervisor requests a dispatch stop and drains active jobs.

## Registration and launch

The current study is `outputs/geometry_parallel_01`. Training seeds are
2027020101–2027020125, partitioned into consecutive groups of five. The common
reserved split is 2027021001–2027021100. Preparation checks both sets against
previously registered layouts and snapshots the orchestration sources.

```bash
.venv/bin/python scripts/run_parallel_geometry_study.py \
  --output outputs/geometry_parallel_01 --phase prepare \
  --replicas 5 --rounds 3 --layouts 100 --model-workers 4 --rollout-workers 4

.venv/bin/python -u scripts/run_parallel_geometry_study.py \
  --output outputs/geometry_parallel_01 --phase all
```

Use `--phase train` to stop after training and the global freeze; `--phase evaluate`
requires that freeze already exists. `--phase audit` verifies the completed study
without model or simulator calls. All phases use the immutable registered budget.

Key artifacts:

- `protocol.json`: scope, seeds, source hashes and global resource limits.
- `campaign.log`, `lifecycle.json`, `progress.json`: supervisor output and status.
- `replica_*/training.log` and `replica_*/progress.json`: individual training.
- `scheduler/{model,rollout}/events/`: resource lease receipts.
- `all_candidates_frozen.json`: study-wide gate and preserved training hashes.
- `replica_*/selected_evaluation/`: per-replica test registrations and results.
- `summary.json`: paired replica outcomes, runtime metrics and verified token accounting.

## Interpretation

Report all five paired success-rate differences, their mean and range, and each
arm's token/runtime costs. The same test layouts are used across replicas, so the
1000 rollout records must not be treated as 1000 independent policy replications.
Five chain pairs remain a small method-level sample. Test success, resource use and
training variability must be reported together; fewer camera calls on failed
rollouts do not establish greater efficiency at matched success.

No historical manifest, selected policy, source snapshot or measured result is
changed by this study. The older reserved-budget stop remains in force.
