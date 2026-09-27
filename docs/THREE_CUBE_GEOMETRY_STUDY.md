# Three-cube geometry study

This separately registered study compares Visual Base and Full geometry on a
vertical stack of three cubes. It uses the current opt-in point-geometry bundle,
while retaining its existing cube observation backend. It is not a novel-shape
perception test. Both arms share the same RGB-D actuator, a 0.40 generic grasp
fraction and the explicit libccd collision pipeline. The completed bridge study
and its learning curves remain unchanged.

## Current status: reduced evaluation completed

Training completed all 200 slots and 30 model requests. The user subsequently
requested lower evaluation cost. A separate amendment selected the first 20
reserved layouts in the original order before any reserved results, preserving
all five paired chains and every revision. It completed 800 evaluations in about
336 seconds with at most 24 workers. The original 100-layout protocol remains
unchanged as provenance and is superseded for dispatch; do not resume its remaining
3200 slots.

| Curve | R0 | R1 | R2 | R3 |
| --- | ---: | ---: | ---: | ---: |
| Base, raw revision | 0% | 49% | 70% | 76% |
| Full, raw revision | 0% | 62% | 72% | 69% |
| Base, training-retained | 0% | 48% | 70% | 74% |
| Full, training-retained | 0% | 62% | 72% | 72% |

Rates average five chains on 20 common layouts (100 outcomes per arm/revision,
with layout and chain dependence). Full leads at R1, but these results do not
show higher final success. Retention uses training evidence; it can reduce test
success and must not be replaced by test-set selection. Mean cumulative generation
tokens at R3 are 133621 for Base and 132580 for Full, including cached input.

Evidence and plots are in
[`compact_evaluation_20/summary.json`](../outputs/geometry_cube_stack_01/compact_evaluation_20/summary.json)
and [`learning_curves.pdf`](../outputs/geometry_cube_stack_01/compact_evaluation_20/learning_curves.pdf).
The amendment also corrected resource accounting for an unchanged cumulative
usage broadcast carrying stale previous-turn usage. All new increments are
verified against cumulative counters. Historical counters, policies, STOP receipts
and result files remain intact. The first compact launch exited during imports
because resolving the Python symlink bypassed the virtual environment; its log
and launch receipt are preserved alongside the successful virtual-environment
launch. No rollout was retried for that launch correction.

## Source disclosure audit and follow-up

All five Base initial prompts included the common executor source, including
color segmentation, deprojection, table fitting and top-surface estimation.
The Base runtime API withheld supplied object geometry, but the actor could use
this source to implement its own estimator. Interpret the results as a comparison
against a source-disclosed visual baseline. The [new contract-only study](THREE_CUBE_CONTRACT_ONLY_STUDY.md)
removes executor source from both arms while preserving the runtime and public API.
Historical prompts, policies and outcomes remain unchanged.

## Original design and budget

- Five independent persistent actor sessions per arm, paired by training layouts.
- Three generated rounds, R1-R3, plus the common observation-only R0.
- Five training layouts per pair; disjoint training sets between pairs.
- One hundred shared reserved layouts across all pairs and revisions.
- Thirty model requests and 200 training slots, including invalid submissions.
- Four thousand reserved slots: 1000 training-selected evaluations and 3000
  additional evaluations covering every frozen revision. Selected results are
  reused, so they do not add to the 4000-slot budget.

Training seeds are 2027090101-2027090125, in consecutive groups of five; reserved
seeds are 2027091001-2027091100. Registration rejects overlap with previous
registered layouts. Development preflights use separate 202710-series seeds and
make no model calls; their outcomes and private evaluator checks never enter an
actor context.

The model and reasoning setting remain `gpt-6-astra`, high. Each actor receives the
shared interface contract and its own training history, with initial images and
bounded failure-triggered additions. The 50000-character policy budget remains.
R0 provides images without a supplied solution. Goals expose a color-sorted support
graph, relative XY constraints and color/shape identities; they provide neither
construction order nor simulator object measurements.

All training must finish and pass source/context audits before any reserved trial.
The full iteration matrix is planned in `protocol.json` before the first model
request. Frozen policy hashes are materialized after training. Retention uses only
training outcomes: preserve every incumbent success and strictly improve; ties
retain the incumbent. Reserved results never select the policy or enter training.

The original support-graph evaluator is reused for the single column. Contact,
relative placement, orientation and velocity checks remain private; final stability
is evaluated for five simulated seconds. Side RGB-D stays at 320x240 with pixel
center sampling; ideal wrist depth remains enabled. Existing cubes are physical
free-joint objects, with no kinematic attachment.

## Parallelism and storage

Global limits are four model requests and 24 simulator rollouts, shared across all
replicas. New training and selected-test leases reduce to 16 below 12 GiB available
RAM, or eight below 6 GiB / above 90% GPU memory. Active leases drain normally.
The remaining curve phase runs a 48-slot action-policy pilot and retains at most
eight workers if the 6 GiB / 90% limits are exceeded. Kernel-backed resource leases
and bounded queues prevent multiplying concurrency by the replica count.

Every process uses one OpenBLAS/OMP thread. Perception and control remain on CPU;
GPU rendering is allowed. A 15 GiB disk reserve stops new dispatch. The study data
is stored at `/mnt/ssd/agentic_policy/geometry_cube_stack_01`, exposed through
`outputs/geometry_cube_stack_01`. No old output directories are overwritten.

Infrastructure errors stop dispatch and retain incomplete requests and trials.
There are no automatic retries of model calls or partially written rollouts.
Completed, audited work is reused on an explicit resume of the same registration.

## Launch and inspect

Run from `aspire_so100`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/run_three_cube_geometry_study.py \
  --output outputs/geometry_cube_stack_01 --phase prepare

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python -u scripts/run_three_cube_geometry_study.py \
  --output outputs/geometry_cube_stack_01 --phase all
```

`--phase all` automatically trains, freezes, evaluates selected policies, evaluates
all remaining revisions, audits results and writes curves. `--phase train` stops
after the global training freeze. `--phase audit` verifies a fully completed run.
The registered limits, task and sources are immutable after preparation.

- `protocol.json`: complete budget, seeds, resource limits and iteration plan.
- `campaign.log`, `lifecycle.json`, `progress.json`: pipeline and training status.
- `resource_status.json`, `hardware.jsonl`: effective concurrency and hardware samples.
- `replica_*/training.log`: per-pair actor and rollout progress.
- `summary.json`: the 1000-slot selected-policy stage, not the entire pipeline.
- `learning_curves/progress.json`: progress of the additional 3000 slots.
- `learning_curves/summary.json`, `learning_curves.{png,pdf,csv}`: raw and retained
  SR curves, paired outcomes and cumulative generation tokens.

Report all five paired chains and their variation, together with tokens, camera
calls and execution costs. The same layouts and policies are reused across curve
points; 4000 records are not 4000 independent method replications. Lower cost on
failed rollouts does not establish better efficiency at matched success.

## Recorded audit correction and resume

The first launch stopped after 150 training rollouts and 25 completed model
requests. The legacy training auditor required four scene objects even though
the three-cube manifest and runtime correctly used three. No reserved rollout
had started. All 150 completed rollouts and all 25 actor contexts passed the
corrected audit chain before resumption.

For this registered study, use `scripts/resume_three_cube_geometry_study.py`.
It changes one audit expression from a literal count of four to the registered
`manifest['object_count']`. The original source files, snapshots, manifests,
policies and measured results remain unchanged. The adapter compiles only the
corrected audit function, preserves all other checks, and applies the same
correction in coordinator and worker processes. It does not change sensing,
actions, prompts, policy retention, success criteria or experiment budgets.

The `audit_object_count_repair/` directory records the exact correction, original
and corrected auditor hashes, adapter snapshot, pre-resume audits, completed
artifact hashes and the original STOP receipt. A new infrastructure failure is
not automatically cleared or retried. Subsequent phases must use the repair
entry point so that the same correction and evidence checks remain active.

```bash
# Register the correction once, after checking all existing evidence.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/resume_three_cube_geometry_study.py \
  --output outputs/geometry_cube_stack_01 --phase register

# Reuse completed work, finish training, then evaluate the full revision matrix.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python -u scripts/resume_three_cube_geometry_study.py \
  --output outputs/geometry_cube_stack_01 --phase all
```

Use `--phase check` for read-only verification of the registered correction and
retained artifacts, or `--phase audit` after the entire campaign completes.
