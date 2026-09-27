# Three-cube geometry study

This separately registered study compares Visual Base and Full geometry on a
vertical stack of three cubes. It uses the current opt-in point-geometry bundle,
while retaining its existing cube observation backend. It is not a novel-shape
perception test. Both arms share the same RGB-D actuator, a 0.40 generic grasp
fraction and the explicit libccd collision pipeline. The completed bridge study
and its learning curves remain unchanged.

## Design and budget

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
