> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Shared-observer geometric abstraction pilot

Status: completed, 200 valid rollouts and six audited actor contexts. See
[the full results](SHARED_OBSERVER_LEARNING_RESULTS.md); no iteration advantage is established.


Registered root: `outputs/shared_observer_learning_pilot_01`.
The user confirmed this as the sole active comparison: **base versus full geometry
abstraction**, with a lightweight system. No separate numerical-only arm is included.
The older numeric/symbolic evaluation has been stopped and preserved.
This is a new exploratory pilot, not a correction or extra replication of the
running numeric/symbolic study. It measures supplied rigid metric/symbolic state
against a minimal surface interface over identical sensing and execution.

## Registered contrast

`basic` is retained only as an internal harness key; its label is **shared surface**.
It uses the mixed rigid observer, not the historical basic visible-top observer.
`symbolic` supplies the full rigid metric and symbolic interface. Both libraries
are byte-identical except for `SURFACE_ONLY = True/False`. Both instantiate the same
`SymbolicSession`; the baseline projects its public outputs through
`SurfaceProjectionSession`. Actor access to trusted backend fields remains forbidden.

Both arms receive RGB-D, calibration, public task identities/relative goals and
robot-only measurements. Public initial-scene assumptions are bound separately
from desired orientation. Complete task nodes and shape/color-only descriptor lists
produce the same canonical observation setup. The initial thin beam is publicly
declared flat along world X; no simulator object pose or dimension is used.

The shared executor uses the same full internal fits in both arms. Projection also
applies to observations returned by actions and state packets. The treatment is
the supplied richer geometric state package, including history/targets and symbolic
operators. It does not isolate an individual operator and is not geometry versus
no geometry. Neither arm uses the checkpoint/prediction prototype or temporal fusion.

## Budget and outcomes

- Training seeds: 910001–910005.
- Reserved seeds: 920001–920020.
- One isolated gpt-6-astra generation chain per arm, high reasoning effort.
- Identical starter R0 and three generated revisions: six model calls total.
- Forty training slots and 160 reserved slots: 200 total.
- Every revision freezes before reserved execution; highest training SR selects
  the final policy, with earliest revision breaking ties.
- Primary outcomes: complete reserved learning curves, first reserved revision
  reaching 80%, mean reserved SR across R0–R3, and training-selected reserved SR.
- Registered secondary outcome: at each stopping budget, retain the best available
  training candidate with earliest ties and report its reserved SR. Never select
  by reserved outcomes or reinterpret this as evolution from a different parent.

Before registration, all 7,281 existing result JSON seed fields under `outputs`
were checked for overlap; none matched. The root preserves the seed-check receipt.
Budgets count invalid submissions and failures. There are no free manual rewrites.
Rollout duration is not optimized. Tokens are recorded, not equalized. Revisions
share layouts; do not pool them as independent Bernoulli samples. A single model
chain cannot establish repeatability, even if the paired seed-level result is strong.

## Gates and artifacts

The runner reuses the established orchestration with scoped process-local hooks;
it does not modify the original runner. Source snapshots include the new driver,
initial-context adapter and surface projection. Its initial `bootstrap` phase runs
the ten R0 training slots, compares saved physical arrays and projected state, and
requires parity before any actor generation. Those same slots are reused in training.
A missing common observation is not excluded or replaced with an easier seed.

The auditor checks exact library equivalence except for the projection flag, saved
starter evidence, inherited rollout/source/actor-isolation gates, and reconstructed
actor prompts plus attached-image hashes. Feedback contains only the actor's own
training outcomes and observations. No older solution, diagnostic report, opposite
arm feedback or reserved result is supplied.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/run_shared_observer_experiment.py \
  --output outputs/shared_observer_learning_pilot_01 --resume --phase train

OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/summarize_shared_observer_experiment.py \
  --root outputs/shared_observer_learning_pilot_01 \
  --output docs/evaluations/shared_observer_learning_pilot_01.json --allow-partial
```

Resume only after checking that the preceding coordinator is terminal. An observation
timeout does not establish termination. The initial registration runs bootstrap only;
no model or reserved execution is implied by registration. Use `--phase all` to
continue training and then evaluate every frozen revision, or `--phase evaluate`
after completed training. Exact context reconstruction requires the registered
source version and deliberately fails on later source drift.

The [development parity check](SHARED_SURFACE_ABLATION.md) passed on two previously
inspected layouts. That evidence validates the interface design, not the learning
claim. The running older replication and all earlier negative results remain intact.

## Bootstrap checkpoint

All ten R0 training rollouts completed with zero invalid slots. All five pairs
passed bit-identical physical trajectory and projected-state comparisons; both
arms succeeded on 2/5 layouts. The [initial audit](../evaluations/shared_observer_learning_pilot_01.json)
passed before launching the six registered actor revision calls with `--phase train`.
This process will stop after training and candidate freezing; reserved evaluation
is a separate continuation. The bootstrap is parity evidence, not learning benefit.

## Completed training checkpoint

Training completed with 40 valid rollouts, zero invalid slots and six model
generations. Exact reconstruction of all six actor contexts passed. The complete
training curves (successes out of five, R0 through R3) are:

| Arm | R0 | R1 | R2 | R3 | Training-selected revision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 2 | 5 | 5 | 5 | R1 |
| Full geometry | 2 | 4 | 3 | 0 | R1 |

All eight candidate versions passed the source/hash freeze checks before any
reserved result existed. Final-round regressions are retained. The evaluation
continuation was launched with `--phase evaluate` to run all 160 registered reserved
slots, without further model calls or policy changes. These training scores do
not establish held-out generalization or an advantage for the full abstraction.

After completion, generate the audited report and clearly labeled figure with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/summarize_shared_observer_experiment.py \
  --root outputs/shared_observer_learning_pilot_01 \
  --output docs/evaluations/shared_observer_learning_pilot_01.json \
  --figure docs/media/shared_observer_learning_pilot_01.png
```

The report also evaluates the registered training-retention rule at each stopping
budget. That secondary curve does not select by reserved outcomes and does not
replace the raw fixed-revision curves. A final figure is refused until the full
campaign passes its audit.

## Training-only R3 regression note

The full-geometry R3 candidate failed all five training layouts without issuing
a move or gripper command. Every trace records the same planner exit:
`No clear workspace with an orientation-preserving IK corridor`. The candidate's
own hypothesis added orientation-preserving path screening after an earlier beam
orientation failure. There was no worker exception or invalid submission in these
five slots. This localizes the regression to planner rejection before actuation;
it does not establish that the underlying scenes were infeasible or that richer
geometry caused the regression. The R1 training-selected policy remains unchanged.
Source evidence is preserved under
`outputs/shared_observer_learning_pilot_01/symbolic/round_003/response.json` and
its five training API logs. This inspection used no reserved outcome or object truth.

## Training context footprint

The paired R0 initial checkpoint for training seed 910001 serializes to 1,957
characters for base and 15,888 for full geometry. In the full packet, the `facts`
value alone occupies 10,686 characters, approximately 67% of the packet. These
are automatically enumerated pairwise relations, already computable through the
public relation query. Full geometry prompts for R1/R2/R3 have 88,782 / 106,203 /
112,201 characters, versus 30,436 / 44,860 / 48,591 for base; both attach eight
images each time. Token usage and hashes are preserved in
`outputs/shared_observer_learning_pilot_01/training_context_footprint.json`.

This supports a specific lightweight change after the freeze: keep object geometry
in the default packet and query derived relations only when needed. It is a size
diagnostic, not measured reasoning speed or a causal explanation of the SR curves.
No packet, model prompt or policy was changed during this evaluation.
