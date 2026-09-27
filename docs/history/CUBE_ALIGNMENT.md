> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Visual alignment of short cube columns

The first alignment prototype is implemented, but **automatic regrasp refinement
is not promoted**. On ten reserved default-physics scenes both the baseline and
fine-servo candidate stack successfully, yet the latter does not improve mean
edge alignment. A separate contact-solver experiment suppresses long-term
simulation drift; it is explicitly a changed-physics result.

| Default-physics reserved result | Frozen baseline | Fine visual servo |
| --- | ---: | ---: |
| Successful three-cube stacks | 10/10 | 10/10 |
| Mean per-tower maximum corner mismatch | 1.12 mm | 1.16 mm |
| Worst corner mismatch | 2.06 mm | 1.85 mm |
| All adjacent pairs within 1 mm | 5/10 | 5/10 |
| Additional 60-second strict stability pass | 1/10 | 0/10 |

All ten paired trials remain in these denominators. The tighter servo improves
five cases and worsens five; there is no consistent policy-only gain. The
original default policy and scene settings remain unchanged.

[Complete results and source hashes](../evaluations/cube_alignment.json) ·
[All three actual contact-profile comparisons](../media/cube_alignment_contact_comparison.png) ·
[Actual precision-policy rollout, seed 2802](../media/cube_alignment_contact_2802.mp4) ·
[Failed refinement, seed 2803](../media/cube_alignment_refine_failure_2803.mp4)

## Contact-solver finding and actual re-test

MuJoCo documents that its regularized soft-contact model can exhibit slow slip,
and describes NoSlip friction post-processing as one suppression option. This
changes the contact computation and can have multi-contact stability tradeoffs;
it is not a guarantee of real-world fidelity. See the
[official solver and slip discussion](https://mujoco.readthedocs.io/en/latest/modeling.html#solver-settings).

An offline audit restored identical final states from development seed 2603 and
reserved seed 2701 with two distractors. With zero NoSlip iterations both towers
failed a further sixty seconds and eventually collapsed. With ten iterations
both passed; the largest final body displacement was 0.0037 mm and 0.0361 mm
respectively. Fifty iterations gave the same reported outcome. These are passive
solver-sensitivity diagnostics, not new policy successes.

Actual policies were then rerun from reset on three fresh development seeds,
2801–2803, using **NoSlip=10 for the entire rollout**. Both policies succeeded in
all three cases, and all six passed an additional sixty seconds. Dimensions,
friction coefficients, task goals and success thresholds were unchanged; the
contact solver was explicitly changed and recorded.

| Seed | Baseline max corner error | Fine-servo max corner error | Extra 60 s |
| --- | ---: | ---: | --- |
| 2801 | 0.65 mm | 1.04 mm | Both pass |
| 2802 | 2.11 mm | 1.79 mm | Both pass |
| 2803 | 1.27 mm | 1.35 mm | Both pass |

This establishes a small reproducible set of stable, roughly 1–2 mm alignments,
not a sub-millimetre or broadly generalized alignment skill. Re-enabling the
experimental regrasp on these same three scenes succeeded in two and failed in
one; the failure overturned the tower. It remains disabled by default. The
successful demonstration video uses the largest-error fine-servo case among
these three, rather than only the best outcome. Videos stop at the original
trial endpoint; the sixty-second continuation is recorded numerically.

NoSlip is opt-in via `run_vision_trial.py --noslip-iterations 10`, currently scoped
to inventory towers. Results record `physics.noslip_iterations`; replay restores
that setting. The standard `evaluate_columns.py` benchmark rejects cached
nondefault-physics trials. Legacy recordings restore the original zero-iteration
setting. This protects earlier evidence from silent changes.

This experiment separates successful stacking from edge registration. A tower
can pass the existing support/stability check while adjacent cube edges differ
by several millimetres. The alignment goal is an approximately 1 mm plan-view
corner mismatch with matching square-edge heading, rather than only placing a
center inside the support polygon.

## Candidate behavior

`policies/stack_cubes_aligned_cpu.py` is an optional CPU-only controller for two
or three cubes. It preserves the frozen column policy's source, bundle and
whole-column balance compensation. Its final visual XY servo target is 0.3 mm
instead of 1 mm, with up to 24 rather than 16 final corrections and a 2 mm
release XY gate. The 3 mm target release clearance and existing contact/support
checks are retained. The servo target is a controller tolerance, not a promise
of equivalent final physical accuracy.

After releasing each upper cube, the arm withdraws and the controller checks
fresh RGB-D side surfaces against the immediately supporting cube. Geometry
reports XY displacement, square-edge heading modulo 90 degrees and matched
plan-view corner mismatch. A covered top face can use earlier visually learned
size/pose as an explicit prior. A single visible vertical face requires narrow
finite-surface bounds on the missing horizontal translation. Unusable geometry
causes the alignment check to skip, rather than inventing a zero error.

## Experimental re-placement

An additional `--refine` build option enables one bounded re-placement of the
newest, unloaded top cube. This is **disabled by default** because development
runs disturbed the supporting cube and sometimes increased final misalignment.
The experimental step reuses the payload-to-gripper transform measured at the
prior release, lifts approximately 14 mm rather than repeating a 60 mm transport,
and bounds the correction bias to 2 mm. It rejects a residual outside a 5 mm / 8
degree envelope and stops if the supporting cube moves more than 1.5 mm. It
never deliberately pushes or regrips a loaded lower cube. These bounds do not
establish reliable correction: the recorded regressions remain evidence against
promoting this step.

Other development alternatives included tighter release clearance, replacing
the balance objective with direct centering, extra settling time and held-payload
attitude commands. These did not consistently improve final alignment. In
particular, a five-axis arm's soft Cartesian IK does not guarantee the requested
attitude, and accurate alignment immediately after release can degrade when the
next level loads the support. The default candidate therefore retains the
original balance objective and transport attitudes.

## Evaluation boundary

The online policy uses RGB-D, wrist RGB, calibrated cameras, robot proprioception,
robot-only IK and public cube identities. It receives no simulator object poses,
dimensions, contacts, masses or evaluator feedback. Pointing subagents and hosted
model calls are not used. Simulation/rendering may use GPU; policy computation
uses NumPy/SciPy on CPU.

The offline evaluator restores the final saved integration state and computes
alignment from object geometry. For each adjacent pair, it matches four
plan-view corners modulo square symmetry and reports the largest Euclidean
corner discrepancy. Per-tower error is the maximum across the two adjacent
pairs. It also reports center displacement and heading mismatch. These private
measurements never feed the controller. A separate sixty-second continuation
holds the recorded final robot controls and applies the existing 3 mm drift /
contact / uprightness criteria. No policy executes during this continuation.

Twenty-seven adaptive development runs use seeds 2601–2603: three baseline runs
and eight successive candidate versions. The first candidate failed two stacks;
later rejected candidates can pass stacking while worsening alignment. These
runs must not be pooled with reserved results or treated as independent samples.

The final candidate and baseline were frozen before a new paired matrix:
seeds 2701–2705, each with zero and two distractors, using deterministic color
permutations. Ten scenes are run for each policy. The configurations share seeds,
so this small matrix is correlated. All initial object yaws remain zero, although
manipulation introduces small attitude errors. This is not evidence for arbitrary
starting yaw, arbitrary shapes, touching clutter, new cameras or real hardware.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/build_inventory_policy.py \
  --controller cube_alignment_policy --output policies/stack_cubes_aligned_cpu.py

MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/evaluate_columns.py --candidate policies/stack_cubes_aligned_cpu.py \
  --output outputs/cube_alignment_reproduction --recipes cube_triple \
  --permute-colors --distractors 0 2 --seeds 2701 2702 2703 2704 2705 \
  --jobs 2 --split development

MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/score_cube_alignment.py --trials outputs/cube_alignment_reproduction \
  --output outputs/cube_alignment_reproduction/alignment.json --durability 60
```

The reserved manifest records candidate hashes; reproducing already inspected
seeds is a reproduction, not a new held-out test. Rebuild only the optional
alignment bundle. Do not overwrite the frozen `stack_column_cpu.py` baseline.

## Contact-profile reproduction and validation

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/run_cube_alignment_contact_suite.py \
  --output outputs/cube_alignment_contact_reproduction --seeds 2801 2802 2803 \
  --noslip-iterations 10 --policies baseline precision

MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/score_cube_alignment.py --trials outputs/cube_alignment_contact_reproduction \
  --output outputs/cube_alignment_contact_reproduction/alignment.json --durability 60
```

The complete work contains 56 actual policy trials: 27 adaptive default-physics
development trials, 20 reserved default-physics trials, six changed-contact-profile
baseline/precision trials, and three rejected refinement trials. Passive
continuations are separate. The regression suite ran 166 tests: 165 passed and
one platform-specific test was skipped. Tests cover square symmetry, translation,
yaw, visual-confidence gating, baseline bundle preservation, recorded contact
settings, replay and prevention of benchmark mixing. No commit or push was made.
