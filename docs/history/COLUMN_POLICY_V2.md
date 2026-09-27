> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# CPU column policy: visual geometry and generalization

The frozen CPU policy passed **79/100 new trials** across the complete declared
upright two-, three- and four-level matrix. This improves the construction
pipeline but **does not achieve the approximately 90% overall target**. Two-
and three-level tasks passed **73/80 (91.3%)**; four-level tasks passed **6/20**.
The latter remain part of the overall denominator. Development results are
reported separately.

## What changed

The side RGB-D renderer now samples depth at the calibrated pixel center for
`inventory_tower`. The historical multisample resolve could bias metric depth.
Camera intrinsics, physics and the private success criteria are unchanged, but
RGB/depth observations changed; this is not a policy-only paired comparison with
the [first milestone](SINGLE_COLUMNS.md).

The CPU geometry model fits cylinder surfaces and cuboid faces instead of
assuming that visible bounds locate the center. Finite face extents bound hidden
translation coordinates. Independent outward face planes reject contaminated
edge pixels. A non-cubic width measured from coarse pixels retains a bounded
sampling interval and can increase when later trusted surfaces require it;
cube dimensions and learned heights remain fixed. Physical axis identities keep
these dimensions attached to the object as it rotates. Contradictory evidence
causes an explicit rejection rather than a fallback to stale geometry.

Tracking assigns same-color pixels to previously observed surfaces and retains
a source grasp model across robot motion. This prevents a repeated capture from
erasing the transform needed to distinguish jaw-closure slip from rigid motion.
The controller persists refined dimensions, previews every placement height,
clears a tall column before returning an uncertain grasp, and withdraws an empty
gripper even if an observation just after release is unusable. Cube columns use
visually estimated upper-stack balance and bounded release-drift compensation.

See [geometry derivations, sensor fixtures and error measurements](COLUMN_GEOMETRY.md).
The 11-observation cylinder diagnostic reduced median center error from 2.436 mm
for visible bounds with historical depth to 0.093 mm for surface fitting with
pixel-center depth. Simulator truth was an **offline error oracle only**.

## Frozen evaluation

Candidate: `policies/stack_column_cpu.py`, implementation commit
`2f8dcaca8fcca28b3e8e359c616b58c98cf0be06`.

```text
621f19b8ed4ba08bac2fa2314f1cf2527b62ee60380e3c10bee9f05ada38f47f
```

The [evaluation protocol](COLUMN_EVALUATION_PROTOCOL.md) declared 100 trials
before execution: 60 named structures with deterministic color permutations,
20 random two-level goals and 20 random three/four-level goals. Every group
contains scenes with zero and two sampled distractors. Candidate source,
individual goals and sensor settings were frozen before any reserved run.
The policy was not changed after viewing reserved outcomes.

The evaluator still requires ordered object contacts, uprightness, support
margin for the entire upper substack, robot withdrawal, low velocity and five
seconds of stability within 3 mm. A policy note saying every level was placed,
or a partial-placement score of one, is insufficient for success.

| Task cell | No distractors | Two distractors | Total |
| --- | ---: | ---: | ---: |
| Two cubes, permuted colors | 5/5 | 5/5 | 10/10 |
| Three cubes, permuted colors | 5/5 | 5/5 | 10/10 |
| Four cubes, permuted colors | 2/5 | 3/5 | 5/10 |
| Two cubes + thin prism, permuted colors | 4/5 | 5/5 | 9/10 |
| Prism + cube + cylinder, permuted colors | 4/5 | 4/5 | 8/10 |
| Cylinder + two cubes, permuted colors | 4/5 | 5/5 | 9/10 |
| Random two-level goals | 10/10 | 8/10 | 18/20 |
| Random three-level goals | 4/5 | 5/5 | 9/10 |
| Random four-level goals | 0/5 | 1/5 | 1/10 |
| **Complete declared matrix** | **38/50** | **41/50** | **79/100** |

By level count, success was **28/30 (93.3%)** for two levels,
**45/50 (90.0%)** for three levels and **6/20 (30.0%)** for four levels.
These are descriptive breakdowns of the same 100 trials, not newly declared
curricula. The named/color group passed 51/60 and the combined random groups
passed 28/40. There were no worker errors; controlled policy stops count as
failures.

All 21 failures remain in the [portable frozen report](../evaluations/column_v2_heldout.json),
with exact goals, source hash, sensor settings, final notes, CPU measurements and
private final checks labeled as offline evidence. Five failed four-cube trials
reached apparent visual completion: three exceeded the 3 mm drift bound, one
collapsed, and one violated the rotation limit. Random four-level failures were
mainly loss of usable target geometry, tipping or incomplete placements near the
highest level. Other failures included thin-prism geometry contradictions,
a fallen cylinder base and residual mixed-column drift/rotation. These labels
summarize observed outcomes rather than establishing a single cause for each.

For context, the historical first milestone passed 25/45 reserved trials. Its
matrix, source and depth sampling differ, so subtracting these percentages would
not be a paired estimate of improvement. The 79/100 result is the evidence for
the new complete declared distribution.

Changing distractor count also changes source poses; these are distribution
cells, not paired causal clutter experiments. Seeds repeat across recipe cells,
and the smaller subgroup rates should not be treated as precise estimates of
all possible tasks.

## Actual frozen-trial replays

These selected examples restore saved physics states at 6x speed. They do not
execute the policy again, and the external camera is for display only. The
selection illustrates both successful construction and the remaining failure;
it does not replace the 100-trial table.

| Selected trial | Outcome | Recorded video | Final frame |
| --- | --- | --- | --- |
| Random three levels, two distractors, seed 1017: green cylinder → green prism → yellow thin prism | Pass | [Replay](../media/column_v2_random3_1017.mp4) | [Frame](../media/column_v2_random3_1017.png) |
| Four cubes, two distractors, permuted colors, seed 1004 | Pass | [Replay](../media/column_v2_quad_1004.mp4) | [Frame](../media/column_v2_quad_1004.png) |
| Random four levels, two distractors, seed 1017: green cylinder → green prism → yellow prism → blue cylinder | Fail | [Replay](../media/column_v2_random4_failure_1017.mp4) | [Frame](../media/column_v2_random4_failure_1017.png) |

[Replay provenance](../evaluations/column_v2_replays.json) records each exact goal,
candidate hash, result and playback speed. The two seed-1017 examples have
different goals and object counts; they are not a paired causal experiment.

## CPU and observation boundary

The online policy uses NumPy/SciPy, side RGB-D, wrist RGB, camera calibration,
robot proprioception, robot-only IK and the public bottom-to-top identity goal.
It receives no simulator object poses, dimensions, contacts, masses or evaluator
feedback. Previously measured visual dimensions and poses are explicit priors,
not privileged observations. Simulation and image rendering can use the GPU.

On an Intel Core i9-14900K with single-thread BLAS, the 100 frozen trials used a
median **4.394 s** of policy compute CPU time per complete attempted task
(p95 **7.585 s**, maximum **8.974 s**). Separate robot-only host IK used median
**0.188 s**, p95 **0.473 s**. The median simulated task duration was **96.575 s**.
Policy CPU time includes worker compilation/import/run and excludes RPC
serialization, host IK, physics and rendering; it is not wall latency or a
per-action control period. Failed early stops remain in the timing denominator.

A final-source blackout trial stopped with `Insufficient tabletop depth.` after
one black frame, zero robot/gripper moves and zero IK calls. It placed no blocks.
The [runtime audit](../evaluations/column_runtime_audit.json) records the permitted
RPC interface, numerical imports, filesystem/GPU isolation, thread caps and
unchanged historical inventory hash/regression. A blackout is a useful negative
control, not by itself a proof of the entire boundary; the code and sandbox
review provide the complementary evidence.

## Development alternatives

The final adaptive development candidate passed 33/36 named color-permuted
trials. On random goals it passed 6/6 two-level, 6/6 three-level and 1/6 four-level
trials. Those seeds informed the implementation and are not held-out evidence.

Additional tools were investigated and retained as research evidence:

- Wrist silhouettes can constrain a translation axis hidden from side depth,
  with real archived examples improving center error from roughly 13 mm to
  0.1 mm. The runtime ablation did not improve the aggregate result, so this
  helper is not in the deployed bundle.
- A free-yaw robot IK search and pitched grasp prototype explored tall reach.
  On the final geometry revision, pitched grasps passed 1/6 versus 6/6 for the
  ordinary random three-level controller. Feasible endpoint attitudes did not
  guarantee a reachable lift/transport path. This prototype was rejected.
- A linear program maximized predicted support margin across all cube levels.
  It passed 3/6 versus 4/6 for the existing weighted balance controller and
  introduced a collapse. It was rejected.

[Final development comparisons](../evaluations/column_final_development.json)
retain exact sources, goals and all failures. Earlier exploratory comparisons
are in [geometry](../evaluations/column_geometry.json),
[manipulation](../evaluations/tall_manipulation_development.json),
[contact/release](../evaluations/contact_release_development.json),
[policy development](../evaluations/column_policy_development.json),
[robustness](../evaluations/column_robustness_development.json), and
[tall control](../evaluations/tall_control_development.json).
These overlapping, adaptively selected experiments must not be pooled into a
generalization success rate.

## Scope and remaining work

Objects start upright with zero yaw in one calibrated simulated camera setup.
The random sampler uses cube, prism and cylinder for interior levels, with a
thin prism additionally allowed at the top. Triangles can be distractors;
triangular caps are representable goals but the policy stops before moving them.
This does not validate all five target shapes, arbitrary orientations, different
cameras, real hardware, bridges or pyramids.

The main remaining challenge is tall manipulation: endpoint reachability does
not establish a viable grasp and collision-free transport/withdrawal corridor.
Four-cube failures also require better dynamic settling and whole-stack balance.
A separate passive continuation audit restored three **development** four-cube
rollouts and held their final robot controls for 60 additional simulated seconds.
Neither previously failed tower ever passed a strict sliding five-second window;
both later collapsed. The previously successful tower passed an early window
but its top cube dropped after another 49.25 s. Thus a five-second pass does not
establish minute-long durability, and longer waiting alone did not repair these
failures. These offline measurements do not relabel the frozen outcomes or feed
the online policy. See the [passive audit](../evaluations/column_settling_audit.json).
Any further revision informed by this frozen matrix needs a new declared set of
reserved seeds. A shorter-tower curriculum must be specified before sampling and
reported separately, rather than removing failures from this matrix.

## Reproduce

```bash
# Rebuild the self-contained CPU policy.
uv run python scripts/build_inventory_policy.py --controller column_policy \
  --output policies/stack_column_cpu.py

# The exact named/color portion of the frozen matrix (60 trials).
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python scripts/evaluate_columns.py \
  --candidate policies/stack_column_cpu.py --output outputs/column_v2_named_reproduction \
  --recipes cube_pair cube_triple cube_quad slender_top mixed_triple cylinder_base \
  --permute-colors --distractors 0 2 --seeds 1001 1002 1003 1004 1005 \
  --jobs 3 --split heldout \
  --frozen-sha256 621f19b8ed4ba08bac2fa2314f1cf2527b62ee60380e3c10bee9f05ada38f47f

# Random portions use the same source and sensor:
# --recipes random --levels 2 --seeds 1006 ... 1015 --distractors 0 2
# --recipes random --levels 3 4 --seeds 1016 ... 1020 --distractors 0 2
# Expand each listed seed explicitly; shell ellipses are not CLI syntax.

MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python -m unittest discover -s tests
```

Validation: 125 tests ran, 124 passed and one platform-specific test was skipped.
The historical inventory policy remains byte-for-byte unchanged and its
six-object regression passed. Complete local simulation artifacts, copied
candidates and manifests are under `outputs/columns_02/`; portable reports and
sensor fixtures are committed separately from these large raw recordings.
