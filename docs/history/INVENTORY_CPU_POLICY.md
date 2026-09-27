> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# CPU instance geometry: sampled inventory stacking experiment

The frozen `policies/stack_inventory_cpu.py` passed **78/80** new scenes: **50/50** with six objects, **19/20** with ten, and **9/10** with two. The task was to place the **red 1.5 × 3 × 6 cm slender prism upright on the blue 3 cm cube**. Distractors were sampled from five shapes in red, yellow, blue and green.

Perception, tracking, planning and control use **NumPy/SciPy on CPU**. Simulation and rendering retain EGL/GPU. This is a geometry and code-as-policy improvement; no GPU neural network was trained or invoked.

## Results

| Scene | Old color-binding diagnostic, development | CPU policy, development | CPU policy, frozen reserved set |
|---|---|---|---|
| 2 objects | 10/10 | 10/10 | **9/10**, seeds 701–710 |
| 6 objects | 6/20 | **19/20** | **50/50**, seeds 701–750 |
| 10 objects | 0/10 | **9/10** | **19/20**, seeds 701–720 |

Development used six-object seeds 601–620 and two/ten-object seeds 601–610. Reserved seeds were declared during development, executed after source freeze and not used for tuning. The old policy is compared on the same development sets; its [original evidence](INVENTORY_TRANSFER.md) remains unchanged.

Changing object count changes the generator's random draws and object positions. Reusing seed numbers across densities does not produce a paired experiment with fixed target poses.

Frozen source SHA-256:

```text
20613dbf213444f77aac9fdcbdf2b6245d249e0770ab362121e5663e4e53394b
```

The private evaluator checks the required object identities, table/base and base/target contacts, support margin, long-axis verticality within 8°, robot withdrawal, distractor contact and five seconds of final stability. Controlled stops, wrong-object stacks and unstable placements count as failures.

## How geometric abstraction drives decisions

1. Backproject calibrated RGB-D into visible points and fit the tabletop plane.
2. Split each color into instances using 3-D depth continuity. Same-color objects no longer share an action target.
3. Compare five shape hypotheses using measured aspect ratios, planar/curved surface normals and sloping-face evidence. Insufficient evidence produces `unknown`. Metric dimensions come from images, not the scene catalog.
4. Assign persistent IDs with color-compatible, bounded spatial association. Ambiguous matches do not take over an existing ID. During a grasp, robot motion predicts the last visually observed payload position, then new depth corrects it. Narrow exposed surfaces can update an existing track but cannot create a new object identity alone.
5. Bind the task to one red `thin_prism` ID and one blue `cube` ID. Localization, grasping, clearing, parking and collision checks use IDs, so other same-color objects remain obstacles.
6. Fit cube/box surfaces, centroids and inferred vertices from the selected instance's isolated RGB-D mask. Visible gripper geometry informs grasp height; bottom clearance and support margin guide placement. Parking cells require observed tabletop depth rather than treating occluded space as verified free space.

Initial bounding-envelope centers summarize instance locations; solid cube/cuboid centroids and vertices come from subsequent model fitting. Hidden corners remain inferred. Partial observations update constrained axes, while missing axes propagate from the last visual pose and robot motion with a consistency check. Complete occlusion can still stop execution.

## Actual recorded example: four red objects

This figure processes six-object seed 604's `camera_001.npz` without simulator segmentation IDs or object poses. The red triangular prism, cube, slender prism and cylinder become `red_0`–`red_3`; only `red_2` is the target. Dots on the right are observed depth, dashed rectangles are visible XY extents, and crosses are inferred location summaries.

![Recorded RGB-D instances and geometry](../media/inventory_cpu_instances_604.png)

The old diagnostic stacked the red cylinder in this scene. The CPU instance policy selected and stacked the correct slender prism:

![Actual physics replay of the CPU policy](../media/inventory_cpu_604.png)

[Correct-object stacking, six-object seed 604](../media/inventory_cpu_604.mp4) ·
[Clear a yellow prism, then stack, ten-object seed 603](../media/inventory_cpu_clear_603.mp4)

Videos restore recorded physics states rather than generating illustrative motion. The large external view is for the viewer; policy inputs remain side RGB-D and wrist RGB.

## Perception and CPU measurements

An offline audit of 80 initial reserved frames found all 520 objects with no extra detections. **516/520** shapes were correct: three yellow cylinders were labeled as prisms and one was `unknown`. **80/80** scenes uniquely bound the correct target and base.

These are initial-frame results, not every-frame trajectory accuracy. Only the offline audit uses private trace labels for matching; the actor does not receive them.

On an **Intel Core i9-14900K**, BLAS and OMP were each limited to one thread. The 80 frames were preloaded, warmed up and processed three times each:

| CPU measurement | p50 | p95 |
|---|---|---|
| Table fitting, instance separation and shape classification per RGB-D frame | **30.6 ms** | **44.4 ms** |
| Cumulative policy CPU computation per six-object episode | **3.57 s** | **4.66 s** |
| Cumulative policy CPU computation per ten-object episode | **4.42 s** | **6.54 s** |

The first row excludes rendering, file loading, tracking, servo and IK; it is not the full control-loop frequency. Episode measurements use worker `process_time`, including candidate compilation/imports, perception, tracking and policy logic, with RPC serialization/waiting subtracted. They exclude worker startup/pre-imported NumPy, simulation, rendering and host IK. Robot IK is measured separately: median cumulative CPU time was **0.103/0.098 s** for six/ten-object episodes. These are measurements on this CPU, not a low-power-device guarantee.

Candidate imports are limited to numerical libraries. The Linux `bwrap` worker cannot see GPU devices; tests check that `/dev/nvidia0` and `/dev/dri` are absent. Host MuJoCo/EGL usage remains separate, with no switch to CPU rendering.

## Remaining failures and scope

Reserved failures:

- Two-object seed 703: uncertain placement caused a return-to-source attempt, then the retry requested an unreachable pose.
- Ten-object seed 704: the target top became occluded during a later grasp after clearing.

Development still failed on six-object seed 620's top occlusion and ten-object seed 602's 600-RPC limit. The latter was terminated by the runner without a complete worker CPU profile; the other **119/120** final task trials have profiles. The original **600-call/120-second** limits were retained for this two-object task.

The policy assumes initially upright, zero-yaw objects and a fixed four-color palette. It clears at most two blockers. Triangular prisms are classified and avoided, but their clearing grasp is not implemented. Shape scores are hypothesis-selection heuristics, not calibrated probabilities. Touching/merged instances, severe occlusion, arbitrary rotations and lighting changes need further evaluation. Different density groups have different poses, so these results do not establish that more objects improve reliability.

## Observation boundary and verification

Online inputs are side RGB-D, wrist RGB, calibration, robot joints/tool state and robot-model-only IK previews. The policy does not receive true object poses/dimensions, simulator segmentation IDs, contacts, the scene catalog or evaluator results. Development diagnostics and offline evaluation may inspect private traces, separately from online action inputs.

Full sensor blackout produced **zero motion commands and zero policy simulation-motion seconds**. Historical prism seed 405 still passed with the unchanged historical source hash. Tests: **57 passed, one platform-specific skip**, covering depth separation, same-color instances, motion association, ambiguous matching, partial surfaces, missing targets, reproducible bundling, input isolation and CPU timing that excludes RPC waits.

The [portable per-trial evidence](../evaluations/inventory_cpu.json) contains final development/reserved results, hashes, CPU data, diagnostics and offline audits. Complete frames, RPC logs and physics states remain under `outputs/cpu_instances_01/`.

## Run and reproduce

```bash
# Execute the CPU policy while retaining EGL simulation rendering.
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python scripts/run_vision_trial.py \
  --task inventory_prism --layout sampled --object-count 6 --seed 701 \
  --candidate policies/stack_inventory_cpu.py --output outputs/cpu_trial_701

# Execute, then view the actual physics replay.
uv run python view_scene.py --vision-task inventory_prism --object-count 10 \
  --seed 603 --vision-policy policies/stack_inventory_cpu.py \
  --vision-output outputs/cpu_clear_view_603

# Visualize a real frame without reading object labels.
uv run python scripts/inspect_instances.py \
  --frame outputs/cpu_trial_701/camera_001.npz --output outputs/cpu_overlay_701

# Offline audit: this script reads private evaluator trace labels.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python scripts/audit_inventory_perception.py \
  --trials outputs/cpu_instances_01/reserved_six \
           outputs/cpu_instances_01/reserved_two \
           outputs/cpu_instances_01/reserved_ten \
  --output outputs/cpu_audit.json

# Edit reusable modules, then rebuild the numerical-only worker source.
uv run python scripts/build_inventory_policy.py --output policies/stack_inventory_cpu.py
MUJOCO_GL=egl uv run python -m unittest discover -s tests
```

Reusable implementations are `src/aspire_so100/instance_geometry.py` and `inventory_policy.py`. The builder combines them with cube/cuboid/gripper geometry, so the isolated worker need not read the checkout or import simulation packages.
