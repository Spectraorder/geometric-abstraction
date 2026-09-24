# Observation and privileged-state audit of the first three experiments

A later, fourth experiment used the agreed **RGB-D + proprioception** boundary: no object ground truth or online structure oracle reached the policy. Starting from an empty task-skill library, it reached 3/3 in two rounds, then 2/2 after freezing. See the [visual learning report](VISION_FROM_SCRATCH.md). The historical audit below is retained; later results do not retroactively change earlier policies.

This audit covers the two-block baseline, initial three-level tower, and tower policy after four rounds of assistant feedback. **All three used privileged state and were not vision-only controllers.**

| Experiment | Inputs actually used by the policy | Image use | Classification |
|---|---|---|---|
| Two-block baseline | Simulator world positions of target and base; robot model and joint state for IK | Keyframes, recordings, browser display | Privileged-state policy |
| Initial three-level tower | Object positions, world half-extents, tool position, joint angles, known kinematics and limits | Reference video, development diagnosis, replay | Privileged-state feedback policy |
| LLM feedback tower | All preceding inputs plus joint arrival feedback; the outer assistant also read true object velocities, positions and structure checks | Failure diagnosis, evidence, recording | Outer image/ground-truth feedback; inner privileged-state policy |

## Source evidence

- `stacking_env.py::_object_state` reads positions from `data.xpos`, orientations from `data.xquat`, and velocities from `data.qvel`. These are not RGB/depth estimates.
- `observation` reads the tool from `data.site_xpos` and joints from `data.qpos`.
- `tower.py::extent` combines `geom_xmat` and `geom_size`; `TowerAPI.get_observation` exposes the result as `half_extent_world`.
- `policies/stack_small_red_on_blue.py` reads true object positions.
- `policies/stack_three_blocks.py` reads target/support positions, extents and tool offsets, then solves IK using joint state.
- `policies/stack_three_blocks_learned.py` adds joint arrival checks. All four actual candidates called `get_observation`; none called `get_overhead_rgbd`.
- The three policy hashes match their recorded demos' `recording.json`, tying this audit to the code that produced those videos.

## Evaluation truth versus policy truth

An evaluator may use ground truth without making the policy privileged. Here, however, the executing policies directly read object ground truth and used it to plan and correct actions.

Both tower policies also called `api.check_structure` and used its result to continue or stop. That result depends on true contacts, orientation, velocity, center of mass and drift; it therefore entered online decisions.

Joint encoders and known robot kinematics can be realistic proprioception. The object poses, geometry bounds and contact/stability oracle are the evidence that these three experiments were not vision-only.

## Images did not select online actions

`scripts/record_demo.py` first executes a policy and records physics states, then restores those states to render RGB, depth and wrist views. `scripts/evaluate_tower.py --visuals` saves initial/final images. Multiple camera panels in a video do not establish visual control.

During direct learning, the assistant inspected final images from rounds 1 and 3. Repairs also used true trajectory measurements, including roughly 54 mm of lift, 0.133 m/s velocity and joint information. The outer learning process cannot be called vision-only either.

## Interpretation of the historical results

The success rates, timing improvements and stability checks remain valid for **MuJoCo simulation, privileged-state-assisted control, a known model and limited layout perturbations**. They do not establish visual perception performance.

Subsequent visual experiments must state their proprioception/calibration allowance, remove object ground truth and structure oracles from the controller API, and keep success evaluation private. Ground-truth feedback used during development must be disclosed separately. This audit itself did not modify or rerun the policies.
