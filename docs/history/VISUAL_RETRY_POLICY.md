> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Visual cube and gripper geometry for 3 cm cubes

The [published policy](../../policies/stack_three_blocks_visual_geometry.py) passed
**198/200 fresh reserved randomized layouts (99%)**, **445/450 development
layouts**, and **10/10 separately reserved nominal layouts** in simulation.
This meets the experiment's target of at least 99% on 200 fresh layouts; it does
not establish a guaranteed 99% population success rate.

The published source is identical to the third frozen candidate, SHA-256
`9d4484224ddcfb8c00e8c930ca68471a3e10b3ebd067a7c9fcc587a62e765e4b`.
No policy parameters changed after the freeze or during reserved validation.

## Visual observations and gripper geometry

Object positions, dimensions, orientations, centroids, and inferred vertices
come from calibrated side RGB-D. Wrist RGB corroborates the held payload.
Robot proprioception and robot-only inverse kinematics support motion planning.
The policy receives no object state, contacts, evaluator feedback, seed, scene
import, or stored layout lookup. The sandbox and source validator remain active.

`api.camera()` and `api.camera('side')` return side RGB-D. `api.camera('wrist')`
returns RGB, intrinsics, camera pose, and timestamp, **without wrist depth**.
The wrist checks colored pixels near the projected payload position; a
disagreement triggers another side-depth grasp check.

The [gripper estimator](../../src/aspire_so100/gripper_geometry.py) finds neutral
robot surfaces near the reported tool position. It measures visible fingertip
reach, lowest visible surfaces, and observed jaw separation when both jaws are
visible. The policy measures the open gripper at the approach pose and uses
its reach to choose grasp height. It also records visible clearance above the
held cube's bottom.

This is **partial visible geometry**. Hidden finger surfaces remain unknown;
observed separation is not a certified physical aperture or a complete collision
model. Grasp height retains calibrated bounds when visibility is insufficient.

![Visible gripper geometry and wrist RGB](../../outputs/visual_retry_03/gripper_visualization/gripper_overlay.png)

The overlay comes from reserved trial 8001. Cyan marks visible gripper points,
magenta marks its lowest visible surfaces, and yellow marks the robot tool
reference. The green sphere in wrist RGB is the existing robot tool marker.
Generate an overlay from saved sensor data and corresponding robot proprioception:

```bash
uv run python scripts/inspect_gripper_geometry.py \
  --frame path/to/side_frame.npz --tool-position X Y Z \
  --wrist path/to/wrist_frame.npz --output outputs/gripper_inspection
```

Trials record robot pose responses in `api_calls.json` for these diagnostics.

## Reachability, placement, and recovery

Some stacking positions leave too little wrist range to keep the top cube level.
When a clear candidate area is available, the policy moves the blue base before
stacking; otherwise it uses the observed base location. It selects candidate areas
from camera-measured clearance to the other cubes. After lifting blue, it combines
the observed cube normal with robot tool rotation to estimate how the cube sits
in the gripper. `api.plan_tool()` previews reachable tool poses at the future
tower height, allowing the policy to favor a clear location with low predicted
cube tilt. The preview reads robot joints and kinematics only, advances no time,
and exposes no object state. The task allows moving the base.

During placement, the controller predicts how a tilted cube will rock flat from
its measured normal and lowest inferred vertex. It accumulates visual alignment
corrections, aiming for 1 mm XY error, and checks gap and support margin before
release. Visible support-plane coordinates are refreshed while occluded
coordinates retain their prior estimates. A small compensation is applied when
the measured red support offset exceeds 1.5 mm. Neither physics parameters nor
the evaluator's success criteria were relaxed.

Each placement gets at most two attempts. After a missed grasp or rejected
placement, the policy measures again and lowers its retry grasp by 2 mm. A test
deliberately left the jaw open for the first close command on seed 101. The
policy recovered and completed the tower in **102.45 simulated seconds**, versus
92.70 seconds normally: **9.75 extra seconds**. A full sensor blackout produced
zero motion and no success.

[Watch the forced-miss recovery replay](../../outputs/visual_retry_03/injected_miss/seed_101/demo.mp4).
The 450 development runs averaged **91.63 simulated seconds** (median 90.52;
range 62.60–110.30). Moving the base adds time compared with the preceding
98% candidate, which averaged 64.88 seconds on its smaller development set.
These are simulated robot trajectory durations, not wall-clock execution times.

## Evaluation record and remaining failures

| Frozen version | Development | Fresh randomized validation | Nominal |
| --- | ---: | ---: | ---: |
| First | 50/50 | 192/200 (96%) | 10/10 |
| Second | 249/250 | 196/200 (98%) | 10/10 |
| Current | 445/450 | 198/200 (99%) | 10/10 |

The original development seeds were 101–105, 1001–1005, 2001–2010, and 3001–3030.
Earlier reserved sets 4001–4200 and 5001–5200 became development evidence for
the current search, yielding 450 development layouts. Current reserved seeds
**8001–8200** were declared before this search and evaluated only after the
source was frozen. Nominal seeds **9001–9010** were separately reserved.

Both reserved failures remain in the reported denominator:

- **8050:** placed both cubes, but green failed the final rotation stability check.
- **8087:** stopped before moving because the initial green dimensions were unreliable.

Development failures were 4011, 4168, 5022, 5042, and 5114. All 650 randomized
trials completed without policy runtime errors. These results cover the current
30 mm cubes, zero-yaw randomized XY distribution, and ideal simulated depth;
hardware and noisy-sensor reliability have not been evaluated.

The test suite ran **30 tests: 29 passed and one macOS-only check was skipped**
on Linux. Tests cover wrist RGB without depth, blackout behavior, robot-only
IK preview, partially visible gripper geometry, and an occluded cube-fit
regression that previously could produce NaN coordinates.

Complete plans, frozen sources and hashes, sensor frames, API logs, physics
rollouts, failures, and fault tests are retained under `outputs/visual_retry_01/`,
`outputs/visual_retry_02/`, and `outputs/visual_retry_03/`. The current
[evaluation record](../evaluations/visual_geometry_3cm.json) preserves the audit
summary, frozen runtime hashes, and planned seed sets in Git. Raw experiment
outputs, overlays, and videos are local artifacts ignored by Git; the media
links above require those artifacts. The standalone policy intentionally embeds
its numerical helpers so the isolated worker needs no repository access; its
validated source is kept byte-for-byte unchanged.

```bash
uv run python view_scene.py \
  --vision-policy policies/stack_three_blocks_visual_geometry.py \
  --vision-layout random_xy --seed 8001 \
  --vision-output outputs/visual_geometry_view_8001
```

Use a new output directory. Saved trials can be replayed with `--vision-trial`.
