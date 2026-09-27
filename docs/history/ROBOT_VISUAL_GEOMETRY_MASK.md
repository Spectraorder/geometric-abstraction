> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Robot geometry removes a false object hypothesis

The remaining large errors in the memory diagnostic came from a known green
visual marker on the robot. In a wrist frame at 9.90 s, the selected green
component contained 181 depth points, all approximately 3.96–4.00 mm from the
encoder-derived tool center. The robot model defines its `ee_site` visual sphere
with radius 4 mm. The actual green block's side-view points were about
177–200 mm from that center.

The development prototype excludes depth points inside this known robot sphere,
using its model radius and same-time robot proprioception. It does not use an
object pose, object contact, segmentation ID or private task state. RGB is not
changed; a copied depth-validity mask marks those samples unavailable. Points at
a different depth remain available even if they project near the marker. Stale
robot timestamps are rejected. Coverage is one supplied visual sphere, not the
whole robot or gripper housing.

The sphere mask uses an explicit 0.5 mm tolerance. Its purpose is to prevent robot
visual geometry from becoming an object observation. This is an analytic,
simulated-marker repair, not a general solution to same-color object recognition.

## Paired replay, including coverage changes

The rule was tested unchanged on the same two development sets. All sensor-only
estimates were saved before private physics was read for scoring. Comparisons
below use identical object/time keys; loss of coverage is reported separately.

| Set / method | Matched estimates | p95 center error | Maximum error |
| --- | ---: | ---: | ---: |
| Successful rollouts: original instantaneous | 122 | 2.689 mm | 80.091 mm |
| Same: robot mask | 122 | 2.689 mm | 4.737 mm |
| Same: robot mask plus bounded memory | 122 | 0.306 mm | 4.737 mm |
| Failed rollouts: original instantaneous | 193 | 2.398 mm | 73.715 mm |
| Same: robot mask | 193 | 2.398 mm | 2.985 mm |
| Same: robot mask plus bounded memory | 193 | 0.619 mm | 2.488 mm |

The first set retains all 122 estimates. In the second, output coverage falls
from 195 to 193: two green poses formerly had errors of 151.33 and 155.00 mm;
after robot geometry is excluded, there is insufficient visible object evidence
and they become unavailable. They are not counted as improved pose estimates.

Unconditional static memory still fails after robot masking: its maximum error
is 45.84 mm in the first set and 149.66 mm in the second. Robot exclusion and
current-evidence checks solve different problems. Neither alone justifies
unconditional temporal retention.

## Evidence and boundaries

- `scripts/robot_visual_mask_prototype.py`: CPU depth exclusion from known robot
  spheres and timestamped pose data. Three tests cover copied input, depth-aware
  exclusion and timestamp rejection.
- `scripts/audit_bounded_memory.py --robot-mask`: replay using the exact frozen
  v2 geometry library; robot radius comes from `get_so100_spec().site('ee_site')`,
  and centers come from logged robot-only proprioception.
- `scripts/summarize_robot_mask_memory.py`: matched-key accuracy and separately
  reported unavailable outputs.
- [Complete paired report](../evaluations/robot_mask_memory_diagnostic.json), with
  raw estimates, model inputs and source snapshots in
  `outputs/bounded_memory_robot_mask_01` and `_02`.

These are inspected training trajectories with ideal simulated depth. No policy
success, iteration-efficiency or real-camera benefit is established by this
replay. The ongoing v2 policy comparison has not been altered. A future production
interface must expose the same robot-model information to both comparison arms
and keep robot geometry provenance separate from visual object estimates.

## Production interface after the v2 comparison completed

The robot sphere filter was promoted only after all 285 v2 rollouts and their
integrity audit completed. Historical candidates, libraries and source snapshots
remain unchanged.

`api.gripper_geometry()` now includes `visual_spheres` for the tool marker in
addition to the existing eight pad boxes. Their centers use forward kinematics
on fresh data populated only with robot joint encoders. A regression test moves
all free object joints and verifies that the returned robot geometry is unchanged.

The action-free scene interface accepts the packet explicitly:

```python
frame = api.camera('side')
scene = geometry.scene(frame, colors, robot_geometry=api.gripper_geometry())
scene.refresh(api.camera('wrist'), 'wrist',
              robot_geometry=api.gripper_geometry())
```

Every supplied robot packet must match the frame's timestamp and use world
coordinates in meters. No robot geometry is silently reused after motion.
Without a packet, the interface retains its previous unfiltered behavior.
`scene.packet()` records per-view `robot_geometry_exclusion` evidence, including
the number of removed depth samples.

The numerical exporter includes `robot_visual_geometry.py`, without a robot API
or simulator dependency inside the scene library. Fourteen interface/protocol
tests, four production mask tests and eight rigid-state tests passed. The real
pick-and-return probe `outputs/online_geometry_smoke_04` completed without a policy
error, visually confirmed the lift, released the object back on the table, and
saved three state records. Its wrist checkpoint excluded 186 robot-marker depth
samples. The probe did not attempt a tower, so its false task-success flag is
expected and is not an SR measurement.
