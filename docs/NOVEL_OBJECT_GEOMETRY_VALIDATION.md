# Novel-block geometry validation

As of 2026-09-27, the explicit libccd/0.40 grasp variant passes 6/6 paired physical
rollouts across three development layouts. These runs diagnose shared execution;
they do not measure a Base/Full advantage or establish held-out generalization.
The [design document](NOVEL_OBJECT_GEOMETRY_DESIGN.md) describes the current
representation, observer, API and regression tests. Saved failures, source
snapshots and frozen learning-study evidence remain unchanged.

## Setup and scoring

The diagnostics use three seeded irregular convex meshes, one object at
a time, and the same one-action controller for Base and Full: initialize the
session, pick red, and place at a public destination surface point. The policy
receives no simulator object dimensions, pose, vertex count or contact results. A sandboxed
worker receives only the public goal, 640x480 pixel-aligned RGB-D, robot state and
robot geometry; ideal wrist depth is enabled. This is a development sensor
configuration, not an extension of the completed 320x240 learning-curve study.

The meshes are physical 30 g free-joint objects. Their visual and collision
geometry agree. There is no kinematic attachment; actuator parameter variants are recorded explicitly.
The unchanged evaluator privately requires a lift greater than 25 mm with robot
contact, then table support without robot contact, upright orientation within
8 degrees and destination XY error at most 8 mm. During the final two seconds,
position drift must remain within 3 mm, linear speed at most 0.015 m/s and angular
speed at most 0.1 rad/s. A policy error still marks the rollout unsuccessful.

## Current contact and grasp variant

These development checks isolate physics from perception. With MuJoCo 3.11.0,
[one-option-at-a-time static controls](../outputs/point_geometry_contact_probe_01/summary.json)
compare the scene defaults, a halved timestep, elliptic friction cones, ten noslip
iterations, a tighter collision tolerance, and the alternative libccd collision
pipeline. Only the libccd variant passes the original stability criterion on both
previously affected fixtures. This local result identifies a collision-pipeline
sensitivity; it does not establish a general defect in MuJoCo. The
[MuJoCo option reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-flag-nativeccd)
documents the native/libccd selection.

[A separate no-action check](../outputs/point_geometry_static_02/summary.json)
uses all three existing fixtures plus six additional generated fixtures. The
compiled scene defaults pass stability on 0/9, versus 9/9 with libccd. Across these
nine cases, the maximum angular speed ranges from 0.320 to 1.534 rad/s under the
original pipeline and from 0.0118 to 0.0212 rad/s with libccd. The threshold stays
0.1 rad/s. There is no policy action, model call, change to friction or mass, or
change to the evaluator. These are static contact checks, not task successes.

The physical runner now accepts `--collision-pipeline scene|native|libccd`.
`scene` remains the default and preserves the compiled options. The selected
backend is applied before initialization and recorded with the effective option
flags, timestep, solver and collision settings. The shared scene and frozen
experiment sources remain unchanged.

The subsequent paired grasp checks use the same three original development
fixtures and separate the changes:

| Run | Collision pipeline | Generic grasp fraction | Strict successes |
| --- | --- | --- | --- |
| `point_geometry_physics_04` | Native, original scene default | 0.70 | 0/6 |
| `point_geometry_physics_05` | libccd | 0.70 | 4/6 |
| `point_geometry_physics_06` | libccd | 0.40 | 6/6 |

In the middle run, seed `2026092701` slips during placement. Although it eventually
rests inside the private goal tolerance, the controller reports a neighborhood
error and never completes its release procedure. It remains a failed rollout.
Replay shows the original grasp primarily engaging the upper part of the object.

The final run uses the actuator's existing `grasp_fraction` parameter. Its tool
height is `observed_center_z + grasp_fraction * observed_height`; reducing 0.70 to
0.40 lowers the grasp by approximately 10–13 mm on these fixtures. It introduces
no shape-specific branch and uses only observed dimensions. The opt-in builder
accepts `library(arm, grasp_fraction=0.4)`; both CLIs accept
`--grasp-fraction .4`. Omitting the override preserves the original generated
library exactly. Both arms receive the same parameter; policy method signatures
and the shared actuator source are unchanged.

| Fixture seed | Final XY error in both arms | Maximum angular speed in the final two-second window |
| --- | --- | --- |
| 2026092701 | 4.40 mm | 0.0137 rad/s |
| 2026092702 | 6.95 mm | 0.0246 rad/s |
| 2026092703 | 1.05 mm | 0.0313 rad/s |

All six final rollouts report a visually confirmed lift and completed release,
and pass the unchanged private placement and stability checks. The
[comparison audit](../outputs/point_geometry_physics_06/comparison.json) verifies
identical controller and observer sources, unchanged evaluation checks, identical
motion commands within each Base/Full pair, and isolated changes between stages.
The collision-only models differ only in the native-CCD flag; the grasp-only
variant preserves the model and initial image and changes only the constructor's
grasp default in each generated library.

These are three development layouts, not six independent layouts or a held-out
success-rate estimate. The deeper grasp was selected after inspecting the failed
case; its suitability for thinner or differently scaled objects is untested.
The 6.95 mm placement is also close to the unchanged 8 mm tolerance. These results
support a usable physical diagnostic configuration, not a measured advantage from
geometric abstraction. The completed learning-curve experiment remains frozen.

To reproduce the explicit variants in new directories:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/validate_polygon_contact_stability.py --output outputs/polygon_contact_recheck --seeds 2026092701 2026092702 2026092703 2026092704 2026092705 2026092706 2026092707 2026092708 2026092709
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/validate_polygon_prism_physics.py --output outputs/polygon_libccd_recheck --collision-pipeline libccd
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/validate_polygon_prism_physics.py --output outputs/polygon_deeper_grasp_recheck --collision-pipeline libccd --grasp-fraction .4
```

## Original configuration

To reproduce in a new directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/validate_polygon_prism_physics.py --output outputs/point_geometry_physics_recheck
```

The default budget is three seeds times two arms. Use `--seeds` and `--arms` for
a smaller probe. The runner refuses to overwrite an output directory and records
source/library hashes, the controller, model binaries, images and calibrated depth,
policy calls, compact geometry, private states, contact checks and replay states.
Private evaluator data stays on the host; it is never included in RPC responses.

## Earlier perception diagnostics

A one-rollout probe is preserved under
[`point_geometry_physics_probe_01`](../outputs/point_geometry_physics_probe_01/summary.json).
The subsequent six-rollout diagnostic is recorded in
[the protocol](../outputs/point_geometry_physics_01/protocol.json) and
[the complete results](../outputs/point_geometry_physics_01/summary.json).
There were no model calls, and no learning-study layouts or results were modified.

| Fixture seed | Observed boundary count | Maximum bottom lift | Outcome in both arms |
| --- | --- | --- | --- |
| 2026092701 | 4 | 56.3 mm | Object remains grasped; current cap boundary is occluded in both views. |
| 2026092702 | 4 | 35.0 mm | Object slips before lift confirmation; `grasp_unconfirmed` correctly prevents placement. |
| 2026092703 | 5 | 56.3 mm | Object remains grasped; current cap boundary is occluded in both views. |

All three initial shape fits were available. None of the six rollouts completed
placement. Base and Full produced identical time-stamped motion commands in every
pair; these pairs verify shared execution, not a learned-policy advantage or six
independent layouts. Transient lift is not evidence of a stable grasp.

[Offline replay of the raw frames](../outputs/point_geometry_physics_01/raw_frame_audit.json)
removes robot masking from the diagnosis. The two grasped fixtures still fail the
nearer-geometry boundary check in both views, so relaxing the robot mask alone
would not resolve the missing evidence. The slipped fixture is visibly back on
the table at the confirmation frame, consistent with the private contact record.

These failures motivated the current [partial-boundary update](NOVEL_OBJECT_GEOMETRY_DESIGN.md#from-pixels-to-geometry).
The original results and generated libraries remain unchanged. Intermediate
single-view checks are preserved under `point_geometry_physics_02` and
`point_geometry_physics_03`.

### After current-edge and cross-view updates

[The paired recheck](../outputs/point_geometry_physics_04/summary.json) uses the
same three seeds, public controller, meshes, cameras, shared actuator and scoring
thresholds. Both arms still produce identical time-stamped motion commands.
No model calls or changes to the frozen learning-curve study were made.

| Fixture seed | Outcome in both arms | Final XY error | Strict physical success |
| --- | --- | --- | --- |
| 2026092701 | Visually confirmed lift, placement and release | 2.35 mm | No: stability velocity threshold exceeded. |
| 2026092702 | `grasp_unconfirmed` after slipping, unchanged | 56.18 mm | No: placement not attempted. |
| 2026092703 | Visually confirmed lift, placement and release, using joint current-view registration during descent | 0.83 mm | No: stability velocity threshold exceeded. |

On these three development layouts, release completion improves from 0/6 to 4/6
rollouts and observation exceptions from 4/6 to 0/6. The strict success count stays
0/6. The completed placements have table support, no robot contact, upright poses
and XY error below 8 mm, but fail the two-second velocity criteria. Do not report
release completion as task success, or paired arms as independent layouts.

A separate [no-action control](../outputs/point_geometry_static_01/summary.json)
leaves each of the two affected meshes untouched for 36 seconds before the same
stability check. Both still fail: maximum angular speeds are 0.320 and 1.534 rad/s,
above the 0.1 rad/s threshold, with drift of 0.216 and 1.331 mm. This establishes
that the mesh/contact configuration can fail the velocity criterion without any
pick/place action. Physics parameters and scoring thresholds were not relaxed.
These results motivated the subsequent contact and grasp controls described above.
