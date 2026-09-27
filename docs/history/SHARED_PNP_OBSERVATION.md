> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Shared PnP observation interface

## Why this interface is needed

The bridge release audit found a visible-top center bias while the object was in
the gripper. Supplying a more accurate scene estimate only to a task planner could
not fix the executor's internal release check. This interface allows the same
execution loop to consume an externally selected numerical observer.

`policies/pick_place_rgbd_shared.py` is unchanged. Its SHA-256 remains
`1fcaad52faad8f93fa34165ef520409376701d1e434cca1ff65b08a0003e0329`.
`ObservedRGBDPickPlace` inherits its grasp, motion, correction, retry and release
implementation and overrides only the observation entry point. With no callback,
the original path is used. Historical candidate files and outcomes remain intact.

## Contract

Select `set_observer(callback)` before observing or picking any objects. The
callback receives a fresh RGB-D frame, unique color, previous measured dimension
model and fitted table plane. It returns center, normal, size, vertical bounds,
model, current timestamp, world frame and meter units. Numeric shape, finiteness,
normalization, vertical-bound consistency and timestamp checks run before caching
the model or using the estimate. Backends cannot be changed after model learning.

`MixedPnPObservation` uses existing CPU fits for cubes, rectangular prisms, thin
prisms and cylinders. Shapes and initial orientation hints are public inputs;
dimensions come from the initial on-table RGB-D. No target, construction order,
simulator object state or robot command enters this observer. It retains the
fit's partial-observability and inferred-orientation flags. A usable point estimate
is not a claim that all coordinates are observed or that release will succeed.

`observe_views` accepts an ordered list of same-time views and uses the first
usable fit. It records failed earlier views; it does not fuse inconsistent poses,
retain an old pose or assume a commanded movement occurred. If neither view is
usable it reports missing evidence instead of fabricating a pose.

The development adapter initially tries the side view, then requests current wrist
depth if fitting fails. It filters known robot visual marker points using the robot
model and same-time encoder geometry. Sensor acquisition and logging stay outside
the pure geometry callback. This is view selection, not multi-view fusion.

## Export and validation

`scripts/build_mixed_pnp_observer.py` exports only numerical geometry, the shared
executor and the callback subclass. Task selection, task targets, support graphs
and specialized controllers are excluded from exported geometry modules. The
manual development planner is appended separately and never supplied to
experimental actors.

Six tests cover identical default observations and inherited motion methods,
stale timestamp rejection, invalid units before action, matching packaged and
exported fits on an actual bridge RGB-D fixture, execution through the new callback,
and same-time alternate-view behavior. The fixture contains only initial and held
RGB-D observations, no simulator object truth.

## Completed development results

| Candidate | Layouts | Success | Policy errors |
| --- | --- | ---: | ---: |
| Archived basic observer, manual planner | 75001–75004 | 1/4 | 0 |
| Mixed rigid observer, side only | 75001–75004 | 1/4 | 2 |
| Mixed rigid observer, side/wrist fallback and robot mask | 75001–75004 | 3/4 | 0 |
| Same fixed mixed candidate, new development layouts | 75101–75108 | 6/8 | 1 |
| Archived fixed basic candidate on the same new layouts | 75101–75108 | 3/8 | 2 |

The first three rows are adaptive development, not independent learning campaigns.
The side-only errors occurred during beam occlusion after both foundations were
released. In the side/wrist four-layout probe, 4 of 195 returned observations used
the wrist view. The remaining failure, seed 75004, retained its first foundation
with a measured 43.7 mm lateral residual after retries.

On the eight new layouts, both candidates succeeded on 2, mixed only on 4, basic
only on 1, and neither on 1. Two basic failures occurred before motion because a
cylinder's top was unavailable to the basic observer. Mixed failed at foundation
placement on 75101 and at beam observation on 75105; the latter was a basic success
and remains a regression. The mixed observer correctly reported that neither
current view provided a usable fit. Six successes are therefore not evidence of
universal reliability.

Both candidate sources were fixed across 75101–75108. The audit verifies matching
initial image hashes for every pair, source hashes against manifests and results,
unchanged task and physics settings, wrist-depth availability, and same-time
observation records. Temporary worker source copies are not retained; source
verification uses the archived campaign file and the runner's recorded hashes.
The basic runs were performed after the mixed outcomes were visible, so this is a
development diagnostic rather than a preregistered confirmation. No actor made a
policy revision in these probes. All twelve layout seeds are now inspected data.

## Experimental interpretation

The completed development outcomes are reported in
[the machine-readable audit](../evaluations/shared_pnp_observer_development.json).
These are fixed manual-planner probes, not agent-generated policy campaigns.
Changing the observer changes perception available both to target construction
and to the otherwise unchanged executor. The mixed variant also has wrist fallback
and robot visual masking. This combination does not isolate a symbolic-interface
effect, nor show that agents improve in fewer iterations.

The next numeric-state and symbolic-state arms must share this exact estimator,
camera acquisition rule, robot mask and executor. Their difference should be the
additional typed references, relations and desired-state operators. All arms must
have the same callback capability; a stronger controller cannot be hidden in a
geometry treatment. Conditional memory remains a separate prototype because its
assumptions have not yet been validated in this mixed-shape execution loop.

## Optional current-surface correspondence recovery

The regression in seed 75105 motivated an opt-in `correspondence=True` mode. When
individual views fail for a known noncube box, it tests the measured lengths
against current face and boundary evidence across same-time cameras. It rejects
ambiguous assignments and wide hidden intervals rather than retaining an old
pose. The original default remains unchanged. See
[the correspondence diagnosis and validation](CUBOID_CORRESPONDENCE.md).
