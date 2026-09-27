# Explicit target binding for online skill reuse

Status: component tests, an accelerated physical check and one paired live
development check are complete. No reserved evaluation has been run for this interface. This is a
separate interface study, not an amendment to any completed or stopped campaign.

## Motivation and unchanged task

The ordinary bridge-plus-two-cube task can finish from one initial node-order plan
because the shared skill computes current destinations internally. That system
comparison tests useful skill composition, but often leaves little model-level
spatial reasoning to compare. The proposed interface makes one responsibility
explicit: the LLM binds the placement surface point from current evidence; the
shared skill executes the pick and placement. Objects, goal, natural-failure
handling and the no-reset rule remain the same. Artificial collapse and other
complex recovery scenarios remain out of scope.

This changes the degree of assistance provided by a skill. Describe it as adapted
skill reuse, not unmodified execution of the previously trained policy. Do not
combine its outcomes with the earlier node-only API results.

## Development interface

The opt-in adapter is `scripts/online_bound_target_skills.py`. Existing
`run_steps(skill, node)` calls retain their original target inference. Explicit
calls use `drive_bound`:

```python
operation = {
    "action": "execute",
    "skill": "stack",
    "node": "red",
    "surface_point_world_m": [0.10, -0.20, 0.44],
}
result = drive_bound(skills, operation, on_boundary=check_current_supports)
```

The coordinates above are illustrative. They specify world-frame XY and the
support-surface Z in meters, not the source object's desired center. A binding is
a caller action intent. It does not assert that the point is currently visible,
occupied by a support, reachable or correct for the task.

Within an explicit call, the adapter does not invoke `current_target`. It passes
the submitted point to the existing segmented placement controller and checks the
observed placement relative to that point. It never silently moves the target to
a newly observed support. A changed support can instead cause the existing motion
boundary callback to cancel the operation and return control to the caller.

The adapter copies and validates the three numeric coordinates, records the
binding, preserves current grasp state on cancellation, and removes the binding
on normal completion or exception. Invalid coordinates are rejected before
motion. The common actuator still performs its normal grasp observations and
corrections toward the submitted goal. Goal construction remains the caller's
responsibility.

`visually_consistent` means consistency with the submitted point under the
existing visual check. It is not task success, contact certification or stability
certification. Returned explicit-call results retain
`task_success_certified=false`; the independent private task evaluator remains
responsible for ending an episode successfully. A deliberately wrong target is
not rescued by a task-specific destination calculation.

## Validation so far

Seven unit tests use fake sensing and motion to check:

- Submitted, deliberately displaced coordinates reach the placement call unchanged.
- A confirmed held object resumes toward the same submitted target without a new pick.
- Cancellation after picking retains holding, does not place, and clears the binding.
- Mutating the caller's input list cannot change an active target.
- Unbound calls dispatch to the original implementation.
- Invalid or nonnumeric coordinates fail before actions.
- Execution errors clear the binding and retain the submitted intent in the result.

These tests establish API behavior, not robot feasibility or a reasoning advantage.
The adapter has not been inserted into any earlier frozen runner. The later live
development check below evaluates the new schema and prompt separately.

## Requirements before a comparative experiment

Both arms must have the same target schema, decision boundaries, skill library,
actuator, recovery choices and timing budget. Neither may receive an automatically
corrected submitted target. Full may obtain current typed geometric references;
Base must retain calibrated RGB-D access and a viable way to inspect metric depth
or perform its own visual calculation. Quantized depth-display images alone are
not equivalent to metric-depth access. A shared sensor-query implementation now exists and passes the archived-frame
checks below. Its integration into model decisions and end-to-end usability still
require validation before a comparative episode.

Full feedback should expose measured references with evidence, rather than a
precomputed task answer disguised as a geometric state. Before any model requests,
verify sensor-grounded reference values, coordinate conventions and both arms'
access boundary on development scenes. Log the actual submitted point and
executed intent so the evaluator can distinguish a reasoning error from a control
failure. Timers must include queries and failed decisions.

Only after those checks should a separate small protocol be registered. Retain
the user's five-layout scope, bounded parallelism and all failures. Do not reuse
reserved outcomes from earlier studies to choose or repair evaluation layouts.
A different action interface is a new experimental question, not evidence that the
previous negative or inconclusive result was incorrect.


## Shared metric-depth access

`scripts/online_metric_depth.py` provides `MetricDepthSensorMixin`, usable by both
arms with the same arguments and outputs. An image response includes a `frame_id`,
image dimensions, pixel convention and observation time. Query integer pixel
coordinates on that displayed frame:

```python
path, metadata = api.image("decision_000_side", "side")
frame_id = metadata["metric_depth_frame"]["frame_id"]
samples = api.sample_depth(frame_id, [[180, 110], [160, 120]])
```

Each valid sample returns optical-axis depth in meters and its world-frame point,
computed from the archived intrinsics and camera-to-world transform. This common
sensor operation supplies visible surface points, not object centers, inferred
hidden surfaces or task-specific targets. Full's treatment remains object-level
geometry and typed references above this shared point-query access.

The query uses the exact RGB-D archive produced when the image was displayed;
it does not recapture or use a different camera frame. At most 16 pixels are
returned per query, with the same limit for both arms. Only the latest displayed
frame per camera is retained. Invalid depth or out-of-frame pixels have no point
value. A frame queried after sensor time has advanced is explicitly historical;
the service does not promote its points into a current object pose. A new image
can supply a new frame identifier. Query counts, processing times and responses
are recorded separately from model-token accounting.

Seven tests cover calibrated backprojection, validity masks, invalid values,
historical timestamps, view-specific replacement, copy isolation, exclusion of
object/evaluator fields, calibration validation, and the no-recapture integration.
An additional replay verified eight initial development frames and 128 sampled
pixels from `outputs/online_reasoning_dev/pairs_01`. The shown PNGs match archived
RGB, both arms receive identical query responses, and depth values are not
quantized. Reprojection through the supplied calibration agrees within 1e-9
pixels. This is algebraic/sensor-access consistency, not proof of physical accuracy
or the LLM's ability to select useful pixels. The report and source snapshots are
under `outputs/online_metric_depth_dev`. No model calls or simulation rollouts were
made for these checks.

The shared depth-query action, response handling, Full reference feedback and
explicit placement parameters are now connected in the development loop below.
They passed mock-actor integration tests and the later live development check
below. No reserved evaluation is registered for this interface.


## Connected development decision loop

`online_bound_actor.py` defines a separate structured decision schema for
`execute`, `sample_depth`, `observe`, `release`, and `wait`. Execution accepts a
skill, node and world surface point; depth queries accept a frame ID and pixels.
Inactive fields must be empty. The persistent model transport reuses the frozen
actor method's bytecode with separate schema/validator globals. It does not patch
the original module or change earlier actors. Existing cumulative token receipts,
image archives, deadline enforcement and physics callbacks remain in that shared
transport; the live smoke check below also audits their recorded behavior.

`BoundDecisionLoop` in `online_bound_decisions.py` connects these actions inside an
already-started episode. It never resets the scene. It uses Astra medium for both
arms, maintains one persistent actor context, and asks for a new decision after
each completed action. Each request receives the remaining wall-clock budget.
There is no node-only queued plan in this interface.

Both arms receive initial side/wrist images and fresh images on `observe`. Base
normally receives a side image after other actions. A depth-query response uses
the previously displayed frame and does not trigger image recapture just to
return its samples. Full receives current center/top/bottom references with their
source and timestamp. Positionally underconstrained references are unavailable;
dense face boundaries are omitted from the default feedback. No computed task
destination or goal residual is supplied by this new reference packet.

`ExplicitSpatialSkills` uses observed scale for shared local tolerances but omits
automatic worksite selection: the actor now chooses placement points. This is a
change to the interface's assistance level, not a repair of the stopped ordinary
study. Relocating an obstruction uses an explicit destination through the same
placement action, rather than an automatically chosen clearing location. Both
arms have identical relocation and release options. The prior task-independent
node-only skill API remains available outside this development loop.

Four integration tests passed without invoking a model or actuator. They verify
schema isolation, invalid cross-action parameters, partial-reference rejection,
and a mocked `image -> sample_depth -> execute` sequence for both arms. Both arms
submit the same sampled coordinate; Base receives no object-reference geometry,
and extra private debug fields from a fake executor are not forwarded. The tests
also verify that delivering depth-query results makes no extra image capture.
These tests complement, and do not replace, the previous seven target-adapter and
seven metric-depth tests.

No new reserved layout, episode budget or model experiment is registered here.
The environment-owning runner and live checks are described below. A passed mock
decision chain alone is not evidence of a geometric-text reasoning advantage.


## Physical and live development checks

The owning runner is `scripts/run_online_bound_development.py`. It accepts only
existing development seeds 2028100101–2028100103, refuses an existing output
directory, caps the episode budget at 180 seconds, resets exactly once, and saves
source snapshots plus private terminal physics. Both arms use the same actor
model and physical executor. The runner records model receipts, camera counts,
feedback preparation time, query counts/time, action time and incomplete requests.

An accelerated physical check on seed 2028100101 used the shared visual driver to
submit explicit targets through the new adapter. It completed the full structure
with 119.20 simulated seconds, about 9.48 wall seconds, and zero model requests.
This is actuator/interface feasibility evidence; the scripted driver computed its
own targets and does not demonstrate LLM reasoning or real-time speed. Artifacts:
`outputs/online_bound_dev/physical_01`.

A single paired live check then used the same existing development seed, two
parallel workers, and a 180-second budget per arm. There was no injected
intervention, reset after failure, retry or budget extension. Its pre-dispatch
settings are in `outputs/online_bound_dev/pairs_01/plan.json`.

| Measurement | Base | Full |
| --- | ---: | ---: |
| Outcome | Timeout | Success |
| Capped completion time | 180.00 s | 156.07 s |
| Completed model requests | 7 | 6 |
| Submitted placement targets | 5 | 4 |
| Metric-depth queries / pixels | 2 / 13 | 1 / 4 |
| Attached images | 6 | 4 |
| Input tokens, including cached input | 96,591 | 106,221 |
| Cached input tokens | 65,664 | 83,328 |
| Output tokens | 645 | 461 |
| Total completed-request tokens | 97,236 | 106,682 |
| Cumulative completed-request wall time | 48.72 s | 35.89 s |
| Camera acquisitions | 99 | 124 |
| Feedback preparation time | 0.10 s | 0.49 s |
| Maximum simulation lag | 0.61 s | 0.58 s |
| Incomplete model requests | 0 | 0 |

Both episodes began from the same physics hash and retained exactly one reset.
Full used geometric references to retain the initial green foundation and place
the other foundation relative to it. Base chose a new origin and submitted a move
for green as well. Full also requested images and sampled depth when a geometric
reference was unavailable. Thus this case exercised both reference-based binding
and the common fallback sensor interface. It did not give Full a precomputed
placement destination.

Full reached verified success sooner in this development case, but used about
9.7% more total tokens and more local camera acquisitions. It had fewer image
attachments, less completed-request wall time, and fewer submitted placements.
The completion-time difference combines decisions about reusing existing work,
model requests and physical actions; it is not an isolated measurement of text
inference latency. Uncached input counts are 30,927 for Base and 22,893 for Full,
but neither count alone establishes monetary cost or end-to-end efficiency.

The independent source/resource audit passed. The additional
`audit_online_bound_development.py` check matched each submitted target to the
actual model response and each metric query to the displayed RGB-D archive,
confirmed Base received no object-reference geometry, and reconstructed terminal
contacts. Full's terminal structure had no robot/object contact; Base still had
robot contact with the blue cube at timeout. Full's original private monitor
recorded five seconds of stability with about 0.57 mm final drift. Terminal replay
does not independently reconstruct the entire stability interval. Reports are
`audit.json` and `interface_audit.json` in the paired output directory.

This is one previously used development layout, not a fresh five-layout comparison.
It is an encouraging mechanism check, not proof of a general geometric-abstraction
advantage. Earlier negative studies remain valid under their different interfaces.
Any reserved comparison must be separately frozen and retain all five shared
layouts, equal budgets, failed attempts and unfavorable resource measurements.

## Optional compact feedback

The [compact geometric text extension](ONLINE_COMPACT_GEOMETRY_TEXT.md) factors
repeated reference metadata without changing the geometric API or action schema.
Development replay reduced Full prompt bytes by 21.1%, with exact reconstruction
and unchanged Base prompts. Seven focused and sixteen existing tests passed.
The runner exposes `--compact-text` as an opt-in at the current 200-second budget;
a subsequent live Raw-Full/Compact-Full pair succeeded in both conditions, with
6.7% fewer compact tokens but no time improvement. See the linked record. Preserve earlier
results and the separate completed reserved study.
