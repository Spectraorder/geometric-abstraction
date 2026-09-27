# Common numeric state and symbolic geometry interface

## Boundary

`MetricScene` records the exact selected mixed RGB-D estimates. `SymbolicScene`
adds references, relations and desired-transform calculations over that same
ledger. Neither performs sensing, task planning, object selection or robot action.
`MixedSession` connects the shared observer, robot visual mask, camera acquisition
and unchanged PnP executor; `SymbolicSession` inherits that integration and adds
only symbolic queries. Both are exported by `scripts/build_mixed_scene.py`.

The current rigid shapes are cubes, rectangular prisms, thin prisms and cylinders.
Colors uniquely identify instances. Shapes and initial orientation hints are
public, while dimensions are learned from initial on-table RGB-D. Triangular-prism
pose estimation is not implemented in this interface.

## Four separate kinds of state

- Current observations carry world coordinates in meters, observation time,
  fitted rotation, axis dimensions, evidence flags and source view. Model-inferred
  vertices are not individually observed points. Finite center bounds use the
  fitted local basis (`center @ rotation`), with explicitly labeled coordinates;
  their validity is conditional on the model and association.
- History retains the last two observations and differences between estimates.
  It is not temporal pose fusion or certified physical displacement. Missing
  observations and observations from an older timestamp cannot be returned as
  current poses. Live sessions check the robot clock even if the caller moved or
  waited without taking another image.
- Desired states are separate caller-selected centers/rotations. New observations
  do not move a target. Desired transforms report remaining unconstrained twist
  and do not certify execution.
- Conditional predictions require explicit opt-in and an unverified condition.
  They record a supplied rigid transform and its source observation, never replace
  observations, and can report a same-time reobservation residual. Explicitly
  labeled predictions and goals are rejected at the observation-ingestion boundary.
  Live experimental sessions keep predictions disabled. Previous negative temporal
  diagnostics remain relevant; this interface does not establish predictive accuracy.

Returned arrays are copied. JSON-compatible packets have finite metric values for
validated observations. `snapshot()` explicitly refreshes objects; `state()` only
reports their current availability. Initial dimension learning must finish before
any task motion. A later unavailable model cannot be initialized in midair.

## Symbolic operators

`reference('object.center')`, `top_face`, `bottom_face`, `long_axis`, `short_axis`
and box `corner_0..7` resolve current geometric features. Long/short axes require
unique lengths; a cube has no unique long axis. These labels do not recover
invisible material point identities across cube symmetries. Rotated box faces use
the current fitted axis length, not the model's initial vertical height.

Cylinders expose an unoriented `axis` and end-cap disks. Rims are model samples,
not physical corners. An upward-facing end cap is not necessarily the topmost
curved surface of a horizontal cylinder.

`relation(predicate, first, second)` supports `above`, `aligned_xy`,
`support_candidate` and `contact`. Results include numeric residuals and
`true / false / unknown` about estimated geometry. A known-false component rejects
a compound support candidate even if another component is unobserved. An unknown
component is not automatically a failure. Support is only a geometric candidate;
contact always remains unknown without a contact sensor. Positive overlap and a
small surface gap do not certify load balance or physical stability.

`align(source_face_reference, target_point, target_normal)` solves the generic
point-and-directed-axis desired transform. The target normal is the desired
outward normal of the source face. A caller mating a bottom face to a top face
chooses opposing normals. The operator does not choose a supporting object,
construction order, world location, action or recovery. A single axis leaves
one rotational degree of freedom; the returned representative explicitly records
that fact. The caller may record the resulting desired center and rotation with
`target`, separately from the observed state.

## Validation and actual experiment packets

Ten tests cover stale/missing poses, copy isolation, fixed targets, separation of
conditional prediction and observation, disagreement on reobservation, rotated
cuboid face locations, cylinder/cube reference semantics, conservative relations,
desired transformations, finite-landmark rejection, identical numerical exports, and robot-clock freshness
without camera updates.

The archived live integration uses two inspected development layouts (75103 and
75105), each with the same manual planner under numeric and symbolic interfaces.
All four executions succeed. Every recorded physics frame is identical between
the two interfaces for each layout. At all five checkpoints per rollout, the
numeric projection of the symbolic packet equals the numeric packet exactly.
Each pair also has identical initial images, scene metadata, physics settings and
camera counts (82 and 79 respectively). This is an interface-parity check, not
success-rate or policy-improvement evidence.

Reproduce the archived-source audit with:

```bash
.venv/bin/python scripts/audit_mixed_scene_integration.py \
  --root outputs/mixed_scene_integration_01 \
  --output docs/evaluations/mixed_scene_integration.json
```

See [the parity results](evaluations/mixed_scene_integration.json) and
[a real final symbolic packet](evaluations/mixed_symbolic_scene_example.json).
The packet is from the archived integration implementation. Subsequent refinements
to three-valued conjunction, provenance guards and explanatory metadata are tested
separately and do not rewrite recorded evidence. The manual planner and its
completed solutions remain excluded from experimental actor contexts.

## Remaining causal experiment

Numerical and symbolic agents must share this same observation and execution path,
including current-view correspondence, wrist acquisition and robot masking.
Differences in geometric accuracy must not be credited to symbolic reasoning.
The existing basic RGB-D PnP comparison measures a broader perception/package
contrast and must be reported separately from the representation contrast.

Freeze the final interfaces, starter policy, budgets, feedback and unused scene
splits before agent revisions. Measure full learning curves and selected-policy
reserved success; retain missing-observation failures and invalid submissions.
The completed cube comparisons did not establish fewer iterations, and the manual
bridge development results are not a substitute for that requested evidence.

## Canonical public observations

A live experiment exposed a schema mismatch: cylinder fits could omit
`axis_dimensions` from `observe` even though `get` and state packets included it,
contrary to the published API. The shared callback now returns the validated
`MetricScene.observe` record. Public `observe` and `get` therefore expose the same
canonical shape fields and copied arrays. There is no additional pose estimation
or motion logic in this repair. A recorded RGB-D test covers both session types
on two cubes, a flat beam and a cylinder. Physical regression and the impact on
experimental validity are recorded in the [learning protocol](MIXED_SCENE_LEARNING_PROTOCOL.md).
