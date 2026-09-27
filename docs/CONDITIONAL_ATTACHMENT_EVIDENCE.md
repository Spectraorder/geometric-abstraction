# Conditional held-point memory: development evidence

A single-object motion hypothesis helps distinguish two recorded failure modes,
but it has not recovered an occluded pose or improved policy success. This replay
uses the five training layouts from the separate guard-threshold counterfactual.
It reads action-boundary visual estimates, public robot proprioception, and RGB-D.
It does not read rollout state, object truth, contacts, or success labels.

After a visually confirmed lift with a fully observed center, retain one center
and its robot pose. For later robot pose `(p1, R1)`, predict
`c1 = p1 + R1 @ R0.T @ (c0 - p0)`, conditional on rigid attachment without slip.
A gripper command or changed focus invalidates the anchor. Missing robot poses or
an unconfirmed/partially observed initial center cannot establish an anchor.
The prediction stays separate from current observations.

## Recorded results

There are two eligible failed placement boundaries. Other cases either release
normally, fail before confirmed lifting, or have no action-boundary geometry.

| Training layout | Current visual center | Conditional prediction evidence | Interpretation |
|---|---|---|---|
| 990202, blue thin prism | Fully observed | Center disagreement 99.4 mm; all 101 current object depth samples lie outside the predicted enclosing sphere plus 5 mm margin | Current geometry contradicts rigid attachment; do not fill the state with the predicted center |
| 990205, red thin prism | Unavailable | All 74 side and 4,455 wrist depth samples remain inside the conservative predicted envelope | Hypothesis is not contradicted, but the center remains unknown; no pose recovery or attachment certification |

The first case's measured center is approximately `[-0.011, -0.248, 0.437] m`,
whereas the rigid-motion prediction is `[-0.013, -0.251, 0.537] m`. Its existing
visual placement error is only about `[-0.2, -5.1, +0.4] mm` in the controller's
center-XY/minimum-Z reference. Thus a large attachment discrepancy and proximity
to the desired placement can coexist. Neither establishes stable support.

The second prediction is approximately `[-0.080, -0.297, 0.470] m`, explicitly
conditional. Agreement with an enclosing sphere is a necessary consistency check,
not enough information to localize the body. The sphere uses dimensions measured
at the confirmed-lift checkpoint, not catalog sizes. The 5 mm margin is a diagnostic
tolerance, not a calibrated uncertainty bound. Green depth checks remain unknown
because this replay does not remove the same-color robot marker.

## Validity checks and a failed shortcut

Four focused tests pass. They check that current depth can contradict a prediction
without certifying attachment; missing depth, background and unmasked green remain
unknown; current missing poses are not replaced by observations; and release,
changed focus, absent current robot pose or an unconfirmed/unobserved anchor prevent
replay predictions.

The first draft's raw blue color mask included distant background. That diagnostic
was withdrawn and preserved with its source. The corrected version uses the same
4–350 mm tabletop work-volume filter as the shared visual observer, with the table
plane measured from initial RGB-D. It does not crop around the predicted center,
which would hide contradictory object evidence. The independent 99.4 mm visual
center disagreement is unaffected by this mask correction.

An attempted replay using only the initial dimension model did not reproduce the
later occluded-fit rejection. A separate logging-only rerun now captures the exact
visual model, visual table plane, and public robot geometry supplied to each fit.
All camera archives, the opaque physics archive, and prior geometry records are
byte/content identical. Nonlogging API calls are identical after excluding only
terminal CPU-profile metadata. The first strict audit rejected that naturally
variable profiling field; its original driver is preserved, and the revised audit
explicitly names the exclusion. No sensing, action or physical-state comparison
was relaxed.

## Exact visual-input replay

At training layout 990205, time 75.9 s, replaying the exact model and masked
side/wrist RGB-D reproduces the original rejection. A four-way diagnostic gives:

| Visual dimension model | Robot mask | Observer result |
|---|---|---|
| Current, after two refinements | Applied | Rejected: hidden interval too wide |
| Current, after two refinements | Omitted | Same rejection and finite bounds |
| Initial visual dimensions | Applied | Accepted, explicitly not fully observed |
| Initial visual dimensions | Omitted | Same accepted estimate |

The initial measured planar dimensions were 58.11 × 29.92 mm. The current model
has grown to 60.68 × 33.02 mm. With this model, two current face directions are
constrained and the remaining center interval is 1.973 mm wide, exceeding the
unchanged 1.5 mm threshold. This isolates the dimension-model contribution in this
case; it does not establish which dimensions or accepted center are more accurate.
No catalog size, object pose or success label enters the replay. In particular,
reverting dimensions merely to pass a narrow-interval gate is not a justified fix.

A diagnostic call exposes the rejected bounds without accepting a live pose. The
rigid-motion predicted center falls outside them by 2.834 mm along one constrained
axis and 0.307 mm along the hidden axis. Thus the earlier enclosing-sphere check
was too coarse to establish compatibility with the tighter visual constraints.
These finite bounds depend on fitted orientation and learned dimensions and are
not calibrated confidence intervals. The discrepancy neither proves slip nor
justifies fusing the predicted point into an observed pose.

## Design implication

A lightweight state should keep three distinct items for the active object:
current visual geometry (possibly unknown), a conditional motion prediction, and
the desired placement. Their discrepancy can inform reasoning without another
image interpretation. A non-contradicted prediction must never become a measured
pose or a confirmed grasp solely because it is plausible. No new all-scene memory,
controller change, or learned-policy SR gain has been demonstrated here. The
development interface now exposes partial geometric constraints when a single pose
is unavailable, with visual dimension provenance. A separate
[constraint-feedback prototype](PARTIAL_GEOMETRY_LEARNING_PROTOCOL.md) passes five
trajectory-equivalence checks and adds one 903-character failed-fit checkpoint.
Prediction disagreement remains an offline diagnostic. No controller recovery or
success-rate gain follows from these interface checks; fresh learning evaluation
is registered separately.

## Artifacts

- Replay: `scripts/replay_attachment_evidence.py`.
- Exact observer replay: `scripts/replay_observer_inputs.py` and
  [four-way evidence](evaluations/observer_input_failure_replay.json).
- Logging rerun and equivalence audit: `scripts/record_observer_inputs.py`, with
  preserved artifacts under `outputs/observer_input_logging_development_01`.
- Tests: `tests/test_attachment_evidence.py` (four passing tests).
- [Complete replay evidence](evaluations/conditional_attachment_failure_replay.json).
- The withdrawn first draft is
  `docs/evaluations/conditional_attachment_failure_replay_unfiltered.json`, with
  its source snapshot under `outputs/visual_guard_threshold_development_01`.
- The unsuccessful initial-model inspection is retained as
  `outputs/visual_guard_threshold_development_01/inspect_occluded_bounds.py`.

These diagnostics are not actor feedback and do not modify the completed
[learning study](VISUAL_INPUT_LEARNING_RESULTS.md).

## Coverage in fresh completed training traces

The unchanged replay was applied, without success labels or actor feedback, to
chain 1 Full R1 and R2 of the new constraint-feedback experiment (eight layouts
per revision). R1 has 20 visually confirmed-lift checkpoints, of which nine have
fully observed centers and eleven do not. R2 has 19, with eleven fully observed
and eight partial or missing. Normal release invalidates a transport anchor.

There are five eligible later placement boundaries, all retries from just two
layout/revision cases. The three R1 boundaries involve green and remain unknown
in the raw-depth check because its robot marker is not masked. The two R2
boundaries are not contradicted by the enclosing sphere, but neither has a fully
observed current center. No pose is recovered. Counts are in
[the coverage receipt](evaluations/constraint_attachment_coverage.json).

This limits the coverage of point-anchor transport on these traces. It does not
justify lowering observation requirements or treating an inferred point as a
measurement. Propagating a conditional center region is a future design direction;
it is not implemented or validated by these point-replay results. These analyses
never enter the running actors' feedback and do not change the frozen experiment.

## Conditional center-region development replay

A separate logging-only rerun of chain 1 Full R2, training seed `992107`,
retains the successful observer's full visual fit. The frozen candidate is
unchanged. All 82 camera archives, the opaque physics archive, original geometry
records, and nonlogging API calls match the parent trial exactly (excluding only
terminal CPU-profile metadata). The receipt is
`outputs/partial_anchor_logging_development_01/equivalence.json`.

The yellow thin prism's visually confirmed lift at 79.05 s is only partially
observed, so the existing point-memory rule cannot anchor it. Its visual cuboid
fit nevertheless contains finite center intervals with widths 1.703, 0.251, and
0.687 mm in the fitted object basis. These are conditional geometric bounds, not
calibrated confidence intervals. The accepted fit has two face-constrained axes
and one inferred axial position.

`scripts/replay_center_regions.py` transports this entire region using public
robot poses, conditional on rigid attachment without slip. It compares the
transported region with current visual constraints at the two failed placement
boundaries. Neither pair of regions overlaps. The minimum uniform outward
expansion of both regions' half-space bounds is 26.579–26.586 mm at 85.75 s and
26.504–26.512 mm at 86.0 s. These values are **not** Euclidean set distances.
The ranges cover every combination of recorded same-time public pose queries;
small differences between repeated queries are retained rather than silently
selecting one. The second comparison independently reuses the original hypothesis
for diagnosis; it is not a tracker accepting a previously contradicted hypothesis.

The contradiction is conditional on the visual dimensions/orientations and the
attachment assumption. It cannot identify slip separately from visual-model
error. The visual length model also changes between the two fits. No estimate is
fused, no pose or grasp is certified, and no control action changes. This shows
that a region can represent a diagnostic case excluded by the fully observed
point rule; it does not demonstrate improved policy success. The replay reads
only visual-fit logs and public proprioception, with no object truth, contacts,
or evaluator labels. It never enters the running actors' feedback.

[Machine-readable evidence](evaluations/conditional_center_region_replay.json)
includes source hashes and limitations. Three `unittest` checks pass for rigid
transport of all corners, overlap/separation, and coordinate-frame invariance.
This isolated offline implementation extends the earlier point-only coverage
analysis; it is not integrated into the frozen geometry interface.
