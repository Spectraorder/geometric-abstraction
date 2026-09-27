# Conditional geometric intervals for partially visible rigid objects

This development prototype adds information about unseen center coordinates
without declaring them directly observed. It is separate from the frozen v2
policy comparison. It has not been used to select or modify any of those policies.

For visible points belonging to a rigid box with measured orientation and side
lengths, transform the points into the box's orientation basis. Along each axis:

```
max(visible_points) - half_width - tolerance <= center
center <= min(visible_points) + half_width + tolerance
```

Every visible point must lie inside the rigid object. Occlusion widens the set of
possible centers; observing opposite extents tightens it. The prototype adds
1.5 mm tolerance for point, dimension and orientation error. These are conditional
geometric intervals, not calibrated confidence intervals. Incorrect association,
wrong shape/orientation or larger measurement error can invalidate them. Empty
intervals report inconsistent evidence. Unobserved orientation causes abstention.

This differs from treating an unconstrained translation direction as a 30 mm
Gaussian uncertainty. Plane observability remains unchanged. The new bounds can
support a symbolic height-order query even when neither object's vertical center
is directly fixed by a visible horizontal face. `above` means center-height
order only; it does not certify physical support, contact or stability.

## Development evidence

All sensor-only estimates were saved before private physics was read for scoring.
The prototype was then evaluated unchanged on a second set containing only failed
rollouts. Both sets are already inspected training trajectories; neither is a
reserved generalization test.

| Diagnostic set | Checkpoints with bounds / all | True center in bounds | Partial states with bounds | Median largest interval width, partial |
| --- | --- | --- | --- | --- |
| Repetition 2, symbolic R1, five successful rollouts | 117/122 | 117/117 | 46/48 | 7.04 mm |
| Repetition 1, symbolic R1, five failed rollouts | 176/195 | 176/176 | 55/63 | 7.12 mm |

The existing point estimates were already good. On partial states, replacing them
with interval midpoints would worsen p95 center error from 3.06 to 3.20 mm in the
first set and from 2.85 to 3.24 mm in the second. Therefore this is an uncertainty
representation proposal, not a replacement point estimator.

Across the same 208 directed height-order queries, the current representation
returned `unknown` 141 times; interval reasoning returned `unknown` 69 times.
This measures information availability under the stated assumptions. It is not
an SR gain, a statistical independence claim, or an accuracy measurement on real
depth cameras. All 293 emitted development intervals covered their true centers,
but these correlated simulated examples do not establish universal coverage.

## Implementation and next boundary

- `scripts/rigid_containment_prototype.py`: action-free box intervals and
  three-valued height-order queries; accepts points, measured orientation and
  dimensions only.
- `scripts/audit_rigid_containment.py`: frozen RGB-D library replay, followed by
  separate private scoring. Sensor-only output and source snapshots are retained
  in `outputs/rigid_containment_audit_01` and `_02`.
- `scripts/summarize_rigid_containment.py`: reproducible information-availability
  comparison over the saved sensor-only estimates.
- Four tests cover rotated partial surfaces, tightening with additional evidence,
  incompatible points and height-order reasoning.

[Measured diagnostic](evaluations/rigid_containment_diagnostic.json) retains
coverage, point-error and relation-transition counts. A future state interface
could expose these bounds alongside metric estimates, preserving the difference
between observed constraints and model-conditioned constraints. It needs tests
with noisy depth, segmentation contamination and uncertain rotation before
promotion. No temporal memory or attachment assumption is introduced here.
