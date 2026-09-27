> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Current-surface cuboid correspondence

## Observed regression

The fixed mixed-observer bridge candidate failed on development seed 75105 while
the archived basic candidate succeeded. At simulation time 69.35 s, the lifted
beam was rotated substantially. The low-level cuboid fit assigned stored lengths
to the nearest world axes. It then compared an observed span requiring about
55.8 mm with the initial 14.89 mm thickness and reported a dimension conflict.
The wrist view alone had only one usable face direction and could not determine
the complete pose. The two individual failures were preserved.

This is an orientation-to-dimension correspondence issue with partial visibility,
not evidence that the object changed its physical dimensions. The initial model
was measured from RGB-D: approximately 58.07 by 29.14 by 14.89 mm. The first two
measurements also carry independently estimated pixel uncertainty; no catalog
object dimensions are used by the new fitter.

## Action-free repair

`cuboid_hypotheses.py` tests all six assignments of measured lengths to the current
fitted face basis. At least one view must establish two independent face normals.
The candidate must satisfy the current face-plane locations and contain the
visible surface within the same finite-bound tolerance. Equal-length axis swaps
are equivalent, and unresolved distinct assignments remain unavailable.

Opposite-facing planes are kept separate when combining cameras. Finite boundary
trimming is performed within each camera before combining bounds: the recorded
wrist cloud has 2,318 points while the side cloud has only 48. Trimming the pooled
cloud can erase the far edge seen only by the sparse side camera. Camera sampling
density must not discard that independent geometric constraint.

Horizontal dimensions can increase only within the initial pixel-derived bounds
when current planes or silhouettes require it. The originally measured height
remains fixed. A missing translation direction must have a narrow finite interval
before the high-level observer accepts the unique hypothesis. The interval is
conditional on association, fitted orientation and model sizes, not a calibrated
probability or an unconditional physical bound.

The returned structure retains the assignment, inferred vertices, constrained
axes, finite position intervals, observation time and contributing views. It
explicitly states that no temporal pose prior was used. A world-up geometric face
is not an original material face identity. No goal, task order, robot command or
simulator object state enters the geometry solver.

`MixedPnPObservation(..., correspondence=True)` enables this recovery after the
individual current-view fits fail. It is opt-in; the historical default and all
archived candidates remain unchanged. The worker exporter includes this numerical
module, with no specialized task controller.

## Evidence and limits

The failing RGB-D frames now produce one compatible assignment, `[1, 0, 2]` in
the fitted basis, rather than the incorrect nearest-world-axis assignment.
The unobserved direction has a conditional interval width of about 0.713 mm.
That coordinate remains marked as inferred; it is not promoted to a directly
observed face-plane coordinate.

The sensor-only estimate and implementation were saved to
`outputs/cuboid_correspondence_diagnostic_01` before private recorded-state scoring.
Offline scoring gives a center error of **1.014 mm** and a long-axis direction
consistent with the actual recorded beam. That score is one development checkpoint,
not a policy result, real-camera accuracy guarantee or evidence of calibrated
interval coverage. The independent physical rerun of seed 75105 succeeds with
no policy error in `outputs/mixed_pnp_correspondence_development_01`; the shared
PnP motion code and manual planner are unchanged.

Four focused tests cover large rotations with noncatalog lengths, a dense camera
not erasing a sparse camera's far face, genuinely ambiguous small adjacent plane
patches, and the actual RGB-D regression including stale-view rejection. The six
shared-observer interface tests also pass. The recorded fixture contains only
RGB-D, calibration and robot-marker exclusion based on robot model/proprioception,
with no object truth.

The complete eight-layout regression in
`outputs/mixed_pnp_correspondence_development_02` finished at **7/8**, compared with
6/8 before this repair. Seed 75105 is recovered; all six previous successes remain
successes, and the foundation placement failure on 75101 remains a failure. The
same candidate was used for every repaired trial. These are already inspected
scenes after targeted development, not a fresh reserved split or an unbiased new
87.5% generalization estimate. The
[regression audit](../evaluations/cuboid_correspondence_regression.json) checks source
hashes, initial scenes, sensor/physics settings and recorded trajectories.

This repair and all development runs do not demonstrate that a symbolic interface
improves agent iteration efficiency. The numerical and symbolic arms must share
this perception implementation in the next controlled comparison. No further
manual controller change was needed to resolve this beam regression.
