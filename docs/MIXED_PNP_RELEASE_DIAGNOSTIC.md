# Mixed-shape PnP development diagnostic

## Boundary and original outcomes

The unchanged generic RGB-D PnP executor was paired with a manual support-graph
planner solely to assess execution limits before the next agentic comparison.
The planner and its solutions remain excluded from experimental actor contexts.
The fixed candidate, hash `8186c330acea08f85d37132bf2eced511146407cbd15b4f30ad3bdd9ffc3e3f1`,
was run on preselected development seeds 75001–75004. Original success was **1/4**.
All four nodes were released in the first three trials; the fourth stopped with
its first foundation still held after placement retries. All policy errors were
null. The separate earlier seed 3601 development run succeeded after a logging-only
serialization repair; it is not included in the four-scene result.

Private replay shows that seed 75002 fails only the cylinder's relative XY check;
seed 75003 fails relative XY checks for a foundation, the beam and the cylinder.
Both final structures meet the other orientation, contact and load-balance checks.
This does not relax the task's original 6 mm relative-position threshold or turn
these two outcomes into successes.

## Sensor replay before private scoring

`scripts/audit_mixed_pnp_release.py` reconstructs the exact final pre-release RGB-D
observation and frozen basic estimator for every completed release. It also fits
the existing mixed rigid observer to the same frame, learning dimensions from the
initial RGB-D only. All 12 replayed basic XY residuals match their recorded values
within 1e-10 m. Both sets of estimates are saved before the scorer opens recorded
physics. The scorer then compares the object center at the observation timestamp
and at the stage's post-release/retreat timestamp.

The script uses the public `thin_beam_bridge` goal and unique colors. It never
executes a new policy or feeds truth into an observer. The first diagnostic attempt
failed while parsing the task log before producing estimates; its error and source
are retained in `outputs/mixed_pnp_release_audit_01`. The completed diagnostic is
`outputs/mixed_pnp_release_audit_02`, with source snapshots and provenance hashes.

| Seed | Beam basic center error (mm) | Beam mixed center error (mm) | Beam true XY error before release (mm) | Cylinder XY movement during release/retreat (mm) |
| --- | ---: | ---: | ---: | ---: |
| 75001 | 4.614 | 0.383 | 4.409 | 6.426 |
| 75002 | 6.929 | 0.602 | 8.124 | 10.044 |
| 75003 | 12.234 | 0.779 | 12.217 | 8.957 |

The release gate believed each beam was within 4 mm of its target. Its visible-top
center can be biased under the gripper, while the existing rigid fit is much more
accurate on these three frames. However, improved target estimates alone do not
remove the cylinder movement after opening the fingers. The interval includes
opening, waiting and retreat; these observations do not identify a single contact
mechanism. Seed 75004 has no completed release and is explicitly absent from this
12-observation diagnostic, while remaining a failed trial in the original 1/4.

These are counterfactual perception measurements on correlated development frames,
not success-rate evidence for geometry, symbolic reasoning or state memory.

## Evaluator consistency repair

The bridge evaluator still required a cube's local positive Z face to remain up.
Unlike the corrected column evaluator, its private support polygon could also use
the wrong face after a cube-equivalent rotation. `InventoryStructureEnv` now picks
the cube face closest to world up, orients its normal upward and uses that face's
vertices for support geometry. Noncube orientation constraints are unchanged.

The old evaluator source and all four original final-state diagnostics were saved
in the development directory before editing. Corrected final-state pass/fail is
unchanged for all four recordings; only the tilted-cube explanation in seed 75004
changes. This is not a retrospective rewrite of original results or a new
five-second stability evaluation.

Validation: all five cube-symmetry tests pass, including 24 equivalent rotations
for both evaluators, a physical bridge with rotated cube foundations, rejection
of actual tilt, and rejection of incorrect beam/cylinder orientations. The existing
support-graph suite also passes.

## Consequence for the next abstraction experiment

Keep the symbolic-plus-metric representation decision separate from perception
improvements. A common execution interface must allow every experimental arm to
use its available pose estimate at the same grasp and release checkpoints. The
current executor always calls its internal basic observer, so supplying a better
scene packet only to high-level planning cannot repair this release-center bias.
Do not hide a stronger observer or recovery controller in the symbolic arm.

Next, validate a shared observer interface and record post-release state before
choosing whether a generic recovery operation is needed. Numeric-state and
symbolic-state arms must share the estimator and execution loop. Keep current
estimates, conditional predictions, desired transforms and observed post-action
changes distinct. Retain cube stacking as a control and reserve fresh scenes only
after these development issues are resolved.

The archived four-scene candidate and its executor remain unchanged.
