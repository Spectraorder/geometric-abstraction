# Surface projection with a shared geometric observer

The development interface in `scripts/surface_projection_prototype.py` provides
a stronger minimal baseline for testing the supplied rigid abstraction. It wraps
the same `MixedSession` used by the numeric arm, projecting only public results.
It does not replace the existing running experiment or its registered contrasts.

The original mixed-shape pilot's basic arm used an older visible-top observer.
Its difference from numeric therefore included perception and execution effects.
The current numeric/symbolic replication instead tests the extra symbolic layer;
both already have a rich numerical abstraction. Neither contrast alone isolates
all of the benefit claimed for giving an agent rigid geometric state.

The proposed matched contrast is:

| Interface | Public information | Observer and executor |
| --- | --- | --- |
| Shared surface | Center, normal, height, 2D dimensions, top/bottom/extent, freshness and coarse observability | Same mixed RGB-D observer and PnP |
| Numeric rigid | Above plus rigid axes, dimensions, vertices, geometric state/history and desired poses | Same |
| Symbolic rigid | Same numeric information plus typed references, relations and desired transforms | Same |

This is a richer-abstraction versus minimal-surface comparison, not geometry versus
no geometry. All arms may still use the same raw RGB-D and robot measurements.
The surface arm retains `position_fully_observed` and `inferred_orientation`; it
never advertises an inferred center as fully measured merely because axes are hidden.
The estimator's internal dimension model is not returned by the surface interface.

Direct observations and action-result observations use the same projection.
State packets retain available/stale/unavailable status, timestamps, reasons,
center, normal and vertical extents. As in the current numeric packet, they do not
repeat every field of `observe()`. The surface arm does not expose rigid history,
vertices, relations or desired-pose operators. This compares the supplied state
package; further component ablations would be needed to identify which operator
causes any benefit.

The wrapper delegates every sensing and action call to the same backend exactly
once. It does not choose a worksite, change source order, recover failures or relax
release checks. Both arms must use the same
[explicit initial context](INITIAL_OBSERVATION_CONTEXT.md), bound by the host.
A future worker validator must continue to block actor access to `backend` and
other trusted internals, as the existing candidate validator does.

## Checks

Three projection tests check real recorded RGB-D agreement, identical public shared
fields, copy isolation and stale-state handling; unchanged action arguments and
filtering of nested observations; and worker-validator compatibility of the initial
context prototype. The initial-context suite adds three tests for descriptor
invariance, independent goal orientations and numerical parity between arms.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s tests -p test_surface_projection_prototype.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s tests -p test_observation_context_prototype.py -v
```

A separate development parity run uses the same support-graph starter in both arms
on already-inspected layouts 75103 and 75105. The fixture's public initial assumptions
are bound independently of goal orientation. It verifies identical action commands,
full saved physical arrays, sensor call counts and outcomes. Physical arrays are
read only after execution for the offline audit, never by the policy or estimator.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/run_shared_surface_parity.py \
  --output outputs/shared_surface_parity_01
```

The completed development check passed on both layouts: all four rollouts succeeded,
with bit-identical saved physical arrays and identical action commands within each
pair. Camera counts match at 67 per arm for seed 75103 and 63 for seed 75105.
The [saved parity report](evaluations/shared_surface_parity_01.json) contains input
hashes. This verifies matched starting execution, not an abstraction learning benefit.

The command requires a new output directory and does not overwrite prior evidence.
Inspect `parity.json` for completion; a launched run is not a passed check. These
two layouts do not measure learning, generalization or the advantage of an evolved
policy. A later learning study still needs frozen fresh splits, matched actors and
feedback, all-revision evaluation, and independent generation chains.
