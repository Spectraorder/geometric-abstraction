# A reusable RGB-D pick-and-place base

The historical `stack_three_blocks_cubes_4cm.py` is preserved unchanged. Its
visual top-surface estimator and basic robot sequence have been extracted into
[`pick_place_rgbd_base.py`](../policies/pick_place_rgbd_base.py).

The mechanism does not read `api.task()`, select object order, choose a worksite,
plan a tower, or decide which object should support another. A separate
[26-line single-column adapter](../policies/stack_from_rgbd_pnp_adapter.py) reads
the public goal and supplies placement requests. This separation makes the
common starting capability explicit for the geometry comparison.

## Interface

```python
pnp = RGBDPickPlace(api)
source = pnp.observe(source_color)
support = pnp.observe(support_color)
surface_point = np.r_[support['center'][:2], support['top']]
result = pnp.pick_and_place(source_color, surface_point)
```

`pick(color)` and `place(surface_point)` can also be called separately. The
placement point is world XYZ: desired payload center in XY and destination
surface height in Z. It must come from the caller's observations/planning,
not simulator object truth. A `released` result means the action sequence opened
and withdrew the gripper; it does not certify stacking, contact or stability.
Uncertain placement returns `placement_unconfirmed` with the payload still held.
The caller owns recovery. Unreachable SDK targets raise an explicit error.

`locate(color, model=None, near=None)` is the perception boundary. A geometric
observation provider can replace it while preserving the same movement methods.
The default implementation uses calibrated side RGB-D, unique color regions,
a top-plane estimate and visible bounds. It already performs basic geometric
arithmetic, but does not expose fitted rigid vertices, a support graph, desired
3D states or geometric relations. "No supplied abstraction" must not be described
as "no geometry of any kind."

## Removed task-specific behavior

- Fixed blue/red/green roles and the two-step red-on-blue, green-on-red program.
- Cross-object release drift memory and the 1.35-gain green placement compensation.
- Three-level tower/prefix drift, support-margin and uprightness checks.
- The historically tuned negative 5 mm stacking contact gate.
- Assumptions that the last object is green or that the task has exactly three objects.

The new adapter takes any unique-color single-column order from the public goal.
It supplies support coordinates and stops on a failed primitive; it adds no
worksite search, base relocation, alignment refinement or learned compensation.

## Retained assumptions and limits

This is a simple tabletop PnP base, not an unrestricted manipulation controller.
Its first object observation assumes an upright, flat-topped object on the table;
learned dimensions are retained after lifting. Hidden extents inferred from
visible surfaces can be biased under occlusion. Brown table segmentation,
calibrated RGB-D and red/yellow/blue/green color thresholds remain explicit sensor
assumptions. Unique colors remain required for unambiguous association.

Robot-specific joint/gripper commands, motion durations and basic grasp/lift/
withdrawal behavior remain. The default grasp offset is 0.30 of measured height,
inherited from the historical controller; it is configurable, not a universal
optimal grasp. Release clearance (3 mm) and position tolerance (4 mm) are generic
primitive parameters. There is no learned dimension prior or fixed 3/4 cm size.
Orientation planning, obstacle avoidance, staging, contact verification and
multi-object stability reasoning belong to a higher layer and are absent.
Online perception and control use CPU numerical code. Object poses/dimensions
are never read from the simulator. No explicit geometry treatment library or
mature column/bridge skill is included in the base.

## Checks and actual smoke runs

Five tests cover camera-only initialization with no task/action calls, invalid
request rejection before movement, holding on uncertain placement, caller-driven
color order and compatibility with the isolated policy language.

The same bundled base and adapter were then run without tuning, with current
3 cm cubes and default NoSlip 0 physics:

| Smoke task | Seed | Outcome |
| --- | --- | --- |
| Two cubes, blue then red | 3901 | Private final evaluator passed; PnP returned released |
| Three cubes, blue/red/green | 3902 | Failed before the first grasp: requested tool position unreachable within 5 mm |

These are two functional examples, not a success-rate estimate. The second
failure was preserved; no task-specific worksite selection was added to make
it pass. The old 4 cm controller's success rates do not transfer to this extraction.

Artifacts are under `outputs/generic_rgbd_pnp_01`. Its tested bundle is retained
exactly; a later formatting-only change to the base's color mask does not change
its AST. [Smoke summary and source provenance](evaluations/generic_rgbd_pnp.json).

To create an executable comparison starter:

```bash
.venv/bin/python scripts/build_rgbd_pnp_policy.py \
  --output outputs/my_pnp_base/stack_policy.py
```

Supply `--adapter path/to/adapter.py` for a different task. The builder concatenates
the common mechanism and adapter into a sensor-sandbox-compatible policy. The
mechanism alone intentionally has no `run(api)` task entry point. Use the bundled
file with `run_geometry_ablation.py --base-policy ...` only in a **new** campaign;
never replace the baseline inside an already frozen comparison. Both comparison
arms should receive identical mechanism, task adapter and robot/sensor interfaces.

The extracted base is now used by the separate
[generic PnP geometry experiment](GENERIC_PNP_GEOMETRY_EXPERIMENT.md).
