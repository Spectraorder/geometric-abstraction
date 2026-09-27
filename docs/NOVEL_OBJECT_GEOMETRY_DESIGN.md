# Points and connectivity behind the existing policy API

Status: implemented as an opt-in development extension. Current-edge registration
handles the recorded grasp-time occlusions, using complementary current views when
necessary. The explicit libccd/0.40 grasp variant completes all six paired rollouts
on three development layouts; this is not a held-out success-rate estimate.
See [validation results and reproduction commands](NOVEL_OBJECT_GEOMETRY_VALIDATION.md).
The completed geometry comparison, frozen sources and saved evidence are unchanged.

The abstraction is point coordinates, per-point evidence and declared edge/face
connectivity. The current observer extracts a convex planar boundary and infers
an extrusion under explicit assumptions. These are separate layers: a general
point-reference API does not establish general unknown-object perception.

## Implementation boundaries

| File | Responsibility |
| --- | --- |
| `prototypes/point_geometry.py` | Shape-independent point, segment and planar-face references; connectivity validation and polygon moments. No sensing or shape classification. |
| `prototypes/polygon_prism/geometry.py` | CPU RGB-D contour fitting, initial extrusion model, current-pose checks and conversion into the existing geometry record. |
| `prototypes/polygon_prism/partial_boundary.py` | Registration of the initialized footprint to current visible edges and side surfaces, with ambiguity and cross-view checks. |
| `scripts/build_polygon_prism_session.py` | Opt-in integration with the observer, ledger, references and compact checkpoints, using checked edits to a generated library. |
| `scripts/validate_polygon_prism_physics.py` | Separate MuJoCo mesh fixtures, sandboxed sensor-only policy execution and private contact/stability scoring. |
| `scripts/validate_polygon_contact_stability.py` | No-action controls using the same fixtures, collision options and stability evaluator. |

There are no production branches for named polygons. Vertex count, angles and
edge lengths come from the observed contour. The observer accepts 3 to 12 resolved
boundary vertices; the hexagonal and trapezoidal fixtures are diagnostic examples.
The public descriptor `shape='polygon_prism'` declares a geometric family rather
than a catalog shape or automatic arbitrary-object classification.

Keep the current color segmentation, unique object colors, calibrated RGB-D and
CPU execution. This path uses NumPy and SciPy and adds no dependencies, SAM2,
learned segmentation, superquadric fitter or scene graph.

## From pixels to geometry

1. Backproject valid depth and isolate the object's color above the measured table.
2. Fit a near-horizontal planar cap from local depth normals and plane residuals.
3. Extract its convex contour and remove unresolved raster-scale turns/chamfers.
4. Project half-pixel mask transitions onto the cap plane, fit straight boundary
   lines by total least squares, and intersect neighboring lines to recover vertices.
5. On the first upright, table-supported observation, estimate height. Construct
   the hidden parallel bottom and side connectivity under the constant-cross-section
   model; label those points and faces as inferred.
6. For a complete later cap, refit its contour and check count, pairwise distances,
   area, perimeter and visible side evidence against the initial model.
7. If the boundary is occluded, match unoccluded boundary lines and current side
   points to the initialized footprint. Enumerate edge correspondences and refine
   a planar rigid transform; reject candidates contradicted by current depth.
8. If one view leaves different plausible completions, validate them against the
   already captured second view's side surfaces. The views must have the same
   timestamp and distinct camera geometry. Remaining ambiguity stays unavailable.

Only height and shape evidence are reused. A previous pose, commanded tool motion
or assumed gripper attachment never supplies the current pose.

The initial cap must be visible, convex, uniformly colored and sufficiently resolved.
Short edges need roughly eight projected pixels to survive raster suppression;
near-parallel incident edges also reduce vertex resolution. Subsequent partial
updates require at least two independent supported edge directions, a conditioned
pose fit, bounded disagreement between candidate centers/vertices and consistent
visible side surfaces. Predicted cap interior must be visible or have current
nearer depth explaining its occlusion; at least 35% remains visible. An unknown
depth hole is not treated as a measured occluder. The known robot-marker exclusion
is tracked separately from missing sensor depth. Image cuts and insufficient or
contradictory evidence still reject the fit.

`boundary_resolution_m` and `vertex_resolution_m` are engineering resolution
estimates, not calibrated statistical confidence intervals.

## One representation, existing calls

Existing policy signatures remain: `Session(api, objects)`, `observe/get`,
`reference`, `relation`, `align`, `checkpoint`, `pick` and `place`. Known inventory
objects retain their original backend. Base keeps its visual-only public interface;
both arms share the observer and actuator. The optional generic grasp parameter
is applied equally to both arms; omitting it preserves the original default.

For Full, `observe(color)` and `get(color)` expose the following fields:

| Field | Meaning |
| --- | --- |
| `vertices` | World-meter XYZ points, bottom first then top for this observer; 2N total. |
| `vertex_sources` | One source per point: current fitted contour, initialized vertex registered to current edges, or inferred parallel bottom. |
| `edge_vertex_indices` | Pairs of point indices: N top edges, N bottom edges, N vertical edges. |
| `face_vertex_indices` | Ordered point boundaries: top, bottom, then side faces. |
| `vertex_id_scope` | Indices identify the current observation only. |
| `observability_condition` | Current pose availability depends on a complete cap or sufficiently constrained current-edge registration, under the initialized extrusion model. |

`top_polygon_m`, `bottom_polygon_m` and `edges_m` remain compatibility coordinate
projections. The ledger verifies that they agree with the points and connectivity.
The old named faces and new indexed faces resolve from the same point data.
Partial updates additionally report `pose_update_mode`, supported-edge count,
edge residual, fit conditioning, candidate count and center spread. Joint updates
include `registration_views`. These are engineering checks, not statistical
confidence intervals. Registered top vertices are explicitly inferred from the
initialized shape, and `top_face` retains that distinction in its semantics.

```python
point = session.reference('red.vertex_0')  # Point with its evidence source.
edge = session.reference('red.edge_0')     # Segment from declared endpoints.
face = session.reference('red.face_0')     # Plane patch from a declared boundary.
top = session.reference('red.top_face')   # Same geometry as face_0.
bottom = session.reference('red.bottom_face')  # Same geometry as face_1.
```

References include world-meter coordinates, observation time and evidence. A
physical face requires declared connectivity, not merely three coplanar points.
The reference layer rejects fractional, negative, repeated or out-of-range
indices, coincident points, degenerate planes, noncoplanar faces and intersecting
face boundaries. Supplied simple concave face boundaries are representable; the
current RGB-D observer still does not reconstruct concave objects.

`center` remains the fitted model centroid. `axis` is the extrusion axis, and
`long_axis`/`short_axis` require a unique geometric direction. A symmetric cap
provides no material yaw. `corner_i` remains box-specific and is rejected here.
No reference certifies contact, stability or grasp feasibility.

Indices may change after motion or refitting. Re-select a point using current
geometry when necessary; requesting the same index again does not establish
correspondence with the same material point. Stale or unavailable measurements
follow the existing rejection path.

## Compact agent context and sensing

```python
record = session.checkpoint(
    'red', 'inspect', refresh=False,
    reference_names=('red.vertex_0', 'red.bottom_face'),
)
```

The checkpoint records only requested points or face centers, their normals when
applicable, semantics, unique `evidence_sources` and observation-local `id_scope`.
Its object geometry retains `observability_condition` and `vertex_id_scope`.
Inferred bottom points therefore keep their evidence labels in text passed to the
agent. Dense vertices, edge endpoints and face connectivity are not copied into
compact records. Segments are queried directly with `reference`; they are not
accepted as point/face-center checkpoint inputs.

`get`, `reference` and `checkpoint(refresh=False)` do not request new camera
frames. Use `observe` or `checkpoint(refresh=True)` for an explicit current update.
Motion invalidates old pose estimates; the compact recorder does not silently
refresh them or reuse stale requested points. Full detailed scene packets expose
points, connectivity and per-point sources only with `details=True`.

The CV update processes RGB-D locally and does not attach images to an LLM request.
This preserves the path to text-only subsequent reasoning, but actual token and
latency savings require measurement in a separately registered policy experiment.

## Build and validate

Run from `aspire_so100`:

```bash
.venv/bin/python scripts/build_polygon_prism_session.py --arm symbolic --output /tmp/polygon_full.py
.venv/bin/python scripts/build_polygon_prism_session.py --arm basic --output /tmp/polygon_base.py
.venv/bin/python -m unittest discover -s tests -p 'test_point_geometry.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_polygon_prism_geometry.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_partial_polygon_geometry.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_polygon_physics_diagnostics.py' -v
```

The builder creates a new file and refuses to overwrite existing outputs. Supply
the generated library through the existing geometry-library injection for a new
development run. Building does not launch training or modify a campaign. Scene
generation and the existing experiment inventory remain unchanged. The builder
and physical runner both accept `--grasp-fraction .4` for the explicit development
variant; the Python equivalent is `library(arm, grasp_fraction=0.4)`. Without this
option, the existing 0.70 default is preserved. Collision selection belongs to the
physical runner, not to the generated policy library.

The four suites contain 33 tests: 5 reference-layer, 18 integration/perception,
7 partial-boundary and 3 diagnostic-configuration tests. They cover generated
irregular convex boundaries with 3 to 12 vertices (20 shapes, initial and moved
views at 720x540), pose changes, depth noise,
occlusion, missing depth, image cuts, visible concavity, inconsistent models,
symmetry, support overlap, alignment and stale geometry. They also verify:

- Shape-independent references, valid face winding and rejection of malformed topology.
- Agreement of indexed faces, named faces and detailed packets.
- Evidence retention in compact text and no implicit camera calls on repeated reads.
- Identical legacy cuboid packets/checkpoints and unchanged actuator code in both arms.
- Matching Base/Full pick/place commands using calibrated current RGB-D.
- Occluded updates after translation/tilt, rejection of unknown depth, contradictory
  background, heavy occlusion, one visible direction and incompatible shape models.
- Recorded held-object pixels, complementary-view disambiguation, and rejection of
  stale, duplicated or mutually inconsistent view geometry.
- Isolated collision options, equal grasp overrides and unchanged velocity scoring.

The portable `tests/fixtures/partial_polygon_rgbd.npz` record includes calibrated
crops from the physical diagnostics. Its adjacent JSON records source hashes,
crop offsets and the private-truth sample offsets. Truth is used only in test
assertions, never as an observer argument.

The synthetic pick/place fixture attaches the object kinematically. It validates
the API and visual update path, not physical grasp success. Separate
[physical diagnostics](NOVEL_OBJECT_GEOMETRY_VALIDATION.md) exercise real contact,
lift and release. Rounded, tapered, concave, heavily occluded and very small objects
are outside the validated observer scope. A failed fit must not supply an assumed
executable pose.

Earlier contour accuracy and timing measurements are preserved separately in
[the archived validation](../outputs/polygon_prism_cv_02/validation.json) with its
[geometry snapshot](../outputs/polygon_prism_cv_02/geometry_snapshot.py). They describe
that historical implementation, not a new benchmark of the current code. The
pixel-center predecessor remains under `outputs/polygon_prism_cv_01`.
