# Cube centroids and vertices from RGB-D

The numerical fitter in `src/aspire_so100/cube_geometry.py` estimates each
colored cube's solid centroid, orientation, eight vertices, and top support
face in world coordinates (meters). It uses RGB, metric depth, camera
calibration, and a side length measured in the initial view. It does not use
simulator object poses, meshes, or segmentation IDs.

A single image does not directly reveal hidden corners. All eight output
vertices are **inferred cube-model vertices**, including the hidden ones.
This implementation needs calibrated RGB-D; it is not an RGB-only detector.

## How the fit works

1. Select a connected color region above the RGB-D-fitted table plane.
2. Unproject valid depth into world XYZ and estimate surface normals from
   neighboring interior pixels, excluding silhouette and depth-edge pixels.
3. Cluster perpendicular visible face normals and fit their planes.
4. Move inward from each visible plane by half the measured cube side to
   recover the solid center; construct vertices from that center and orientation.

The mean of visible surface samples is also returned as `surface_centroid` for
diagnostics, but it is not the solid cube center. The distinction matters when
the gripper hides part of a face: the old visible-top bounds could shift even
if the actual cube did not.

The `reliable` flag requires three observed perpendicular faces, fitted plane
RMS below 0.8 mm, and more than 80% of selected points consistent with the cube.
It is a quality gate, not a calibrated probability or an error guarantee.
One- or two-face fits mark unconstrained coordinates and remain unreliable.
Vertex order follows the fitted axes; cube symmetry means persistent physical
corner identities across frames are not provided.

The cube side prior is measured while each cube is upright on the table in the
initial view, then reused within the trial. That assumption would need revisiting
for non-cubes, initially tilted objects, or objects initially stacked together.

## Policy integration and measured results

The results in this section were recorded with **4 cm cubes**. The current
scene uses 3 cm cubes; see the [current policy report](VISUAL_RETRY_POLICY.md).

`policies/stack_three_blocks_geometry.py` bundles the fitter into the existing
isolated policy. It replaces the baseline center/extent estimates when the fit
passes its quality gate, and checks placement margin against the inferred top
face's edges. Otherwise it retains the baseline perception. The policy still
uses RGB-D/proprioception only, with the existing release-drift compensation
and physical evaluator unchanged.

| Frozen policy | Training 101–105 | New random-XY validation 1001–1005 |
| --- | --- | --- |
| Earlier evolved 4 cm policy | 5/5 | 3/5 |
| Cube geometry candidate | 5/5 | 5/5 |

Both policies ran on the same five new validation layouts, with separate saved
rollouts. The earlier policy failed 1002 and 1004; the geometry candidate passed
them. The candidate was frozen before validation and was not adjusted afterward.
This is evidence of improvement on this small sample, not universal reliability.

An offline diagnostic sampled 40 recorded training frames (120 cube observations).
92 fits passed the quality gate. For those fits, comparison with saved simulator
states found mean centroid error **1.62 mm**, 95th-percentile centroid error
**1.76 mm**, and 95th-percentile maximum inferred-corner error **2.14 mm**.
Simulator geometry was used only as a reference for this offline accuracy check,
never as an input to fitting or policy execution. These numbers describe the ideal
simulated RGB-D sensor, not a calibrated real-camera accuracy claim.

The test suite passed 24 tests with one macOS-only skip. New tests cover analytic
rotated-cube images, missing faces, invalid depth, and rotated support edges.

## Inspect and use

Run the geometry policy:

```bash
uv run python view_scene.py --vision-policy policies/stack_three_blocks_geometry.py \
  --vision-layout random_xy --seed 1002 --vision-output outputs/geometry_view_1002
```

The output directory must be new. To inspect saved RGB-D without executing a policy:

```bash
uv run python scripts/inspect_cube_geometry.py \
  --frame outputs/cube_4cm_evolution_01/round_005/trials/seed_101/camera_019.npz \
  --size-frame outputs/cube_4cm_evolution_01/round_005/trials/seed_101/camera_000.npz \
  --output outputs/cube_geometry_01/visualization
```

`--size-frame` must show upright cubes resting directly on the table. The tool
writes `geometry.json` with numeric centers/vertices and `geometry_overlay.png`.
Dashed edges represent model inference, yellow crosses mark solid centroids,
and underconstrained fits are labeled uncertain.

The reusable API is:

```python
from aspire_so100.cube_geometry import estimate_cube_geometry

geometry = estimate_cube_geometry(frame, "red", side_m=measured_side)
center = geometry["centroid"]       # (3,), world XYZ meters
corners = geometry["vertices"]      # (8, 3), inferred world XYZ meters
usable = geometry["reliable"]
```

Rebuild the standalone policy after changing the fitter or integration wrapper:

```bash
uv run python scripts/build_geometry_policy.py --output outputs/new_geometry_candidate.py
```

The bundle is needed because the isolated policy may import only numerical
libraries. Rebuilding changes the candidate and requires reevaluation. The
builder requires a new output file and preserves evaluated policy snapshots.
This builder assembles the earlier cube-only controller, not the current
cube-and-gripper policy.

Local evidence: `outputs/cube_geometry_01/` contains `training/`, paired
`validation/` trials and summary, `validation_plan.json` with frozen source hashes,
`frozen/` sources, `accuracy.json`, and `visualization/` outputs.
