# Fasten Screws: original RoboDojo models

Both scene modes now use the actual RoboDojo `factory_nut` and `factory_bolt`
meshes, downloaded from the official dataset and converted from USDZ. See
[asset provenance, license and conversion](../assets/robodojo_fasteners/README.md).
The two-pair inspection scene below remains available alongside the
[three-pair fastening learning task](FASTEN_SCREWS_LEARNING.md), with fixed bolts,
compliant thread engagement and 0/20/50/100 scoring.

Branch: `fasten-screws-scene`, based on `trajectory-phantom-visualization`.

The models come from [RoboDojo Fasten Screws](https://robodojo-benchmark.com/doc/sim-tasks/fasten-screws/),
whose task matches screws and nuts by color. This SO-100 inspection scene has two
pairs: red screw + red nut, and blue screw + blue nut. All four are movable.
Screws stand on their hex heads with shafts pointing upward; nuts lie flat with
their bores pointing upward. The color order is crossed between the two rows.
The layout is fixed for this first inspection stage; `--seed` does not randomize it.

```bash
uv run --no-sync python view_scene.py --scene-only --scene-task fasten_screws
```

Use the Viser URL printed by the command. The existing SO-100, tabletop,
wrist camera and side-mounted RGB-D camera are included. There is no policy
running in this scene. Trajectory and phantom review are available for recorded
screw-task policies through the version-aware fastening replay adapter.

| Geometry | Size |
| --- | --- |
| Original bolt | 36.000 × 41.550 × 52.500 mm; 26,122 triangles |
| Original nut | 35.281 × 40.671 × 19.111 mm; 15,080 triangles |
| Screw-motion adapter | Nominal 3 mm pitch; 3 net turns / 9 mm depth |
| Colors | Original red, blue and yellow diffuse colors |

The original mesh geometry is retained without decimation or rescaling. USD
transforms are applied and the origin is moved to bottom-center. Authored shading
normals are retained, with fallback directions for invalid zero-length normals.
MuJoCo lighting is different from Isaac Sim's OmniPBR shading.

The original six nut edge-box colliders preserve the open hole for bulk/grasp
contact. The bolt uses hidden shaft/head proxies. Group 2 contains original
visual surfaces; group 3 contains contact proxies. RoboDojo's PhysX SDF settings
are not portable merely by loading the meshes, and have not been reproduced here.

The **scene-only viewer** implements gravity and ordinary contact only; it does
not run fastening control. The learning environment adds a solver-resolved helical
constraint and an approximate entry shoulder. The thread faces themselves are **not contact-resolved**:
bulk contact uses a smooth shaft and original nut edge boxes. This is still a
simplified fastening simulation, not calibrated
thread contact, friction or cross-threading physics. See the learning document
for the scoring and engagement conditions.

Current learning scene version is **`fasten_robodojo_v2`**. See the
[solver constraint and entry-stop correction](FASTEN_THREAD_CONSTRAINT.md).
Historical `fasten_robodojo_v1` remains replayable with its original spring model. It replaces the procedural
surfaces and retains the solver dry-friction fix from `fasten_helical_v3`. See the
[gravity-drift correction](FASTEN_SCREWS_LEARNING.md#gravity-drift-correction-v3).
Historical `fasten_helical_v1`, `v2`, and `v3` trials retain their original geometry/physics when replayed;
they must not be relabeled as evidence for the corrected scene. A new learning
campaign needs a new output root. No learning is started by these fixes.

To save overview and close-up renders on a rendering-capable node:

```bash
uv run --no-sync python scripts/render_fasten_scene.py --output outputs/fasten_screws_robodojo_preview
```

Images are written to `outputs/fasten_screws_robodojo_preview/`, including
`nut_internal_thread.png`, a close-up looking into the nut.
