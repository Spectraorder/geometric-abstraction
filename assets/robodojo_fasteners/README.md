# Original RoboDojo fasteners

These are converted **original benchmark meshes**, not procedurally recreated
lookalikes. Downloaded using the Hugging Face CLI, scoped to the fastener assets.

- Dataset: [RoboDojo-Benchmark/RoboDojo](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo).
- Pinned dataset revision: `43dacb13d3051a9ccd421f4e7e08e4e3eb29dfab`.
- Original files: `Assets/Object/RoboDojo/Rigid/factory_nut/00000/object.usdz`
  and `Assets/Object/RoboDojo/Geometry/factory_bolt/00000/object.usdz`.
- Task config: [fasten_screws.yml](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/726e9aabfaa642203722eb126f5eaf0f37f3e1ad/task/RoboDojo/config/fasten_screws.yml).
- Dataset-declared license: Apache-2.0, retained in [upstream/README.md](upstream/README.md).
  See [LICENSE](LICENSE) and [NOTICE](NOTICE). The separate benchmark code repository
  uses MIT; that is not the license selected for these dataset assets.

`upstream/` retains the unmodified USDZ files and geometry metadata. `converted/`
contains lossless vertex/triangle arrays and original shading normals for MuJoCo.
The manifest records source and generated SHA-256 hashes, original prim paths,
mesh counts, dimensions, and origin translations. Only these transformations
are applied: authored USD transforms, conversion to meters, bottom-center origin,
and triangulation. No decimation, rescaling, thread redesign, or hole filling.
Invalid zero-length source normals receive geometric fallback directions;
vertices and faces are not changed by this shading repair.

Both assets' five color variants share their geometry. We vendor variant `00000`
(yellow) once and apply the original red/blue/yellow diffuse colors at runtime.
OmniPBR shader code is not imported; MuJoCo lighting/material rendering differs.

| Original visual mesh | Vertices | Triangles | Dimensions (mm) |
| --- | ---: | ---: | --- |
| factory_nut | 13,089 | 15,080 | 35.281 × 40.671 × 19.111 |
| factory_bolt | 78,366 | 26,122 | 36.000 × 41.550 × 52.500 |

The six authored nut edge-box colliders are also converted and retained for
grasp/table contact. Both USDZ files include PhysX SDF mesh-collision settings;
these are **not** reproduced in MuJoCo. The imported meshes are visible surfaces,
the nut's edge boxes and simple shaft/head proxies handle bulk collision, and our
existing compliant helical model with solver dry friction handles fastening.
This is an asset port, not a validated port of RoboDojo's physics or evaluator.

The bolt's nominal 3 mm pitch is measured from the imported mesh; the compliant
adapter uses that pitch and an approximate phase offset. It is not a guarantee
of exact triangle-to-triangle fit. Three net turns correspond to 9 mm of depth
in this adapter, replacing the historical procedural scene's 4 mm / 12 mm values.

To regenerate, from the project root (isolated tool dependencies, not the robot
runtime environment):

```bash
uv run --no-project --with usd-core --with numpy python scripts/import_robodojo_fasteners.py
```

Runtime does not require USD, Isaac Sim, network access or downloads. Do not
overwrite historical trial scenes: the asset port is `fasten_robodojo_v1`.
