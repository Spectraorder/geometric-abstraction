# Transfer to 3 cm cubes

This records the initial, unchanged-policy transfer test. The subsequent
[3 cm policy evolution](CUBE_3CM_EVOLUTION.md) improved training to 8/10 and
passed 9/10 new reserved layouts.

The current scene uses 3 × 3 × 3 cm cubes (`CUBE_SIDE_M = 0.03`). Shared
dimensions update all box geometry, initial center heights, randomized layout
clearance, and the single-stack evaluator's expected height. Block mass,
friction, robot, camera, and success criteria were not changed.

Two frozen policies were evaluated without retraining or edits:

| Policy | Seeds 101–105 | Seeds 1001–1005 | Total |
| --- | --- | --- | --- |
| `stack_three_blocks_geometry.py` | 2/5 | 2/5 | 4/10 |
| `stack_three_blocks_cubes_4cm.py` | 2/5 | 2/5 | 4/10 |

Both passed **102, 103, 1003, 1005** and failed
**101, 104, 105, 1001, 1002, 1004**. Each used the same initial layout for a given
3 cm seed. Seed numbers were reused from the earlier 4 cm experiment, so this
is a size-transfer check rather than a new held-out split. Since layout sampling
uses block dimensions to reject overlaps, identical seed numbers need not
produce identical XY arrangements across the two cube sizes.

The policy still measures dimensions from RGB-D: initial geometry-policy logs
report approximately **29.9 mm** on each axis. No 3 cm dimension or object
coordinates were supplied to the policy. The preserved source hashes are:

- Geometry: `8724bc2392efc72d6d09a2041920342fd164a7273624998dd71799cd57fc2624`
- Earlier evolved policy: `4cc3b4ccd976a91ceecc98d016064468313e9c0c968614cdca77133109bd7c8b`

## What failed

The geometry policy stopped after placing red on seeds 101 and 104 because it
reported uncertain support. On 101, it measured about 10.9 mm lateral offset
after withdrawal, leaving little margin on a cube with a 15 mm half-width.
Other failures included lost support contacts, insufficient support margin,
or instability after placing green. Better visual geometry alone did not
improve total success over the older policy at this smaller size.

No policy was tuned using these results. A subsequent evolution run should
address release displacement and the smaller support footprint, with a new
reserved validation split.

## Verification and artifacts

Both single-stack and tower scenes were compiled and checked for 30 mm box
edges and settled center heights at table height + 15 mm. All 20 trials saved
results and physics rollouts. The test suite passed 24 tests with one
macOS-only skip.

Evidence is in `outputs/cube_3cm_eval_01/`:

- `metadata.json`: scene size, evaluated seeds, source hashes, experiment scope.
- `frozen/`: exact policy sources used for the comparison.
- `results.json` and `summary.json`: per-trial outcomes and aggregated results.
- Policy/seed folders: RGB-D frames, notes, traces, results, and `rollout.npz`.
- `stack_three_blocks_geometry/seed_102/demo.mp4`: successful replay at 2×.
- `visualization/geometry.json` and `geometry_overlay.png`: fitted 3 cm cube
  centers and inferred vertices from the initial RGB-D view.

Replay this successful 3 cm trial with the current scene:

```bash
uv run python view_scene.py \
  --vision-trial outputs/cube_3cm_eval_01/stack_three_blocks_geometry/seed_102
```

Historical 4 cm state replays require restoring the matching 4 cm scene;
existing rendered videos remain accurate for their original experiments.
