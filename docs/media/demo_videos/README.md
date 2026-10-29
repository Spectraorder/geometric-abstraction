# Demo rollout videos

Continuous 1280×720, 30 fps rollouts of the six paper tasks, recorded for the
[project page](https://geometric-abstraction.github.io/#videos). They show the
physical tasks in the same simulated SO-100 setup. **They are not study episodes
and are not the study's LLM agents.** For sparse replays of actual study episodes,
see [`docs/evaluations/videos/`](../../evaluations/videos/README.md); for results,
see the [evaluation index](../../evaluations/README.md).

| Video | Controller | Seed | Speed |
| --- | --- | ---: | --- |
| [single_policy_bridge.mp4](single_policy_bridge.mp4) | `structure_policy` (sensor-only), `thin_beam_bridge` recipe | 1501 | 3× |
| [single_policy_stacking.mp4](single_policy_stacking.mp4) | `cube_alignment_policy` (sensor-only), `cube_triple` tower | 2701 | 3× |
| [single_policy_ring.mp4](single_policy_ring.mp4) | `ring_peg_starter` (sensor-only) | 1 | 1× |
| [online_tower.mp4](online_tower.mp4) | `structure_policy` (sensor-only), demo-only `bridge_tower` recipe | 1501 | 4× |
| [online_ring.mp4](online_ring.mp4) | `ring_peg_starter` (sensor-only) | 2 | 1× |
| [online_relocation.mp4](online_relocation.mp4) | Scripted pick-and-place in the online relocation scene; **reads simulator object poses** | 1 | 2× |

All seeds are development seeds, disjoint from every reserved evaluation set. Each
rollout succeeded under its scene's own success check. The online tasks' study
skills load trained policies from the experiment archives, so the online videos use
scripted controllers, not the online agent. The `bridge_tower` recipe is registered
at runtime by `scripts/demo_videos/demo_recipes.py`; repository recipes are unchanged.

## Regenerate

Recording saves every physics step to `rollout.npz`; rendering replays those states
with a free camera and hides the side camera's non-colliding mast. Rendering does
not execute the policy again.

```bash
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
D=scripts/demo_videos; OUT=outputs/demo_videos

# Single-policy Bridge + cylinder and Bridge-tower (structure_policy)
uv run python scripts/build_inventory_policy.py --controller structure_policy --output $OUT/structure.py
uv run python $D/run_trusted_trial.py --task inventory_structure --layout sampled \
  --structure-recipe thin_beam_bridge --object-count 4 --seed 1501 --candidate $OUT/structure.py --output $OUT/bridge
uv run python $D/run_trusted_trial.py --task inventory_structure --layout sampled \
  --structure-recipe bridge_tower --object-count 5 --seed 1501 --candidate $OUT/structure.py --output $OUT/bridge_tower

# Three-cube stacking
uv run python scripts/build_inventory_policy.py --controller cube_alignment_policy --output $OUT/cube_align.py
uv run python $D/run_trusted_trial.py --task inventory_tower --tower-recipe cube_triple --layout sampled \
  --object-count 3 --seed 2701 --candidate $OUT/cube_align.py --output $OUT/stacking

# Ring-on-peg (seeds 1 and 2) and tower relocation
uv run python $D/run_ring_demo.py --seed 1 --output $OUT/ring_1
uv run python $D/run_ring_demo.py --seed 2 --output $OUT/ring_2
uv run python $D/run_relocation_demo.py --seed 1 --output $OUT/relocation

# Render, e.g.
uv run python $D/render_hd.py --clean --trial $OUT/bridge --output single_policy_bridge.mp4 --speed 3
uv run python $D/render_hd.py --clean --trial $OUT/ring_1 --output single_policy_ring.mp4 --speed 1 --distance .58
```

`run_trusted_trial.py` runs `scripts/run_vision_trial.py` unchanged, except that the
policy worker starts as a plain isolated subprocess rather than under `bwrap`. The
`bwrap` sandbox exists for LLM-generated policies; these are the repository's own
hand-written controllers.
