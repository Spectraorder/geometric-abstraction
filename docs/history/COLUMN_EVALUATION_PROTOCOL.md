> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Reproducible single-column generalization evaluation

The CPU policy target is approximately 90% complete, physically stable towers
across a declared task distribution. Report whole-tower success rather than
visual completion messages or partial placement scores. Keep the existing
private evaluator's contact, uprightness, support-margin, velocity and
five-second stability requirements unchanged.

## Declare coverage before evaluation

`scripts/evaluate_columns.py` supports the Cartesian product of explicit random
level counts and distractor counts. Each output contains:

- `candidate.py`: the frozen source used for every trial;
- `manifest.json`: source path and SHA-256, split, every recipe, seed, public
  goal, identity order, level count, object count and distractor count;
- per-case and aggregate `summary.json` files retaining every result and its
  declared trial, including failures.

An existing output directory cannot be reused with a different source, split,
seed set or task matrix. The runner also validates cached results against the
source hash, goal and object count. `--plan-only` writes the source and manifest
without starting simulations. `--split heldout` additionally requires the
explicit `--frozen-sha256` recorded at the freeze point.

`random` still defaults to three levels. Pass `--levels 2 3 4` to evaluate all
three counts; the object count for each trial is its level count plus the
requested distractors. Distractor changes also change scene layouts, so results
at different distractor counts are not a paired causal clutter experiment.

```bash
# Development example: six cells, 30 trials, no reserved seeds.
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python scripts/evaluate_columns.py \
  --candidate policies/stack_column_cpu.py \
  --output outputs/columns_development_matrix \
  --recipes random --levels 2 3 4 --distractors 0 2 \
  --seeds 901 902 903 904 905 --jobs 3

# Inspect the exact matrix and source hash without running it.
uv run python scripts/evaluate_columns.py \
  --candidate policies/stack_column_cpu.py \
  --output outputs/columns_development_plan \
  --recipes cube_pair cube_triple cube_quad mixed_triple slender_top cylinder_base random \
  --levels 2 3 4 --distractors 0 2 --seeds 901 902 --plan-only
```

When an explicit level is supplied, its case directory is named `random_l2`,
`random_l3` or `random_l4`; a distractor matrix appends `_d0`, `_d2`, etc.
Single named-recipe runs without distractors retain their existing directory
names. Individual custom identity orders remain available through
`scripts/run_vision_trial.py --tower-order ...`.

## Color permutations and exact custom orders

Add `--permute-colors` to apply a deterministic permutation of all four color
labels to every named or random goal. Shapes, layer count, ordering and
same-color relationships are preserved. The seed maps to a color bijection by
sorting SHA-256 hashes of `column_colors_v1:<seed>:<color>`; it does not depend
on Python's randomized hash or a particular random-generator version.

Each declared trial records the source recipe/order, full color mapping,
permutation algorithm, executed custom order and public goal. Case directory
names append `_colors`. Execution uses the runner's existing `--tower-order`
interface; the policy still sees only its public ordered goal, not the seed or
permutation. Cached results must match both that goal and the recorded scene
order. Parallel execution across seeds is retained.

```bash
uv run python scripts/evaluate_columns.py \
  --candidate policies/stack_column_cpu.py --output outputs/columns_color_plan \
  --recipes cube_pair cube_triple cube_quad mixed_triple slender_top cylinder_base \
  --seeds 901 902 903 --permute-colors --plan-only
```

Manifests also declare the sensor profile. Column experiments use pixel-center
depth with offscreen multisampling disabled. Legacy or mismatched sensor
profiles and blackout diagnostics cannot be reused in the same matrix, even
when candidate hashes match.

## Declared second-milestone freeze

Development used seeds 901–909 and older archived failures. Before running any
new reserved trial, the following 100-trial matrix was declared in
`outputs/columns_02/evaluation_plan.json`. All three groups share one frozen
candidate and the pixel-center sensor profile.

| Group | Recipes | Seeds | Distractors | Trials |
|---|---|---|---|---:|
| Named structures with color permutations | Six named recipes above | 1001–1005 | 0 and 2 | 60 |
| Random two-level identities and shapes | `random`, 2 levels | 1006–1015 | 0 and 2 | 20 |
| Random tall structures | `random`, 3 and 4 levels | 1016–1020 | 0 and 2 | 20 |

Freeze and record the exact candidate hash before executing any reserved trial.
Run the full declared matrix, retain controlled stops and worker errors as
failures, and report both the overall 100-trial count and every subgroup.
No source tuning is allowed after inspecting these results. If these failures
inform another revision, use a fresh declared seed set for that revision.
Seeds repeat across task cells; trials are not independent samples of every
possible scene, and subgroup counts remain small.

This matrix covers cube, prism, thin-prism cap and cylinder targets.
Triangular prisms may occur as distractors but are not random target caps in
the current task sampler. Triangular-cap construction needs its own declared
challenge set and successful manipulation before claiming coverage of all
five shapes. Likewise, initial yaw remains zero and camera/calibration are
fixed: this protocol does not establish orientation, camera or real-world
transfer.

Do not silently remove tall or difficult recipes after seeing failures.
A reachability-constrained curriculum must define its generation rules before
sampling, and its success rate must be reported separately from unrestricted
2–4-level random recipes. Offline feasibility analysis may inspect simulator
geometry, but online policy decisions must continue to use public goals,
visual measurements and robot-only kinematics.
