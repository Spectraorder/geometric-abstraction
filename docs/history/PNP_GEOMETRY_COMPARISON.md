> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Stacking comparison from a shared RGB-D base

This protocol compares policy improvement with and without a supplied geometric
abstraction, starting from the same existing RGB-D pick-and-place/stacking policy.
It replaces the empty-function starting point of the earlier diagnostic pilots.
Historical pilots and their failed results remain separate.

## Frozen shared starting point

The user selected `policies/stack_three_blocks_vision.py`. Its source is copied
unchanged to the campaign's `base_policy.py`, hashed, executed once on each
training layout, and supplied verbatim to both arms. The shared baseline results
are revision zero in both learning curves. No adaptation to the current scene is
manually inserted. The older `stack_small_red_on_blue.py` is excluded because its
original interface reads simulator object state.

Both arms receive the same initial images, baseline source, baseline reduced
feedback, RGB-D camera APIs, robot SDK, numerical libraries, public three-cube
goal, and robot-model finger pad geometry. Side and wrist depth are enabled for
both. Object dimensions and poses must come from visual observations; evaluator
object truth never crosses the runtime policy boundary.

The baseline already contains color segmentation, depth backprojection, simple
center estimates and numerical placement corrections. Therefore "raw" means
**no supplied explicit geometric-abstraction library**, not absence of all
geometric calculation. This is a comparison of added reusable abstraction on an
existing controller.

## Treatment

The geometry arm additionally receives a frozen, action-free numerical library:

- Cube face fitting, centers, orientations, inferred vertices and observability.
- Constructed stack targets from measured sizes and an explicit visual base anchor.
- Translation, symmetric cube rotation and vertex errors to those targets.
- Estimated support gap and center-to-edge margin, without contact certification.
- The historical table, visible-gripper and point-to-support-edge helpers.

Both arms may revise their complete base policy, including perception and control.
The raw arm may implement geometric reasoning itself. The treatment library does
not choose grasps, issue actions, retry, query object truth, or include mature
column/bridge controllers. Both arms share robot-model geometry; it is not an
extra sensor granted only to the geometry arm.

## First pilot budget

Campaign directory: `outputs/geometry_base_ablation_01`.

| Setting | Value |
| --- | --- |
| Task | Blue/red/green three-cube stack, unique colors, no distractors |
| Scene | `inventory_tower`, sampled, three objects, `cube_triple` |
| Physics | Default NoSlip 0 |
| Training layouts | Seeds 3701–3705 |
| Reserved layouts | Seeds 3801–3810 |
| Initial baseline | One shared execution per training layout |
| Improvements | Three revisions per arm, five training trials each |
| Generation | Same `gpt-6-astra`, high reasoning, 900-second per-call timeout |
| Code budget | 48,000-character maximum, 18,000 target |
| Selection | Highest training success including revision zero; earliest tie |
| Frozen evaluation | Both policies frozen before either sees reserved layouts |

There are at most 55 actual rollout slots: five shared baseline, thirty candidate
training and twenty reserved. The five shared baseline executions appear in both
curves, but are not forty separate training executions. Invalid candidates consume
their allocated slots and remain explicitly distinguished from executed rollouts.

Model contexts are independent: the same initial base, then only each arm's own
hypotheses, source, reduced success/error feedback and selected sensor images.
They do not receive this conversation, repository access, other skills, historical
successful controllers, other-arm feedback or reserved outcomes. Model generation
can run concurrently; physics rollouts are serialized. All prompts, responses,
usage, candidate hashes, trials and failures are retained.

## What this experiment can establish

Measure training success by revision/rollout budget, best-so-far success,
iterations to a stated training threshold when reached, and paired reserved
success after freezing. Also retain model tokens/generation time, camera calls,
policy CPU compute and host trial wall time. Fixed revisions are matched; tokens
are measured rather than assumed equal.

One short campaign per arm is a diagnostic pilot, not statistically conclusive
proof. A higher result is not assumed in advance. Replication across independent
campaigns is required before a general efficiency claim.

This experiment does **not** test online repeated VLM checking. Both resulting
controllers execute on CPU without a VLM at runtime. Host trial wall time also
includes process startup, simulation, rendering, private evaluation and artifact
IO; policy CPU intervals exclude RPC and rendering. Neither metric alone proves
that structured text reasoning is faster than an online VLM inspection loop.
That requires a separate matched runtime agent/checker experiment.

## Reproduction and implementation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -u \
  scripts/run_geometry_ablation.py \
  --output outputs/geometry_base_ablation_new \
  --base-policy policies/stack_three_blocks_vision.py \
  --rounds 3 --train-seeds 3701 3702 3703 3704 3705 \
  --heldout-seeds 3801 3802 3803 3804 3805 3806 3807 3808 3809 3810
```

Use a fresh directory. `--prepare-only` freezes the manifest/contracts/sources
without executing policies or generating candidates. Omitting `--base-policy`
retains the older empty-starter protocol and its wrist-RGB default. Source changes
after freeze stop the campaign; they are not silently incorporated.

Thirteen experiment tests pass, covering original isolation guarantees plus
shared source identity, matched sensor contracts, immutable target functions,
rejection of privileged/treatment-dependent starters, and revision-zero selection.
Results will be exported separately after all frozen trials complete.
