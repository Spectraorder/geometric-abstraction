# Rendered task illustrations

The primary paper figures combine a workflow diagram on the left with three
task columns on the right, with starts above goals. Intermediate task panels are omitted to save
space. These figures accompany the [experiment results](../README.md).

- **[Single-policy workflow + tasks (PDF)](single_policy_workflow_tasks.pdf)** ·
  [PNG](single_policy_workflow_tasks.png) · [editable SVG](single_policy_workflow_tasks.svg).
- **[Online workflow + tasks (PDF)](online_workflow_tasks.pdf)** ·
  [PNG](online_workflow_tasks.png) · [editable SVG](online_workflow_tasks.svg).

The figures omit the condition legend and model settings; these remain in the
experiment reports. A small Geometric abstraction (Full only) module marks the
additional supplied interface. Dashed arrows indicate this condition-specific
path: geometry is available to the executing policy in single-policy acquisition,
and observations yield geometric text for LLM decisions in online execution. Each figure is 12 × 3.8 inches (3600 × 1140 pixels at 300 dpi),
43% shorter than the previous 12 × 6.7-inch layout at the same width. Flow labels
retain the key distinction: revision between reset rollouts versus decisions
within a continuing episode. Right-hand tasks use two rows and three columns.
Online Ring's start panel follows the shared lift; that prefix remains included
in episode cost. The object is a thick hexagonal ring.

The original three-stage renders below remain available as source illustrations.

**These are constructed simulator configurations, not recorded policy trajectories.**
They use the task scene assets and object dimensions. Object poses are set for
illustration; no policy, model request, physics rollout, or success evaluator is
run. The arrows indicate task progression, not a measured motion path. These
figures do not contribute observations to the reported success rates.

## Figures

| Setting | Task | Three-panel figure | Stages |
| --- | --- | --- | --- |
| Single policy | Bridge + cylinder | [PDF](single_bridge.pdf) · [PNG](single_bridge.png) | Separate objects; span two foundations; place the cylinder |
| Single policy | Three-cube stacking | [PDF](single_stacking.pdf) · [PNG](single_stacking.png) | Separate cubes; place the middle cube; complete the ordered stack |
| Single policy | Ring over peg | [PDF](single_ring.pdf) · [PNG](single_ring.png) | Ring beside the peg; align the opening; lower and release |
| Online | Bridge-tower construction | [PDF](online_tower.pdf) · [PNG](online_tower.png) | Separate objects; construct the bridge; stack above the beam |
| Online | Ring placement | [PDF](online_ring.pdf) · [PNG](online_ring.png) | After shared grasp/lift; observe and align; release around the peg |
| Online | Tower relocation | [PDF](online_relocation.pdf) · [PNG](online_relocation.png) | Source stack and destination; temporary placement; rebuild on the destination |

Combined three-row galleries:

- [Single-policy tasks (PDF)](single_policy_tasks.pdf) · [PNG](single_policy_tasks.png).
- [Online tasks (PDF)](online_tasks.pdf) · [PNG](online_tasks.png).

Each individual frame is a native **1200 × 900** MuJoCo render, with 4× offscreen
multisampling. Files ending in `_00.png`, `_01.png`, and `_02.png` correspond to
the three stages. They are rendered from geometry rather than enlarged from the
low-resolution policy inputs. Composite PNGs use 300 dpi; PDFs retain vector text
and embedded render images. The policy observation resolution is unchanged.

The robot and debug markers are hidden to expose object relationships. Suspended
rings depict alignment configurations, not unsupported equilibrium states. Each
panel keeps the same viewing angles but adjusts its framing to the current object
extent. Panels therefore do not share a metric image scale. Colors identify
objects within each task; they do not encode Base/Full conditions or match a
particular reserved episode. Seed 1 selects scene assets/colors; layouts and
intermediate poses are constructed for display.

Online Ring starts after the shared grasp/lift prefix, highlighting the subsequent
online alignment and release decisions. This does not omit the prefix from episode
cost accounting. Relocation illustrates one possible temporary placement, not a
required or supplied plan. Both foundation positions remain unchanged in the
illustration. The online studies allow recovery without resetting; these diagrams
show intended progression rather than the distribution of observed failures.

## Paper captions

Editable LaTeX figure blocks and captions are in
[task_illustrations.tex](../../paper/task_illustrations.tex). They now reference the
two compact workflow/task figures.

**Single-policy acquisition.** The LLM revises a policy between reset rollouts;
frozen policies are evaluated on held-out layouts without test feedback. The
Full policy additionally queries supplied geometric abstraction during execution.
Right:
start (top) and goal (bottom) for Bridge + cylinder, three-cube stacking, and
hexagonal ring-on-peg placement. Task panels are constructed simulator
illustrations, not recorded trajectories.

**Online execution.** The LLM observes and acts in the same scene until success
or a 200-second timeout, without resetting after failure. The dashed Full-only
path supplies geometric text for reasoning. Right: start (top) and
goal (bottom) for bridge-tower construction, ring-on-peg placement, and tower
relocation. Ring starts here after the shared grasp/lift, which remains included
in episode timing. Task panels are constructed simulator illustrations, not
recorded trajectories.

Robots are omitted and panel framing varies. Model settings, observation
conditions, training-only retention, skill assistance, and timing details remain
in the experiment reports rather than crowding the figure.

## Reproduction and provenance

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/render_paper_task_schematics.py
```

Compose the primary workflow/task figures from the existing native renders:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/compose_experiment_task_figures.py
```

This composition step makes no simulator or model calls. Its
[workflow manifest](workflow_manifest.json) records the source images, model
protocols, source hashes, and output hashes. PDFs preserve vector diagram elements
and text; SVGs retain editable text and embed their raster task panels.

The renderer creates temporary scene models and writes only this figure bundle
(and small review previews under `/tmp`). It does not modify frozen study sources
or original experiment outputs. [manifest.json](manifest.json) records source
hashes, output hashes, object identities, camera parameters, and all illustrated
configuration vectors. The states are visualization inputs, not evaluation data.

Some studies retain full rollout states, while others archive intermediate RGB-D
and terminal physics only. A consistent set of high-resolution **recorded**
three-stage trajectories cannot be recovered for all six tasks from those
archives. This bundle uses the same explicit schematic convention throughout.
