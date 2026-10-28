<div align="center">

# Do Agentic Policies Need to Look at Every Step?<br>Geometric Abstraction for Reasoning and Action

**CoRL 2026 Workshop on Agentic Robotics** (Poster)

[Yihang Liu](mailto:yihang.liu.pc@gmail.com)<sup>1</sup> &nbsp;·&nbsp;
[Yiming Chen](mailto:ychen3868@gatech.edu)<sup>2</sup> &nbsp;·&nbsp;
[Mingzhe Zheng](mailto:mzhengar@connect.ust.hk)<sup>3</sup>

<sup>1</sup>Independent Researcher &nbsp;&nbsp;
<sup>2</sup>Georgia Institute of Technology &nbsp;&nbsp;
<sup>3</sup>The Hong Kong University of Science and Technology

<!-- TODO: replace the placeholder links once they are public. -->
[**Paper**](#) &nbsp;|&nbsp; [**arXiv**](#) &nbsp;|&nbsp; [**Project page**](#) &nbsp;|&nbsp; [**Videos**](docs/evaluations/videos/README.md) &nbsp;|&nbsp; [**BibTeX**](#citation)

<img src="docs/media/agentic_geometry_method.png" width="90%" alt="Geometric abstraction between perception and reasoning">

</div>

## Overview

Multimodal language models can reason directly about robot scenes, but do they
need to reinterpret raw images at every decision step? We maintain task-relevant
3D information as a **geometric abstraction** that is grounded locally from RGB-D
and read directly by the language model:

- **Typed, evidence-aware references** to object parts (`red.bottom_face`,
  `blue.top_face`, centers, edges, vertices), each marked *observed*, *inferred*, or
  *unavailable*.
- **Geometric operators** (`reference`, `relation`, `align`, `checkpoint`) that
  compose those references into placements and support checks.
- **Compact textual checkpoints** after each action, reporting current geometry,
  evidence status, and the residual to the goal. Images stay available on demand.

Shape may persist across steps, but pose is always re-estimated from the current
observation. Perception, geometry, and control run on CPU; MuJoCo simulation and
rendering may use the GPU.

We evaluate the interface on a simulated SO-100 arm in two settings:

| Setting | What the LLM does | Tasks |
| --- | --- | --- |
| **Single-policy acquisition** | Revises an executable Python policy between reset rollouts (R0–R3); no LLM calls during execution | Bridge + cylinder, three-cube stacking, ring-on-peg |
| **Continuous online execution** | Observes, reasons, and acts in one scene until success or a 200 s timeout, with no reset after failures | Bridge-tower, ring-on-peg, tower relocation |

**Base** and **Full** agents share the task specification, calibrated RGB-D access,
skills, and action interface. Full additionally receives the geometric abstraction.
Neither agent sees simulator object poses, dimensions, contacts, or evaluator state.

## Installation

### Requirements

- Linux with an NVIDIA GPU and EGL for MuJoCo rendering. Headless is fine. CPU-only
  works for the perception and reporting code.
- Python **3.14+**.
- [uv](https://docs.astral.sh/uv/) for the locked environment.
- To run new LLM experiments, you also need the
  [OpenAI Codex CLI](https://github.com/openai/codex) (`codex`) and an account with
  access to the models used in the paper (`gpt-6-astra`). Reading the results,
  running tests, and viewing scenes do not need model access.

### Setup

```bash
git clone https://github.com/Spectraorder/geometric-abstraction.git
cd geometric-abstraction
uv sync --locked            # installs mjlab 1.6.0, MuJoCo 3.11, and pinned dependencies
```

For headless rendering, and to keep CPU timing comparable to the paper:

```bash
export MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

For LLM experiments, log in once with the Codex CLI. Actor sessions copy
`$CODEX_HOME/auth.json` (default `~/.codex/auth.json`) into an isolated home for
each run:

```bash
codex login
```

### Verify the installation

```bash
uv run python -m unittest discover -s tests -p 'test_online_*.py'   # reporting / interface tests
uv run python -m unittest discover -s tests                          # full suite
```

The tests use small fixtures in `tests/fixtures/` and make no model requests.

## Quick start

**View the scene and object library.** This opens an interactive Viser viewer:

```bash
uv run python view_scene.py --scene-only --seed 51                 # six sampled objects
uv run python view_scene.py --scene-only --scene-task catalog      # all 20 objects: 5 shapes x 4 colors
```

**Run a sensor-only RGB-D controller and replay it.** First build a standalone
policy from a canonical controller, then run it:

```bash
uv run python scripts/build_inventory_policy.py \
  --controller structure_policy --output outputs/generated/structure.py
uv run python view_scene.py --vision-task inventory_structure \
  --vision-policy outputs/generated/structure.py --vision-output outputs/demo_structure
```

**Rebuild the paper figures** from the committed data:

```bash
uv run python scripts/render_paper_task_schematics.py      # task and workflow illustrations
uv run python scripts/render_agentic_geometry_method.py    # method figure
```

## Results

### Single-policy acquisition (paper Table 2)

Mean ± SD test success over five training chains on 100 shared held-out layouts,
using training-only policy retention.

| Task | Condition | R1 | R2 | R3 | Tokens R1–R3 (k) |
| --- | --- | ---: | ---: | ---: | ---: |
| Bridge + cylinder | Base | 21.2 ± 22.0 | 53.0 ± 11.7 | 56.0 ± 9.1 | 125.7 |
|  | **Full** | **45.2 ± 25.4** | **63.4 ± 5.9** | **65.4 ± 3.5** | 127.0 |
| Three-cube stacking | Base | 17.2 ± 29.4 | 53.6 ± 26.2 | 70.8 ± 26.5 | 138.1 |
|  | **Full** | **57.2 ± 32.6** | **72.2 ± 7.2** | **81.4 ± 12.3** | **123.4** |
| Ring-on-peg | Base | 0.0 | 0.0 | 0.0 | 166.8 |
|  | **Full** | 0.0 | 0.0 | **38.0 ± 34.9** | **163.1** |

### Continuous online execution (paper Table 3)

Five paired layouts per task, 200 s budget, no reset. Timeouts count as 200 s.

| Task | Condition | Success | Time (s) | Model (s) | Tokens (k) | Requests | Images |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bridge-tower | Base | 2/5 | 195.9 | 46.2 | 101.4 | 7.2 | 6.6 |
|  | **Full** | **4/5** | **174.1** | 45.7 | 107.8 | 6.4 | **4.0** |
| Ring-on-peg | Base | 5/5 | 119.8 | 94.4 | 233.8 | 12.6 | 12.4 |
|  | **Full** | 5/5 | **74.2** | **41.8** | **93.0** | **6.6** | **2.0** |
| Tower relocation | Base | 2/5 | 190.6 | 80.0 | 181.4 | 11.4 | 11.0 |
|  | **Full** | **3/5** | **164.3** | **50.6** | **151.2** | **8.2** | **4.0** |

On online ring-on-peg, Full reduces mean completion time by **38.0%**, model-request
time by **55.7%**, and recorded tokens by **60.2%**.

**Extended evaluation.** After the paper experiments, each online task was extended
to **50 paired layouts** (300 episodes). Those results, figures, and audit trail are
in [`docs/evaluations/online_50/`](docs/evaluations/online_50/README.md). The paper's
five-pair exports are unchanged in
[`docs/evaluations/online/`](docs/evaluations/online/README.md).

All tables are exported as CSV/JSON with publication figures (PDF, PNG, editable
SVG). Start from the [evaluation index](docs/evaluations/README.md).

## Reproducing the paper

### 1. Regenerate tables and figures from recorded evidence

These exporters read the frozen experiment archives. They run no simulation and
make no model requests:

```bash
uv run python scripts/export_single_policy_results.py          # -> docs/evaluations/single_policy/
uv run python scripts/export_online_results.py                 # -> docs/evaluations/online/     (paper, 5 pairs)
uv run python scripts/export_online_extension_results.py       # -> docs/evaluations/online_50/  (50 pairs)
uv run python scripts/export_online_extension_results.py --check   # audit only, writes nothing
```

> **Note:** The raw archives (`outputs/`: RGB-D observations, model transcripts,
> rollouts) are not in Git because of their size. A fresh clone contains code,
> fixtures, reports, and the exported CSV/JSON/figures, which can be used directly.
> Regeneration requires the archives. <!-- TODO: add the data download link. -->

### 2. Re-run the experiments

Each study has a driver and a protocol document that specifies seeds, budgets,
and audits. Running them requires the Codex CLI and model access (see
[Installation](#installation)).

| Study | Driver (`scripts/`) | Protocol |
| --- | --- | --- |
| Single-policy Bridge + cylinder | `run_bridge_contract_study.py` | [docs](docs/BRIDGE_CONTRACT_ONLY_STUDY.md) |
| Single-policy three-cube stacking | `run_three_cube_contract_study.py` | [docs](docs/THREE_CUBE_CONTRACT_ONLY_STUDY.md) |
| Single-policy ring-on-peg | `run_ring_peg_comparison_study.py` | [docs](docs/RING_PEG_COMPARISON_STUDY.md) |
| Online bridge-tower | `run_online_compact_study.py` | [docs](docs/ONLINE_COMPACT_TOWER_200_STUDY.md) |
| Online ring-on-peg | `run_online_ring_checked_study.py` | [docs](docs/ONLINE_RING_STUDY.md) |
| Online tower relocation | `run_online_relocation_study.py` | [docs](docs/ONLINE_TOWER_RELOCATION_STUDY.md) |
| Online 50-pair extension | `run_online_extension_study.py` | [docs](docs/ONLINE_SAMPLE_EXTENSION_50.md) |

Single-policy drivers run in phases (`prepare`, `run`, `check`, `summarize`):

```bash
uv run python scripts/run_ring_peg_comparison_study.py --root outputs/my_ring_study --phase prepare
uv run python scripts/run_ring_peg_comparison_study.py --root outputs/my_ring_study --phase run
```

Use a new `--root` for new runs. The drivers refuse to overwrite or resume a
completed registered study. The online ring-on-peg agent reuses the grasp-and-lift
prefix selected from the single-policy ring training, so run that study first.
See the [script guide](scripts/README.md) for all entry points.

## Repository structure

```
geometric-abstraction/
├── src/aspire_so100/   # RGB-D perception, geometric state, symbolic references, MuJoCo integration
├── policies/           # Task controllers, starters, and policy sessions (Base / Full interfaces)
├── scripts/            # Experiment drivers, online agent loops, audits, exporters, renderers
│   └── online_reporting/   # Evidence validation, statistics, and plotting for the online report
├── prototypes/         # Opt-in geometry extensions
├── tests/              # Unit and integration tests with small fixtures
├── assets/             # SO-100 robot model
├── docs/
│   ├── evaluations/    # Exported results (CSV/JSON) and publication figures
│   ├── paper/          # LaTeX fragments for the experiments and appendices
│   └── ...             # Method notes, study protocols, and historical reports
└── view_scene.py       # Interactive scene viewer and policy replay
```

Key implementation files:
- [`pnp_observation.py`](src/aspire_so100/pnp_observation.py): block geometry from RGB-D.
- [`symbolic_scene.py`](src/aspire_so100/symbolic_scene.py): typed symbolic references.
- [`ring_peg_geometry.py`](scripts/ring_peg_geometry.py): ring observer.
- [`online_ring_loop.py`](scripts/online_ring_loop.py): online textual checkpoints.

The [methodology notes](docs/GEOMETRIC_ABSTRACTION_METHODOLOGY.md) explain the design.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{liu2026geometric,
  title     = {Do Agentic Policies Need to Look at Every Step? Geometric Abstraction for Reasoning and Action},
  author    = {Liu, Yihang and Chen, Yiming and Zheng, Mingzhe},
  booktitle = {CoRL 2026 Workshop on Agentic Robotics},
  year      = {2026}
}
```

## Acknowledgements

The simulation builds on [MuJoCo](https://github.com/google-deepmind/mujoco) and
[mjlab](https://github.com/mujocolab/mjlab). The SO-100 model in `assets/` is the
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) description
of [The Robot Studio's SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(Apache 2.0). The
closed-loop execution design was inspired by
[Code as Policies](https://code-as-policies.github.io/) and ASPIRE.

## License

The code in this repository is released under the [MIT License](LICENSE).

The SO-100 robot model in [`assets/trs_so_arm100/`](assets/trs_so_arm100/) keeps its
original [Apache 2.0 license](assets/trs_so_arm100/LICENSE). The paper is licensed
under CC BY 4.0.
