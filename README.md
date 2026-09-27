# CPU geometric abstraction for agentic manipulation

This project converts calibrated RGB-D observations of simple rigid blocks into
compact, target-focused geometric state for policy reasoning. Perception, geometry
and policy execution run on CPU; simulation and rendering may use GPU.

The current comparison is **Visual Base versus Full geometry** with the same
hidden visual pick/place actuator. Neither policy receives simulator object poses,
dimensions, contacts or evaluator state. The task is a low-rise bridge with two
cubes, a thin prism and a cylinder, with unique colors and varied layouts.

## Read first

- [Run experiments](docs/RUN_EXPERIMENTS.md): setup, registration, training, testing and audit.
- [Documentation index](docs/README.md): method, interface, protocol and results.
- [Method](docs/METHOD.md): what the abstraction contains and how it updates.
- [Latest results](docs/PARTIAL_GEOMETRY_BUDGET_LIMITED_RESULTS.md): preliminary,
  budget-limited evidence, including failures and regressions.

Historical source archives and generated policy copies are kept in Git history,
not duplicated in the active tree.

The latest campaign stopped at **103 reserved rollouts**, following 192 training
rollouts. In the tested chain, first-revision SR is 20% for Base and 75% for Full,
from a common 40% starter. This does not establish a stable advantage across
independent learning chains. Do not resume evaluation without a new user request.

## Local environment and checks

The project requires Python 3.14+ and uses `uv.lock`:

```bash
uv sync --locked
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  uv run python -m unittest discover -s tests -p 'test_partial_geometry.py' -v
```

To regenerate paired outcome analysis from existing local records without new
rollouts or model calls:

```bash
uv run python scripts/analyze_budget_limited_pairs.py
```

The analysis requires the retained `outputs/constraint_geometry_learning_replications_01`
folder. Raw rollouts and local credentials are ignored by Git; this repository
contains code, fixtures, reports and selected visual evidence, not the complete
raw dataset. Do not assume a fresh clone includes experiment recordings.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `policies/` | Canonical controllers, shared sessions, compact state and starters |
| `src/aspire_so100/` | CPU perception/control and simulation integration |
| `scripts/` | Frozen experiment drivers, auditing and analysis |
| `tests/` | Interface, geometry, provenance and recorded-fixture checks |
| `docs/` | Current documentation, evaluation summaries and figures |
| `docs/history/` | Earlier designs, experiments and the old project guide |

Current experiment actors do **not** load a task-skill library. They receive the
registered interface, starter and their own bounded training feedback. Historical
controllers and scripts remain when needed to reproduce earlier results or build
shared components. Start from the documentation index rather than old launch commands.
