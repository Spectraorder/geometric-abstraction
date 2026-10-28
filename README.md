# Geometric abstraction for agentic manipulation

This project converts calibrated visual observations of simple rigid objects into
metric geometry, symbolic references, and compact text for language-model policies.
Perception, geometric estimation, and policy control run on CPU; simulation and
rendering may use GPU. The experiments use a simulated SO-100 robot in MuJoCo.

We study two settings:

- **Single-policy acquisition:** an LLM revises a Python policy between reset
  rollouts. The three tasks are Bridge + cylinder, three-cube stacking, and
  ring-on-peg placement. Each task uses five paired training chains, R0–R3,
  and 100 shared held-out layouts.
- **Continuous online execution:** an LLM uses shared prior skills in one scene
  until success or a 200-second timeout, without resetting after failures.
  Bridge-tower, Ring-on-peg, and Tower relocation each have 50 paired layouts.

Base and Full share the action interface within each task. Full additionally
receives supplied geometry and feedback. Runtime policies do not receive simulator
object poses, dimensions, contacts, or private evaluator state.

## Start here

- [Results, data, and figures](docs/evaluations/README.md): the six paper comparisons.
- [Single-policy results](docs/evaluations/single_policy/README.md) and
  [50-pair online results](docs/evaluations/online_50/README.md): tables, plots, and provenance.
- [Methodology](docs/GEOMETRIC_ABSTRACTION_METHODOLOGY.md): geometric grounding and the policy interface.
- [Paper sources](docs/paper/README.md): editable LaTeX and figure links.
- [Script guide](scripts/README.md): reporting commands and experiment entry points.
- [Documentation index](docs/README.md): protocols, implementation, and historical studies.

## Local setup and verification

The project requires Python 3.14+ and uses `uv.lock`:

```bash
uv sync --locked
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python -m unittest discover -s tests -p 'test_online_*.py'
```

With the original experiment archives available locally, verify the current online
report without exporting files, or regenerate data and figures into a separate directory:

```bash
uv run python scripts/export_online_extension_results.py --check
uv run python scripts/export_online_extension_results.py --output /tmp/online-report
```

These commands make no model requests and run no experiments. The default export
destination is `docs/evaluations/online_50`. The report uses 282 original episodes
and 18 audited repeats replacing both arms of nine affected layout pairs;
original outcomes, costs, and the reporting amendment remain available.

Raw experiment archives under `outputs/` are ignored by Git and may reside on a
separate disk. A fresh clone contains code, fixtures, reports, and selected visual
evidence, but not every raw observation or model transcript. Regeneration needs
those archives; the published CSV/JSON exports can be read without them.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `src/aspire_so100/` | CPU perception/control and simulation integration |
| `policies/` | Controllers, shared sessions, geometric state, and policy scaffolds |
| `prototypes/` | Opt-in geometry extensions and development components |
| `scripts/` | Experiment drivers, audits, reporting, and rendering |
| `scripts/online_reporting/` | Current online evidence validation, statistics, and plotting |
| `tests/` | Geometry, interface, accounting, and provenance checks |
| `docs/evaluations/` | Consolidated data and publication figures |
| `docs/paper/` | Paper text and appendices |
| `docs/history/` | Earlier designs and experiment guides |

Registered runtime sources and recorded evidence are frozen. Use the matching
study protocol when inspecting an experiment; older launch commands do not resume
or extend completed studies. Reporting code reads existing evidence separately
from simulation and actor execution.
