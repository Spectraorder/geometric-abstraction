# Script guide

Run commands from the repository root with the locked environment (`uv sync --locked`).
For published results, start with [the evaluation index](../docs/evaluations/README.md).

## Current data and figure exports

| Entry point | Output |
| --- | --- |
| `export_single_policy_results.py` | Three single-policy studies in `docs/evaluations/single_policy/` |
| `export_online_extension_results.py` | Current 50-pair online report in `docs/evaluations/online_50/` |
| `export_online_results.py` | Historical five-pair online report in `docs/evaluations/online/` |

These exporters read archived evidence and make no model requests or simulation
rollouts. They require the original `outputs/` archives, which are not included
in a fresh clone. The committed CSV/JSON data and figures are usable on their own.

```bash
# Audit the current online report without writing any exports.
.venv/bin/python scripts/export_online_extension_results.py --check

# Rebuild into a separate directory for comparison or figure work.
.venv/bin/python scripts/export_online_extension_results.py --output /tmp/online-report

# Update the canonical online export after reviewing changes.
.venv/bin/python scripts/export_online_extension_results.py
```

The online CLI delegates to `online_reporting/`:

| Module | Responsibility |
| --- | --- |
| `evidence.py` | Read and hash-check original evidence, verify usage, and apply the registered paired substitutions |
| `analysis.py` | Pair layouts, select cohorts, compute summaries/intervals, and construct success-time curves |
| `plotting.py` | Render the five publication figures as PDF, PNG, and editable SVG |

Each evidence collection has its own source-hash ledger. Statistical helpers do
not read files; plotting receives already-validated records and an explicit output
directory. `online_interrupted_usage.py` verifies interrupted token recoveries.
The export provenance records both source hashes and reporting module hashes.

Both arms of every selected repeat pair replace their originals, including worse
outcomes. Historical exclusions still define the quality-sensitivity cohorts.
Original attempts and superseded costs remain separate exports. These rules are
part of the report, not options for choosing more favorable results.

## Tests

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -p 'test_online_*.py'
```

The reporting tests use small fixtures without the private experiment archives.
With those archives present, `--check` additionally verifies the complete study.
For a reporting refactor, export to a temporary directory and compare all CSV,
JSON, and figure files with the committed export. Code hashes in `provenance.json`
change when reporting code changes; measured values should not change.

## Paper illustrations and supplementary material

| Entry point | Purpose |
| --- | --- |
| `render_paper_task_schematics.py` | Constructed task and workflow illustrations; not policy trajectories |
| `render_agentic_geometry_method.py` | Editable method diagram |
| `render_geometric_abstraction_pitch.py` | Editable pitch figure |
| `render_submission_replays.py` | Sparse observation-frame videos with source and timing labels |
| `package_paper_supplement.py` | Submission ZIP; currently packages the historical five-pair online export |

See the [illustration guide](../docs/evaluations/task_illustrations/README.md) and
[submission guide](../docs/evaluations/SUPPLEMENTARY_MATERIALS.md) for exact options
and prerequisites.

## Experiment drivers and frozen runtime

| Study | Driver / protocol |
| --- | --- |
| Single-policy Bridge + cylinder | `run_bridge_contract_study.py` / [protocol](../docs/BRIDGE_CONTRACT_ONLY_STUDY.md) |
| Single-policy stacking | `run_three_cube_contract_study.py` / [protocol](../docs/THREE_CUBE_CONTRACT_ONLY_STUDY.md) |
| Single-policy Ring | `run_ring_peg_comparison_study.py` / [protocol](../docs/RING_PEG_COMPARISON_STUDY.md) |
| Original online Tower | `run_online_compact_study.py` / [protocol](../docs/ONLINE_COMPACT_TOWER_200_STUDY.md) |
| Original online Ring | `run_online_ring_checked_study.py` / [protocol](../docs/ONLINE_RING_STUDY.md) |
| Original online Relocation | `run_online_relocation_study.py` / [protocol](../docs/ONLINE_TOWER_RELOCATION_STUDY.md) |
| Online expansion | `run_online_extension_study.py`, registered resume drivers / [protocol](../docs/ONLINE_SAMPLE_EXTENSION_50.md) |
| Targeted repeats | `run_online_targeted_repeat.py` / [protocol](../docs/ONLINE_TARGETED_REPEAT.md) |

These studies are complete. Resume scripts record specific repairs and are not
general retry entry points. Runtime modules and some older exporters are pinned
by registered source hashes, so they remain at their original paths. Refactor
current reporting separately; consult the matching protocol before changing or
running a driver. Earlier development and audit scripts remain for traceability.
