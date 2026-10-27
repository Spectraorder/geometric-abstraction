# Paper experiments: results, data, and figures

This is the common entry point for the three **single-policy acquisition** tasks
and three **continuous online** tasks. All six primary studies are complete.
Their experimental units and resource budgets differ, so they are reported
separately rather than pooled into a single success rate.

## Single-policy acquisition

[Results report](../SINGLE_POLICY_RESULTS.md) ·
[Setup and protocols](../SINGLE_POLICY_EXPERIMENTS.md) ·
[CSV/JSON and figure index](single_policy/README.md) ·
[Paper LaTeX](../paper/single_policy_experiments.tex)

Five paired training chains per task; R0–R3; five training layouts per pair;
100 shared held-out test layouts. Policies are retained using training success
sets only. Each study has 4,000 test slots; the three studies total 12,000 test
slots, 600 training slots, and 90 completed model generations, plus two interrupted
Ring requests with unknown tokens. Bridge uses the new contract-only study.

| Task | Base final retained SR | Full final retained SR | Difference |
| --- | ---: | ---: | ---: |
| Bridge | 56.0% | 65.4% | +9.4 pp |
| Three-cube stacking | 70.8% | 81.4% | +10.6 pp |
| Ring over peg | 0.0% | 38.0% | +38.0 pp |

Figures: [SR lines and cumulative token bars](single_policy/iteration_sr_tokens.pdf),
[editable SVG](single_policy/iteration_sr_tokens.svg),
[raw and retained learning curves](single_policy/learning_curves.pdf),
[cumulative-token curves](single_policy/learning_curves_tokens.pdf).

## Continuous online execution

[Expanded results and figure index](online_50/README.md) ·
[Expansion protocol](../ONLINE_SAMPLE_EXTENSION_50.md) ·
[Original five-pair report](../ONLINE_EXPERIMENTS.md) ·
[Paper main text](../paper/online_experiments.tex) ·
[Paper appendix](../paper/online_appendix.tex)

Fifty paired layouts per task (original 5 + new 45), with one selected episode
per arm/layout and 200 seconds without reset after failure. The revised analysis
uses **282 original episodes + 18 audited repeats**, replacing both arms of all
nine timing/GPU-flagged pairs. It contains 2,546 completed requests and 77
interruptions; four usages are recovered and 73 remain unknown. Shared prior
skills and action interfaces are matched within each task. Original outcomes,
new-cohort summaries, runtime epochs, and paired sensitivities remain available.

| Task | Success, Base / Full | Mean capped time, Base / Full | Mean recorded tokens per episode, Base / Full |
| --- | ---: | ---: | ---: |
| Bridge-tower | 12/50 / 28/50 | 198.03 / 185.26 s | 112,502 / 129,467 |
| Ring-on-peg | 49/50 / 50/50 | 115.85 / 77.23 s | 214,514 / 101,316 |
| Tower relocation | 15/50 / 42/50 | 186.43 / 139.18 s | 185,936 / 88,338 |

Timeouts count as 200 seconds. Tokens include verified terminal usage and exclude
unknown usage. Full's recorded token use increases on Tower; improvements are
task-dependent. Two independently verified terminal-image archive recoveries
remain included. The [replacement manifest](online_50/replacement_manifest.json)
records the post-hoc reporting amendment and all original/repeat mappings.
Superseded attempts and startup costs remain separate from selected-episode means.
The original-attempt dataset and the sensitivity excluding all nine affected
pairs are preserved; no best-of-two selection is used.

Figures: [success, time, and token bars](online_50/outcome_bars.pdf),
[success versus elapsed time](online_50/success_vs_time.pdf),
[model time and token use](online_50/model_resources.pdf),
[paired effects and sensitivity](online_50/paired_effects.pdf),
[every paired completion time](online_50/paired_completion_times.pdf).
All are available as PDF, PNG, and editable SVG.

The [targeted repeat](../ONLINE_TARGETED_REPEAT.md) is complete and incorporated.
The sample remains 50 pairs per task; it does not add independent layouts or
isolate geometry/feedback components. The [original pilot exports](online/README.md)
remain unchanged.

## Task illustrations

[Workflow diagrams, tasks, and captions](task_illustrations/README.md) ·
[Single-policy workflow + tasks](task_illustrations/single_policy_workflow_tasks.pdf) ·
[Online workflow + tasks](task_illustrations/online_workflow_tasks.pdf) ·
[Paper LaTeX](../paper/task_illustrations.tex)

Each primary figure pairs a workflow diagram with start/goal task panels;
intermediate panels are omitted. Native task renders are 1200 × 900.
PDF, PNG, and editable SVG versions are available.
These are explicitly labeled constructed task schematics, not recorded policy
trajectories or additional experimental evidence. Policy input resolution and
all original study outputs are unchanged.

## Supplementary submission bundle

[Submission guide](SUPPLEMENTARY_MATERIALS.md) ·
[Sparse replay videos](videos/README.md) ·
[Generated ZIP](../../dist/paper_supplement.zip)

The existing ZIP contains the **original five-pair online results**; it has not
yet been rebuilt with `online_50/`. It contains self-contained `single_policy/`, `online/`, and `videos/`
folders, a common README, and file hashes. Six selected successful Full examples
use archived observations with explicit sparse-frame labels. Online terminal
frames are separately labeled renders of archived states; no experiment is rerun.
The video guide records selection, timing, and source coverage.

Rebuild from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/render_submission_replays.py
python scripts/package_paper_supplement.py
```

The ZIP and its checksum are generated under ignored `dist/`; video sources and
metadata remain in this directory. The paper's PDF appendix is submitted separately.

## Using the materials

The result directories contain outcomes, resource tables, publication PDF/PNG
figures, data dictionaries, and source hashes. The reports provide study
attribution, protocol differences, all failures, and interpretation limits. The
LaTeX fragments contain editable main-text prose, result tables, and captions.
Figure paths in those fragments assume compilation from `docs/paper/`; adjust
paths when transferring to another paper project. A LaTeX compiler is not
installed in the export environment, so only structural checks were run here.

Reproduce data and figures from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/export_single_policy_results.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/export_online_results.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/export_online_extension_results.py
```

These commands read frozen evidence and run no simulations or model requests.
Original RGB-D archives and private traces remain in the study outputs; this
bundle consolidates results and analysis data, not every raw sensor file.
