# Paper fragments

The main text uses the single-policy workflow and learning curve, followed by the
online workflow, an outcomes/cost bar chart, and a model interaction cost plot.
Detailed single-policy numbers and the online numerical table, success-versus-time
plot, and paired-effects plot are in appendix fragments. Online figures and prose
now use the completed 50-pair expansion; original five-pair exports remain intact.
The shared [geometry appendix](geometric_abstraction_appendix.tex) explains
RGB-D grounding, boundary and opening geometry, evidence, and text feedback,
with short stacking and ring examples.

| Setting | Main text | Appendix |
| --- | --- | --- |
| Single policy | [single_policy_experiments.tex](single_policy_experiments.tex) | [single_policy_appendix.tex](single_policy_appendix.tex) |
| Online | [online_experiments.tex](online_experiments.tex) | [online_appendix.tex](online_appendix.tex) |

```latex
% Preamble: \usepackage{graphicx}
\input{single_policy_experiments}
\input{online_experiments}

% After the paper's main sections:
\appendix
\input{geometric_abstraction_appendix}
\input{single_policy_appendix}
\input{online_appendix}
```

The main sections already contain the workflow/task figures. Do not also include
those figure blocks from `task_illustrations.tex`.

All image paths are relative to the paper root:

- `figures/single_policy_workflow_tasks.png` (main text)
- `figures/iteration_sr_tokens.png` (main text)
- `figures/online_workflow_tasks.png` (main text)
- `figures/outcome_bars.png` (main text)
- `figures/success_vs_time.png` (appendix)
- `figures/model_resources.png` (main text)
- `figures/paired_effects.png` (appendix)

The local `figures/` entries link to canonical exports. When transferring the
fragments to another paper project, copy the actual PNG files into its `figures/`
folder. No LaTeX compiler is installed in this workspace; source structure,
cross-references, and image paths are checked.

The single-policy table uses the `all_100` rows of
[mean_curves.csv](../evaluations/single_policy/mean_curves.csv): success is mean
plus/minus sample SD across five chains; tokens are cumulative per-chain means
through R1, R2, and R3, matching the main figure.

The online appendix table uses [arm_summary.csv](../evaluations/online_50/arm_summary.csv).
All time/token entries are per-episode means across 50 episodes, including
failures. Model time uses `mean_all_model_seconds`, including interrupted waits,
matching the main-text resource plot; tokens use `mean_total_tokens` divided by
1,000. Requests and Images use `mean_completed_requests` and `mean_images`;
Images counts new attachments across all request attempts, not camera frames.
Unknown interrupted usage remains excluded. The current report uses all 18
audited repeats for nine historically flagged pairs; original outcomes and
superseded costs remain available alongside paired sensitivities in the
[expanded report](../evaluations/online_50/README.md). This reporting amendment
was made after viewing the repeats, applies to both arms, and retains worse
outcomes too. Intervals resample paired layouts, not individual model requests.
Four terminal usages are verified from original CLI/session counters; 73 selected
interrupted usages remain unknown. Completed requests still require a decision
returned within the episode. Figures and the appendix table use the same
selected 300-episode dataset.

Detailed outcomes and protocols remain in the
[single-policy report](../SINGLE_POLICY_RESULTS.md) and
[expanded online report](../evaluations/online_50/README.md). The
[five-pair pilot report](../ONLINE_EXPERIMENTS.md) remains historical.

The geometry appendix is based on the
[methodology notes](../GEOMETRIC_ABSTRACTION_METHODOLOGY.md),
[block observer](../../src/aspire_so100/pnp_observation.py),
[symbolic references](../../src/aspire_so100/symbolic_scene.py),
[ring observer](../../scripts/ring_peg_geometry.py), and
[online ring feedback](../../scripts/online_ring_loop.py).
The detailed [ring study](../ONLINE_RING_STUDY.md) specifies the shared grasp
assistance and the geometry fields exposed to Full.
