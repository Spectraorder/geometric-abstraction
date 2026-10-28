# Documentation index

## Current paper results

The [combined experiment index](evaluations/README.md) collects the three
single-policy and three online tasks, with data, figures, and source attribution.

| Setting | Results and data | Paper text |
| --- | --- | --- |
| Single-policy acquisition | [Report](SINGLE_POLICY_RESULTS.md), [exports](evaluations/single_policy/README.md) | [Setup](SINGLE_POLICY_EXPERIMENTS.md), [main text](paper/single_policy_experiments.tex), [appendix](paper/single_policy_appendix.tex) |
| Online, 50 paired layouts per task | [Results and figures](evaluations/online_50/README.md), [expansion](ONLINE_SAMPLE_EXTENSION_50.md), [paired repeats](ONLINE_TARGETED_REPEAT.md) | [Main text](paper/online_experiments.tex), [appendix](paper/online_appendix.tex) |

The online report incorporates all nine audited repeat pairs, including regressions.
Original attempts, runtime epochs, unknown token usage, and paired sensitivities
remain available. The [original five-pair report](ONLINE_EXPERIMENTS.md) and
[exports](evaluations/online/README.md) are historical results.

For figure and submission work, see the [paper source guide](paper/README.md),
[method and pitch figures](media/README.md),
[task illustrations](evaluations/task_illustrations/README.md), and
[supplementary materials](evaluations/SUPPLEMENTARY_MATERIALS.md). The existing
supplement ZIP still contains the original five-pair online report.

## Method and implementation

- [Geometric abstraction for agentic policy](GEOMETRIC_ABSTRACTION_METHODOLOGY.md):
  geometric grounding, typed references, textual state, and worked examples.
- [Short method overview](METHOD.md) and [Base/Full interface](VISUAL_INPUT_COMPARISON.md).
- [Implementation appendix](paper/geometric_abstraction_appendix.tex).
- [Point-based geometry design](NOVEL_OBJECT_GEOMETRY_DESIGN.md) and
  [validation](NOVEL_OBJECT_GEOMETRY_VALIDATION.md): observer scope and evidence labels.
- [Policy build instructions](../policies/README.md): canonical controllers and sessions.
- [Script guide](../scripts/README.md): reporting commands, module map, and frozen drivers.

## Protocols for the paper studies

| Task | Protocol |
| --- | --- |
| Single-policy Bridge + cylinder | [Contract-only bridge](BRIDGE_CONTRACT_ONLY_STUDY.md) |
| Single-policy three-cube stacking | [Contract-only stacking](THREE_CUBE_CONTRACT_ONLY_STUDY.md) |
| Single-policy Ring-on-peg | [Paired ring study](RING_PEG_COMPARISON_STUDY.md) |
| Online Bridge-tower | [200-second study](ONLINE_COMPACT_TOWER_200_STUDY.md) |
| Online Ring-on-peg | [Staged alignment and release](ONLINE_RING_STUDY.md) |
| Online Tower relocation | [Relocation study](ONLINE_TOWER_RELOCATION_STUDY.md) |
| Online expansion and repeats | [50-pair expansion](ONLINE_SAMPLE_EXTENSION_50.md), [targeted repeat](ONLINE_TARGETED_REPEAT.md) |

Single-policy actors revise one executable program between rollouts. Online
agents instead use shared prior skills and make decisions within an episode.
These settings have different resource budgets and experimental units.

## Development and diagnostics

- [Event-driven execution](ONLINE_EVENT_DRIVEN_DEVELOPMENT.md): segmented actions,
  exception handling, and bounded interaction history.
- [Reasoning-state development](ONLINE_REASONING_STATE_DEVELOPMENT.md): geometric
  feedback, physical failures, and contact-evaluator diagnostics.
- [Explicit target binding](ONLINE_TARGET_BINDING_DEVELOPMENT.md) and
  [compact geometric text](ONLINE_COMPACT_GEOMETRY_TEXT.md).
- [Ring training pilot](RING_PEG_SINGLE_POLICY.md) and
  [timeout diagnosis](RING_PEG_TIMEOUT_DIAGNOSIS.md).
- [Conditional motion memory](CONDITIONAL_ATTACHMENT_EVIDENCE.md) and
  [release/reacquisition prototype](EXPLICIT_OPEN_GRIPPER_PROTOTYPE.md).

Development checks and prototypes are separate from the paper comparisons.
Their reports retain failed checks as well as successful ones.

## Earlier studies

These documents explain how the interfaces developed; their launch commands and
progress statements describe the corresponding historical study.

- [Budget-limited geometry results](PARTIAL_GEOMETRY_BUDGET_LIMITED_RESULTS.md),
  [registered protocol](PARTIAL_GEOMETRY_LEARNING_PROTOCOL.md), and
  [earlier visual-input comparison](VISUAL_INPUT_LEARNING_RESULTS.md).
- [Three-cube study with executor source disclosed](THREE_CUBE_GEOMETRY_STUDY.md).
- [Online bridge-tower pilot](ONLINE_BRIDGE_TOWER_PILOT.md) and
  [results](ONLINE_BRIDGE_TOWER_RESULTS.md).
- [Controlled recovery](ONLINE_CONTROLLED_RECOVERY.md) and
  [results](ONLINE_CONTROLLED_RECOVERY_RESULTS.md).
- [Explicit-target comparison](ONLINE_TARGET_BINDING_STUDY.md) and
  [results](ONLINE_TARGET_BINDING_RESULTS.md).
- [Original five-pair Tower results](ONLINE_COMPACT_TOWER_200_RESULTS.md).
- [Early geometry-training reports](history/early_geometry/README.md): diagnostic,
  five-round, observation-only, and interface-contract studies.
- [Earlier experiment guide](RUN_EXPERIMENTS.md) and [history index](history/README.md).

Frozen sources, protocols, and original results remain unchanged. See `AGENTS.md`
for campaign-specific execution constraints; reporting does not restart a study.
