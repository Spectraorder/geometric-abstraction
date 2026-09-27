# Current experiment documentation

Start with [Running experiments](RUN_EXPERIMENTS.md) for commands and budget accounting.
For background, use the documents below; historical material is optional.

The new [three-cube stacking study](THREE_CUBE_GEOMETRY_STUDY.md) uses the updated
geometry bundle and plans all iteration curves before training.

For paper preparation, see [Geometric abstraction for agentic policy](GEOMETRIC_ABSTRACTION_METHODOLOGY.md),
covering the text and symbolic interface, current policy training, and planned continuous online evaluation.

1. [Method](METHOD.md): compact geometry, state updates and limitations.
2. [Base/Full interface](VISUAL_INPUT_COMPARISON.md): matched input and action boundary.
3. [Latest results](PARTIAL_GEOMETRY_BUDGET_LIMITED_RESULTS.md): all retained reserved batches and paired outcomes.
4. [Registered protocol](PARTIAL_GEOMETRY_LEARNING_PROTOCOL.md): original design and provenance.
5. [Previous completed comparison](VISUAL_INPUT_LEARNING_RESULTS.md): independent chains and mixed/negative evidence.

The current campaign is stopped at the user's test budget. The protocol retains
its original bytes for hash verification, including historical progress statements
and the original 480-slot plan. Those statements do not authorize restarting it;
the latest results and `AGENTS.md` describe the current execution boundary.

## Supporting diagnostics

- [Point-based novel-block geometry](NOVEL_OBJECT_GEOMETRY_DESIGN.md): current design, API and tests.
- [Novel-block validation](NOVEL_OBJECT_GEOMETRY_VALIDATION.md): physical diagnostics, isolated variants, results and reproduction commands.
- [Conditional motion memory](CONDITIONAL_ATTACHMENT_EVIDENCE.md): offline point/region transport, contradictions and uncertainty.
- [Explicit release/reacquisition](EXPLICIT_OPEN_GRIPPER_PROTOTYPE.md): separate human-designed counterfactual, not agent-learning evidence.

The motion-memory and release/reacquisition documents remain here because the registered protocol links them.
Their prototypes are not additional treatment arms.

## Historical material

Earlier designs, task controllers, papers and experiments are in
[the history index](history/README.md). Machine-readable evidence remains in
`evaluations/`, and figures remain in `media/`, preserving source paths and hashes.
The archives retain failures as well as successes; they are not current instructions.

There is no active task-skill library in this experiment. Old task guidance and
actor-loop code are retained in Git history rather than a mirrored source archive.
Current actors receive the registered interface and their own training feedback.

## Development after cleanup

Task controllers now have one canonical source under `policies/controllers/`.
Standalone candidates are generated on demand; shared geometry, prediction and
relation capabilities remain in `src/aspire_so100`. See the
[policy build instructions](../policies/README.md).

Historical manifests and raw results are unchanged. New experiments register
current sources; the cleaned tree is not required to reproduce every historical
source layout. Use Git history if an old implementation needs inspection.
