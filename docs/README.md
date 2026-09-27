# Current experiment documentation

Use these documents in order; the historical archive is not required reading.

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

- [Conditional motion memory](CONDITIONAL_ATTACHMENT_EVIDENCE.md): offline point/region transport, contradictions and uncertainty.
- [Explicit release/reacquisition](EXPLICIT_OPEN_GRIPPER_PROTOTYPE.md): separate human-designed counterfactual, not agent-learning evidence.

These two documents remain here because the registered protocol links them.
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
