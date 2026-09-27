# Current experiment documentation

Start with [Running experiments](RUN_EXPERIMENTS.md) for commands and budget accounting.
For background, use the documents below; historical material is optional.

The new [three-cube stacking study](THREE_CUBE_GEOMETRY_STUDY.md) uses the updated
geometry bundle and plans all iteration curves before training.

For paper preparation, see [Geometric abstraction for agentic policy](GEOMETRIC_ABSTRACTION_METHODOLOGY.md),
covering the text and symbolic interface, single-policy training, and continuous online evaluation.

The [online bridge-tower pilot](ONLINE_BRIDGE_TOWER_PILOT.md) completed 20 episodes
with shared skills, no resets within an episode and explicit latency metrics.
The [results](ONLINE_BRIDGE_TOWER_RESULTS.md) report equal 60% success, fewer Full
image attachments, and limited end-to-end time savings.

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

The historical single-policy comparison uses no task-skill library: its actors
receive the registered interface and their own training feedback. The separate
online bridge-tower pilot adapts two training-selected policies into shared skills.
Old task guidance and actor-loop code remain in Git history.

## Development after cleanup

Task controllers now have one canonical source under `policies/controllers/`.
Standalone candidates are generated on demand; shared geometry, prediction and
relation capabilities remain in `src/aspire_so100`. See the
[policy build instructions](../policies/README.md).

Historical manifests and raw results are unchanged. New experiments register
current sources; the cleaned tree is not required to reproduce every historical
source layout. Use Git history if an old implementation needs inspection.

For the separate optimization attempt, see
[event-driven online development](ONLINE_EVENT_DRIVEN_DEVELOPMENT.md): cooperative
motion boundaries, exception-triggered reasoning, retry suppression and bounded
conversation history. These changes do not modify the completed pilot.

The [controlled online recovery study](ONLINE_CONTROLLED_RECOVERY.md) completed
five paired layouts with four concurrent workers. Its [results](ONLINE_CONTROLLED_RECOVERY_RESULTS.md)
show equal 4/5 success, shorter average first requests and fewer Full image
attachments, but higher Full total token use and no material completion-time gain.

[Constraint-aware online development](ONLINE_REASONING_STATE_DEVELOPMENT.md) records
compact support-chain evidence, relevant retry gates, remaining physical failures,
and a correction for unnamed robot collision meshes in future private evaluation.
Earlier online result reports include the resulting retrospective audit limitation.

The [explicit-target online comparison](ONLINE_TARGET_BINDING_STUDY.md) completed
five paired layouts with four workers and no resets within episodes. Its
[results and editable figure](ONLINE_TARGET_BINDING_RESULTS.md) report Full 3/5
versus Base 0/5 at 180 seconds, with fewer image attachments but higher recorded
token use. Subsequent explicit-target budgets are 200 seconds; historical outcomes
remain unchanged.

[Compact geometric text](ONLINE_COMPACT_GEOMETRY_TEXT.md) is an opt-in development
adapter with exact metadata reconstruction. Offline replay reduced Full's newly
supplied prompt bytes by 21.1%. A subsequent live pair at 200 seconds succeeded in
both conditions and used 6.7% fewer compact tokens, with no timing improvement;
this remains development evidence on one reused scene.

The completed [200-second geometric-text study](ONLINE_COMPACT_TOWER_200_STUDY.md)
ran five new paired layouts with four workers. Its
[results and paper figure](ONLINE_COMPACT_TOWER_200_RESULTS.md) report Full 4/5
versus Base 2/5, 11.1% lower mean capped time and fewer images, but more recorded
tokens and no mean per-request speedup. All ten outcomes and audits are retained.

The [contract-only stacking follow-up](THREE_CUBE_CONTRACT_ONLY_STUDY.md) removes
executor implementation from both actor prompts and registers all 800 evaluations
before training. The source-disclosed study remains separate evidence.
