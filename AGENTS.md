# Project direction

Build varied towers from the five-shape, four-color block inventory. The historical first
curriculum covers upright single columns. Current work prioritizes low-rise
support graphs, multiple foundations, bridges and stepped structures rather
than increasing height. Preserve reusable geometry capabilities; keep historical
results honest, but use Git history for old generated policies and compatibility.
Do not keep duplicate build artifacts solely to reproduce an earlier experiment.
Task controllers live in `policies/controllers`; shared geometry lives in `src`.

Use English for maintained documentation, code comments and docstrings.
Preserve recorded prompts, observations and experiment artifacts as evidence;
do not translate historical raw inputs or silently rewrite measured results.

Keep online perception, tracking, planning and control on CPU. GPU simulation
and rendering are allowed. Public task goals may specify object identities and
order, but runtime policies must not receive simulator object poses, dimensions,
contacts or evaluator state. Keep development and frozen evaluation separate.


# Current experiment budget

The user reduced reserved evaluation for
`outputs/constraint_geometry_learning_replications_01` to approximately 50–100
rollouts. Dispatch was stopped and its final in-flight rollout completed, retaining
103 reserved rollouts. Do not resume this campaign or extend its reserved-test
budget without a further user request for more tests. The original 480-slot
protocol remains unchanged as provenance; it is not current authorization to run
all remaining slots. Continue read-only analysis and report the study as budget
limited. See `user_budget_stop.json` in the study root and
`docs/PARTIAL_GEOMETRY_BUDGET_LIMITED_RESULTS.md`.

For subsequent geometry-abstraction experiments, the user selected three generated
policy-improvement rounds per arm (R1–R3), in addition to the shared R0 baseline.
Keep five shared training layouts and 20 shared test layouts unless changed by the
user. Preserve the completed five-round pilot and its test evidence unchanged.
The CLI already defaults to three rounds; register any new campaign separately.

The current geometry comparison removes the supplied task-solving starter and
recipe-specific prompt advice. Use `scripts/run_unseeded_geometry_experiment.py`
and `docs/UNSEEDED_GEOMETRY_EXPERIMENT.md` for this variant. R0 captures images
without acting; R1 is the first generated solution. Keep the declarative task goal
and shared generic actuator, but do not supply construction order or prior learned
policies. Preserve the historical capable-starter runs as separate evidence.

For new neutral-start runs with explicit shared API guidance, use
`scripts/run_interface_contract_geometry_experiment.py` and
`docs/INTERFACE_CONTRACT_EXPERIMENT.md`. Both arms receive the same interface
contract and task-independent checkpoint adapter. Do not alter or resume the
completed `geometry_unseeded_01` study with this changed prompt/starter.

For `outputs/geometry_interface_contract_01`, the user explicitly authorized five
rounds this time (R1–R5), overriding the usual three-round default for this run.
The budget is 60 training slots including R0 and ten model generations across both
arms. Training uses the five shared development layouts; reserved evaluation is
registered separately and has not been requested with this training launch.

For subsequent interface-guided runs, the user raised the policy source limit to
50000 characters. Use `scripts/run_expanded_budget_geometry_experiment.py` for all
phases; it scopes the larger limit to a distinct registered variant. Keep the
30000-character runner, shared contract and completed R5 invalid-slot records
unchanged. Raising the limit alone does not authorize another campaign or a rerun.

The user subsequently authorized test evaluation of the completed
`geometry_interface_contract_01` training. Evaluate the training-selected Base R4
and Full R4 on its 20 originally registered reserved layouts, four workers and
40 total rollouts, under `selected_evaluation_20/`. Use
`scripts/evaluate_selected_interface_policies.py`; this separate selected-policy
registration does not complete the original full-revision evaluation protocol.

The user authorized launching `outputs/geometry_parallel_01` after improving
parallel scheduling: five independent chains per arm, three generated rounds,
five paired training layouts per replica, and 100 common fresh test layouts.
Use `scripts/run_parallel_geometry_study.py` and `docs/PARALLEL_GEOMETRY_STUDY.md`.
Global limits are four model requests and four rollouts for this study. Training
must finish and freeze across all replicas before selected-policy tests. Budget:
30 model calls, 200 training slots, 1000 test rollouts. Keep incomplete requests
and STOP receipts for inspection; never silently retry or overwrite them.

The user subsequently authorized evaluating every frozen iteration (R0-R3) of
`geometry_parallel_01` on the same 100 test layouts to obtain learning curves.
Use `scripts/evaluate_geometry_learning_curves.py`, with the amendment under
`outputs/geometry_parallel_01/learning_curves`. Reuse 1000 selected-policy results;
add 3000 rollout slots, no model calls. Up to 16 global rollout workers are allowed,
with a 32-rollout hardware pilot and a registered fallback to eight workers.
New rollout data lives at `/mnt/ssd/agentic_policy/geometry_parallel_01_curves`,
linked through each replica's `curve_rollouts`; keep all historical evidence.
Show both raw and training-retained curves and cumulative token budgets. This is
a post-hoc extension after inspecting selected results, not a new preregistered
confirmation. For future studies, plan evaluation of every saved revision after
all training freezes; reserved outcomes must never enter the training loop.

For novel-object work, the user narrowed scope to simple blocks and modest
geometric irregularity. Keep color segmentation and CPU perception; do not add
SAM2 at this stage. Prefer a small polygonal-prism extension behind existing policy
interfaces over a general shape framework. EMS and multi-part reconstruction are
not prerequisites. See `docs/NOVEL_OBJECT_GEOMETRY_DESIGN.md` for the current scope;
do not alter sources frozen by running experiments.

The user clarified that novel-object abstraction must not be defined by named
shape categories. Use point coordinates with evidence and edge/face connectivity;
named shapes are validation examples only. Preserve existing policy call signatures.
Distinguish the general representation from the current convex-extrusion observer's
geometric assumptions, and do not claim arbitrary-object perception from a general
point-reference API.

The user authorized a new complete three-cube stacking study using the updated
geometry bundle: `outputs/geometry_cube_stack_01`, via
`scripts/run_three_cube_geometry_study.py`. Use five paired chains, R0-R3, five
training layouts per pair and 100 common reserved layouts, with every revision
evaluated after all training freezes. Budget: 30 model requests, 200 training
slots and 4000 reserved slots total. Global limits are four model requests and
24 rollouts, with the registered hardware fallback. The shared 0.40 grasp and
libccd physics define this new variant. Cubes retain the existing cube observer;
this is not a novel-shape perception result. Data lives on the SSD with a project
output symlink. See `docs/THREE_CUBE_GEOMETRY_STUDY.md`. Preserve previous bridge
studies, sources and evidence; never feed reserved outcomes back to actors.

The three-cube campaign stopped on a legacy four-object training-audit check.
The user authorized fixing it and continuing the existing study. Use
`scripts/resume_three_cube_geometry_study.py` for subsequent phases of this run;
its recorded audit-only correction preserves the original frozen sources and
all completed evidence. See `audit_object_count_repair/` in the study output and
`docs/THREE_CUBE_GEOMETRY_STUDY.md`. Do not discard new STOP receipts or retry
incomplete attempts automatically.
