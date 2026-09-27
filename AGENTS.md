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


The user subsequently authorized reducing the three-cube evaluation cost while
implementing an online experiment. All 200 training rollouts and 30 model
requests are complete. The separate `compact_evaluation_20` amendment evaluates
all R0-R3 candidates on the first 20 reserved seeds, selected before reserved
results, for 800 slots. This evaluation is complete. Use
`scripts/evaluate_compact_three_cube.py` for its checks and analysis; do not resume
the superseded 4000-slot pipeline. Preserve both original STOP receipts. The
amendment verifies token-counter transitions and ignores unchanged broadcasts;
it does not change actor policies or physical evaluation.

The active development task is a continuous online bridge plus two-cube tower.
Use five shared new evaluation layouts. The proposed pilot has two repetitions
per arm and layout (20 episodes); freeze seeds, model settings and timing after
separate development checks. Both arms share the same adapted bridge and stacking
skills and Astra/Luna routing. Astra plans initially and replans; Luna handles
routine decisions. Retain one environment and geometric session per episode;
never reset after an unsuccessful action. Simulation advances during model waits,
with measured wall/simulation time and lag. All runtime perception remains sensor
based. Do not expose private evaluator truth to the model or skills. New work
lives in isolated `scripts/online_bridge_*` modules and opt-in builders; preserve
existing registered sources. See `docs/ONLINE_BRIDGE_TOWER_PILOT.md`.


Online development passed three layouts, a physical clearing diagnostic, and
paired continuous recovery. The frozen reserved pilot is
`outputs/online_bridge_tower_01`, via `scripts/run_online_bridge_pilot.py`:
five shared fresh layouts (2028101001-2028101005), two episodes per arm/layout,
20 total episodes, 360 seconds each, and one episode at a time with balanced arm
order. Astra medium handles initial planning/replanning; Luna low handles routine
decisions. Both arms share identical adapted skills and the clearing primitive.
Freeze runtime sources/assets and package/client versions for the campaign.
Do not modify the registered implementation or tune against reserved outcomes.
Preserve STOP receipts and incomplete attempts; no automatic retries. Source checks
run before and after every episode. Use the recorded progress and process handle
to distinguish a running experiment from a stopped one.

The frozen online bridge-tower pilot completed all 20 episodes and passed the
independent audit. Both arms achieved 6/10 successes; Full used fewer image
attachments but did not improve final success. See
`docs/ONLINE_BRIDGE_TOWER_RESULTS.md`. Preserve the frozen runtime and evidence;
do not rerun or tune this reserved study. Further development needs a separate
variant and evaluation split. The compact three-cube evaluation is also complete.

The user authorized a separate event-driven optimization attempt, including
segmenting physical actions. Development modules are `online_event_skills.py`,
`online_event_memory.py` and `run_online_event_development.py`. The first real-time
paired development layout succeeded with one model request per arm. A separate
180-second Full exception diagnostic timed out but exercised interruption,
Luna recovery decisions, retry suppression and context rotation without reset.
See `docs/ONLINE_EVENT_DRIVEN_DEVELOPMENT.md`. These are development checks, not
an extension or rerun of the completed reserved pilot. No new reserved study is
registered. Preserve all prior experiment sources and evidence.

A separate controlled online recovery study completed under
`outputs/online_recovery_01`, using `scripts/run_online_recovery_study.py`. It uses
five fresh paired seeds 2028102001-2028102005, ten episodes, a 240-second recovery
budget, and balanced batches of up to four processes. A common accelerated
sensor-controlled prefix places both foundations; an evaluator-only 25 mm right
foundation displacement precedes the single recovery clock. No reset or further
state edit is allowed during recovery. See `docs/ONLINE_CONTROLLED_RECOVERY.md`.
This is foundation repair, not validated general clearing. Freeze sources before
dispatch; preserve setup failures and all attempts, and do not tune on these seeds.

The controlled recovery study completed all ten episodes and passed an independent
audit. Both arms achieved 4/5 success. Full had shorter average initial requests
and fewer image attachments, but higher total token consumption and no material
overall time gain. See `docs/ONLINE_CONTROLLED_RECOVERY_RESULTS.md`. Preserve this
reserved study; further improvements require separate development and registration.

Further development lives in `online_reasoning_state.py`, `online_reasoning_skills.py`,
`online_contact_scene.py` and `run_online_reasoning_development.py`. Four paired
180-second development episodes and a 360-second difficult-scene pair are complete;
they did not establish a Full recovery or token advantage. Their source snapshots
and receipts are retained under `outputs/online_reasoning_dev`. No new reserved
comparison has been run. See `docs/ONLINE_REASONING_STATE_DEVELOPMENT.md`.

A private development diagnosis exposed unnamed robot collision meshes missed by
the old geometry-name contact helper. New development uses ContactCompleteEnv,
which groups contact geometry by body ancestry and archives terminal physics.
The two corrected accelerated checks passed/failed as expected and their terminal
contacts were independently reconstructed. Do not silently change frozen evaluators
or claim the earlier online successes were retrospectively re-certified. Preserve
the limitation documented in both online result reports. Remaining development
issues include jaw-housing clearance and routing complex failures to a planner.


The user narrowed current online development to ordinary bridge-plus-two-cube
construction to conserve time and resources. Do not expand into injected collapse,
loaded-stack displacement, or multi-level clearing stress scenarios. Keep natural
failures in the same episode; no reset, outcome filtering or automatic reruns.
The unfrozen reasoning-development runner defaults to 180 seconds and no longer
accepts intervention fixtures. Shared jaw-clearance development passed all three
ordinary accelerated checks in `clearance_preflight_03`, without model calls;
this does not establish a geometric-text reasoning or efficiency advantage.
Future small comparisons should use five fresh paired layouts, one episode per
arm/layout (ten total), at most four workers, and a separate frozen protocol.
Do not launch additional model experiments merely for this scope cleanup, or
resume completed complex recovery studies. Preserve all historical evidence and
registered protocols. This scope supersedes earlier suggestions to pursue complex
failure recovery; see `docs/ONLINE_REASONING_STATE_DEVELOPMENT.md`.


The ordinary follow-up has a dedicated preparation/dispatch entry point,
`scripts/run_online_ordinary_study.py`, and design in
`docs/ONLINE_ORDINARY_TOWER_STUDY.md`. Use `outputs/online_ordinary_tower_01` for
five fresh pairs (2028103001-2028103005), ten episodes, 180 seconds each and at
most four workers. Both arms share the clearance screen and Astra medium planner;
no artificial intervention. Preparation freezes sources and launches no model
requests. Once registered, preserve the sources and do not screen or replace seeds
based on outcomes. The scheduler refuses prior attempts and stops new batches on
infrastructure errors. Ordinary does not imply guaranteed success. Keep earlier
negative results and do not claim a geometric-text advantage from physical checks.

The ordinary comparison stopped after its first batch: four attempted slots and
six unstarted, under `outputs/online_ordinary_tower_01`. Layout 2028103001 succeeded
in both arms with one model request each (130.98/131.43 seconds Base/Full). Layout
2028103002 failed shared conditional-bound initialization before model calls.
Preserve the STOP receipt and all results; do not resume, replace the failed seed,
or tune using this reserved outcome. Sources and resource accounting passed audit;
terminal contacts were independently reconstructed. Total completed model usage
was 23,532 tokens, with no incomplete requests. No geometric-text advantage was
established. See the recorded partial outcome in `docs/ONLINE_ORDINARY_TOWER_STUDY.md`.

A development-only feedback-cost review is recorded in
`docs/ONLINE_FEEDBACK_COST_REVIEW.md`. The optional `online_decision_frames.py`
prototype is not adopted: it reduced repeated camera calls, but strict RGB output
equivalence failed and measured savings were only milliseconds. Preserve its failed
checks under `outputs/online_feedback_cache_dev`; do not silently wire the cache
into frozen or new evaluations as a validated optimization. No model calls or
manipulation actions were made for this review. The ordinary task's one-plan
execution does not exercise repeated model-level spatial reasoning; do not claim
that it proves a real-time text advantage or add artificial difficulty to force one.

An isolated explicit-target adapter now lives in `online_bound_target_skills.py`,
with design and limits in `docs/ONLINE_TARGET_BINDING_DEVELOPMENT.md`. It preserves
node-only calls and adds a development path that passes caller world surface points
to the existing actuator without task-specific target replacement. Seven mock
execution tests cover target preservation, cancellation, legacy behavior and input
validation. No model or physical evaluation has used it. Before any comparative
study, provide both arms viable calibrated metric-depth access, validate the
sensor-grounded reference boundary and register this changed interface separately.
Do not describe the component tests as task success or a geometric-text advantage.

Shared calibrated depth access for the explicit-target development interface is
implemented in `online_metric_depth.py`. It queries pixels on the exact displayed
RGB-D archive, returns visible surface samples without object/goal inference,
marks historical timestamps and retains only the latest displayed frame per view.
Both arms have the same 16-pixel query limit. Seven unit tests and a replay of eight
development frames/128 pixels passed; no model calls or rollouts were run. Reports
and source snapshots are in `outputs/online_metric_depth_dev`. Live actor schema,
query response handling and Full reference feedback remain unvalidated; do not
claim an end-to-end fair comparison or a geometric advantage from these tests.

The explicit-target development interface now has `online_bound_actor.py` and
`online_bound_decisions.py`: a separate decision schema, shared metric queries,
evidence-qualified Full references and one placement per decision. The persistent
transport is reused with independent schema globals; frozen actors are unchanged.
Four mock integration tests passed, including both-arm image/query/target dispatch
and Base feedback exclusion. `ExplicitSpatialSkills` omits automatic worksite
selection and uses explicit relocations for clearing; this changed assistance
level must be disclosed. No live model or physical episode has used this loop.
An owning runner, physical smoke test and live usage/schema validation remain
before a new registration. Keep ordinary scenes and the five-layout resource scope.

The explicit-target runner `run_online_bound_development.py` is now implemented;
it permits only the three existing development seeds and budgets up to 180 seconds.
A zero-model accelerated physical check passed on 2028100101. A two-worker live
Base/Full check then finished under `outputs/online_bound_dev/pairs_01`: Base timed
out at 180 seconds, Full succeeded at 156.07 seconds. Model requests were 7/6 and
total tokens 97,236/106,682 (Base/Full); there were no incomplete requests. Full
retained an existing foundation, submitted fewer placements and used fewer image
attachments, but consumed more total tokens. Independent receipt, sensor-query,
submitted-target and terminal-contact audits passed. This is one reused development
layout, not reserved evidence or an isolated text-latency effect. Preserve the
results, including the higher Full token cost. No reserved explicit-target study
is registered yet; use a separate five-layout protocol before evaluation.

The next bounded comparison uses `run_online_bound_study.py` and
`outputs/online_bound_tower_01`: five fresh paired seeds 2028104001–2028104005,
one episode per arm/layout, ten total, 180 seconds each and balanced 4/4/2 batches.
It uses the tested explicit-target interface, shared metric queries and Astra
medium, with no injected intervention. Freeze the protocol before dispatch;
retain failures and stop new batches on infrastructure errors. No replacement
seeds, automatic retries or budget extension. See
`docs/ONLINE_TARGET_BINDING_STUDY.md`. The earlier positive development pair is not
reserved evidence; report total tokens and unfavorable outcomes as well as speed.


The user subsequently changed the online episode budget to 200 seconds. Apply
this equally to both arms in future explicit-target development and registrations.
All ten slots of `online_bound_tower_01` had already started with fixed 180-second
deadlines; preserve their protocol, source snapshots and actual outcomes. Finish
source verification before updating working runtime code. This timing amendment
does not authorize automatic reruns or another campaign. Keep five paired layouts,
at most four workers, ordinary scenes and continuous recovery without resets.

The explicit-target study `outputs/online_bound_tower_01` completed all ten
180-second episodes in 552.10 seconds and passed resource/source and interface
audits. Full succeeded on 3/5 layouts, Base on 0/5. Full used fewer images and
less completed model-request time but more recorded tokens; incomplete usage is
unknown in one Base and two Full requests. See `ONLINE_TARGET_BINDING_RESULTS.md`
and `analyze_online_bound_results.py` for all paired outcomes and reproducible
figures. This is exploratory evidence for the interface package, not an isolated
latency or token advantage. Preserve the frozen study and do not tune or rerun
its reserved layouts. Later timing uses the authorized 200-second budget.

An opt-in compact-text development adapter is implemented in
`online_compact_geometry.py` and `online_compact_actor.py`. The explicit-target
development runner accepts `--compact-text`, records the serialization mode and
archives the adapter sources. Default behavior and reserved registration remain
unchanged. Seven focused and sixteen existing explicit-target tests passed.
Development-only replay of thirteen archived prompts under
`outputs/online_compact_text_dev/replay_01` preserved every feedback field and
reduced Full prompt bytes by 21.1%, including its first-turn explanation; Base was
byte-identical. No model requests or rollouts were run for this check. Byte savings
are not token or latency evidence. Live development at 200 seconds must precede
any adoption into a new reserved comparison. See `docs/ONLINE_COMPACT_GEOMETRY_TEXT.md`.

The compact-text live development check completed under
`outputs/online_compact_text_dev/live_01`, using `run_online_compact_check.py`:
two concurrent Full episodes on existing development seed 2028100101, raw versus
compact text, 200 seconds each. Both succeeded (157.12/158.52 s) with six requests,
four images and no incomplete usage. Compact used 99,620 total tokens versus
106,784 raw (6.7% fewer), but model-request and completion times were slightly
longer. All twelve actual model prompts passed independent reconstruction and
transcript checks; source/resource, submitted-target, sensor-query and terminal
contact audits passed. See `docs/ONLINE_COMPACT_GEOMETRY_TEXT.md`. This validates
one development pair, not a reserved or Base/Full 200-second comparison. Preserve
the raw default and historical evidence; freeze any adopted format separately.

Under the continuing experiment objective, the next bounded comparison is
`outputs/online_compact_tower_200_01`, via `run_online_compact_study.py`: five fresh
paired seeds 2028105001–2028105005, one Base/Full episode each, 200 seconds and
balanced 4/4/2 batches. It adopts the development-validated compact Full text;
Base prompts and shared action skills remain unchanged. Freeze sources before
dispatch and preserve every outcome. Independent prompt reconstruction supplements
the existing resource, sensor and physical audits. Stop after the ten fixed slots;
no automatic retries, replacements, budget extensions or further studies selected
to obtain a positive result. See `docs/ONLINE_COMPACT_TOWER_200_STUDY.md`.

The fixed `online_compact_tower_200_01` comparison is complete: ten episodes in
611.62 seconds, Full 4/5 and Base 2/5 successes, mean capped time 174.13/195.91 s.
All source/resource, interface and compact-prompt audits cover ten episodes and
passed. Full attached fewer images but used more recorded tokens and had longer
mean completed requests. One Base and three Full requests have unknown incomplete
usage. The empirical advantage is task-level and exploratory; do not claim a
universal latency, reliability or token advantage. See
`docs/ONLINE_COMPACT_TOWER_200_RESULTS.md` for every paired result and editable
figures. The authorized fixed comparison is finished. Preserve all evidence and
do not automatically dispatch further episodes or grow the sample for a preferred
outcome. Follow-up experiments require a new user request and separate registration.

The user authorized a new single-policy stacking comparison after auditing Base
prompt disclosure. Use `run_three_cube_contract_study.py` and
`outputs/geometry_cube_stack_contract_01`: five paired chains, R0–R3, five training
layouts per pair and 20 fresh shared test layouts, all revisions evaluated after
training freezes (30 model generations, 200 training slots, 800 test slots).
Both arms omit executor source while keeping the same runtime and public API;
Full retains geometric queries. Global limits remain four model requests and
24 rollouts with resource fallback. See `THREE_CUBE_CONTRACT_ONLY_STUDY.md`.
Preserve the old source-disclosed experiment. Stop on infrastructure errors and
retain incomplete attempts; do not retry, replace layouts or expand the budget.
