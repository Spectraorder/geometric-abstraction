> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Geometric state for text-based agentic policy reasoning

## Current comparison and completion audit

The current comparison is **Visual Base versus Full geometry only**. Base receives
raw RGB-D, calibration, public task labels, robot proprioception/CAD and qualitative
action feedback; it is not supplied object coordinates. Both arms share the same
hidden visual PnP actuator and may compute geometry from pixels. Full additionally
receives supplied visual geometry and compact target checkpoints. This tests a
perception/geometry package, not an isolated symbolic-versus-numeric effect.
Earlier baselines with supplied centers used RGB-D estimates, not simulator truth;
those studies remain separate historical evidence.

The [completed visual-input study](../VISUAL_INPUT_LEARNING_RESULTS.md) contains
three independent learning chains and 600 valid rollouts. It found mixed results:
training-selected SR averaged 56.67% for Base and 65% for Full, with paired chain
differences of +40, -20 and +5 percentage points. A repeatable iteration advantage
was not established.

The subsequent [constraint-feedback study](../PARTIAL_GEOMETRY_BUDGET_LIMITED_RESULTS.md)
completed all 192 training rollouts and froze all 24 candidates before evaluation.
The user capped reserved evaluation near 100; 103 reserved rollouts were retained
and execution stopped. Only chain 1 has reserved results. At R1, Base scores 20%
and Full 75%, compared with their common 40% starter. Training-selected versions
score 40% and 65%. This is positive preliminary evidence in one chain, with later
regression, not a completed replication. Do not resume testing without a further
user request. The original 480-slot protocol records the original plan rather
than the current authorized budget.

The following audit distinguishes implemented capabilities from empirical claims.
It does not declare the research goal complete.

| Requirement | Current evidence | Remaining boundary |
| --- | --- | --- |
| CPU RGB-D geometry serialized for agent use | Frozen visual observers, geometric session, and real actor feedback; [interface contract](../VISUAL_INPUT_COMPARISON.md) | Simple known shape families, unique colors and initial orientation assumptions; no arbitrary-shape or same-color recognition claim. |
| Lightweight state at action boundaries | `GeometryStepState` retains one object, one goal and one preceding center; current study Full checkpoint median 366 characters, maximum 1,074 | Record length is not model latency or reduced reasoning cost. Initial visual-model acquisition still observes the scene. |
| Separate current evidence and desired state | Current-time checks, explicit unknowns, requested references and goal residuals in `policies/geometry_step_state.py` | Desired geometry does not certify stability or imply arbitrary orientation is executable by the common actuator. |
| Evaluate persistent state and predicted change | [Point and center-region replays](../CONDITIONAL_ATTACHMENT_EVIDENCE.md) test transport under a rigid-attachment hypothesis against subsequent visual evidence | Prediction is an offline diagnostic, not an accepted current pose; no measured policy-SR benefit from motion memory. |
| Relevant primary research | Reading notes below cover Mao, Wu and Goldberg and link primary papers | These motivate design choices, not evidence for this implementation's success. |
| Controlled learning comparison and validity repairs | Completed three-chain study; current study has common starter checks, matched scenes/actuator, global freeze, verified actor contexts and retained quota failures | Current reserved evaluation is incomplete and stopped after interim results were visible. |
| Higher held-out SR in fewer revisions | Strong gains in individual chains, including current R1; previous completed replication is mixed | Repeatable learning efficiency remains unproven. More analysis of the same chain cannot create independent replications. |
| Preserve negative evidence | Prior studies, failed revisions, partial batches and separate manual counterfactuals remain documented | Do not pool incompatible studies or substitute test-selected versions. |

The bounded CPU replay measures 134.80 ms median on one difficult recorded frame;
it does not establish end-to-end realtime operation or speedup over VLM reasoning.
Rollout elapsed time is not the current optimization objective.

## Current representation decision

Use a **typed symbolic/metric core**, serialized as JSON or concise text for the
agent. Numerical geometry stays on CPU. Text represents the same state rather than
a separate store of coordinates. Keep current evidence, conditional prediction and
intent distinct. The proposed minimal motion-memory summary remains outside the
frozen learning interfaces; its offline diagnostic value does not justify claiming
an online learning gain.

The decisive unresolved empirical requirement is repeatability across independent
policy-evolution chains. Existing reserved data cannot settle that requirement for
the latest method. Additional tests require a new user instruction under the current
budget cap. No further model generations, policy repairs or test-based selection
are implied by this audit.

The sections below preserve the design history and narrower diagnostic evidence.
Earlier numeric/symbolic comparisons and temporal fusion proposals are historical;
they do not reintroduce a third arm or enable hidden-coordinate persistence in the
current system.

## Action-relevant compact feedback development

Following the completed compact pilot, placement checkpoints now automatically
include the active object's attempted surface target and signed translation residual
from its current visual extent. No global scene snapshot or implicit camera refresh
is added. The existing `place()` contract is center XY plus lowest world Z, rather
than a material face-center target. Center goals remain separate, and stale evidence
produces an unknown residual. See the [development contract and evidence](GEOMETRY_STEP_LEARNING_PROTOCOL.md#post-study-development-placement-intent-at-checkpoints).
This is an interface improvement, not a demonstrated learning or SR gain.

## Conditional held-point replay

A small state hypothesis is to retain only the active object's estimated center in
the robot tool frame at a confirmed held checkpoint. Given tool poses `(p0, R0)` and
`(p1, R1)`, the conditional next center is `p1 + R1 @ R0.T @ (center0 - p0)`.
This is conditional on a rigid grasp without slip, not a new sensor observation.
Release, an intervening gripper command, missing robot/visual evidence or a focus
change invalidates the replay anchor. No persistent global scene is required.

`scripts/replay_held_geometry.py` reads only the 40 completed corrected training
trials' `geometry_step` records and logged robot proprioception. It does not read
`rollout.npz` state arrays, result labels, simulator object poses or reserved data.
Nine checkpoint pairs qualify; six have fully observed position flags at both ends.
Those six come from **two policy executions of the same initial seed**, including
four small-motion retries. The first pair in each execution has visual center changes
of 77.88 mm and 49.38 mm, with prediction-to-later-visual residuals of 2.26 mm and
1.13 mm. The six-pair median is 0.26 mm, but is dominated by retries and must not be
reported as general tracking accuracy. Four tests cover rigid-frame rotation,
release/gripper-command invalidation, missing evidence and invalid robot rotations.

This supports further testing of a one-object conditional point representation.
It does not validate predictions during unobserved intervals, prove attachment or
slip absence, or demonstrate a policy SR benefit. Visual fits may share bias.
The predictor is an offline diagnostic and is **not integrated into the frozen
controller or observation state**. See [the complete replay evidence](../evaluations/held_geometry_replay.json).

## Target availability and controlled held-point probes

Six separate development probes used two layouts and cube, thin-prism and cylinder
targets, each with five prescribed robot offsets after a visually confirmed pick.
The original session completed 15/30 motion checks: all three targets on the second
layout were blocked during initialization by an unrelated unobservable beam.
`outputs/held_geometry_development_01` preserves those failures and the original
probe driver. No task success claim is made; the probes do not build the bridge.

A development-only library patch catches initial per-object observation failures
and records that object as unavailable. It leaves valid objects usable and does not
invent missing dimensions or poses. The same six probes then completed 25/30 checks:
cube and cylinder probes on the second layout recovered, while that layout's beam
remained unavailable. The three first-layout trajectory files are byte-identical to
the original probe, confirming no change on the already observable cases. A fixture
test also verifies that an unknown neighbor does not erase a valid target estimate.
The patch is confined to `outputs/held_geometry_development_unknown_01`; the learning
experiment's runtime and candidates remain frozen.

Of the 25 measured pairs, 15 have fully observed position flags at both endpoints
(cube/cylinder), with median prediction-to-visual residual 0.085 mm and maximum
0.68 mm. These are consecutive prescribed movements, not independent trials or
simulator-ground-truth errors. The other ten pairs retain their partial-observation
flags and are excluded from that full-position summary, not deleted. No disturbances
were injected; grasp attachment and behavior through invisible intervals remain
unproven. The useful system result is isolating unknown state to the affected object,
plus feasibility evidence for a conditional single-object point representation.

Evidence: [availability/trajectory receipt](../evaluations/initial_unknown_probe.json),
[original probes](../evaluations/held_geometry_development.json), and
[probes with per-object unknown initialization](../evaluations/held_geometry_development_unknown.json).

## Lightweight system direction

The user requests one base/full comparison and a lightweight system. The active
path is calibrated RGB-D -> CPU rigid fitting -> a small object-keyed state ->
the existing PnP executor. Typed JSON/text is a view of that state. Keep geometry
queries as local numerical functions; use robot CAD and encoders for the gripper.

The default reasoning record now targets **one active object after each semantic
action step** (pick, place, check). Keep its current estimated geometry, one desired
reference point, remaining translation and the change from the preceding observation.
Keep only that preceding observation in recorder memory; external experiment logs
may retain records for auditing without replaying the whole history to the agent.
Do not emit all object pairs or a global scene snapshot by default. Query supporting
objects or nearby obstacles only when the current operation requires them. A bridge
placement can require two support faces; "one active object" does not imply ignoring
task-relevant surrounding geometry.

The small implementation is `policies/geometry_step_state.py`. It consumes existing
observer output and performs no sensing or fitting. The caller updates it at an
action boundary with the fresh active-object fit. An unavailable or stale fit stays
unknown; a gap clears the previous estimate so it cannot silently become a current
pose or a motion measurement. Goals remain distinct from observations. Changes are
differences between estimates, not certified physical motion. Request corners, edge
endpoints or face centers only when needed. A face normal alone does not determine
in-plane alignment. The current helper supports named point and face-center goals;
full orientation targets remain outside this helper.

A training-only replay of all five R1 trials in the shared-observer pilot covers
64 recorded pick/place/settled/ready checkpoints. With the same compact JSON
serialization, the global state has median **16,152.5 characters** and the focused
record **562 characters**, a **96.53% aggregate character reduction**. The projection
uses only recorded visual geometry, public object identities and policy-chosen desired
centers. It does not rerun physics, consume held-out outcomes or measure an SR/latency
benefit. Reproduce with `python scripts/replay_geometry_step_state.py`; provenance and
all projected records are in
[evaluation data](../evaluations/geometry_step_state_replay.json).
Eleven tests cover freshness, uncertainty, reference identity, goal/observation
separation, per-step changes, actual RGB-D session integration and worker validation.
`policies/geometry_step_session.py` adds `checkpoint(color, step, refresh=False,
action_status=None)`. It reuses a fresh action observation through `get`; only an
explicit refresh calls `observe` on that object. The full interface optionally accepts
`goal_center` and requested `reference_names`. Changing focus discards previous-object
recorder memory. Support queries remain ordinary explicit observer/reference calls.

The common `policies/support_graph_step_starter.py` replaces global snapshots with
before-pick, post-pick, post-place and settled checkpoints. Pick/place commands retain
the original macro's transit height and action order. A paired real simulation on
development seed 75103 completed the bridge in both arms, with byte-identical physical
arrays, 54 camera calls per arm and 12 target-only records each. The largest full
geometry record was 512 characters including its log envelope. This verifies runtime
integration on one development layout, not learning or held-out SR. See the
[integration receipt](../evaluations/geometry_step_integration.json) and frozen development
artifacts in `outputs/geometry_step_development_01`.

The completed learning experiment retains its original frozen feedback. The completed [target-only study](GEOMETRY_STEP_LEARNING_RESULTS.md) is
`outputs/geometry_step_learning_pilot_01_corrected` (selected Base 80%, Full 70%;
no iteration advantage);
its predecessor is preserved as withdrawn for a generic validator naming defect.
No reserved outcome motivated the correction, and no submitted policy was edited.
The [target-only learning protocol](GEOMETRY_STEP_LEARNING_PROTOCOL.md) consumes
`geometry_step` records instead of old `planner_state` snapshots; both arms use the
same bounded feedback rule. Merely appending small records to global feedback would
not reduce context.

No database, runtime VLM loop, general history system or additional orchestration
service is needed. Conditional predictions remain optional research evidence and do
not replace current measurements.

The completed base/full pilot stays frozen. Its full arm includes the existing
metric/symbolic operators; interface consolidation happens after that registered
comparison, with parity checks before adoption. No extra numerical-only arm or
new state framework is being added to the active experiment. Lightweight here means
a small interface and bounded state as well as CPU execution; passing a CPU test
alone would not establish that the software has been simplified.

## Research objective

Design a CPU abstraction from RGB-D and robot proprioception to compact,
queryable geometric text. Test whether it reaches higher held-out task success
in fewer policy-improvement iterations than matched interfaces without it.
Rollout elapsed time is not an optimization objective at this stage. Success
must be measured, not assumed; negative campaigns remain part of the record.

The immediate scope is simple rigid objects with unique colors, starting with
three cubes. Runtime object poses, dimensions, contacts and evaluator truth are
forbidden. Simulation/rendering may use GPU. The system should later support
mixed shapes and support graphs without encoding a completed tower controller.

## Reading notes and design implications

These are design interpretations of primary sources, not reproduced results.

| Work | Relevant idea | Implication and boundary for this project |
| --- | --- | --- |
| [PDSketch, Mao et al., NeurIPS 2022](https://pdsketch.csail.mit.edu/data/papers/2022NeurIPS-PDSketch.pdf) | Factor object state and express sparse, local action effects; primitive execution is a separate layer. | Represent which object factors an action might change. Do not require an LLM to implement low-level joint control to use a state abstraction. Their learned symbolic planning results are not evidence for our coding-agent setup. |
| [3D Dynamic Scene Representations, Xu et al., including Jiajun Wu](https://arxiv.org/html/2011.01968v2) | Object permanence, amodal completion and temporal continuity; aggregate observations and predict dynamics. | Missing visual evidence should retain uncertain object memory. Use rigid transforms and analytic geometry first, rather than adopting the learned volumetric network under our CPU constraint. |
| [Scene Graph Memory, Kurenkov et al., ICML 2023, including Jiajun Wu](https://proceedings.mlr.press/v202/kurenkov23a/kurenkov23a.pdf) | Accumulate partially observed graph history with observation times; reason about potentially changed relations. | Timestamp facts and distinguish remembered relations from current observations. Their object-search setting differs from millimeter-scale manipulation. |
| [ReKep, Huang et al.](https://arxiv.org/html/2409.01652v2) | Express constraints over tracked 3D keypoints and optimize actions; tracking under occlusion is a major failure source. | Expose geometric relations and residuals; preserve keypoint identity across views. For cubes, use rigid object landmarks and frame symmetry rather than independent point tracks. Their perception stack and optimizer are not imported here. |
| [VoxPoser, Huang et al., including Jiajun Wu](https://voxposer.github.io/) | Ground language-generated affordances and constraints in 3D value maps for motion planning. | Separate agent intent from numerical grounding. Our compact object/face/point representation is chosen for simple rigid blocks; it is not an open-vocabulary dense value map. |
| [CaP-X, Fu et al., including Jiajun Wu and Ken Goldberg, v2](https://arxiv.org/abs/2603.22435v2) | Study coding agents across abstraction, interaction and perceptual-grounding levels; structured execution feedback and other test-time mechanisms affect outcomes. | Control the primitive set and feedback budget. Separate stronger action skills from improved state information. Do not conflate supplying a geometry library with demonstrating that text state improves agent reasoning. |
| [Dex-Net 2.0, Mahler et al., including Ken Goldberg](https://arxiv.org/abs/1703.09312) | Depth-conditioned robust grasp estimation combines geometry, uncertainty and physical success metrics. | Geometric alignment is not contact or grasp success. Keep visual grasp evidence and uncertainty explicit. We do not adopt its learned grasp network in this CPU prototype. |
| [LERF, Kerr et al., including Ken Goldberg](https://arxiv.org/abs/2303.09553) | Ground language queries in consistent 3D fields across views. | Persistent world grounding matters, but semantic recognition and radiance-field optimization are outside this unique-color, simple-geometry study. |

## Representation decision: symbolic and metric state, serialized as text

The core interface is typed symbolic state with numeric geometry. JSON and short
text are serializations for the agent, not competing state estimators. This is an
engineering decision, not evidence that this representation already improves
policy success. The completed v2 comparison did not establish fewer revisions;
see [the completed experiment](ONLINE_GEOMETRY_EXPERIMENT.md) and
[the next experimental boundary](GEOMETRY_NEXT_EXPERIMENT.md).

- **Entities:** rigid objects and named geometric features such as a center,
  top face, bottom face, axis, or cylinder rim. Each feature carries its frame,
  units, observation time and whether it is observed or inferred.
- **Relations:** predicates with `true`, `false` or `unknown`, accompanied by
  metric residuals and supporting evidence. A measured face gap is not proof of
  physical contact; projected overlap is not proof of stable support.
- **Goals:** explicit constraints on features and relative transforms, separate
  from the current estimate. The generic geometry solver computes a desired
  pose and remaining degrees of freedom; the policy selects the task and actions.
- **State history:** retain evidence and timestamps; conditional memory and
  predicted motion must carry their assumptions and be invalidated when new
  evidence contradicts them. Do not promote an intended motion to an observation.

The default agent packet should report the relevant objects, changed facts,
unknown quantities and goal residuals. Detailed vertices remain queryable.
Free-form prose may summarize the packet but must not become a second,
independently maintained source of coordinates or facts. No hard-coded tower
recipe, color-specific action, or simulator object state belongs in this layer.

An earlier proposal separated basic PnP, numeric rigid state, and symbolic
operators. The user's later scope decision supersedes that proposal: the active
comparison has only Visual Base and Full geometry. Consequently, any measured
benefit belongs to the supplied perception/geometry package and cannot isolate
symbolic reasoning from improved sensing. Conditional motion-memory summaries
remain offline prototypes; desired-state operators are available but do not
establish successful execution or a learning benefit.

## Proposed representation

Objects own rigid local landmarks: centers, face centers, axes and vertices.
World points are derived from an object transform. Cube axes are matched modulo
24 face symmetries so a fitter relabeling axes does not spuriously change the geometric representation.
These canonical labels do not recover invisible material-point identity on an
unmarked cube. Top/bottom faces denote geometric roles, not painted face IDs.
Landmarks on hidden faces are inferred from the rigid model, never labeled as
individually observed vertices. A cylinder will require an axis and rims rather
than fabricated physical corners; it is outside the first memory adapter.

Keep four concepts distinct:

1. **Visual evidence:** measured face-plane directions, residual, camera and time.
2. **Current estimate:** evidence plus uncertain memory; retain hidden coordinates
   while only observed directions update. Covariance bookkeeping is heuristic,
   not a calibrated success probability.
3. **Conditional prediction:** a rigid displacement, such as encoder-derived tool
   motion *if the object remains grasped*. A commanded destination is not an
   observation or a guaranteed action effect. Contradictory new evidence records
   a prediction error and invalidates the hidden-motion assumption.
4. **Desired state:** task-selected target geometry. Updating an observation must
   not move the goal. Reanchoring requires an explicit operation.

The agent should normally see compact text containing object identity, estimated
position, observation age, measured directions, change since last update,
prediction condition and discrepancies, target residuals, and candidate support
relations. Full landmarks remain queryable when needed. Do not dump every pixel
or every vertex into every prompt. Relations expose signed gap and support margin,
not hidden simulator contacts or certified stability.

This is an engineering hypothesis about an effective interface, not a novelty
claim about object memories, rigid models, state estimators or scene graphs.

## Implementation checkpoint

- The shared PnP observation interface now accepts the existing mixed rigid CPU
  estimator at every visual checkpoint, with optional same-time wrist fallback.
  The unchanged manual planner/executor pairing produced 6/8 versus 3/8 for the
  archived basic observer on matched new development layouts. This combines
  perception, extra-view use and robot masking; it is not evidence for symbolic
  representation or fewer agent revisions. See
  [the complete development boundary](SHARED_PNP_OBSERVATION.md).

- `src/aspire_so100/geometric_state.py`: action-free cube memory, conditional
  SE(3) predictions, stable landmark labels, desired targets and text packets.
- The first implementation accepts `observe_cube` fits and explicit rigid
  transforms. It does not issue actions, automatically declare a grasp, or solve
  a task. Predictions require an explicit condition from the caller.
- `geometry.state()` is available in the optional state-enabled numerical library.
  Existing frozen campaigns retain their source snapshots; new runs require new
  directories and manifests.
- Cube column evaluation now handles all equivalent cube faces, including the
  support-plane normal and in-plane margin axes. Noncube upright requirements
  remain. Physical tests retain rejection of real tilt and horizontal cylinders.
- The target anchor schema is documented and validated with a descriptive error.
  Arrays are not silently interpreted as complete observation dictionaries.

## Experimental design

The previous experiment supplied geometry code, but its actor feedback was mainly
short notes and images. It also allowed wholesale controller rewriting; control
failures overwhelmed the reasoning comparison. The next study must expose the
abstraction as actual structured text, with identical shared execution capability.

First establish a reusable grasp/place layer on development scenes. It must be
shared by all arms, not secretly improved only for geometry. Keep task order,
worksite selection, target construction and checks in the agent's policy layer.
Then freeze source, prompts, initial policy, scene distribution and trial budgets.

Use staged ablations:

| Arm | Additional representation |
| --- | --- |
| Raw observation baseline | Same sensors, public task, robot model and shared action primitives; no supplied object-state text |
| Instantaneous geometry | Current fitted objects/relations and observability, without persistent history |
| Geometric memory | Same fits plus timestamps, rigid landmark continuity and change events |
| Memory with conditional predictions | Previous arm plus explicit robot-motion-conditioned hypotheses and residuals |

Start with a diagnostic comparison before scaling to replicated campaigns. Match
model, policy-improvement iterations, training scenes, observations and execution
primitives. Give each actor only its own results. Distinguish policy revisions from
within-rollout sensing/replanning steps. Record tokens and source/feedback sizes,
but do not select by wall time. No runtime VLM is required for the CPU policies.

Primary outcomes are learning curves by revision, training iterations to a frozen
success threshold, and paired held-out success after model selection. Repeat with
independent campaign contexts before making a general learning-efficiency claim.
Include color-order/layout transfer, followed by shape/composition transfer. A
stronger claim requires a benefit on unseen layouts, not only training recovery.

Intrinsic state checks should include occlusion, cube-axis swaps, stale memory,
unexpected object motion, predicted grasps that fail, and measurement innovation.
Private simulator truth may evaluate accuracy offline after state records are
frozen; it must never enter the state estimator or actor input.

## Current evidence and remaining work

The old 55-rollout campaign remains unchanged in
[its report](GENERIC_PNP_GEOMETRY_EXPERIMENT.md). It does not prove an abstraction
benefit. The new state unit tests establish representation invariants only; they
are not task-success or sample-efficiency evidence. The actual camera replay, shared-execution development and feedback integration
are now complete. The corrected matched pilot is complete: its first revision is
4/5 in both arms from a 2/5 common baseline; reserved success is 5/10 raw versus
6/10 symbolic feedback. The research goal remains active,
because this is not yet evidence of a supplied-abstraction advantage.

## Decision: symbolic plus metric, with text as serialization

The user explicitly delegated this representation choice. The core representation
is now typed symbolic entities and relations with numerical geometry, rather than
free-form natural-language memory. JSON/compact text is the interface to the agent.
`sym = geometry.symbols(memory)` exposes typed objects and three-valued facts;
`geometry.resolve(memory, "red.bottom_face")` exposes a plane patch;
`geometry.predicate(memory, "above", ["red", "blue"])` returns a truth value,
metric residual, evidence requirements and provenance.

Predicates are `true`, `false` or `unknown` about estimated geometry. A partial
view may constrain height enough for `above` but not XY enough for alignment.
`support_candidate` is explicitly geometric, while `contact` remains unknown
without a contact sensor. The uncertainty bands are heuristic translation bands,
not calibrated probabilities or a guarantee of orientation/contact stability.

The first camera-only replay processed 107 RGB-D frames over 63.25 simulation
seconds from the preserved raw revision-3 seed-4101 rollout. It produced compact
text with a median of 837 characters. Camera acquisition, file IO and agent
reasoning were excluded from the measured CPU work. This validates an executable
sensor-to-state-to-text path, not improved pose accuracy or task success. No
conditional attachments were assumed in that offline replay.

## Execution plan recorded before feedback integration

The following staged plan led to the current pilot. Items 1, 2 and 4 are now
implemented for instantaneous feedback; the memory/prediction component of item 3
remains a separate experiment. Do not declare the research goal complete from
unit tests, replay coverage or compact serialization alone.

1. Validate and improve a new shared PnP primitive on development-only scenes.
   Preserve `pick_place_rgbd_base.py` and previous bundles as historical starting
   points. Inspect excessive transit heights (`destination_z + 0.15`) and actual
   versus previewed tool tracking; avoid restoring tower/color-specific corrections.
2. Instrument shared executions with the same sensor schedule. Emit typed metric
   state and selected change events to the treatment actor's feedback, with exact
   payloads archived. Existing feedback truncates notes, so a separate structured
   channel is required. Do not merely add a callable library that the actor never
   sees as data.
3. Use shared frozen execution code and task-independent primitive semantics;
   let both actors revise task planning, target selection and checks. Initially
   compare raw versus state, then remove history/prediction to identify which
   parts help. Prediction must have encoder-derived transforms and explicit
   grasp hypotheses; the current replay intentionally does not invent these.
4. Freeze a fresh protocol and disjoint seeds after development validation.
   All scenes inspected so far, including the old reserved split, are development
   evidence only for the next study. Maintain independent model contexts and
   held-out evaluation; replicate any promising learning-efficiency result.

Validation to date: 3 cube-evaluator tests, 8 rigid-state tests, 13 existing
experiment tests, 10 memory tests, 6 symbolic-predicate/reference tests and 8
existing tower tests passed. The 107-frame replay's measured summary and final
symbolic packet are available as
[replay metrics](../evaluations/geometric_state_replay_02.json),
[typed state](../evaluations/geometric_state_example.json) and
[compact text](../evaluations/geometric_state_example.txt).

## Shared execution development: planning opportunity established

The new `policies/pick_place_rgbd_shared.py` retains the original simple visual
surface estimator and task-free PnP interface. It shortens excessively high
transits, uses a configurable grasp-height fraction inherited from the historical
3 cm controller, and falls back to robot-only IK with an achievable orientation.
It does not add a tower plan, fixed object roles, target-site selection, explicit
cube vertices or cross-object release compensation. The original extraction is
unchanged. The builder accepts an explicit `--base` for selecting this version.

Fifteen development rollouts on seeds 51001–51005 establish the following:

| Development probe | Result | Interpretation |
| --- | --- | --- |
| Shared PnP with original simple single-column adapter | 1/5 stable towers; no runtime errors | Remaining failures stopped on uncertain placement |
| Same PnP, single-object table placement | 5/5 releases; offline final XY errors 1.8–4.1 mm, settled objects | Primitive execution works at accessible targets; not a tower-success measure |
| Same PnP, manually supplied workspace-selection adapter | 5/5 stable three-cube towers | High-level geometric planning can unlock existing execution capability |

The workspace adapter uses visual sizes and free spacing, robot-relative candidate
locations, and robot-only IK previews before relocating the base. It is an
explicit development ceiling probe, **not an agent-generated policy or evidence
of an abstraction learning benefit**. Do not supply its source, chosen coordinates,
or results to either experiment actor. Both actors must discover their own policy
from the frozen common starter and their own training feedback. These development
seeds must not become reserved evaluation scenes.

[Measured development report](../evaluations/shared_pnp_development_01.json) preserves
all candidates' hashes and outcomes. Table-placement errors use private state only
in the offline diagnostic, after execution; no private state enters the primitive.
Eight old/new PnP tests cover its task boundary, source compatibility and robot-only
fallback behavior.

A new `scripts/geometric_feedback.py` exports actual initial/final side RGB-D
observations as symbolic/metric text and typed facts. It also records exact image
and sensor references so both actors can receive the same underlying frames.
It reads neither evaluator results nor traces. This first feedback adapter is
**instantaneous geometry with initially measured dimensions**, not temporal
memory; memory and conditional predictions remain explicit later ablations.
The exporter is now integrated into `run_symbolic_policy_experiment.py`.
The frozen pilot uses fresh training seeds 52001–52005 and reserved seeds
53001–53010. Both actors receive matched RGB/depth views; the treatment receives
additional derived symbolic/metric text. The common baseline achieved 2/5 on
these new training layouts. The completed comparison selected revision 1 in both arms and measured 5/10 raw
versus 6/10 symbolic reserved success. It demonstrates no learning-efficiency
advantage; later source-budget failures further limit interpretation.
See [the experiment protocol](SYMBOLIC_FEEDBACK_EXPERIMENT.md).


The first feedback pilot uncovered a validator defect: it rejected a numerical
`nonlocal` closure counter even though that does not modify the executor. The
original run is retained as invalidated. Corrected pilot 02 reuses both unchanged
first proposals and the same baseline, excludes both contaminated old second
proposals, and applies the corrected rule equally. Its manifest and protocol
explicitly record this amendment. No reserved feedback has been exposed.


## Next research step after the negative feedback pilot

Keep the symbolic-plus-metric representation, but test its intended online use:
supply an action-free state API during execution, with the same frozen PnP
actuator and equal high-level planning freedom in both arms. The current feedback
pilot required both actors to recreate detailed runtime perception; providing a
few endpoint summaries did not measurably accelerate improvement.

Freeze a new protocol and new reserved scenes before doing this comparison. Use
an adequate common planner budget and an explicit concise-source target to avoid
having most later slots measure character-count compliance. Include a fixed,
shared rule for showing unsuccessful as well as successful training trajectories;
this pilot's first two image layouts were both successful. Start with instantaneous
metric state versus raw, then separately test symbolic relations and temporal
memory/conditional predictions. Do not merge these treatments or silently drop the
negative feedback-only results. Replicate any promising result with independent
agent contexts before claiming improved learning efficiency.


## Online interface and temporal diagnostic checkpoint

The online `SceneGeometry` interface is implemented and exercised during a real
pick-and-return development rollout. Pilot `online_geometry_pilot_01` now compares
the supplied online instantaneous state plus feedback against raw RGB-D planning,
with identical shared actuation and newly frozen scenes. The completed v1 pilot reached 80% training success in one revision for geometry
versus two for raw, with reserved success 15/20 versus 13/20. This is promising but
not statistically robust or independently replicated. See [the protocol](ONLINE_GEOMETRY_EXPERIMENT.md).

The temporal replay supplies a substantive counterexample to naive persistence:
on 341 matched object/frame pairs, center-error p95 is 3.24 mm for instantaneous
geometry, 12.07 mm for static pose memory and 8.02 mm for conditional encoder
transport. Static memory's worst error is 142.81 mm during motion. Future memory
work must model motion hypotheses, revoke attachment assumptions explicitly and
avoid presenting stale hidden coordinates with unjustifiably small uncertainty.
The active pilot retains dimensions/goals but disables temporal pose memory.


## Current continuation checkpoint

The v1 online pilot is complete and audited: 75 actual rollouts, no invalid code,
geometry/raw training curves 0–4–4–4 versus 0–2–4–4, selected reserved success
15/20 versus 13/20. Preserve this as preliminary evidence; paired p=0.625 does
not support a strong held-out advantage claim by itself.

A real wrist-marker association failure led to a separately tested fusion repair.
After v1 completed, v2 was integrated and tested without changing shared actuation.
Three independent v2 repetitions are now running under
`outputs/online_geometry_v2_replications_01`, with all training frozen before any
reserved evaluation. Next: monitor the live coordinator, audit per-replica results,
and assess whether iteration and held-out gains repeat. Do not mark the research
goal complete merely from the earlier promising pilot or the geometry replay.


## Further development while the v2 repetitions remain frozen

Two separate, action-free prototypes address gaps in the representation:

- [Conditional rigid containment intervals](RIGID_CONTAINMENT_DIAGNOSTIC.md)
  express model-conditioned coordinate bounds under occlusion. The development
  diagnostic has better relation availability, but no point-estimate or policy
  success gain has been shown.
- [Point and axis goal constraints](RIGID_GOAL_CONSTRAINTS.md) convert geometric
  intent into a desired rigid transform while exposing unconstrained twist.
  Unit tests and a real sensor example verify geometry only, not execution.

Neither prototype changes the online v2 method or any candidate in its ongoing
independent repetitions. ReKep, kPAM and PARL motivate the separation of state,
intent and numerical grounding; they do not substitute for our own controlled
policy evidence.


## Current checkpoint after the completed independent comparison

The [three v2 campaigns](ONLINE_GEOMETRY_EXPERIMENT.md) are complete and audited.
They did not establish faster policy improvement from the added abstraction.
The goal remains unachieved. [Next experiment boundaries](GEOMETRY_NEXT_EXPERIMENT.md)
retain that evidence and separate physical execution, perception quality and
symbolic reasoning support.

[Evidence-checked memory](BOUNDED_GEOMETRY_MEMORY.md) improved paired development
percentiles, and [known robot visual geometry](ROBOT_VISUAL_GEOMETRY_MASK.md)
removed large false object estimates while explicitly reporting lost coverage.
The latter is now integrated through optional same-time robot packets; the other
new components still require online policy validation.

## Compact visual state and the current evidence boundary

The subsequent [visual-input comparison](../VISUAL_INPUT_LEARNING_RESULTS.md)
removed supplied centers and dimensions from Base while preserving a shared
hidden visual PnP actuator. Across three independent policy-evolution chains,
training-selected reserved success averaged 56.67% for Base and 65% for Full,
with paired chain differences of +40, -20, and +5 percentage points. This is
mixed evidence, not an established repeatable iteration advantage. Historical
stronger-baseline comparisons and their negative results remain separate.

The [partial-constraint experiment](../PARTIAL_GEOMETRY_LEARNING_PROTOCOL.md)
compares the full geometry package with visual Base using eight training layouts
per chain. All training completed; reserved testing stopped at the user's reduced
budget. Its [preliminary report](../PARTIAL_GEOMETRY_BUDGET_LIMITED_RESULTS.md) records
all available results and the current execution limit.

Across all nine evolved candidates per arm, Full has 1,226 recorded training
checkpoints with median length 366 characters and maximum 1,074; Base has 707 with
median 113 and maximum 293. These counts cover all recorded training checkpoints,
not just model-attached feedback. Full actor input totals 241,517 tokens versus
224,849 for Base, so these measurements do not demonstrate token or latency savings.

Full candidates read supplied geometry, but none has literal `reference`,
`relation`, `align`, `target`, or `goal_error` call sites. Static attribute counts
do not resolve every receiver, alias, or internal reasoning step. Partial-constraint
feedback appears only in chain 1 Full R3 and cannot explain the earlier R1 gain.
Any benefit therefore identifies the whole supplied geometry package. Complete
training analyses are `interface_usage_training_complete.json` and
`feedback_coverage_training_complete.json` under
`outputs/constraint_geometry_learning_replications_01`; earlier progress snapshots
remain historical evidence.

### State decision supported by the development evidence

Keep a single active-target record with three distinct roles:

1. **Current evidence:** same-time visual geometry or conditional center bounds;
   absent evidence remains unknown. Preserve whether an axis is observed or
   inferred and the measured-size/orientation assumptions.
2. **Conditional prediction:** at most one lift anchor transported by public
   robot poses under an explicit rigid-attachment hypothesis. A release command,
   target change, or contradictory current evidence invalidates that hypothesis.
   It is never relabeled as an observation or accepted pose.
3. **Intent:** the desired point/face relation and a residual only when its
   geometric reference is available. Do not infer stable contact from positional
   proximity or confuse the executor's logical held flag with physical grasp.

Store numerical regions internally; expose a short result such as
`attachment_hypothesis: contradicted` with provenance and a quantitative
consistency residual when useful. An agent can request the underlying interval
if it needs a different geometric query. That proposed summary is not integrated
into the current study and cannot explain its results. A discrepancy should mean
that the *joint assumptions* fail; it cannot by itself attribute the error to
slip rather than visual association, dimensions, or orientation.

The [conditional attachment diagnostic](../CONDITIONAL_ATTACHMENT_EVIDENCE.md)
provides a real reason for this separation. Fully observed point anchors have
limited coverage on the new training traces. In one behavior-equivalent logging
replay, a partially observed lift still has finite visual center intervals.
Transporting those intervals produces regions incompatible with subsequent
current-view constraints. This expands diagnostic coverage without inventing a
point observation, but it has not recovered a pose, changed an action, or improved
policy success. Regions and their comparisons remain conditional, not calibrated
probabilistic uncertainty.

This decision requires neither persistent all-scene poses nor a growing event
history. Any future online test of prediction summaries must be a separately
frozen intervention; it must preserve Base/Full action parity and retain the
current study's complete results, including regressions.


### CPU replay cost of the frozen abstraction

A bounded microbenchmark uses the frozen Full library and the recorded two-camera
occlusion fixture from training-development seed 990205 at 75.9 s. After three
warmup runs, 30 fresh-session repetitions produce identical constraint-only text.
Median wall time is **134.80 ms**, with p95 **145.89 ms**; median process CPU time
is **134.79 ms**. Checkpoint construction and JSON serialization together take
**0.044 ms** median wall time. The timed path includes robot masking, failed
single/joint-view point-pose fits, the additional constraint fit, and serialization.
The replay API performs no camera transport, rendering, model call, or actuation.

This is one repeated difficult frame, with one thread requested through the
OpenBLAS/OMP environment and a learning campaign running concurrently. It excludes
initial visual-model learning, module imports, sensor IO, and policy reasoning.
The 625-character benchmark record uses a fixed short exception label and no
placement goal; it is not the length of the original complete action record.
The result supports measuring the current CPU cost but does not establish a
representative control frequency, end-to-end realtime operation, or speedup over
VLM reasoning. Rollout elapsed time remains outside the optimization objective.

[Raw timing samples and source hashes](../evaluations/target_abstraction_cpu_replay.json)
are produced by `scripts/benchmark_target_abstraction.py`. This read-only replay
never enters actor feedback and changes no frozen policy or library.
