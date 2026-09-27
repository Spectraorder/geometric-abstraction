# Constraint-aware online feedback development

This is separate development after the frozen
[controlled recovery comparison](ONLINE_CONTROLLED_RECOVERY_RESULTS.md). The
reserved seeds and runtime of that comparison are unchanged. This work does not
turn its mixed result into a positive result.

## Representation change

The previous text exposed local placement residuals and pairwise support
predicates. A child could be locally aligned to a misplaced parent. The new
projection explicitly distinguishes `local_goal_geometry` from
`support_chain_geometry`: the latter is the conjunction of the local geometric
constraints of all ancestors in the public support graph. It lists violated and
unknown ancestor identities without choosing a repair action.

For example, when `red` is aligned with `blue` but `blue` is displaced from its
own goal, the representation can say:

```json
{
  "id": "red",
  "local_goal_geometry": "true",
  "support_chain_geometry": "false",
  "violated_ancestors": ["blue"]
}
```

These are observation-based geometric constraints, not contact or stability
certificates. False evidence dominates unknown evidence in a conjunction. Missing
translation evidence is unknown and cannot produce an executable residual.
The observer and low-level control remain unchanged. This projection requires
only existing sensor-grounded records and the public task graph; it does not
consult evaluator poses or outcomes.

The Full projection omits redundant absolute coordinates, retains available goal
residuals, and makes each violated or unknown constraint explicit. On the six
recorded turns of the earlier development recovery diagnostic, the serialized
geometry shrank by approximately 16–24%. This is a character-count measurement,
not a claim about total prompt tokens, inference latency or task performance.
Additional instructions also have a cost, so complete request receipts remain
the resource measure.

## Shared scheduler change

Failed execution retries now depend on the operated object, its support ancestors,
and holding state. Motion of an unrelated object does not unlock a retry. Loss of
visibility does not count as useful new pose evidence; newly available geometry,
a displacement above 1 cm or a changed holding state can unlock one. Clearing
retains all-object dependencies because free-space selection needs them.

After repeatedly requesting a blocked operation, the scheduler polls that same
operation's dependencies rather than waiting for arbitrary motion anywhere in the
scene. Both arms use this gate and receive the same qualitative holding constraints.
Only Full receives the geometric constraint projection. Shared instructions explain
that alignment to a misplaced support does not satisfy the original task, and that
upper obstructions may need clearing before their supports can move.

## Validation and live checks

Nine unit checks cover transitive violations, unknown evidence, definite support
violations under partial translation evidence, no Base geometry leakage, graph
cycles, relevant/irrelevant movement, holding changes, copied failure state and
all-object clearing dependencies. The new runner also passes the existing plan
continuation, failure-queue invalidation and context-rotation scheduler tests.

`outputs/online_reasoning_dev/pairs_01/plan.json` records four development episodes:
seeds 2028100101 and 2028100102, Base and Full, 180 seconds each, four parallel
processes. The first seed is a normal construction check and the second is a
known difficult development layout. No reserved seed is used. All attempts and
source snapshots are retained. These runs start from the ordinary unassembled
scene and do not use the external foundation intervention.

Implementation:

- [Constraint projection and retry gates](../scripts/online_reasoning_state.py)
- [Separate development runner](../scripts/run_online_reasoning_development.py)
- [Semantic regression tests](../tests/test_online_reasoning_state.py)

The development runner is an explicit fork because the original runner is frozen
by a completed study. It preserves the shared physical skills and actuator. This
variant changes feedback, shared instructions and retry scheduling together, so
comparison with earlier runs would not isolate a single component.

## Completed development results

The four 180-second episodes finished and passed an independent source/resource
audit. They used the first constraint-feedback revision, before the subsequent
observability and private-contact corrections described below.

| Layout | Base | Full | Base / Full requests | Base / Full total tokens |
| --- | --- | --- | ---: | ---: |
| 2028100101 | Success, 128.07 s | Success, 126.90 s | 1 / 1 | 11,638 / 11,909 |
| 2028100102 | Timeout, 180 s | Timeout, 180 s | 5 / 7 | 68,095 / 104,669 |

A follow-up paired development run on layout 2028100102 used 360 seconds for both
arms, allowing time for multiple physical repairs after the late failure. It used
strict positional evidence for retry/motion checks, but still preceded the new
private contact evaluator. Both timed out: Base made 17 requests and used 223,161
tokens; Full made 18 requests and used 248,613 tokens. All usage was available,
reset count remained one, and measured maximum simulation lag stayed below one
second. Artifacts are in `recovery_360_01/` under the same development root.

These checks do not demonstrate a recovery-success or total-token advantage.
Full sometimes selected the correct lower support to repair, but explicit
constraint text alone did not produce reliable execution or consistently valid
plans. The event-driven scheduler also leaves most failure decisions to the fast
model, even though these are replanning problems rather than routine progress.
Routing and physical clearance remain development issues; no new reserved study
has been started from these results.

## Evidence qualification correction

The earlier retry and motion guards compared fitted centers even when not all
translation directions were observed. The new `observed_center` adapter requires
`fully_observed=true`, finite coordinates and three dimensions before accepting
a center as evidence of motion. Missing/partial evidence does not assert movement.
A regression test covers this boundary. Existing local pick/place observations
and shape fitting remain unchanged.

An accelerated check on the difficult layout still detected a fully observed
support displacement of about 14.6 mm. The observability correction therefore
does not explain away that physical failure.

## Physical contact diagnosis and evaluator correction

The private development contact diagnostic found the robot's `Fixed_Jaw` body
contacting the already placed magenta cube while picking the final blue cube.
Contact was recorded at simulation time 132.25 s; magenta displacement exceeded
8 mm at 132.47 s. Later contacts included the yellow beam and green foundation.
The evidence is under `contact_diagnostic_02/private_contacts.jsonl`. These private
records were never supplied to the actor or skills.

The original contact helper matched geometry names. Some robot collision meshes
are unnamed, so the helper missed these contacts even though their owning robot
bodies were identifiable. The first geometry-name diagnostic missed the robot
contact; the refined body-name diagnostic exposed it. The public gripper geometry
also explicitly covers finger pads, not the entire housing. Object-footprint
screening alone does not reserve the swept space of that housing.

New development runs use [ContactCompleteEnv](../scripts/online_contact_scene.py).
Its private contact groups combine geometry names with body ancestry, including
unnamed collision meshes. Three tests cover unnamed geoms, unnamed child bodies
and namespace boundaries. The original frozen environment is preserved.

New runs additionally save a private compiled model and terminal physics state.
An independent offline reconstruction of the two accelerated development checks
confirmed no object/robot contacts in the successful 2028100101 terminal state,
and contact with the held blue cube in the unsuccessful 2028100102 state. The
normal scene passed the corrected success criterion; the difficult scene still
failed. These accelerated checks are not latency or model-reasoning measurements.

The earlier online success counts remain outcomes under their registered contact
helper. They have not been retrospectively certified against the corrected
body-ancestry check: those episodes did not archive enough terminal physics to
perform this contact reconstruction. The newly discovered defect does not by
itself establish that any recorded successful episode was a false success.


## Ordinary-scene clearance checks and reduced scope

A shared workspace screen now reserves sampled robot jaw-housing pickup space
around objects needed later in the construction. It uses public robot geometry,
robot-only IK and visual object estimates. Partial visual positions contribute
conditional planning bounds, not executable fully observed poses. Candidate
placements must also meet position and orientation checks. This sampled screen
is not a continuous collision-free trajectory certificate.

The latest accelerated checks in
`outputs/online_reasoning_dev/clearance_preflight_03` succeeded on all three
development layouts (2028100101–2028100103), using the corrected private contact
evaluator and no model calls. Earlier failed checks remain recorded. These are
physical feasibility checks, not evidence of a Full reasoning, token or latency
advantage; they do not retrospectively validate previous studies.

The user subsequently removed high-complexity scenarios from current development.
Continue with ordinary unassembled bridge-plus-two-cube scenes. Do not inject
collapses or loaded-stack shifts, expand into multi-level clearing stress tests,
or extend timeouts repeatedly to obtain success. Natural failures still consume
the episode budget without resets. The unfrozen development runner defaults to
180 seconds and no longer accepts an intervention fixture or preparation hook.
A future small comparison should use five fresh paired layouts, one episode per
arm/layout and at most four concurrent workers under a separately frozen protocol.
No additional model experiment was launched for this scope change.
