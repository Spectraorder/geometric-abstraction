# Online bridge-tower composition pilot

Status: all 20 reserved episodes completed and passed the final audit under
`outputs/online_bridge_tower_01`. See [results and limitations](ONLINE_BRIDGE_TOWER_RESULTS.md). The frozen design uses two
repetitions per arm/layout, 20 episodes, a 360-second wall-clock budget, and
sequential execution with balanced arm order. Do not tune on reserved outcomes.

## Latest 200-second comparison and timing analysis

The separate five-layout comparison is complete: Full succeeded in 4/5 episodes
and Base in 2/5. Among successful episodes only, mean total completion time was
167.66 seconds for Full versus 189.77 seconds for Base; mean cumulative completed
model-request time was 36.95 versus 45.84 seconds. These are success-conditioned
subsets with different layouts, not matched averages. Full also has three
interrupted requests excluded from these model-time subtotals.

The [complete success-conditioned analysis](ONLINE_COMPACT_TOWER_200_RESULTS.md#timing-conditioned-on-success)
includes medians, request counts and the sole jointly successful layout, alongside
the all-episode metrics. Mean per-request timing favors Full only after filtering
to successes; it does not establish a general model-latency advantage. These
200-second results do not change the historical 360-second pilot below.

## Current development scope

The user narrowed subsequent development to ordinary bridge construction followed
by two cubes stacked at the beam center. Do not add artificial collapses,
loaded-stack displacements, or multi-level clearing stress scenarios. Natural
action failures still remain in the same scene and count against the time budget;
there is no reset, failure filtering, or automatic retry of an episode.

The current explicit-target development runner uses a 200-second wall-clock
budget, as requested by the user. Both arms receive the same budget, including
perception, model requests, actions and settling. The older reasoning-development
runner remains a historical 180-second variant. No additional model episodes are
launched by this budget change.
For a future small comparison, retain five shared fresh layouts with one episode
per arm/layout (ten total), at most four concurrent workers, and the same budget
and physical skills in both arms. Register that comparison separately before
execution; do not reuse development layouts as held-out evaluation.

The completed 20-episode protocol below and all previous failures remain historical
evidence. Its 360-second budget and repetition count are unchanged. See
[current development notes](ONLINE_REASONING_STATE_DEVELOPMENT.md) for the latest
physical checks and their limits. The separate
[ordinary-scene comparison](ONLINE_ORDINARY_TOWER_STUDY.md) fixes the next small
design and provides a bounded parallel runner.

## Question and task

Does geometric text help an LLM compose existing manipulation skills and recover
from failures within a fixed wall-clock budget in a new scene?

The task combines two previously studied constructions: a bridge and a stack.
Build two foundations and a spanning beam, then place two cubes in a vertical
column at the center of that beam. There are five distinct objects. The previous
bridge task included a cylinder cap; the new goal replaces that cap with the
first cube and adds a second cube above it.

```text
                    upper cube
                    lower cube
                ===== beam =====
                foundation   foundation
                ________________________ table
```

All objects start separately on the table. The public goal identifies the objects
and desired support graph, without supplying measured poses or dimensions. The
composition is new even though its component skills were trained previously.
Initial positions, color-to-role assignments and reachable construction locations
vary across reserved layouts. Start with the existing shapes and dimensions;
novel shapes, changed friction and object-size shifts are separate questions.

## Shared skill experience and treatment

Use the same two frozen executable skills and the same brief skill descriptions
in both arms. Select their source policies using a rule fixed from training
evidence, independently of reserved performance. Record the exact source policy,
training chain, selected revision, adapter changes and final executable hash.
The two-entry library can be presented directly; a retrieval system is unnecessary.

The conceptual composition is:

```text
build_bridge(left, right, beam)
stack_on(object, support)
```

The LLM chooses the skill, binds public object identities, and decides whether to
continue, obtain another observation, or repair an earlier subgoal. A call uses
current observations to determine action targets. Each skill makes a bounded
attempt and returns control on completion or a recoverable failure. Bridge
execution needs interruption points between placements so the agent can address
an incomplete subgoal without redoing valid work. Shared primitive actions remain
available for repair when the two high-level calls are insufficient.

Current learned policies are complete `run(api)` programs. Extracting and
parameterizing their behavior into reusable skill calls is an adaptation that
must be documented and checked on development scenes. Do not describe adapted
skills as unmodified frozen policies. Keep skill implementations identical across
arms after this development stage. Their private geometric execution backend may
be shared without exposing its estimates to the Base agent.

| Component | Visual Base | Geometry Full |
| --- | --- | --- |
| Goal, identities and calibration | Same public information | Same public information |
| Skills and prior examples | Same frozen library | Same frozen library |
| Initial visual context | RGB-D access and initial RGB | Same access and initial RGB |
| Subsequent default LLM observation | Current image and qualitative action status | Compact geometric text and qualitative action status |
| Additional observations | May request current RGB-D | May request current RGB-D or images |
| Geometric references and residuals | No supplied metric packet; may derive estimates from permitted sensors | Supplied current references, observability and residuals |
| Model, reasoning setting and action interface | Shared | Shared |

This comparison measures the online geometric-interface package, including its
observation presentation. It does not isolate text serialization from the supplied
geometric estimator. Skill-internal sensing and actions have the same boundaries
in both arms. Default image attachment differs as part of the treatment; record
actual attachments and requests rather than assuming a token or latency advantage.

## Implemented development interface and model routing

The development runtime uses a JSON decision interface, with executable skills
bounded at individual placements. `execute(bridge, node)` handles a foundation or
the beam; `execute(stack, node)` handles one single-support cube. Public color keys
bind objects, rather than hard-coded role names. The controller may also request
an observation, release the gripper, wait, or request replanning. No model-produced
Python executes in the host. The same shared skill code serves both arms.

The bridge source is replica 01's training-selected R1 from
`geometry_parallel_01`; the stacking source is replica 01's training-selected R2
from `geometry_cube_stack_01`. Source hashes are checked before loading. The
adapters retain the shared initialized session and expose bounded portions of
those policies; they add continuation for an already held object and an explicit
open-gripper recovery action. Workspace selection uses the bridge policy's
observed table/geometry and robot-only IK. The stack adapter uses the trained
support-refresh and occlusion logic. These are adapted skills, not unchanged
whole-episode policies.

Astra (`gpt-6-astra`, medium reasoning) supplies the initial decision and explicit
replans. Luna (`gpt-6-luna`, low reasoning) supplies routine decisions. Two
consecutive unconfirmed skill calls also trigger Astra. These development routing
settings are identical across arms. A single isolated conversation persists across
model switches; a three-request smoke test verified Astra-to-Luna context
continuity and subsequent Luna cache usage. Per-turn token usage is the verified
difference of cumulative counters, not the sum of cumulative CLI receipts.

Implementation entry points:

- [Persistent episode](../scripts/run_online_bridge_episode.py)
- [Five-object scene and episode clock](../scripts/online_bridge_scene.py)
- [Shared sensor API and skill adapters](../scripts/online_bridge_skills.py)
- [Persistent model client and accounting](../scripts/online_bridge_actor.py)
- [Five-color geometry bundle](../scripts/build_online_bridge_geometry.py)
- [Fixed-sequence development preflight](../scripts/run_online_bridge_preflight.py)

The initial fixed-sequence preflight reached a five-second full-structure stable
interval. A second layout collapsed during the upper placement. These accelerated
checks establish feasibility and a recovery challenge; they are not online model
results or real-time latency evidence. Development artifacts are under
`outputs/online_bridge_tower_dev`, separate from reserved evaluation.

### Development evidence so far

The first model-driven Full episode completed in 179.5 seconds with seven
decisions and one environment reset. Its trace exposed two unnecessary repeated
skill calls, including a mistaken reading of an absolute coordinate as a residual.
The development update now labels placement residuals and tolerances explicitly,
and both arms' bridge adapter checks an existing placement before grasping again.

A subsequent paired development run on seed 2028100101 succeeded in both arms:

| Measurement | Visual Base | Geometry Full |
| --- | ---: | ---: |
| Time to verified success | 139.97 s | 138.17 s |
| Model decisions | 5 | 4 |
| Newly attached images | 6 | 2 |
| Input tokens, including cache | 90275 | 77986 |
| Output tokens | 279 | 222 |
| Model-request wall time | 19.46 s | 18.10 s |

Full skipped an already aligned foundation. This single development pair does
not establish a latency or success-rate advantage. Both arms used the same updated
skill check. Current development additionally records initial physics hashes and
explicit sustained-success evidence, and supports metric depth images on an
observation request. The original 300-second budget was extended to 360 seconds for both arms during
development. Reserved evaluation began only after these development changes were frozen.

A second paired layout (2028100102) produced a natural collapse after the top
placement. Both arms continued without resetting, but timed out at the original
300-second budget. Initial joint/free-body positions and velocities matched
exactly across arms. This exposed a missing recovery operation: moving a fallen
object away from the construction site. A shared `clear` primitive now searches
observed tabletop locations outside the worksite, screens observed object
footprints, checks robot reachability, and uses the same pick/place controller.
It returns unavailable when required geometry is missing; it never teleports an
object or supplies private task truth. An accelerated physical diagnostic cleared
the fallen blue object with a single environment reset for the whole sequence.

The compact feedback update removes repeated target coordinates and relation
residual dictionaries. It retains current centers and vertical bounds, observation
status, orientation compatibility, target-minus-current placement residuals,
translation checks, and three-valued support relations. Common tolerances and
reference semantics are stated once. Depth images with an explicit metric display
scale remain available through `observe` in both arms.

On the third paired development layout (2028100103), Base succeeded in 147.42
seconds with five decisions and six image attachments; Full succeeded in 139.42
seconds with four decisions and two attachments. These remain development
observations, not reserved results. In the updated 360-second recovery check of
layout 2028100102, Base completed clearing and reconstruction in 339.47 seconds
with 13 decisions, and Full in 316.57 seconds with 10 decisions. Both episodes
kept the failed scene and completed without another reset. These natural recovery
trajectories need not have identical intermediate failure states; reserved analysis
will compare all episodes from matched initial layouts.

The frozen pilot runner is [run_online_bridge_pilot.py](../scripts/run_online_bridge_pilot.py).
It freezes scene/robot assets, runtime sources, package versions and the model
client version. It balances arm order across two repetitions of five shared layouts, executes one
episode at a time, preserves every attempt, and rejects source changes after
registration. The summarizer reports failures at the time budget and groups repeat
outcomes by layout. A runtime infrastructure error stops dispatch without an
automatic retry. Registration requires evidence from all three development layouts
and a successful online episode in each arm.

## Continuing execution and recovery

An episode has one initial environment reset. The environment, initialized object
models, gripper state, skill state and LLM conversation persist until success or
timeout. Episode histories do not carry into other reserved episodes. Previously
trained skill experience is fixed throughout evaluation.

At skill boundaries or failures, acquire a current observation, present the
permitted feedback, request one short decision, and execute a bounded action.
Do not request a complete revised policy or perform offline training each turn.
Do not reinitialize the geometric session after objects have left the table:
the current session learns dimensions from initial table-supported observations.

For example, a placement can leave the lower cube beside the beam. If current
observations still support the bridge geometry, the agent can attempt that cube
placement again before stacking the upper cube. If the beam has moved, the agent
must reassess the bridge first. If the cube pose is unavailable, the feedback marks
it unavailable and the agent may obtain another view. A stale pose is never made
current by retaining the conversation or by assuming a successful grasp.

Recoverable manipulation errors are episode events, not permanent failure flags.
Successful recovery can still result in episode success. Neither a `released`
status nor a geometric support candidate certifies physical task completion. A
private common evaluator checks the full structure and stability without returning
object truth or diagnostic evaluator details to the agent. Runtime infrastructure
failures are recorded separately and never silently retried as new episodes.

## Timing and size

The development set contains three layouts, excluded from evaluation. These
checks covered skill composition, continued execution after errors, observability,
and request latency. Runtime choices were frozen before reserved evaluation.

The reserved pilot uses five shared new layouts, as requested by the user.
With two independently initialized episodes per arm and layout, this gives
20 episodes total (5 layouts x 2 arms x 2 repetitions). Pair initial scenes across arms and balance
execution order. Repetitions use independent episode histories; they are not
independent scene layouts or new policy training chains.

The development budget started at 300 seconds and was increased to 360 seconds
for both arms after observing the time required for an initial construction,
clearing, and reconstruction. The reserved protocol uses this common
360-second budget. Include initialization observations,
geometric grounding, model requests, local queueing, actions and final success
verification. Simulator startup and loading fixed assets may precede the clock
under the same rule for both arms. Cached inputs still count as input tokens;
record cache usage separately and preserve raw request accounting.

The online runtime targets one simulated second per wall-clock second, including
while the LLM is responding. Actions are paced and the environment owner advances
physics during model waits. Request time is charged to the deadline. The report
records wall time, simulated time and maximum scheduling lag, flagging episodes
that exceed the registered two-second lag threshold. This differs from the legacy
independent-rollout driver, which pauses physics during worker waits; that driver
does not define the online timing contract.

Use one measured episode at a time initially, interleaving arms, to limit local
resource contention in the latency comparison. Avoid overlap with the existing
large simulation campaign. Throughput parallelism can be evaluated separately.

## Outcomes and fair recovery analysis

The primary metric is success within the common wall-clock budget. Report a
success-versus-time curve, and mean capped completion time, assigning the full
budget to episodes that time out. For a budget B and time to verified success T,
these are P(T <= B) and the sample mean of min(T, B); define T as infinity on
failure. Show per-layout paired outcomes and counts, with repeats grouped by
layout. This pilot supports a feasibility comparison, not a precise broad claim
about statistical superiority.

Also record time from feedback availability to the next action dispatch, model
request latency, perception time, action time, input/output/cached tokens, attached
images, camera acquisitions, decisions, and repair attempts. Record acquisition
timestamps and action-time revalidation so delayed reasoning cannot execute an
unvalidated stale target. Timestamp shared observation packets consistently.

Both arms receive the same initial scenes. Their subsequent states and failures
may differ because they take different actions. Include every episode in the
main success and time metrics. Do not compare recovery success only on each arm's
own natural failures as though those were matched cases. Such conditional counts
are descriptive, with failure type and denominator reported.

If direct recovery comparisons become necessary, add a separately registered,
small set of identical partial-assembly or faulty-placement starting states.
Preserve the same permitted shape-initialization history and gripper state for both
arms; do not silently seed current poses from a hidden state snapshot. These are
new episodes with matched starts, never resets inserted after an agent's failure.
Select these scenarios from development evidence before either arm is evaluated.
This diagnostic is optional and is not included in the registered 20 episodes.

## Implementation boundary and readiness

The frozen observer supports four unique colors. The opt-in online bundle adds
magenta as a fifth distinguishable instance, with color-to-role permutations
shared across arms. Existing frozen observer sources remain unchanged. Validate segmentation
in both side and wrist views, and provide the same palette information to Base.
Avoid introducing a general instance-segmentation system for this pilot.

A new runtime must retain a single environment and session across decisions,
return control after skill failures, and support an actual wall-clock deadline.
The implemented loop has no separate decision-count termination: normal episodes
continue until verified success or the wall-clock deadline.
Persistent exact-time reference checks also require a coherent observation
snapshot and current action validation when physics runs continuously. Repeatedly
invoking the current independent-rollout driver does not implement this loop.

Keep this pilot's task adapter, skill bundle, prompts, observation schedules,
model settings, seeds and timing implementation in a new registration. Do not
modify the existing bridge or three-cube study snapshots. Before a paid pilot,
validate five-object perception, bridge-plus-stack physical feasibility, bounded
skill calls, recovery without resets, and accounting under a fixed time budget.


## Reserved launch

The fresh reserved seeds are 2028101001-2028101005. Development uses
2028100101-2028100103. The source and asset snapshot, package versions, client
version, exact skill policy hashes, model routing and all 20 slots are recorded in
`outputs/online_bridge_tower_01/protocol.json`. Registration is sealed with a
separate hash and is checked before and after every episode.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/run_online_bridge_pilot.py \
  --root outputs/online_bridge_tower_01 --phase prepare

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python -u scripts/run_online_bridge_pilot.py \
  --root outputs/online_bridge_tower_01 --phase run
```

Inspect `progress.json` and `campaign.log` for execution, and each episode's
`feedback.jsonl`, `dispatches.jsonl`, `actions.jsonl`, model receipts and result for
evidence. After all slots complete, `summary.json` and `online_curves.{png,pdf}`
report paired layout outcomes, success-versus-time, capped completion time,
per-model latency, tokens, new image attachments and timing-limit violations.
Incomplete requests retain raw logs and are flagged rather than assigned zero
usage. No episode is silently restarted, and normal execution has no independent
decision-count cutoff before its wall-clock deadline.

## Independent evidence audit

The read-only [auditor](../scripts/audit_online_bridge_pilot.py) checks archived
evidence independently of the episode receipts. It verifies source registration,
persistent conversation history and model routing, prompt and image provenance,
the Base/Full feedback boundary, token-counter increments, and simulated-time
advance between feedback and action dispatch. It also checks that temporary
credentials have been removed from completed episodes. Final auditing invokes
the outcome checks for paired initial states, shared skills and sustained success.

After all episodes finish, save a new audit report:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/audit_online_bridge_pilot.py \
  --root outputs/online_bridge_tower_01 \
  --output outputs/online_bridge_tower_01/audit.json
```

During execution, `--partial` audits only the completed prefix. Use a new output
filename for each report; existing reports are never overwritten. The auditor
records its own source hash and the hashes of the inspected artifacts. It was
added after registration as an analysis tool and does not change the frozen
runtime or the experimental treatment.
