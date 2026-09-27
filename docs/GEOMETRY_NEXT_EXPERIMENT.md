# Next experiment boundary after the completed v2 comparison

**Current status: completed.** The final corrected mixed-shape pilot has 300 valid
rollout records and zero invalid slots; the full source, pairing, and freeze audit
passes. Training-selected reserved results are basic 0/20 (R0), numeric 8/20 (R3),
and symbolic 12/20 (R2). This is one actor chain per arm; numeric versus symbolic
has paired p = 0.125 and still requires independent replication. See the
[completed results and full learning curves](MIXED_SCENE_LEARNING_RESULTS.md).
The chronological checkpoints below preserve the development and repair history;
their earlier pending/running descriptions are no longer the current status.

## Evidence that determines the next action

The three independent cube-stacking campaigns did not show faster improvement
with the added rigid/symbolic interface. Reserved results were 28/30 versus 29/30,
30/30 versus 28/30, and 26/30 versus 29/30. The basic RGB-D PnP baseline already
provides surface position and size estimates. Keep this completed comparison as
a negative/mixed result; do not repeatedly change seeds until it looks positive.

The state diagnostics identified concrete implementation issues and opportunities:
robot visual geometry can become an object hypothesis; missing coordinate ranges
are more informative than an unconditional large uncertainty; static memory needs
current-evidence checks; desired geometry needs an explicit constraint interface.
The robot visual mask and shared PnP observer callback have now been checked in
live integration rollouts. Temporal memory, interval and desired-transform
components remain independent prototypes.

## Immediate development work

1. Establish a physical-execution ceiling for the existing uniquely colored
   `thin_beam_bridge` support graph using the same generic PnP executor. Use
   development-only scenes and a clearly labeled manual planner. Do not supply
   that planner or its solutions to experimental actors. Keep the three-cube task
   as a control rather than replacing or hiding its result.
2. Audit which failure causes the action interface lets the agent influence.
   `pick(color)` currently chooses its own grasp using the executor's surface
   observer. Supplied scene geometry cannot directly choose that grasp target.
   Any future geometry-target parameter must be available identically to every
   arm, while motion, gripper timing and feedback control stay common. A stronger
   controller must not be hidden inside the abstraction treatment.
3. Consolidate the interval, evidence-checked memory and desired-transform
   prototypes into a queryable interface only after their online integration is
   verified. Generalize shape observations through the existing mixed rigid
   observers; retain geometric identities, uncertainty and symmetry semantics.
   No colors, tower plans, site coordinates or recovery policy belong in the
   abstraction layer.
4. Freeze a new protocol and previously uninspected splits after development.
   All current training and reserved scenes are inspected evidence for future
   method changes. The old v2 coordinator is historical; its completed seed split
   must not become a fresh generalization claim for the repaired method.

## Planned causal controls, not yet a registered campaign

A useful follow-up distinguishes three levels rather than calling a baseline
with a pose observer "no geometry":

- Basic RGB-D PnP: the existing surface-estimate interface and raw RGB-D.
- Numeric rigid state: the improved perception, robot context and metric state,
  without additional symbolic relation or desired-constraint operators.
- Symbolic/metric interface: the same numeric estimator, plus typed references,
  conditional facts and geometric goal operators serialized for agent reasoning.

The last two must share the same perception implementation. Their difference can
then address representation and reasoning support, rather than simply improved
pose accuracy. All arms retain the same CPU actuator, sensor access, public task
information and robot-model inputs. Rollout time remains outside the objective.

Choose campaign size and task variants only after the execution ceiling is known.
Then record the training budget, selection rule, reserved split and interpretation
before generating candidates. Report complete learning curves, selected-policy
reserved success, actual abstraction use, and independent campaign variation.
Preserve invalid submissions and failed rollouts. Intrinsic precision, fewer
unknown relations and compact text are supporting diagnostics, not substitutes
for the requested policy improvement.

## First mixed-shape execution probe

The fixed manual planner with the unchanged PnP executor succeeded on 1/4 new
development scenes (75001–75004). The release audit separates pre-release visible-top
center bias from post-release cylinder movement. The existing mixed rigid fit
reduces beam center error on the same three completed pre-release frames, but was
not used by those policies and establishes no SR gain. A cube-symmetry evaluator
repair does not change these four final-state pass/fail outcomes.

See [the release diagnostic](MIXED_PNP_RELEASE_DIAGNOSTIC.md). The next development
step is a shared observer interface at the executor's visual checkpoints, followed
by post-release state validation. All arms must have identical access to that
interface; numeric and symbolic arms must use the same estimator. No new causal
campaign has been registered or launched from these measurements.

## Shared observer integration completed

A task-free callback now lets the unchanged shared PnP motion loop consume the
existing mixed rigid estimator. The side-only integration had 1/4 success and two
occlusion errors. Same-time wrist fallback plus robot visual masking reached 3/4
on those inspected layouts. Its frozen candidate then reached 6/8 on preselected
new development layouts 75101–75108; the archived basic-observer candidate reached
3/8 on the same layouts. The paired outcomes include one basic-only success and
one common failure. All outcomes and source checks are retained.

See [the shared observer interface and results](SHARED_PNP_OBSERVATION.md). This is
positive development evidence for the combined geometric observation path, not a
symbolic-versus-numeric comparison or an iteration-efficiency result. Numeric and
symbolic arms must share this path. No new agent campaign has been registered.

Next resolve whether the remaining workspace and beam-visibility failures require
an action-interface change before freezing a campaign. Current state must expose
unavailable observations and measured action residuals, not substitute an intended
pose. Keep any recovery capability identical in all arms. The new development
layouts are inspected evidence and cannot be reused as an unseen reserved split.

## Current-view correspondence regression resolved

The remaining beam regression was caused by assigning measured cuboid lengths to
the nearest world axes after a large rotation. A task-free, opt-in correspondence
solver now tests length permutations against current face planes and per-camera
finite bounds. It rejects ambiguity, does not use a temporal pose prior and keeps
inferred coordinates explicit. The fixed repaired candidate succeeds on 7/8 of
the already-inspected eight-layout regression, retaining every earlier success.
The first-foundation placement failure on 75101 remains. See
[the diagnosis, tests and physical verification](CUBOID_CORRESPONDENCE.md).

This is enough execution capability to proceed with the representation comparison;
the remaining failure must stay in the data. Next expose a common mixed numeric
state with explicit missing observations and changes, then add symbolic references
and task-independent goal constraints on top of those same measurements. Use
current fitted face vertices/axis lengths when resolving a face: a model's initial
height is not necessarily the extent along the current upward face after rotation.
The manual site selector and completed solutions remain withheld from actors.
Freeze task distribution, budgets and fresh train/reserved splits only after this
interface is ready. The current development success is not the requested learning
curve evidence, and the earlier negative cube comparisons remain unchanged.

## Mixed-state interface integrated

`MetricScene` and `SymbolicScene` now share the same current mixed RGB-D estimates.
Numeric and symbolic live sessions share camera acquisition, robot masking,
correspondence recovery and the PnP motion loop. Explicit history, unavailable
observations and desired states are separate; conditional prediction stays disabled
in live sessions. Typed face/axis references and generic desired alignment are
implemented above the numeric state without task or controller logic.

The two-layout integration completed all four executions. Archived numeric and
symbolic candidates produced identical physics trajectories and identical numeric
state at all five checkpoints per rollout. This establishes interface parity only;
there were no agent revisions. Ten state tests cover provenance and shape semantics.
See [the state interface](MIXED_SCENE_STATE.md) and its reproducible audit.

The next work is the controlled learning harness: freeze the starter, final API,
feedback projections and fresh scene split, then generate isolated actor revisions.
Retain the basic-PnP package contrast and the numeric-versus-symbolic representation
contrast as separate questions. Existing manual solutions and development failure
analyses must not enter actor contexts. No new learning campaign has started yet.

## Three-arm learning pilot registered

The matched starter, isolated observation libraries, source snapshot and fresh
training/reserved distribution are frozen in `outputs/mixed_scene_learning_pilot_01`.
See [the protocol](MIXED_SCENE_LEARNING_PROTOCOL.md). Unlike the earlier comparison,
all policy revisions will be evaluated on the reserved layouts after a global
freeze, permitting a direct reserved learning curve without evaluation feedback
entering policy evolution. The budget is 60 training slots and 240 reserved slots.

Automatic approval review rejected launching model generation because explicit
permission to transmit the named source/sensor payload to the model service was
not established. The user has been asked for that permission. Only local R0
simulation is running while it is pending; no model call or reserved evaluation
has begun. This is not a completed or blocked-goal claim.

The local R0 bootstrap has now finished: basic 2/5, numeric 3/5, symbolic 3/5.
All five numeric/symbolic trajectories match exactly. The fifteen local outcomes
and their failures are preserved, with zero model generations and zero reserved
rollouts. There is no new evidence of iteration efficiency yet. Transmission
permission is still pending; all dependent model work remains unstarted.

The user subsequently authorized the specified model transmission and requested
the experiment run. The frozen campaign has resumed from its completed R0 trials.
No runtime/library code or protocol was changed, and no reserved result has entered
actor feedback. The earlier permission blocker is resolved.

## Live pilot interface defects repaired transparently

The running pilot exposed two infrastructure defects. First, a stale protected-name
list rejected a permitted SciPy import; the isolated actor namespace makes that
restriction unnecessary. Second, the public observation contract promised a field
that raw cylinder fits could omit, although the state ledger already normalized it.
The latter caused five symbolic R2 exceptions and invalidates attributing those
failures to reasoning. Both defects and all raw data remain documented in the
[protocol](MIXED_SCENE_LEARNING_PROTOCOL.md).

The shared callback now returns the canonical ledger observation. A recorded
RGB-D regression covers both interfaces; the same initial physical trajectories
and state packets are preserved. Replaying all R0/R1 pairs establishes exact prior
feedback and trajectories, and reconstructing all six retained R1/R2 generation
inputs establishes byte-identical prompts and images. Therefore those outputs are
reused unchanged in `outputs/mixed_scene_learning_pilot_01_schema_corrected`.
Corrected R2 outcomes feed three replacement R3 calls. Original R3 outputs remain
archived and count in service usage; no original R3 policy was executed. Reserved
layouts are still untouched. There is still no demonstrated learning benefit.
