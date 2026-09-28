# Sensor-grounded ring-over-peg policy training

This development pilot trains one Full policy on five layouts with three generated
revisions. It is not a Base/Full comparison, a reserved evaluation, or evidence
of generalization. Each rollout resets its scene; the executable policy then
observes and acts without additional model calls during that rollout.

## Physical and sensor development evidence

The user's original privileged script succeeded on development seeds 1, 2 and 3.
It reads simulator poses and contacts, so those outcomes establish physical
feasibility only. Neither that source nor its traces are supplied to the actor.
The original scene and oracle are preserved in `ring_peg_scene.py` and
`ring_peg_oracle.py`.

The separate sensor-driven script in `check_ring_peg_sensors.py` achieved **2/3**
successes in `outputs/ring_peg_dev/sensor_physical_05/`. It measured object geometry
from RGB-D and used robot-only kinematics for pad motion. Seed 3 failed to lift
the ring: current images correctly placed it on the table, but the development
script continued transport without checking lift, eventually requesting an
unreachable target. This failure is retained and is not a perception success-rate
estimate. Earlier failed development attempts remain alongside the final check.
An observation-only policy also completed through the isolated RPC process.

## Observation and action boundaries

`ring_peg_geometry.py` fits a visible cap plane and retains separate outer and
inner polygon boundaries. It verifies depth below the cap inside the opening.
Contours are simplified at the measured pixel scale; no six-vertex template,
object dimensions, scene layout or simulator pose is imported. The observer's
present scope is a resolved convex outer boundary and one convex opening in a
rigid, constant-thickness extrusion, initially upright and table supported.

Initialization measures thickness and footprint. Under later partial occlusion,
the observer fits that initialized footprint to visible current boundary samples
and a current cap plane. It rejects inadequate, poorly conditioned, or ambiguous
fits and checks agreement with visible background. Equivalent completions are
compared at the initial/current contour resolution; this is a deterministic
resolution rule, not a calibrated confidence bound. Returned hidden vertices are
model-inferred. No pose is propagated from gripper motion or a previous object pose.

The cylindrical peg is observed from its cap. Its initial measured position may
be retained under the public assumption that the peg is physically fixed to the
world; this is distinct from tracking the movable ring.

`ring_peg_robot.py` exposes robot-only previews and motion targeting the center of
the fixed finger pad, with explicit roll and jaw commands. It includes both jaw
collision geometries for geometric clearance calculations. It never selects an
object wall or placement target, detects object contacts, or substitutes truth
coordinates. The policy chooses grasp points, opening, approach, lift, transport,
release and recovery. This new action interface is isolated from the frozen cube
studies and could be supplied unchanged to a future Base arm.

The restricted worker receives only explicit public RPC results. The simulator,
layout generator, object geometry constants, private contact checks and final
success evaluator stay in the host process. Policy source is restricted to
numerical imports and functions, runs without network/repository access, and
cannot reset the environment. The actor receives contracts and observations,
not perception, actuator or development-controller source.

## Fixed training budget

- Output: `outputs/ring_peg_full_train_01`.
- One persistent Full actor context, Astra high, no supplied task-solving policy.
- Five new training layouts: 2030100101–2030100105.
- R0 only observes; R1–R3 are complete model-generated executable policies.
- Three model requests and 20 rollout slots, including R0.
- Five parallel rollout workers, one BLAS/OpenMP thread per worker.
- 50,000-character policy limit, 120-second rollout wall limit, 600 API calls.
- Strict training retention: preserve every incumbent success and add a success;
  ties retain the incumbent. Preserve every rejected proposal and failed rollout.
- No held-out evaluation or additional model revisions are authorized by this pilot.

The runner waits for the separately registered contract-only stacking training to
freeze before requesting a model, avoiding an additional concurrent request above
that study's four-request cap. Training sources and assets are frozen at registration.
Infrastructure failures stop the new job and retain all attempts; no automatic retry.

Report all per-layout training outcomes, the retained revision, input/output and
cached-input tokens, image attachments and model-request time. Cumulative persistent
session counters are differenced between turns; cached input is already part of
input tokens. Physical development outcomes and training outcomes remain separate.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python \
  scripts/run_ring_peg_training.py --output outputs/ring_peg_full_train_01 --phase prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -u \
  scripts/run_ring_peg_training.py --output outputs/ring_peg_full_train_01 --phase train
```

## Repair development and second training pilot

The first registered pilot completed with **0/5 successes at every revision**;
R0 was retained. This is a negative training result, not a reserved estimate.
Initialization failed for three layouts: two exposed an unsupported `magenta`
association and one had too few samples on the inner contour. Later failures
included unavailable post-grasp geometry and policy-level grasp/placement choices.

The opt-in `ring_peg_development.py` adapter leaves every original frozen source
unchanged. It adds magenta association, uses calibrated 640x480 RGB-D instead of
320x240 for local perception, tests background consistency at camera-ray/cap-plane
intersections, and verifies closing-axis alignment after the fixed-pad IK offset
has converged. The previous background test used projected measured background
points, which can disagree with the cap footprint under perspective. No simulator
object state supplies perception or action targets. No edge-count template is added.
The adapter's installation context is process-local and must not be used concurrently
by threads with different runtimes.

Development artifacts are retained separately:

- `outputs/ring_peg_dev/repair_initial_01`: initialization succeeded on all five
  previously used training layouts. These layouts remain training/development data.
- `outputs/ring_peg_dev/repair_policy_01`: the unchanged first-round policy from the
  original pilot visually verified lift on all three original development seeds,
  but completed insertion on none. Current post-lift estimates remained available;
  the policy stopped during alignment or tilt checks.
- `outputs/ring_peg_dev/repair_physical_01`: the earlier sensor development controller
  also completed insertion on none of the three seeds with the repaired interface.
  Two attempts ended in unreachable transport targets and one missed insertion.
  The controller does not reliably verify lift before transport. These outcomes
  are preserved; the earlier 2/3 result does not validate the repaired controller.

Eleven focused and existing interface tests passed, and the original training
source hash audit still passed. These checks establish interface improvements,
not successful task learning.

The user's continuing ring-task objective is pursued with a separate bounded
pilot, `outputs/ring_peg_full_train_02`, through
`run_ring_peg_repaired_training.py`. It reuses the five training layouts, starts a
new persistent actor with an observation-only R0, and permits R1–R3: three model
requests and twenty rollout slots, with at most five rollout workers. Policies are
not seeded with the development controller, privileged oracle, or previous trained
programs. The public contract reports the revised sensor/kinematic semantics, not
a task-solving procedure. Training retention, isolation, time/call limits and token
accounting remain unchanged. All repair sources and development evidence are frozen
before dispatch. No Base comparison or reserved test is included in this pilot.

## Capable starting policy for subsequent training

The second neutral-start pilot also completed at 0/5 for R1, R2 and R3. All five
layouts had working initialization and visually verified lifts. Final placement
and release remained unsuccessful. The user then requested a better starting
point for comparing training progress, explicitly changing the observation-only
R0 assumption for a new variant.

The maintained starting policy is
[`ring_peg_starter.py`](../policies/controllers/ring_peg_starter.py). It reuses the
second pilot's R3 grasp search and adds hand-edited visual transport and release:

1. Choose a rim grasp from the observed outer and inner boundaries and robot-only
   jaw previews. Verify actual lift using a fresh ring estimate.
2. Keep the ring, both fingers and the previewed opening above the observed peg.
   Correct lateral position from current hole geometry. Retain the commanded pad
   target to prevent repeated lateral commands from accumulating gravity/tracking
   error. This controller memory does not replace a fresh object observation.
3. Check that the peg can pass through the observed aperture, open above the peg,
   and retreat. The unchanged private evaluator determines release and stability.

No simulator object pose, private contact state, layout-specific coordinates or
privileged oracle source enters the policy. This is nevertheless a **task-specific
capable starter**, including prior training and human engineering. It is not a
neutral baseline or evidence of learning from scratch.

### Development validation

All attempts are preserved. No new model requests or reserved tests were used.

| Check | Layouts | Complete task success |
| --- | --- | --- |
| First starter, before command-memory correction | Existing development seeds 1–3 | 1/3 |
| Final starter | Same development seeds 1–3 | 3/3 |
| Exact final starter | All five existing training layouts | 4/5 |

The final validation lives in `outputs/ring_peg_dev/capable_starter_02/` and
`outputs/ring_peg_dev/capable_starter_training_01/`. Training seed 2030100101
stopped because current ring geometry became unavailable after lift; no pose was
substituted from grasp motion. Seeds 2030100102–2030100105 passed the unchanged
insertion, robot-release and two-second stability evaluator. These are reused
development/training layouts, so 4/5 is a starting-point measurement, not a
held-out success-rate estimate. The retained failed first version must remain
part of the development record.

### Training entry point and comparison

`run_ring_peg_seeded_training.py` supplies this exact policy as R0 and includes it
in the first actor prompt. Preparation validates the starter's exact source hash
against all five development outcomes and freezes it with the adapter and public
contract. The registered output is `outputs/ring_peg_seeded_train_01`; preparation
alone does not dispatch a model request. The available training phase maintains
five layouts, R1–R3, three model requests and twenty rollout slots including R0,
with at most five rollout workers. It executes and records R0 again under the
registered runtime before generating any revision. All previous campaigns remain
unchanged.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python \
  scripts/run_ring_peg_seeded_training.py \
  --output outputs/ring_peg_seeded_train_01 --phase prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python \
  scripts/run_ring_peg_seeded_training.py \
  --output outputs/ring_peg_seeded_train_01 --phase train
```

Report the actual registered R0 success and rollout cost alongside every revision,
raw and training-retained curves, and cumulative model resources. Preserve the
same strict retention rule: a candidate must retain every incumbent success and
add at least one. Starting from 4/5 leaves limited room on these five training
layouts; do not degrade the starter or select easier tests to shape a curve.
Use a separate frozen evaluation for generalization claims.

This entry point is currently **Full-only**. A later Base/Full comparison must
match access to prior task experience and action primitives; supplying this
geometry-dependent source only to Full would confound representation with prior
skill. Do not use the old zero-start Base as its matched baseline. The historical
zero-start Full pilots can be reported as separate conditions with their different
prior experience made explicit.

## Launched API-scaffold training variant

The user considered the capable 4/5 starter too strong for measuring policy
acquisition and authorized a simpler starting point followed by training. The
capable policy remains available as a future skill; its registered campaign
`ring_peg_seeded_train_01` remains unstarted and is not the active experiment.

The new `policies/controllers/ring_peg_api_starter.py` supplies only:

- current observation with explicit unavailable-state handling;
- a robot-only preview of a caller-selected target;
- forwarding of explicit motion and jaw commands with error handling;
- command-target bookkeeping for caller-selected increments.

R0 reads the public task, robot and object observations without moving. The actor
must write the grasp search, target selection, action sequence, geometric
alignment, release and recovery itself. No capable starter or earlier trained
candidate enters its context. Generic command bookkeeping preserves a requested
height despite robot tracking error; it does not propagate an object pose or
certify contact, reachability along a path, or task success. This assistance must
be shared by any later Base/Full comparison. The expected non-acting R0 success
rate is not tuned to an intermediate target.

The separately registered output is `outputs/ring_peg_scaffold_train_01`, using
`run_ring_peg_scaffold_training.py`. It uses the repaired 640x480 sensor interface,
one fresh persistent Full actor, the five existing training seeds, R0–R3,
three model generations and twenty rollout slots. At most five rollouts run
concurrently. Training retention, 120-second/600-call rollout limits and token
accounting remain unchanged. Each revision and rollout verifies frozen runtime,
starter and contract hashes. Preserve failed attempts; no automatic resume,
replacement layouts, additional rounds or reserved tests are included.

Seventeen ring tests passed before registration, including checks that R0 never
moves, unavailable observations remain explicit, unreachable previews do not
execute, and incremental targets retain commanded rather than measured height.
The registered R0 executions verify the actual isolated observation flow before
the first model request. Report all outcomes even if the new scaffold does not
improve learning; prior zero-success pilots remain separate negative evidence.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python \
  scripts/run_ring_peg_scaffold_training.py \
  --output outputs/ring_peg_scaffold_train_01 --phase prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -u \
  scripts/run_ring_peg_scaffold_training.py \
  --output outputs/ring_peg_scaffold_train_01 --phase train
```

## Information-enriched scaffold (prepared, not launched)

The API-scaffold pilot completed at 0/5, 0/5 and 3/5 for R1–R3, selecting R3.
The user subsequently requested slightly more information in the initial policy.
The separate guided variant supplies
[`RING_PEG_STARTER_GUIDANCE.md`](RING_PEG_STARTER_GUIDANCE.md) in the first actor
prompt and a shorter explanation in `ring_peg_guided_api_starter.py`.

The additions explain reference semantics, material versus empty-space points,
fixed/moving pad direction conventions, jaw-angle versus metric-gap interpretation,
height-aware finger clearance, measurement resolution, and useful public failure
diagnostics. They contain no grasp ranking, action sequence, trajectory, scene
coordinates or previously successful program. This is disclosed qualitative prior
knowledge. It is informed by development failures and is not a preregistered
confirmation of an information-only benefit.

The starter's executable syntax tree is identical to the original API scaffold
apart from its module docstring. Thus R0 still only observes; the change is in
information supplied to the actor, not inherited physical execution capability.
Both successful and failed geometric queries retain their original semantics.
For a future Base/Full comparison, match qualitative guidance and robot-interface
knowledge across arms while documenting Full-specific observation fields.

Use `run_ring_peg_guided_training.py` and the separate prepared output
`outputs/ring_peg_guided_train_01`. The guidance document, starter, runtime and
prompt are frozen together. The available training configuration retains five
training layouts, R0–R3 and three model calls, but this information update does
not dispatch another campaign. Nineteen ring tests passed, including exact
executable-starter equivalence and prompt/source coverage. The completed
`ring_peg_scaffold_train_01` source audit still passes.

The user subsequently authorized dispatch of `ring_peg_guided_train_01` under
its existing frozen configuration. The preparation-only status above describes
the earlier information update; the authorized training now uses the fixed
three-generation/twenty-rollout budget. See the campaign's `launch.json` and
`progress.json` for actual execution state. No reserved evaluation is included.
