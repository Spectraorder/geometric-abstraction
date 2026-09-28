# Continuous online tower relocation

The user authorized one additional online task after the completed Tower and Ring
comparisons. This separate study tests reuse of shared manipulation skills to
rearrange an existing structure. It does not rerun or modify earlier studies.

## Task and public interface

Four equal rigid cubes begin as a source-middle-upper column plus a separate
destination cube. The goal is a destination-middle-upper column, with source and
destination at their original positions. Colors and initial positions vary by
seed. The public goal describes final supports, roles, and equal size; it gives no
numeric object dimensions, initial coordinates, temporary target, or action order.
The agent can choose temporary placements and repair errors in the same scene.

Both arms receive the same adapted bridge/stack pick-and-place skills and the
same explicit world surface-point schema. A skill performs one placement; it does
not supply a relocation plan or silently replace a submitted destination. The
agent may also relocate a foundation back to its original position after an
accidental displacement. Shared grasp fraction is 0.70, selected in development
to reduce jaw contact with the cube below. The prior source hashes are recorded
in each episode's `skill_provenance.json`. No model calls occur inside skills.

The preassembled column cannot use the old assumption that each object's initial
height above the table is its size. An isolated geometry builder instead measures
the table-supported destination cube from RGB-D, transfers only its dimensions
under the public equal-size assumption, and observes each other cube's current
pose independently. A missing pose stays unavailable. No reference pose is copied,
and no simulator size or pose enters this initialization. Shared local skills use
this observer in both arms; Base is not an end-to-end geometry-free robot.

Base receives initial side/wrist images and side images after physical actions,
with qualitative outcomes. Full receives current center/top/bottom references
encoded using the existing compact metadata format, plus initial/requested images.
Both can request fresh images and up to 16 metric depth samples from the exact
displayed RGB-D frame. The comparison evaluates supplied perception, references,
and feedback together, not text serialization in isolation.

## Fixed evaluation design

- Five new paired seeds: 2034101001–2034101005; overlap checked before registration.
- One episode per arm and seed: ten episodes, with every attempt retained.
- 200 seconds immediately after the sole initial reset, including sensor/skill
  initialization, model waits, local perception, execution, and settling.
- One persistent `gpt-6-astra` medium context per episode, HTTP-only transport.
- Physics advances during model waits; actions are paced to simulation time.
- At most four episode workers, in balanced 4/4/2 batches.
- No injected disturbance, reset after failure, replacement layout, automatic
  retry, or outcome-dependent budget extension.
- Infrastructure errors stop subsequent batches after active workers finish.

Scene setup creates the initial column in the simulator keyframe before the
single reset. This is evaluator-owned initialization, not an agent action or a
truth-driven controller. Source and destination locations are sampled in a
bounded reachable region away from the home gripper sweep. Policies see only
public sensor data and robot information.

The private criterion requires correct support contacts, upright cubes, low
velocities, adequate support margins, no unintended object contacts, and no robot
contact (including unnamed meshes). During the five-second success hold, both
foundations must remain within 5 mm of their initial positions, and all objects
must drift no more than 3 mm from the hold's start. Displacement before the hold
can be repaired; it is not an irreversible episode failure. The monitor and
terminal physics are archived separately from model feedback.

Primary metrics are successes out of five and mean capped completion time, with
failures counted as 200 seconds. Report every pair, complete/incomplete request
time, tokens with unknown interrupted usage separate, image attachments, camera
acquisitions, depth queries, actions, and simulation lag. Cached input is already
included in total input. Do not add overlapping timers or equate fewer images
with fewer local sensor acquisitions. Flag lag above two seconds. Five pairs are
exploratory and cannot establish broad recovery or generalization reliability.

## Development record

Only seeds 1 and 2 are used for development. A fixed sensor-driven script removes
the upper cube to a temporary table location, moves the middle cube to destination,
and replaces the upper cube. This sequence is a physical check and is not included
in actor prompts or skills.

All earlier attempts remain in `outputs/online_relocation_dev`:

- `physical_01` exposed a legacy skill-order dependency on public grid fields;
  the adapter now initializes an observation list without supplying grid goals.
- `physical_02` and `physical_03` collided with the initial gripper neighborhood.
  Increasing grasp height alone did not fix the source-layout collision.
- `physical_04` through `physical_06` completed the target structure but failed
  the development evaluator's original irreversible foundation-displacement flag.
  The final task instead checks initial-position restoration during the success
  hold and permits recovery. These historical failures are not relabeled.
- `physical_07` succeeded on both seeds with the final goal semantics and shared
  grasp fraction 0.70. These are accelerated sensor-driven physics checks, not
  real-time LLM results or a reserved success-rate estimate.
- `physical_final` repeats both checks with the final runtime snapshot; live
  validation uses both arms on both development seeds under `live_01`.

Registration requires both physical checks to succeed, both live interfaces to
exercise at least two model decisions, Base to exercise calibrated depth access,
and independent audits to pass. A favorable Full-vs-Base outcome is not a gate.
No reserved episode is dispatched before registration and source freeze.

## Implementation and evidence

- Scene and private monitor: `scripts/online_relocation_scene.py`.
- Equal-size initializer, shared skills, and actor: `scripts/online_relocation_runtime.py`.
- Development/episode runner: `scripts/run_online_relocation_episode.py`.
- Registration and bounded scheduler: `scripts/run_online_relocation_study.py`.
- Independent task/interface/resource audit: `scripts/audit_online_relocation.py`.
- Focused checks: `tests/test_online_relocation.py`.

The adapter reuses frozen shared implementations through scoped runtime bindings;
existing source files and completed experiment artifacts remain unchanged. Source
snapshots include the shared code, generated geometry, and prior skill sources.
Audits verify submitted targets, Base geometry exclusion, actual compact prompts,
calibrated depth, token-counter deltas, initial-state pairing, terminal support
geometry, and the private continuous hold. Incomplete transcript delivery and
unknown tokens remain explicitly reported.

The registered study is `outputs/online_relocation_200_01`, stored on the SSD.
Registration checked 262 prior manifests/protocols for seed overlap and froze
the validated runtime. The ten-slot campaign completed and all source/resource,
interface, compact-prompt, and relocation-specific audits passed. Read `protocol.json`, `progress.json`, and `lifecycle.json` for its current
status. A launcher-only virtualenv-path correction occurred before any episode or
model request; the failed startup and repair receipts are retained. Frozen runtime
sources and the registered budget were unchanged.

## Completed pre-registration validation

Five focused tests passed, including sensor-only size transfer, unavailable-pose
handling, actor isolation, and restoration of a displaced foundation. Both final
accelerated physical checks succeeded. All four live development episodes passed
resource, source, actual-prompt, metric-depth, target-binding, and terminal-geometry
audits. Results were mixed and are preserved without further outcome-driven tuning:

| Development seed | Base | Full | Completed requests, Base / Full |
| --- | --- | --- | ---: |
| 1 | Success, 126.22 s | Success, 136.22 s | 5 / 4 |
| 2 | Success, 178.32 s | Timeout, 200.00 s | 7 / 7 |

The Full timeout includes one incomplete request with unknown token usage. These
two development layouts are not part of the five reserved pairs. No favorable
Full outcome was required by the registration gate.

## Reserved outcome

All ten episodes completed. Base succeeded on 2/5 layouts and Full on 3/5.
Mean capped completion time was 190.61/164.32 seconds (Base/Full); recorded tokens
were 906,885/755,815 across five episodes per arm. Both arms have two interrupted
requests with unknown token usage. All results, including Full's failure on a
Base-successful layout, remain in the denominator. See the
[three-task online report](ONLINE_EXPERIMENTS.md) and
[unified data bundle](evaluations/online/README.md) for paired outcomes, request
accounting, figures, and limitations. No additional runs were performed.
