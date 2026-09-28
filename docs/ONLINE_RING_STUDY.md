# Continuous online ring-over-peg reasoning

This is the second online task, complementing bridge-tower construction with
geometry-dependent alignment and release. The agent operates in one scene until
success or a 200-second timeout. Natural failures remain in the scene; no reset,
object teleport, injected disturbance, or replacement layout is permitted.

## Shared skills and agent responsibility

Both arms receive the same learned grasp-and-lift skill. Its source is selected
only from the completed paired Ring study's training summaries: highest retained
training success, with replica and revision order breaking ties. This chooses
Full replica 5 R3 (4/5 training layouts), source SHA-256
`5575db08bd2c9c1764038e78d8c389f29d3c75455140e4535f2d329fafdc7864`.
The selector never reads reserved outcomes. Selection occurs after the prior
study, so this is reuse of trained task experience, not learning a novel task
from scratch.

The trusted local skill uses sensor geometry for wall selection, finger-clearance
checks and lift verification in both arms. A mandatory boundary returns control
to the agent before the learned placement procedure executes. Only a qualitative
outcome and the previous jaw-opening command enter shared feedback. This is
matched visual grasp assistance, not a comparison of entire perception/control
stacks. The original trained policy is not copied into either model prompt.

After grasping, the agent chooses explicit world coordinates for the fixed
finger pad, wrist roll, motion duration, whether to request another observation,
and when/how far to open the jaw. The common actuator performs robot-only IK
and executes the submitted target without replacing it with a hole-alignment
coordinate. Release opens the jaw, retreats vertically, and allows settling;
it does not align the ring first. The agent can request another grasp or issue
additional moves after an unsuccessful action, in the same episode.

Shared qualitative experience explains wall pinching, possible slip/tilt,
transport height, current evidence, pad-versus-hole coordinates and release.
There is one persistent Astra medium context per episode. All requests use the
same HTTP-only transport; no additional language-model calls occur inside a skill.

## Information boundary

Both arms have initial side/wrist RGB, calibrated metric pixel queries (up to
16 pixels on the exact displayed frame), images on request, robot encoder and
public jaw-geometry information, and qualitative action feedback. A pixel in the
opening may show the background; it is not an observed hole-center surface.
Queries retain image timestamps and explicitly identify historical samples.

Base receives side RGB after physical actions. Full instead receives compact
current object records: hole center, axis, inscribed radius, ring height and
minimum height, peg top/axis/radius, and evidence/time fields. Dense boundary
vertices are omitted. Initial and requested images remain available to Full.
An unavailable observation supplies no replacement pose. The peg is declared
fixed, so an agent may reason using an earlier measured peg position while
preserving its original timestamp. No simulator object truth, contacts or private
success diagnostics enter prompts.

The contrast includes supplied geometric perception and feedback as a package.
The shared grasp skill internally uses geometry in both arms. Runtime sensing,
images attached to a request, and input tokens are separate cost measures.

## Development and fixed reserved design

Development is restricted to seeds 1 and 2. Sensor-driven physical checks precede
four live episodes, one per arm and seed. Early physical-driver failures remain
under `outputs/online_ring_dev/physical_01` through `physical_04`; corrections
addressed an occluded fixed peg, a release condition in the scripted driver, and
release opening. These are development checks, not model performance estimates.
The corrected seed-1 physical driver succeeded in `physical_05`. Final validation
is recorded in `physical_final`; live checks are recorded separately in `live_01`.

Reserved registration requires successful interface/resource audits of both live
pairs and at least one successful physical check, with all development failures
preserved. A successful Base development episode is not required. No change is
made in response to reserved outcomes.

- Study: `outputs/online_ring_200_01`.
- Five fresh paired seeds: 2033101001–2033101005.
- One episode per arm/layout, ten episodes total, no automatic retries.
- 200 seconds from immediately after the sole initial reset, including skill and
  sensor initialization, model requests, local computation, actions and settling.
- Physics advances during model waits and actions are paced to simulation time.
- Up to four processes, balanced batches of two pairs, two pairs, and one pair.
- Sources, shared skill, client/package versions, settings and development
  evidence are frozen before dispatch.
- Infrastructure errors stop later batches; completed and incomplete attempts
  are retained. No extra rounds, replacement seeds or automatic budget extension.

The private evaluator checks that the peg axis passes through the ring opening,
the ring center lies below the peg top, the ring does not touch the robot, and
motion stays low for two seconds. Drift must remain within 3 mm during that hold.
This continuous drift check is stricter than the single-policy evaluator's
endpoint-drift check. Robot contact grouping includes unnamed meshes through body
ancestry. Terminal physics is archived for independent reconstruction; private
monitor samples record the complete stability interval.

## Outcomes and interpretation

Report successes out of five, every paired outcome, and mean capped completion
time (timeouts count as 200 seconds). Also report completed and incomplete model
request time, per-request latency, input/output/cached tokens, unknown interrupted
usage, image attachments, camera acquisitions, depth queries, actions and local
feedback time. Do not add cached input to total input a second time, and do not
sum timers with overlapping scopes as if they were disjoint wall-clock costs.

Simulation lag above two seconds is flagged in the summary and limits claims of
real-time behavior. Record concurrent machine load; parallel campaign time is
not episode latency. Five pairs provide exploratory evidence. Faster completion
would not by itself prove faster model inference, fewer tokens, or an isolated
serializer advantage. Include unfavorable outcomes.

## Entry points

`run_online_ring_episode.py` runs development seeds only. `run_online_ring_checked_study.py`
is the coordinator entry point for registration, source checks, dispatch and
audited summaries. It adapts the legacy resource auditor to repository-relative
source paths; episode code and recorded development results are unchanged.
`run_online_ring_study.py` supplies the shared scheduler and episode worker. The schema, loop, skill wrapper and
continuous scene are in `online_ring_actor.py`, `online_ring_loop.py`,
`online_ring_skills.py`, and `online_ring_scene.py`. Earlier tower, single-policy
Ring and currently running bridge sources remain unchanged.

## Completed development and reserved evaluation

Eight focused tests passed. Final physical checks completed without infrastructure
errors: seed 1 succeeded, while seed 2 did not complete. Both outcomes and every
earlier driver attempt remain archived. The four subsequent live development
episodes all succeeded and passed source/resource, actual prompt/image delivery,
Base exclusion, explicit-target, metric-depth and terminal-physics checks.

| Development seed | Base time | Full time | Base / Full completed requests | Base / Full images |
| --- | ---: | ---: | ---: | ---: |
| 1 | 128.89 s | 73.23 s | 14 / 6 | 14 / 2 |
| 2 | 129.79 s | 67.31 s | 14 / 6 | 13 / 2 |

These two reused development layouts are not reserved evaluation evidence.
The fresh five-pair study completed all ten episodes under `online_ring_200_01`;
source/resource and interface audits passed. Both arms succeeded on 5/5 layouts.
Mean completion time was 119.79 seconds for Base and 74.25 seconds for Full,
with Full faster on every paired layout. The [online report](ONLINE_EXPERIMENTS.md)
gives all outcomes, resource measures, interrupted-usage accounting, and limits.
Read the frozen `protocol.json`, final `summary.json`, and audit receipts for
source evidence. Development outcomes are not included in the reserved success rate.
