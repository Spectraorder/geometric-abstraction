# Controlled online bridge-tower recovery

The ten-episode comparison is complete. See the [full results](ONLINE_CONTROLLED_RECOVERY_RESULTS.md).

## Question and scope

Does current geometric text help an agent identify and repair a changed support
before continuing a previously started bridge-tower task? Both arms share the
same trained bridge/stack adapters, local geometric controller, event scheduler,
retry gates and model routing. The treatment is the model-facing representation.

This is a separate study. It does not change or extend the completed natural-
failure pilot, and it cannot establish that text always makes model requests faster.

## Development evidence and scenario choice

Three intervention families were checked using prescribed recovery operations on
separate development seed 2028100101. A 25 mm foundation displacement was recovered
without resetting. Beam displacement and displacement of a loaded upper stack
were not reliably recovered, even with a prescribed plan. Failures included lost
visual geometry, collapse and support movement during grasping. All attempts are
retained under `outputs/online_recovery_dev/feasibility_01` and `feasibility_02`.
The foundation case also succeeded on development seed 2028100103.

The small reserved comparison therefore tests foundation repair, not arbitrary
obstruction clearing. This scope was selected using physical development checks,
not a difference between Base and Full on reserved outcomes. The harder families
need further control development before they can isolate reasoning performance.

## Episode and information boundary

Each process creates one environment and resets it exactly once. The common
sensor-controlled prefix initializes shape/workspace and places both foundations.
It executes with accelerated physics before the recovery clock starts. The
experiment harness then translates the right foundation by +25 mm along world Y,
preserves its orientation, zeroes its velocity and settles physics for one second.
This deterministic state edit represents an external intervention; it is not an
agent action or a simulated physical push. Its private before/after poses and
state hashes are archived, never included in model feedback.

The recovery clock starts once, after setup. Physics then advances at one-to-one
wall time during actions and model waits. There are no subsequent resets or state
edits. The actor must repair the scene and complete the bridge plus two-cube tower.
Success uses the unchanged private physical support graph and five-second stability
criterion. The 240-second recovery budget includes observation, model requests,
execution and waiting. Prefix cost is reported separately: this measures adaptation
from a matched interrupted construction state, not end-to-end construction latency.

Both arms receive the same public task, generic disturbance notification, executed
prefix, and historical side/wrist RGB images captured immediately before the event.
Base receives current RGB images and qualitative execution status. Full receives
current sensor-grounded geometry, residuals and relation evidence. Either can
request current RGB-D images. Neither receives which body was moved, the imposed
shift, a prescribed repair plan, or evaluator truth. Only scripted development
checks use the prescribed plan. Runtime guards and skill observations are shared.

## Reserved comparison

The registration contains five new seeds 2028102001–2028102005, one episode
per arm: ten total. These seeds do not overlap development or the completed pilot.
No reserved outcome may enter development, prompting or skill selection. A failed
common prefix remains a setup-failed slot; it is not replaced with a new layout.
Report all ten slots, and separately identify episodes that reached the event.

Freeze runtime sources, assets, generated geometry, package/client versions and
slots before dispatch. Run up to four processes, in balanced batches containing
both arms of each pair; use one BLAS/OpenMP thread per process. Preserve every
attempt and stop new batches on infrastructure failures. No automatic retries.
Parallel model service and CPU contention remain possible; record wall/simulation
lag per episode. A lag above two seconds invalidates a strict real-time latency
claim for that episode and must be reported rather than silently excluded.

Report paired success, capped recovery time, first model-request duration,
completed and interrupted model requests, input/cached/output tokens, attachments,
local observation cost, action duration and suppressed repeated operations. Cached
tokens are included in input tokens. Unknown usage on interrupted requests is not
zero. With five layouts, any representation advantage is preliminary and
scenario-specific; both positive and negative outcomes must be shown.

## Paired development check

The concurrent real-time development pair on seed 2028100101 reached success in
129.32 seconds for Base and 125.17 seconds for Full. Both selected the displaced
foundation and the same four-operation recovery/completion plan on their first
request. Post-event physics hashes matched exactly. Each used one model request:
9.40 seconds for Base and 5.60 seconds for Full, with 11,959 and 11,998 total tokens
respectively. Maximum simulation lag was below 0.14 seconds. This is development
evidence of feasibility; the latency difference from one request is not a result
on the reserved layouts.
