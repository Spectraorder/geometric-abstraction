# Event-driven online development

This is a separate development variant. The completed
[20-episode pilot](ONLINE_BRIDGE_TOWER_RESULTS.md) and its registered sources remain
unchanged. Development uses only seeds 2028100101–2028100103. No new reserved
comparison is implied by these checks.

## Why a placement took approximately 30 seconds

A skill call previously enclosed a complete pick, lift, transfer, visual alignment,
release, settling and retreat. In one recorded development placement, the logged
commands consumed 29.30 seconds: 20.75 seconds of arm motion, 4.03 seconds of jaw
motion and 4.52 seconds of waits. These are controller-selected durations under
one-to-one simulation pacing, not a hardware lower bound. Most individual moves
last 1.1–2.0 seconds. Visual alignment may repeat a correction, and release plus
retreat has several conservative waits.

Segmenting execution makes it interruptible; it does not by itself shorten the
physical trajectory. Shorter motion or settling durations need separate physical
validation. Accelerated simulation smoke checks cannot establish real-time speed.

## Implemented changes

- The shared RGB-D controller's pick and place methods are compiled into local
  generators with yield points after motion and wait commands. The original
  numerical targets, durations, branch logic and visual checks are preserved.
  This also applies to the clearing primitive. All simulation access remains on
  one thread; a yield does not reset the scene or issue a model request.
- A placement is a resumable sequence of bounded commands. The scheduler observes
  current supports between commands while an object is held. A positively
  observed support-center displacement above 1 cm interrupts the sequence and
  preserves holding state. Missing geometry alone is not evidence of movement.
  Interrupted generators are closed: a subsequent action computes a fresh plan
  rather than continuing with saved local targets after model latency.
- Astra supplies an initial short plan. JSON `plan` entries name remaining
  operations as `bridge:red` or `stack:blue`, using the actual public color keys.
  Local execution continues on `visually_consistent` outcomes. Failures discard
  the queued tail and return control to the model. No private success predicate
  is sent to the model.
- Failed operations are blocked while current visual evidence is materially
  unchanged. A new available pose or a center displacement above 1 cm unlocks a
  retry; a changed holding state also unlocks it. A repeated blocked decision leads to local observation polling until
  evidence changes or the wall deadline; it does not repeatedly call a model.
  This conservative gate may wait out an unrecoverable episode and is not a
  guarantee of recovery.
- Astra escalation tracks distinct failed operations. Explicit replanning for
  the same failure set is deduplicated. Luna handles other exception decisions.
- A conversation window contains at most three completed/requested turns before
  a fresh model context receives bounded public memory: four recent events and
  the last attempted operation for up to five objects. The environment, shape
  initialization, gripper state and controller session persist. Old coordinates
  are not promoted into current observations. Detailed raw logs remain on disk.
- After the plan ends, the scene settles locally for 5.5 seconds while physics and
  the private success monitor continue. This avoids asking the model to poll for
  success after every one-second wait.

Both arms use the same scheduler, local guards, skills and memory rules. The
execution backend may use geometry in both arms, while Base model feedback remains
images plus qualitative status. Full receives current geometric text. This
variant also bounds each adapter invocation to one pick/place attempt, returning
an unsuccessful attempt to the scheduler instead of hiding repeated full retries
inside a skill. It is a new adapter, not a claim of whole-skill behavioral parity
with the old pilot. Motion-generator parity is tested at the low-level controller.

## Checks and artifacts

The targeted tests cover command/result parity for pick and place, cancellation,
retry gating under unchanged/noisy/new evidence, bounded memory, plan validation,
automatic plan continuation, and queue invalidation after failure. The accelerated
five-object preflight succeeded with one reset and sustained success evidence;
its wall time is not a real-time result.

Entry points:

- [Cooperative skills](../scripts/online_event_skills.py)
- [Memory and retry gates](../scripts/online_event_memory.py)
- [Development runner](../scripts/run_online_event_development.py)
- [Targeted tests](../tests/test_online_event.py)

Artifacts live under `outputs/online_event_dev/`. `preflight_01` is an accelerated
scripted physical check. `pair_01` uses actual Astra/Luna requests with real-time
physics on development layout 2028100101. Each episode records source hashes,
model contexts/receipts, public feedback, decisions, scheduler events and timed
motion boundaries. Source hashes identify the development version; this is not a
new frozen reserved protocol.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python -m unittest tests.test_online_event -q

# Use a fresh output directory for each attempt.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/run_online_event_development.py \
  --output outputs/online_event_dev/new_full_episode \
  --seed 2028100101 --arm symbolic --budget 360
```

## Initial measured development pair

On layout 2028100101, both arms reached sustained success in one continuing
real-time episode, with exactly one completed Astra request and no interrupted
request. Routine progress therefore required no Luna calls in this pair.

| Metric | Previous Base | Event Base | Previous Full | Event Full |
| --- | ---: | ---: | ---: | ---: |
| Time to success | 139.97 s | 127.47 s | 138.17 s | 128.92 s |
| Completed model requests | 5 | 1 | 4 | 1 |
| Total tokens | 90,554 | 11,472 | 78,208 | 11,902 |
| Input tokens including cache | 90,275 | 11,408 | 77,986 | 11,840 |
| Cached input tokens | 53,888 | 0 | 39,680 | 0 |
| Output tokens | 279 | 64 | 222 | 62 |

Previous measurements are the earlier `online_bridge_tower_dev/pair_01`
development runs, not the reserved pilot. Their usage receipts are complete.
Recorded total token reductions are 87.3% for Base and 84.8% for Full; completion
time reductions are 8.9% and 6.7%. These are descriptive comparisons on one reused
development layout, at different request times and with a changed skill adapter.
They do not establish a general latency effect or an advantage for Full over Base.
Full was slightly slower and used slightly more tokens than Base in the new pair.

Base recorded 96 motion boundaries and Full 98; their largest completed segments
were 2.37 and 2.44 seconds. Both retained the original physical command durations.
The result receipts, token sums, single-reset timing, success intervals and
credential cleanup were checked. This pair exercised automatic plan execution;
it did not exercise exception-time model routing or context rotation. Those paths
have targeted tests, and a separate short recovery diagnostic is recorded below.

The pair predates two follow-up checks in the development runner: validation of
the first requested operation (as well as the plan tail), and explicit started
versus completed operation counters. The old `executed_operations` field counts
returned operations; a final operation can be interrupted by verified success
before its return. Source hashes distinguish these development versions.


## Short exception-path diagnostic

`recovery_01` ran Full on development layout 2028100102 for 180 seconds. It timed
out without a reset. The observed support movement interrupted the queued top
placement at a command boundary. The next identical stack request was suppressed;
the model requested observations, released uncertain holding, and issued another
placement attempt. This exercised exception-time Luna routing, retry suppression,
continued physical execution and two bounded conversation windows.

There were six completed requests (one Astra, five Luna), totaling 97,153 tokens:
96,770 input, including 63,744 cached input, and 383 output. No request usage was
missing. Model-request time was 30.04 seconds, and maximum simulation lag was
0.614 seconds. This short single-arm diagnostic is not a recovery success result
or a matched efficiency comparison with the old 360-second pilot.

A final local refinement includes holding state in the retry gate, so an explicit
release can unlock a retry even when an object's measured center stays nearly
unchanged. A targeted regression test covers this condition; the recorded model
runs predate this gate refinement. The retained live traces already show successful
release and subsequent redispatch through changed visual evidence. Eleven targeted
tests now pass. The original frozen pilot still passes its source check.
