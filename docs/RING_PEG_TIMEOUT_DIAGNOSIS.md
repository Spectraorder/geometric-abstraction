# Ring comparison: repeated Base R2 timeout diagnosis

This read-only diagnosis covers the original and authorized retry of
`ring_peg_comparison_01/replica_03/basic/round_002`. No model request, rollout,
configuration change or additional recovery was dispatched for this diagnosis.

The immediate failure is an interrupted Codex Responses WebSocket stream. The
campaign's 900-second subprocess deadline subsequently terminates the request.
Available local evidence does not identify whether the silent stream originates
in the client, an intermediary network component, or the remote service. It does
not establish that the model is simply reasoning for too long.

## Observed sequence

Times below are seconds since the respective invocation timestamp. Persisted
response items establish activity, but do not record every low-level network
frame; the last persisted item is not necessarily the last received frame.

| Event | Original attempt | Authorized retry |
| --- | ---: | ---: |
| Last persisted response item | 175.42 | 78.61 |
| First WebSocket idle-timeout warning | 498.31 | 401.60 |
| Second WebSocket idle-timeout warning | 823.72 | 726.44 |
| Outer request termination | 900.01 | 900.01 |

Both requests initially received response items. Neither saved a final assistant
response, candidate, generation receipt, or completed new turn. Both CLI runs
were terminated before their advertised five stream retries were exhausted.
The roughly 323-second gaps between persisted activity and the first warning
are consistent with an idle wait, but do not independently reveal the exact
WebSocket timer configuration.

The outer deadline is implemented by `subprocess.run(timeout=...)` in
`scripts/geometry_actor_session.py:135`. The registered manifest sets it to 900
seconds. A failed model lease writes campaign-wide STOP through
`scripts/experiment_parallel.py:65`, so new work stops and in-flight work drains.
This explains why an individual request prevents the final training freeze and
reserved evaluation. It is intentional failure handling, not a rollout deadlock.

## Checks against alternative explanations

- The original and retry prompts, schemas, image lists/hashes, commands, and
  completed-session prefix hashes match. The R1 transcript remains the exact
  prefix of both attempted R2 transcripts. The recovery preserved session identity
  and rebuilt the live home from the completed checkpoint. No transcript
  compaction was recorded. A corrupt local restore is not supported by these checks;
  a session-specific remote failure is still possible.
- The new R2 prompt is 4,416 characters with two images. This is incremental input,
  not the complete persistent context. Completed requests used new prompts up to
  29,556 characters, and several completed session archives were larger. There is
  no logged context-length rejection. Payload size alone is not an established
  explanation; image/base64 and opaque response bytes are not token estimates.
- Five other generations completed during the retry's 900-second window. The
  evidence does not support a campaign-wide service outage or exhausted account
  quota. No rate-limit, authentication or context-limit error was found in the
  inspected stderr/event logs. A transient or request-specific service problem
  cannot be excluded.
- During resumed execution, 218 hardware samples had at least 25.77 GiB available
  RAM and 19,137 MiB free GPU memory; peak one-minute system load was 3.80. Resource
  exhaustion on this workstation is not supported.
- A different request, replica_02 Base R3, logged the same WebSocket idle-timeout
  warning but eventually completed in 842.8 seconds. The symptom is therefore not
  unique to the failed actor identity.
- The installed and archived CLI version is `0.155.0-alpha.16.3`. Several completed
  initial turns also emitted a terminal rollout-flush warning about a thread not
  found. That warning is not unique to the interrupted chain and does not establish
  the cause of its later stream timeout.

Completed Base requests took a median 523.8 seconds (13 requests, maximum 858.7),
versus 356.0 seconds for Full (15 requests, maximum 469.5). These summaries exclude
the two interrupted requests and are diagnostic, not a fair final efficiency
comparison. Some Base requests are already close to the outer deadline; a single
long idle wait can consume their remaining allowance. Increasing that deadline
would allow more recovery time but would not repair the underlying stream stall.

## Separate policy failures

The failed actor's preceding R1 had two layouts with unavailable fitted ring caps
and three with no wall passing the generated clearance checks. These are policy
perception/planning outcomes, not the cause of the R2 transport exception.

Replica_04 Base R3 generated 50,183 characters, exceeding the registered 50,000
limit by 183. Its five invalid-policy slots are a separate recorded failure.
They must not be silently reclassified or used to justify changing this campaign's
source limit. The timed-out R2 never reached source validation.

## Resources and next diagnostic step

Both interrupted attempts only rebroadcast the preceding completed turn's usage
counters. The new requests' final usage is unknown, not zero. Together they account
for approximately 1,800 seconds of observed request wall time and four image
attachments. Preserve both original and retry evidence.

The smallest informative next experiment is a separately recorded transport
comparison using the same model, reasoning setting, prompt, images and completed
R1 history. Test whether the installed client can use HTTP/SSE instead of the
failing WebSocket path before adopting a transport amendment. This is a proposed
live diagnostic, not a validated fix or authorization to launch another request.
Do not simultaneously change transport, context, reasoning effort and timeouts,
as that would obscure the cause and change the training condition.

Official OpenAI configuration documentation exposes provider WebSocket support,
stream retry count and stream idle timeout. It describes the latter two as SSE
settings (defaults: five retries and 300,000 ms); those descriptions alone do not
prove the exact WebSocket behavior of this installed alpha build:
[Configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

## Evidence locations

All paths below are relative to `outputs/ring_peg_comparison_01/`:

- `resume_01/interrupted_attempt/`: original invocation, prompt, events and stderr.
- `resume_01/interrupted_session.jsonl`: original interrupted transcript.
- `resume_01/original_codex_home/`: original actor home preserved before recovery.
- `replica_03/basic/round_002/`: retry invocation, events, stderr and failure receipt.
- `replica_03/basic/.actor_session/codex_home/sessions/`: retry transcript.
- `replica_03/basic/round_001/session.jsonl`: completed context checkpoint.
- `hardware.jsonl`: original and resumed resource samples.
- `replica_02/basic/round_003/model.stderr`: a completed request with the same warning.
- `STOP.json`, `lifecycle.json`, `progress.json`: current stopped campaign state.
