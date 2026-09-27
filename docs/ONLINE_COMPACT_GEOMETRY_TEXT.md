# Compact geometric text for online decisions

The opt-in `--compact-text` development path factors repeated reference metadata
without removing or rounding geometric values. An offline replay of the existing
paired development scene reduced Full's six newly supplied prompts by 21.1% in
UTF-8 bytes, including the initial explanation of the format. Base's seven prompts
were unchanged. A subsequent two-worker live development check at 200 seconds
succeeded in both conditions and recorded 6.7% fewer total tokens with compact
text, but slightly longer completion and model-request times. The offline byte
reduction and the live token result are separate measurements.

## Representation

The existing reference API and action schema remain unchanged. Only the geometric
part of the text sent to the actor is encoded. Metadata fields shared by all
eligible references are moved to `reference_defaults`. A reference explicitly
marked `$shared: true` inherits these fields; its local fields take precedence.
References without this marker inherit nothing. Each message is self-contained:
metadata from an earlier observation is never carried forward implicitly.

For example, this excerpt encodes one available center and one unavailable face:

```json
{
  "encoding": "shared_reference_metadata_v1",
  "reference_defaults": {
    "coordinate_frame": "world",
    "units": "meters",
    "observation_time": 1.0,
    "source": "mixed_rigid_rgbd_observation",
    "position_fully_observed": true
  },
  "geometry": {
    "time": 2.0,
    "objects": [{
      "object": "red",
      "references": {
        "center": {
          "$shared": true,
          "type": "point",
          "reference": "red.center",
          "position_m": [0.12, -0.27, 0.415]
        },
        "top_face": {"status": "unavailable", "reason": "Occluded"}
      }
    }]
  }
}
```

The center retains observation time 1.0 even though the scene record time is 2.0.
The face does not acquire a position, timestamp or observability claim from the
defaults. This is an illustrative excerpt; a record this small may remain in its
original form if the encoding envelope would increase its size.

Only fully observed references are eligible for sharing. Different timestamps,
frames, sources and inference flags remain explicit instead of being merged.
Unknown metadata is retained. Numeric values, including signed zero and the
original floating-point precision, are preserved. The encoder rejects nonfinite
JSON values and reserved-key collisions, and the actor checks exact canonical-JSON
round-trip equality before dispatch. The decoder and encoder never query sensors,
infer task targets or change physics.

## Development validation

Inputs are the original prompts and feedback from
`outputs/online_bound_dev/pairs_01`, seed 2028100101. No reserved layout was used
for this replay. The verification tool rejects non-development seeds and checks
that every original prompt matches its archived public feedback before encoding.
It then expands each compact record and checks all feedback fields, including
non-geometric fields. It retains input hashes and exact codec source snapshots.

| Measure | Base | Full |
| --- | ---: | ---: |
| Archived prompts | 7 | 6 |
| Encoded geometric prompts | 0 | 6 |
| Original prompt bytes | 14,634 | 40,327 |
| New prompt bytes, including format explanation | 14,634 | 31,836 |
| Reduction | 0% | 21.1% |
| Mean local transform time, offline microbenchmark | 0.008 ms | 0.398 ms |

The local timing repeats each transformation 100 times in the replay process;
it excludes model requests, sensor updates, images and actuator work. It is not
an online latency measurement. Byte counts cover newly supplied prompts, not the
persistent context retransmitted or cached by the model service. Tokenization and
model behavior must be measured separately. No token-saving percentage is claimed.

Seven focused tests cover exact reconstruction and immutability, unavailable and
partial references, heterogeneous metadata, empty/small records, malformed input,
actor transport preservation and source provenance in the opt-in runner. Sixteen
existing explicit-target tests also pass. The replay made zero model requests
and ran zero physical episodes. Its report and transformed prompts are in
`outputs/online_compact_text_dev/replay_01`.

## Opt-in execution and provenance

`CompactBoundActor` wraps the existing actor transport, preserving image arguments,
deadline, model routing, returned decisions and usage accounting. Its explanation
is supplied on the first encoded turn only. Base text remains byte-identical;
using the same opt-in flag in both arms still uses the same actor wrapper. The
transport archives the actual transformed prompt, while `feedback.jsonl` retains
the original public feedback for reconstruction and auditing.

The development runner accepts `--compact-text` and records
`geometry_serialization: shared_reference_metadata_v1` in settings/results. It
archives both new implementation files with the usual episode sources. The
flag defaults off, and the existing reserved runner does not enable it. The
current development budget is 200 seconds, with one initial reset and no reset
after failure. Choosing this flag does not alter the shared skill interface,
controller, camera acquisition or evaluation rule.

The first live development check is complete below. A shorter prompt alone is
not sufficient evidence that the model interprets inheritance correctly during
manipulation. Any reserved comparison requires its own frozen registration; the
completed 180-second comparison remains unchanged.

Implementation: [codec](../scripts/online_compact_geometry.py),
[actor adapter](../scripts/online_compact_actor.py),
[development entry point](../scripts/run_online_bound_development.py),
[offline replay](../scripts/verify_online_compact_text.py), and
[focused tests](../tests/test_online_compact_geometry.py).

## Live development check at 200 seconds

Two Full episodes used the same existing development seed, 2028100101, with one
raw-reference actor and one compact-reference actor. They ran concurrently with
identical initial physical states, Astra medium, explicit-target skills, sensors,
evaluator and 200-second budgets. The difference was reference serialization and
its first-turn explanation. No Base condition or reserved layout was used.
The frozen record is `outputs/online_compact_text_dev/live_01`; the runner is
[run_online_compact_check.py](../scripts/run_online_compact_check.py).

| Measure | Raw Full | Compact Full |
| --- | ---: | ---: |
| Task result | Success | Success |
| Time to success | 157.12 s | 158.52 s |
| Completed model requests | 6 | 6 |
| Incomplete requests | 0 | 0 |
| Input tokens, including cached input | 106,296 | 99,120 |
| Cached input tokens, included above | 83,328 | 78,208 |
| Uncached input tokens | 22,968 | 20,912 |
| Output tokens | 488 | 500 |
| Total tokens | 106,784 | 99,620 |
| Total completed model-request time | 36.27 s | 37.49 s |
| First request duration | 7.12 s | 9.58 s |
| Image attachments | 4 | 4 |
| Camera acquisitions | 127 | 126 |
| Newly supplied prompt bytes | 40,565 | 32,198 |

Compact text reduced recorded total tokens by 7,164 (6.7%) in this pair. Both
conditions followed the same high-level six-decision sequence: position a
foundation, place the beam, observe, query depth, and place two upper cubes.
Both retained an existing foundation. Numeric targets and observation times were
chosen in separate live episodes and need not be identical; this is not a replay
of identical prompts. The byte totals include the compact format explanation.

Compact completion was 1.40 seconds slower, and completed model-request time was
1.22 seconds longer. The result supports this adapter's live compatibility and a
token reduction on one reused development scene, not an online latency improvement
or a general success-rate claim. Service variability and differing observations
limit causal interpretation of the small time difference. Do not infer Base/Full
200-second success rates or combine this pair with reserved 180-second outcomes.

The two episodes completed in 161.20 seconds of campaign wall time. Independent
[verification](../scripts/audit_online_compact_check.py) passed for all twelve
actual prompts: it reconstructed their geometry without importing the production
codec and matched the original public feedback, then confirmed each prompt in
its actual model transcript. Existing audits also verified token counter deltas,
source hashes, submitted targets, pixel-depth queries, reset counts and terminal
robot contacts. Both episodes had one initial reset, and both successful terminal
states had no robot contact. No incomplete usage or infrastructure failures were
reported. The final auditor source is archived alongside `compact_audit.json`.

This development check supports retaining compact text as an available variant.
It does not change the default raw-reference behavior or automatically launch a
new reserved campaign. The next representation comparison should freeze its
chosen text format and the common 200-second budget before dispatch.
