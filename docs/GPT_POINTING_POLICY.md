# Live pointing inside the bridge assembly policy

Live RGB-only pointing recovered **one of ten previously failing bridge trials**
under the existing short-horizon success criterion. The paired frozen baseline
recovered none. Both previously successful controls still passed without making
any pointing requests. However, the recovered tower **failed an additional
60-second offline stability check**. This is a useful track-recovery result, not
an established long-term stable assembly success.

Twenty-four new simulator rollouts were executed: baseline and pointing variant
for each of ten archived failures and two success controls. These are selected
development cases, not a new population success-rate estimate.

| Group | Frozen baseline | Pointing variant |
| --- | ---: | ---: |
| Ten previous failures, original final check | 0/10 | 1/10 |
| Two success controls, original final check | 2/2 | 2/2 |
| Additional 60-second check of the one recovered tower | Not applicable | Failed |

[Paired results and live request evidence](evaluations/gpt_pointing_policy_failures.json)

## Actual execution evidence

- [Recovered case, original short-horizon check](media/gpt_pointing_policy_recovered_1608_d2.mp4)
- [Cylinder located, but assembly still fails](media/gpt_pointing_policy_cylinder_failure_1607_d0.mp4)
- [All live RGB requests and replies](media/gpt_pointing_policy_requests.jpg)
- [The recovered blue-beam observation](media/gpt_pointing_policy_recovery_frame.png)

The videos render saved integration states from these new policy executions.
The recovered video ends at the original successful result; it does not depict
the separate extended stability experiment. Both MP4s were probed and completely
decoded successfully.

## Intervention

`policies/assemble_structure_pointing.py` is an opt-in variant. It invokes a
hosted RGB-only pointer when initial task-instance selection or subsequent
object localization/geometry raises an error. The response proposes a pixel;
local color, valid-depth, tabletop, and connected-component gates must accept
that pixel. The candidate track is then reassociated and the existing geometry
code is retried. No dimensions, pose, support margin, grasp, or placement
threshold was relaxed.

An override is valid only for the same timestamp and identical RGB image. A
point cannot silently take over another protected task instance. Abstentions,
invalid pixels, stale responses, and timeouts never produce synthetic positions.
The existing three-attempt perception loop can request points on successive
observations; the host limits each trial to four requests.

The worker retains its network-disabled filesystem sandbox. Its optional
`point_target` RPC sends only a public goal identity and observation timestamp
to the host. The host exposes the current camera RGB and target text through a
mailbox, pauses simulation while awaiting a reply, and accepts only a bounded
pixel or abstention bound to that request ID and image hash. No simulator object
state, depth, calibration, failure explanation, or evaluator result is provided
to the subagent. CPU geometry remains in the worker; hosted inference means this
variant is not CPU-only end-to-end. The default frozen CPU candidate is unchanged.

Six fresh-context subagents serviced twelve live requests. Their shared-workspace
input restrictions were instructions, not an enforced image-only filesystem
allowlist. This is a session-operated experimental mailbox, not an unattended
production inference service. No OpenRouter credential or API was used.

## Per-case results

All twelve fresh baselines reproduced their archived success flags and terminal
notes. For every paired trial, physical action commands before the first point
request matched the baseline prefix. Eleven complete action sequences remained
identical; the recovered case is the sole changed sequence.

| Seed | Distractors | Original issue | Point requests | Outcome |
| --- | ---: | --- | ---: | --- |
| 1601 | 0 | Successful control | 0 | Still successful |
| 1605 | 0 | Held beam contradicts learned dimensions | 1 | Same geometry rejection |
| 1607 | 0 | Cylinder track missing | 3 | Pixel recovered; curved surface cannot constrain center |
| 1609 | 0 | Initial red beam identification | 1 | Still lacks acceptable beam geometry |
| 1610 | 0 | Blue cylinder unavailable | 3 | Three abstentions; cylinder has no visible mask pixels |
| 1601 | 2 | Successful control | 0 | Still successful |
| 1602 | 2 | Beam staging/placement rejected | 0 | Same failure; no localization exception |
| 1603 | 2 | Initial yellow beam identification | 1 | Still lacks acceptable beam geometry |
| 1605 | 2 | Foundation workspace obstructed | 0 | Same failure; no localization exception |
| 1607 | 2 | Held beam contradicts learned dimensions | 1 | Same geometry rejection |
| 1608 | 2 | Held blue beam track missing | 1 | Completes original check; extended stability fails |
| 1610 | 2 | Held beam contradicts learned dimensions | 1 | Same geometry rejection |

In seed 1608 with two distractors, at simulation time 73.4 s the baseline loses
`blue_4`. The subagent points to `(199,124)` in the current 320 x 240 RGB frame.
Local validation selects 75 pixels and reassociates the track. Unmodified
geometry, finger-clearance and wrist-grasp checks then pass. The policy places
the beam, adds the cylinder, confirms all four nodes, and finishes at about
135.1 s of simulation time. It passes the original final physical check.

Restoring that final integration state and controls, then applying the existing
60-second monitor after 0.8 s settling, produces drift greater than 3 mm for the
beam and cylinder, relative-XY failures, and a support-equilibrium failure.
Therefore the one recovered execution should not be reported as a durable
success. This extra test covers only that selected tower, not a durability rate
for the full matrix.

## Point quality versus policy success

All twelve replies were also checked **after the rollouts** by a separate
privileged replay-segmentation evaluator: nine proposed pixels hit the requested
object and three abstentions corresponded to zero visible target pixels. The
abstained blue cylinder existed in the scene but was not visible in those camera
frames. Replay timestamps, camera calibration, valid-depth masks and depth align
with the archived observations. These private labels are offline scoring only;
they never enter the pointer or runtime locator.

Perfect point decisions on this small set did not remove geometry, grasp,
workspace, or stability failures. The remaining work is reliable geometry from
partial views, cylinder pose recovery, grasp verification, and stable placement.
For the recovered track, the data now provide a concrete case for improving the
CPU tracker rather than requiring a hosted pointer indefinitely.

The mailbox records wall wait separately from policy compute and simulation.
Those waits include session coordination, image inspection and agent tool use;
they are not standalone model API latency measurements. Simulator time is paused
while waiting, so these experiments do not establish real-robot real-time safety
or throughput.

## Reproduction and verification

Build the optional candidate without overwriting the frozen baseline:

```bash
.venv/bin/python scripts/build_inventory_policy.py --controller structure_policy \
  --pointing --output policies/assemble_structure_pointing.py
```

Run `scripts/run_vision_trial.py` with that candidate and a new
`--pointing-mailbox` directory. An external session must service each
`request.json` by inspecting only its referenced PNG and atomically writing the
corresponding `response.json`. The response must preserve `task_id` and
`image_sha256`, with `status`, `point_px`, and an optional visual `reason`.
Do not rerun the suite unattended without a pointer service: it will time out.

The exact paired commands, manifest and source observations are stored locally
under `outputs/pointing_policy_01/`. The maintained summary tool is
`scripts/summarize_pointing_policy_suite.py`; offline identity scoring uses
`scripts/score_pointing_replays.py` only after predictions are frozen.

The full repository suite ran **158 tests: 157 passed, one platform-specific
skip**. It includes numerical-only candidate validation, worker isolation,
blackout rejection, request/image binding, coordinate validation, timeout
handling, and expiration of pixel overrides on a changed observation. A fresh
build exactly reproduces the tested candidate.

- Frozen baseline SHA-256: `80bfb487e1b35ec08ceb83e06c298dc40354eca3df17fbf1ce2ff1aad50e9504`
- Pointing candidate SHA-256: `b5ec4b8e742f7b4938564b7be8a5461f2cb0d83c42aefc248add0d85aec8a68e`
