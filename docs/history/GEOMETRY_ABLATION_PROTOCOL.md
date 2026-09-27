> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Controlled three-cube geometry experiment

The current protocol is version 3, written after the unsuccessful first pilot.
Its purpose is to test whether supplied visual geometric observations reduce the
iterations needed to build a CPU stacking controller. It remains a diagnostic pilot,
with one independent campaign per arm. It cannot establish statistical superiority.

The [historical pilot 03 report](GEOMETRY_ABLATION_PILOT_03.md) and original prompts,
responses, results and replays are retained. Never merge its results with the new run.

## Capability boundary

Both arms start with an empty `run(api)`, the same robot SDK description, numerical
libraries, side RGB-D, wrist RGB, camera calibration and robot proprioception. They
receive their own campaign feedback only. Existing controllers, historical policies,
old failure reports, pointing and simulator object truth are excluded from the actor.
Both arms may write their own perception, geometry and control functions.

The geometry arm additionally gets the frozen, action-free `geometry` object:

| Function | Output |
| --- | --- |
| `table(frame)` | Visually fitted tabletop plane |
| `cube(frame, color, model=None, table_plane=None, prior=None)` | Learned size, fitted center, inferred vertices/support face, observability flags |
| `gripper(frame, tool_position, support_z=None)` | Visible finger surfaces and partial clearance/opening estimates |
| `margin(point, top_vertices, normal)` | Signed point-to-edge support margin |

The cube-only adapter excludes general column controllers and trackers. A dimension
model is learned from an initially upright tabletop observation; it must be retained
when lifting or stacking. Colored pixels below the table do not count as object size
evidence. `model_valid` describes stored dimension availability; `geometry_reliable`
and `position_fully_observed` describe the current pose. Partial poses must not be
mistaken for fully observed positions. With a supplied center prior, only observed
translation axes update; the prior does not make hidden axes observed. Orientation
can remain underconstrained. There is no object contact or grasp-success oracle.

Before freezing this version, a sensor-only check used the previous training scenes
3101–3103. All nine cube observations returned dimension models; seven full poses were
marked reliable, and two partial red-cube poses remained explicitly uncertain. This
is an interface check, not a stacking success result. No reserved scenes were used
for this check. A recorded training sensor frame is retained as a regression fixture.

## Frozen rerun

- Output: `outputs/geometry_ablation_pilot_04`.
- Three policy revisions per arm, three training slots per revision.
- New training seeds: 3301, 3302, 3303. New reserved seeds: 3401–3405.
- Three cubes, blue/red/green bottom-to-top, sampled XY, initial upright orientation,
  no distractors, default physics (`noslip_iterations=0`).
- Same `gpt-6-astra`, high reasoning effort, 900-second generation limit per proposal.
- Explicit source budget in both prompts: target 18,000 characters, hard cap 48,000
  characters including whitespace/comments. Maximum assembled prompt: 100,000 characters.
- Invalid or oversized code consumes its revision and training slots, but is recorded
  as `candidate_validation` with `rollout_executed=false`. The exact generated source
  is preserved; the runner does not replace it with a sentinel policy or enter physics.
- Generate both arms independently in parallel each round. Execute simulator trials
  serially, alternating which arm runs first. No arm can see the other's context.
- Select by highest training success count, earliest revision on ties. Freeze both
  candidates before evaluating either on the reserved scenes. No reserved feedback
  goes to the actor or selects another candidate.

The 28 evaluation slots comprise 18 training and 10 reserved slots. Three additional
observation-only bootstrap trials supply initial training images. Generated source,
rollout slots, actual policy executions and code-validation failures are reported
separately. Tokens and generation time are measured, not forced equal. This is not a
token-matched ablation. The changed geometry treatment, seeds and budget mean this run
is not a paired efficacy comparison against pilot 03.

## Isolation and reproducibility

Each proposal uses a fresh temporary directory and context with user config, project
instructions, host skills and tools disabled. Only contract/history text and copied
sensor PNGs enter the prompt. Model events are audited before execution; unexpected
tool calls/items stop the campaign. Normal CLI authentication is temporary; `.apikey`
is not used. Infrastructure errors are preserved and stop the run without free retries.

The existing network-disabled policy sandbox accepts only numerical imports and the
sensor/robot RPC API. The separate geometry library is supplied only to its treatment
arm, in a separate namespace with a read-only wrapper. Policies cannot read the repo.

`manifest.json` freezes settings and source hashes. `source_snapshot/` preserves the
exact implementation files, and `contract_raw.txt` / `contract_geometry.txt` preserve
the effective contracts. Source, library or candidate changes reject cached results.
Human repairs to generated controllers are prohibited within the frozen campaign.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/run_geometry_ablation.py \
  --output outputs/geometry_ablation_pilot_04 \
  --train-seeds 3301 3302 3303 \
  --heldout-seeds 3401 3402 3403 3404 3405
```

`--prepare-only` creates the manifest and snapshots without model calls. A repeated
command resumes completed results only when hashes/settings match; partial interrupted
calls/trials remain for inspection. Do not overwrite a frozen run to change settings.

`progress.json` retains training curves and the active phase. On completion, the runner
writes `summary.json`, `report.json` and `learning_curves.png`. Reports distinguish
budget/code rejections from actual control failures. CPU compute profiles exclude
RPC/rendering/host IK and cannot establish end-to-end realtime behavior. Comparing
geometry checks with repeated VLM checks remains a separate quality-matched experiment.
