> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

> Historical pilot 03 protocol and results. The current runner uses a newer protocol;
> the commands below are archival, not commands to rerun against the current source.

# Three-cube geometry-abstraction pilot

This experiment tests whether a supplied, reusable visual geometry library helps a
policy-writing model develop a stacking controller with fewer revisions and rollouts.
It does not initialize either arm from a successful repository policy. Existing
controllers remain available for ordinary development but are excluded from the actor.

## Frozen pilot design

- One independent campaign per arm: `raw` and `geometry`.
- Three revisions per arm; three training rollout slots per revision.
- Training layout seeds: 3101, 3102, 3103. Reserved seeds: 3201–3205.
- Three cubes only, blue/red/green in bottom-to-top order. Random sampled initial XY,
  initial upright orientation, no distractors, fixed cameras and calibration.
- Default physics (`noslip_iterations=0`) in both arms.
- Identical side RGB-D, wrist RGB, robot proprioception, robot-only IK and low-level
  motion API. Wrist depth is not part of this experiment.
- Same model (`gpt-6-astra`), high reasoning effort, 900-second generation timeout,
  60,000-character prompt cap and 24,000-character generated source cap.
- Revision and rollout budgets are fixed. Token usage and elapsed generation time
  are measured rather than forced equal. Do not call this a token-matched ablation.
- Choose the candidate with highest training success count; ties select the earliest
  revision. Freeze both arms before evaluating either on reserved scenes.

There are 18 training rollout slots and 10 reserved trials, plus three shared
observation-only bootstrap rollouts that provide initial sensor images. Bootstrap
rollouts are setup cost, not generated policy attempts. Invalid candidates consume
revision and rollout slots and count as failures; invalid Python need not enter the
simulator. Infrastructure failures stop the campaign for inspection rather than being
silently reclassified as policy failure or automatically retried until successful.

## Allowed capabilities

Both arms receive the same task and API contract, an empty `run(api)` starter, initial
sensor images, and their own previous hypotheses, reduced feedback and latest code.
They may write their own perception and control functions using NumPy, SciPy and math.
No ready-made pick/place/stack, recovery, clearing, alignment or pointing controller is
supplied. Runtime model calls are unavailable.

The geometry arm additionally receives a read-only `geometry` object:

| Method | Evidence and output |
| --- | --- |
| `table(frame)` | Table plane fitted from calibrated RGB-D |
| `cube(frame, color, model=None, table_plane=None, prior=None)` | Visually measured dimensions, fitted center, inferred corners and support face, observability flags |
| `gripper(frame, tool_position, support_z=None)` | Visible finger surfaces, lowest visible surface, partial opening/clearance estimates |
| `margin(point, top_vertices, normal)` | Geometric point-to-support-edge distance |

The library has no robot API argument and executes no actions. It uses numerical
helpers extracted from the existing perception modules, with controllers and trackers
excluded. First-view cube dimensions require an unobstructed tabletop observation;
later calls retain the visually learned model. Hidden vertices and underconstrained
poses are estimates, not simulator truth. This treatment measures the value of
pre-engineered geometric observations, not learning without prior engineering.

## Isolation and evidence

Each proposal runs in a new temporary directory and fresh model context. User config,
repository instructions, host skills, shell, apps, browser and agent tools are disabled.
Only the explicit contract, own campaign history and copied sensor PNGs are supplied.
Model event logs are audited before candidate execution; unexpected tools/items fail
closed. Normal model authentication is inherited through a temporary protected auth
file, removed with the temporary directory. The project's `.apikey` is not used.

The policy executes in the existing network-disabled worker sandbox. Only the geometry
arm receives the separate frozen library; its function globals are separate from the
candidate's globals. A slotted wrapper prevents replacing the exported methods. The
baseline can implement its own geometry, but cannot import this library or the repo.

`manifest.json` freezes scene/sensor/physics settings, source hashes, library hash,
model settings and budget. Each proposal retains its exact prompt, input-image hashes,
raw model events, response, generated source, usage, duration and source hash. Existing
results must match candidate, library, seed, task, scene, physics and sensor settings.
Implementation changes after freezing abort the run; use a new experiment version.

The actor receives only success, policy error, its own sensor-derived notes and rendered
sensor images from training. Private evaluator fields remain host-side. Reserved
results are never fed back or used to choose another policy in this campaign.

## Running and interpreting

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/run_geometry_ablation.py \
  --output outputs/geometry_ablation_pilot_03
```

Use `--prepare-only` to inspect a manifest before any model calls. Re-running with the
same frozen implementation resumes completed proposals and trials; interrupted partial
proposals/trials remain preserved and require inspection. Do not edit generated
candidates. A new source/settings configuration requires a new output directory.

`summary.json` reports per-revision training success, generated policy hashes, actual
model usage/time and reserved results. Report latest-candidate and best-so-far training
success separately; neither is an unbiased generalization estimate.

This small pilot cannot establish statistical superiority, broad generalization, or
90% success. A confirmatory study needs repeated fresh campaigns, new reserved seeds,
predeclared budgets and uncertainty intervals across campaigns. If both arms fail,
inspect the bottleneck and version the next protocol before gathering new evidence;
do not insert human repairs into only one arm.

The runtime CPU profile excludes RPC, rendering and host IK. It is not end-to-end
feedback latency. Comparing geometric checking against repeated VLM checking is a
separate quality-matched experiment and is not performed here.

The first preflight directory (`geometry_ablation_pilot_01`) was stopped before any
generated candidate ran: the CLI emitted an unstable-feature notice as an error item.
The next version suppresses that startup notice while retaining fail-closed event
auditing. Its partial invocation remains archived and is not an efficacy result.

The second preflight (`geometry_ablation_pilot_02`) timed out after 300 seconds
before receiving its first complete candidate. An independent minimal model request
succeeded in 7.13 seconds. Version 2 uses 900 seconds for every generation in both
arms; no policy success/failure or reserved outcome informed this budget change.

## Completed pilot 03 results

The run completed all 18 training slots and 10 reserved slots. These results do not
support a geometry benefit under this protocol; they are a small diagnostic pilot.

| Arm | Revision 1 train | Revision 2 train | Revision 3 train | Frozen revision | Reserved |
| --- | ---: | ---: | --- | ---: | ---: |
| Raw sensors | 0/3 | 1/3 | 0/3, source cap | 2 | 1/5 |
| Geometry | 0/3 | 0/3 | 0/3, source cap | 1 | 0/5 |

Geometry revision 1 stopped on every training layout because its controller rejected
all tested upright approaches. Revision 2 stopped on every training layout while
initializing the three cube models. These are controller/interface-use failures, not
worker exceptions. The actor's proposed explanations remain hypotheses.

Both third revisions exceeded the 24,000-character source cap (raw: 25,154;
geometry: 24,829). The original responses are preserved. The runner substituted a
failing sentinel for each oversized candidate, so six slots did not test the proposed
controllers. A protocol defect is that the numerical source cap was in the manifest
but not explicitly in the actor prompt. These six failures must not be interpreted
as manipulation failures or evidence about either proposed third revision.

The baseline generated its own face-based perception in revision 2, as allowed by the
protocol. Do not describe the baseline as incapable of geometry reasoning.

Total completed generation time was 875.1 seconds for raw and 766.4 seconds for
geometry; this excludes aborted preflights and simulator trials. Each arm has only one
campaign. Neither generalization superiority nor a realtime/VLM comparison is established.

Evidence: [machine-readable report](../evaluations/geometry_ablation_pilot.json) and
[learning curves](../media/geometry_ablation_pilot.png). Full prompts, model responses,
sensor frames and replays remain in `outputs/geometry_ablation_pilot_03`.

Before another experiment, explicitly communicate source/output limits, diagnose
geometry initialization and IK stopping behavior on development evidence, and freeze
a new protocol. Do not repair one arm's generated controller by hand. Any subsequent
generalization evaluation must use new reserved scenes; seeds 3201–3205 are now used.
