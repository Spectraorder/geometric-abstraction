# ASPIRE SO-100 stacking

新增 Mac RGB-D + 本体感知三层塔学习：空任务技能库，两轮达到 3/3，冻结后验证 2/2。见 [运行指南](docs/VISION_FROM_SCRATCH.zh-CN.md)、[可视化与后台部署](docs/VISION_DEPLOYMENT.zh-CN.md) 和 [历史输入审计](docs/OBSERVATION_AUDIT.zh-CN.md)。录像和公开证据随仓库保存在 `web/vision/`。[固定公网 Demo](https://aspire-so100-vision.vercel.app/) 已部署；Mac 后台服务和临时隧道启动方法见部署指南。

This project adapts the three useful pieces of the ASPIRE workflow to the local
`mjlab` SO-100 scene:

1. A closed-loop execution API records every observation, IK motion, gripper
   command, wait, pre/post state, status, and optional RGB keyframe.
2. Code-as-policy candidates implement `run(api)` and are ranked on fixed debug
   seeds.
3. A small skill library stores validated repair patterns for later candidates.

All box-shaped blocks are currently **3 × 3 × 3 cm cubes**. The initial task is
**stack `small_red_block` on `blue_block`**; the object names are retained for
policy compatibility. The other red cube and other shapes remain in the scene
as distractors. A trial succeeds
when the red block is centered on, touching, and stable on the blue block at the
correct surface-relative height. Orientation is reported but is not required.

Historical recordings and campaign success rates below describe their recorded
scene geometry; the 4 cm results are not results for the current 3 cm scene.

## Visual cube and gripper geometry

`policies/stack_three_blocks_visual_geometry.py` passed **445/450 development**,
**198/200 fresh reserved randomized layouts (99%)**, and **10/10 nominal** layouts
in simulation. It uses side RGB-D and wrist RGB to measure visible gripper reach
and cube geometry, then moves the blue base to a clear area selected using
robot-only reachability previews. It predicts how a tilted cube will settle and
permits a bounded retry. A deliberately missed grasp recovered with **9.75
additional simulated seconds**. See the [results, gripper overlay, and recovery
replay](docs/VISUAL_RETRY_POLICY.md).

```bash
uv run python view_scene.py --vision-policy policies/stack_three_blocks_visual_geometry.py \
  --vision-layout random_xy --seed 4001 --vision-output outputs/visual_geometry_view_4001
```

## Earlier improved 3 cm cube policy

`policies/stack_three_blocks_cubes_3cm.py` passed **8/10 training** layouts and
**9/10 new reserved** layouts (2001–2010), compared with **4/10** for the original
geometry policy on both sets. The fitter and evaluator are unchanged; the repair
raises the grasp, reduces release compensation, and increases final contact
clearance. See [the 3 cm evolution report](docs/CUBE_3CM_EVOLUTION.md).

The unchanged geometry policy and earlier evolved policy each passed **4/10**
randomized layouts in the initial transfer test. RGB-D estimated the side length at about
29.9 mm, but placement/support failures remain. See
[the 3 cm evaluation](docs/CUBE_3CM_EVALUATION.md) for per-seed results and evidence.

```bash
uv run python view_scene.py --vision-policy policies/stack_three_blocks_cubes_3cm.py \
  --vision-layout random_xy --seed 2002 --vision-output outputs/cube_3cm_view_2002
```

Saved state replays require the matching scene size: older 4 cm `rollout.npz`
files should not be replayed with the current 3 cm scene. Their existing MP4s
retain the original geometry.

## Evolved policy for 4 cm cubes

The historical `policies/stack_three_blocks_geometry.py` additionally fits cube
centroids, orientation, and eight inferred vertices from RGB-D. It passed 5/5
training seeds and 5/5 new validation seeds (1001–1005), compared with 3/5 for
the earlier evolved policy on those same new seeds. See
[visual cube geometry](docs/VISUAL_CUBE_GEOMETRY.md) for the method, uncertainty,
overlays, and run commands.

`policies/stack_three_blocks_cubes_4cm.py` reached **5/5 training** layouts
(101–105), then **4/5 reserved** layouts (201–205) after freezing. The same
frozen policy passed **2/3 nominal** layouts (81–83). It uses RGB-D measurements
for grasp height and release-drift compensation. See
[the evolution report](docs/CUBE_4CM_EVOLUTION.md) for changes and remaining failures.

These results require the original 4 cm scene. Use the saved MP4 recordings
to review them; running these policies in the current scene is a new 3 cm
transfer experiment.

## Run the baseline

To watch the robot stack in the browser, run `uv run python view_scene.py`.
This simulates the policy on seed 51, then loops playback of the physical
rollout with pause, speed, and reset controls. Use `--seed 54` to view another
instance, or `--scene-only` for the original passive scene viewer.
The **Arm camera** panel shows the wrist-mounted camera image; an orange
camera outline follows the arm in the 3D scene. The image updates at up to
10 FPS. On headless hosts the viewer selects EGL automatically; you can also
set `MUJOCO_GL=egl` explicitly.

The newer sensor-only RGB-D policy has a separate Viser playback mode. It runs
the frozen policy once with its `camera()`/`proprioception()` API, saves the
actual MuJoCo rollout, then opens interactive Viser playback with the side
RGB-D and wrist-camera panels:

```bash
uv run python view_scene.py --vision-policy policies/stack_three_blocks_vision.py \
  --seed 81 --vision-output outputs/vision_viser_81
```

The output directory must not exist yet. To replay this same trial without
running the policy again, use `uv run python view_scene.py --vision-trial
outputs/vision_viser_81`. On Linux, the policy subprocess requires `bwrap`;
on macOS, it uses `sandbox-exec`. The older `--task tower --candidate` mode
uses a different API and cannot execute the RGB-D policy.
Viser's **1x** setting paces the saved states at their original simulated-time
duration (subject to render performance); the separate MP4 demo is labeled 2x.
The printed success result describes the selected rollout; playback does not
guarantee that every seed succeeds.

```bash
uv run aspire-so100 --split debug --seeds 51
```

Save RGB keyframes around every primitive with:

```bash
MUJOCO_GL=egl uv run aspire-so100 --split debug --seeds 51 --render
```

Outputs are written to `outputs/stack_small_red_on_blue/<split>_<timestamp>/`.
Each candidate has a per-seed `trace.json`; the run-level `manifest.json` ranks
candidates by success rate and mean partial-credit score.

## ASPIRE repair loop

Copy `policies/stack_small_red_on_blue.py`, change one hypothesis, then compare
the candidates on the same debug seeds:

```bash
uv run aspire-so100 --split debug \
  --candidate policies/stack_small_red_on_blue.py \
  --candidate policies/my_repair.py
```

Inspect failed primitive states and frames, revise the candidate, and repeat.
Once a candidate is fixed, evaluate it exactly once on the untouched held-out
split:

```bash
uv run aspire-so100 --split heldout --candidate policies/my_repair.py
```

The default partitions are debug seeds 51–55 and held-out seeds 1–10. Do not use
held-out traces to tune a policy. Record repairs that generalize across debug
seeds in `skills/stacking.md`.

## Automatic LLM propose / execute / repair loop

### Learning across randomized RGB-D layouts

`scripts/learn_randomized_vision.py` evaluates the existing vision policy, then
asks the LLM to improve it using sensor images and post-trial feedback. All
three block colors share XY bounds `[-0.14, 0.14] × [-0.32, -0.21]` metres,
with 25 mm edge clearance. Yaw is unchanged. Training uses seeds 101–105;
only after 5/5 pass is the frozen candidate tested on held-out seeds 201–205.
Held-out feedback is never sent to the learner. This is continuation from the
existing policy, not learning from an empty policy library.

The current 5/5 training policy is saved in
`policies/stack_three_blocks_random_xy.py` (SHA-256
`0c8f31b2949a83b20acae430497a775590aae113439de7c89cb11f37df908349`).
Two reserved layouts, 201 and 202, were run and both passed. The other three
reserved layouts were not run; this is not a 5/5 held-out result.

```bash
uv run python -u scripts/learn_randomized_vision.py \
  --root outputs/random_xy_01 --model gpt-6-astra --rounds 0 --jobs 2
```

The same command resumes completed rounds. It retains the best scored policy
as `best_policy.py` and a successful replay directory in `best_trial.txt`.
When retrying an interrupted trial without `result.json`, it preserves the partial
files and previous log under `seed_<seed>.interrupted-*` before rerunning that seed.
Each attempt saves `rollout.npz`, cameras, notes, results and logs; MP4 rendering
is optional. `--baseline-only` runs no LLM calls. `--rounds 0` keeps trying until
all five training layouts pass or interruption. There is no cumulative token cap;
each proposal has bounded feedback, up to four images, and a 300-second timeout.

To execute any selected policy on a random layout and then view it at 1x:

```bash
uv run python view_scene.py --vision-policy policies/stack_three_blocks_random_xy.py \
  --vision-layout random_xy --seed 301 --vision-output outputs/random_view_301
```

To replay an existing attempt without executing it again, pass its directory
to `view_scene.py --vision-trial`.

### Original campaign runner

The autonomous runner invokes the installed **Codex CLI** using its existing
login and configured model (or an explicit `--model`). It sends the task API,
validated skill memory, generated code, rollout state/error traces, and selected
camera images to that LLM. It starts without a hand-written tower policy.

```bash
uv run python -m aspire_so100.agent_loop \
  --rounds 3 --output outputs/tower_campaign_01
```

Each round asks the LLM for a new Python program and its hypothesis, executes
the program on debug seeds 51–53, and supplies the best and latest programs and
failure evidence to the next LLM call. The full code, prompts, model events,
per-primitive traces, scores and campaign state are saved under the output path.
Each model call has a 300-second timeout; each trial has a 120-second timeout.
The runner stops after its round budget or a full debug pass.

### Compact context and usage controls

The feedback compiler in `src/aspire_so100/context.py` adapts the idea of
selective context construction described by
[GPT-Policy](https://github.com/cheng-haha/GPT-Policy). This is an independent
implementation for our simulator, not an installation of its hardware control
framework. No upstream source was copied. Its current LICENSE says a license
for redistribution/commercial use has not been selected.

Before each proposal, local code groups matching seed outcomes, retains each
distinct error and a representative trace, selects up to six physical changes
or errors, rounds numbers to four decimals, and includes complete source for
the best and latest candidate. Full precision traces remain on disk. Memory
is bounded to three tower skills within 4,000 characters. Summary selection
may omit useful intermediate events; smaller prompts do not prove better task
performance. No extra LLM call is used for compression.

```bash
uv run python -u -m aspire_so100.agent_loop \
  --model gpt-6-astra --rounds 3 \
  --max-prompt-chars 32000 --max-reported-tokens 60000 \
  --output outputs/gpt6_compact_01
```

The runner finds `codex` on `PATH` or in the newest installed OpenAI VS Code
extension. If Codex is installed elsewhere, pass `--codex-bin /path/to/codex`
or set `ASPIRE_CODEX_BIN`; a shell `PATH` export is not required.

`--max-prompt-chars` (default 32,000) bounds our supplied text for every proposal
and curator call. If code plus essential context will not fit, the runner stops
before calling the model rather than cutting code. Codex's own instructions and
other overhead are outside this character limit. Characters are not tokens.

Each call writes `proposal/usage.json` (or `curator/usage.json`), also recorded in
`campaign.json` under `llm_calls`. These are the CLI's observed usage counters,
not a dollar invoice. `--max-reported-tokens` (default 100,000) stops before the
next call once cumulative reported input plus output tokens reach the threshold;
**a running call can overshoot it**, so this is not a hard spending cap. Missing
usage is reported as unknown and blocks the next call. Cached/reasoning counters
are preserved separately without adding them again to input/output totals.

Preview the next prompt from an old campaign without any model call or simulation:

```bash
uv run python -m aspire_so100.agent_loop \
  --preview-from outputs/gpt6_tower_02/campaign.json \
  --output outputs/context_preview_01
```

The preview writes `prompt.txt`; it does not resume that campaign. Output
directories must be new. Already-running processes must be restarted to use
the new compiler; old artifacts are not rewritten.

Continue from the best and latest evidence in a prior campaign with a new
output directory:

```bash
uv run python -u -m aspire_so100.agent_loop \
  --continue-from outputs/gpt6_compact_02/campaign.json \
  --model gpt-6-astra --rounds 5 \
  --max-prompt-chars 32000 --max-reported-tokens 60000 \
  --output outputs/gpt6_compact_03
```

`--rounds` is the number of additional proposals. Prior policies, hypotheses,
scores and trace paths are inherited, while the token counter applies to new
calls in the continuation. Round numbering continues from the old history.
The actor stops proposing immediately after a 3/3 debug result, then runs the
fixed validation seeds and promotes a skill only if both validation trials pass.
`best_policy.py` is atomically refreshed after every scored round, including
when the prior campaign remains best. It is therefore available while an
open-ended campaign is running and after token-budget or keyboard interruption.

For an open-ended continuation, set `--rounds 0`. To make Ctrl+C or a 3/3 pass
the normal stopping conditions, also set `--max-reported-tokens 0`; this removes
the cumulative token guard but retains the per-call prompt-size limit and
timeouts:

```bash
uv run python -u -m aspire_so100.agent_loop \
  --continue-from outputs/gpt6_compact_02/campaign.json \
  --model gpt-6-astra --rounds 0 \
  --max-prompt-chars 32000 --max-reported-tokens 0 \
  --llm-timeout 300 --trial-timeout 120 \
  --output outputs/gpt6_compact_03
```

Ctrl+C records `status: interrupted`. Continue again by using that campaign as
`--continue-from` and choosing another new output directory.

View proposed and demonstrated capability changes for every round without an
LLM call:

```bash
uv run python -m aspire_so100.skill_report \
  --campaign outputs/gpt6_compact_02/campaign.json
```

Every active campaign also maintains `skill_progress.json`, and each new round
writes `round_NNN/skill_update.json`. A round is labeled `partial` only when the
evaluator demonstrates a capability such as a stable two-block stack. Proposed
hypotheses and failed behaviors are not called learned skills. `skills/learned/`
is updated only after a 3/3 debug pass and a separate 2/2 validation pass.

On a full debug pass, the selected program is frozen and tested on separate
validation seeds 61–62. Only if those also pass does a further LLM call distill
repair guidance into `skills/learned/*.json`, with the policy hash and evidence
path. Later campaigns read those entries automatically. Held-out seeds 1–10
are never used by this loop. Failed or unvalidated proposals are not promoted.

The current learning task is a three-block tower in a dedicated uncluttered
workspace: blue base, small red block, then a 4 cm green cube. Generated code
uses observations and motion/gripper primitives; the fixed evaluator checks
the entire support chain, robot release, face alignment, speed, COM margin,
and drift. A two-second two-block check and a final five-second whole-tower
check are required. These are finite-horizon simulation checks, not a promise
of stability under arbitrary future disturbances.

Replay a generated policy, including a failed attempt, with:

```bash
uv run python view_scene.py --task tower \
  --candidate outputs/tower_campaign_01/best_policy.py --seed 51
```

Tower playback executes the selected Python file; use policies from a trusted
campaign. Trials launched by the automatic runner use `bwrap` with networking
disabled, host files read-only, the user's home hidden, a fresh temporary
directory and a writable trial-output directory. API lint is an additional
check, not a Python security boundary. Infrastructure failures stop the campaign
instead of being misreported as failed robot policies.

Status: a live GPT-6 campaign produced policies and failed debug trials.
Compact context and usage guards have offline regression checks; their effect
on task success and live token usage still needs measurement. No learned tower
skill or successful generated tower policy is claimed yet.

## Scope

### Overhead RGB-D camera

Both the shape scene and tower scene include `table/overhead_camera` (name
retained for compatibility), now on a visible right-side mast at world
`(0.48, -0.60, 1.10)`, aimed toward `(0, -0.30, 0.45)`, with a 60-degree
vertical field of view. The decorative mast has no collision response.
This is a workspace view in the spirit of a
bimanual robot's top camera, not a calibrated replica of a particular YAM rig.
Viser shows a **Side-mounted RGB-D camera** folder with RGB and depth images at
up to 5 Hz and a cyan camera frustum, alongside the existing wrist camera.

```bash
uv run python view_scene.py --scene-only
# Or inspect the same camera during policy playback:
uv run python view_scene.py --task tower \
  --candidate outputs/gpt6_compact_03/best_policy.py --seed 52
```

Generated policy code can call `api.get_overhead_rgbd()` on demand. It returns
aligned `rgb` (uint8, 240x320x3), `depth_m` (float32, 240x320), `valid_depth`,
`intrinsics`, `camera_to_world`, and simulation `time`. Depth is distance along
the optical axis in meters, not Euclidean range or height above the table.
Camera coordinates are right/down/forward and pixels are indexed `[v,u]`.
The UI displays a fixed grayscale mapping: white at 0.25 m, black at 1.00 m;
tabletop depth varies across this angled view. Raw depth is not display-normalized.
This ideal simulated sensor has no real-camera noise or missing-depth model.

Rendering requires a working OpenGL backend (EGL on the headless GPU node).
New trials capture before/after image observations automatically. Each PNG sheet
contains side RGB, a depth preview and wrist RGB, saved as `initial.png` and
`outcome.png` under `round_NNN/debug/seed_N/`. Corresponding NPZ files keep raw
metric depth, validity mask and calibration locally. Generated policies may
also call `get_overhead_rgbd()` at intermediate steps.

The next proposal attaches up to four sheets to **Codex/OpenAI** with `--image`:
before/after for one representative seed of the best and latest policies.
If prior trials have no images (including old campaigns), the runner renders
and attaches a fresh initial scene. It never pretends that this is a historical
rollout image. Every proposal's `images.json` records exactly which files were
attached, and usage records include `image_count`. Images consume model tokens
outside the text-character budget; the small fixed selection limits growth.

This makes the coding LLM's learning context multimodal, with simulator object
poses retained as auxiliary inputs. It is not continuous VLM control at each
motion step, nor an image-only perception pipeline. The model writes a policy,
the local simulator executes it, and subsequent revisions receive images and
text feedback. This path was tested locally with a simulated worker and a
mocked model command; improved task success is not yet established. Restart the
viewer/learner to load these changes.

This is a lightweight SO-100 adaptation of ASPIRE's architecture, not the
paper's full CaP-X reproduction. It uses camera feedback plus simulator object
poses, without SAM3/GraspNet perception. The automatic loop generates one proposal per round
and reuses the best and latest candidates; it does not yet run a parallel
population or implement human turn-taking. The original `aspire-so100`
command still evaluates manually supplied single-object policies and does
not call an LLM; use `aspire_so100.agent_loop` for autonomous discovery.
