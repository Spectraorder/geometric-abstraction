# ASPIRE SO-100 stacking

This project adapts the three useful pieces of the ASPIRE workflow to the local
`mjlab` SO-100 scene:

1. A closed-loop execution API records every observation, IK motion, gripper
   command, wait, pre/post state, status, and optional RGB keyframe.
2. Code-as-policy candidates implement `run(api)` and are ranked on fixed debug
   seeds.
3. A small skill library stores validated repair patterns for later candidates.

The initial task is **stack `small_red_block` on `blue_block`**. The larger red
block and the other shapes remain in the scene as distractors. A trial succeeds
when the red block is centered on, touching, and stable on the blue block at the
correct surface-relative height. Orientation is reported but is not required.

## Run the baseline

To watch the robot stack in the browser, run `uv run python view_scene.py`.
This simulates the policy on seed 51, then loops playback of the physical
rollout with pause, speed, and reset controls. Use `--seed 54` to view another
instance, or `--scene-only` for the original passive scene viewer.
The **Arm camera** panel shows the wrist-mounted camera image; an orange
camera outline follows the arm in the 3D scene. The image updates at up to
10 FPS. On headless hosts the viewer selects EGL automatically; you can also
set `MUJOCO_GL=egl` explicitly.
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

## Skill layer, kinematics and tower plans (added 2026-09-21)

A verified pick-and-stack skill layer that reaches 100% on both benchmark
tasks and generalises to arbitrary tower plans on randomized layouts. It
uses only the `TowerAPI` surface (`get_observation`, `move_to_joint_positions`
/ `move_end_effector`, `set_gripper`, `wait`), so the same code runs on
`StackingEnv`, under the tower trial runner, and in the plan harness.

| Module | What it does |
|---|---|
| `src/aspire_so100/skills.py` | `Skills.grasp` (yaw-aligned, centred between the open pads, **lift postcondition** rejects misses and slips), `Skills.place` (live re-perception of the support, descent closed on the held block's **lowest point**, fixed-pad back-off before retreat, geometry-aware placement check), level inspection (`level_ok`, `first_bad_level`), parking on free spots |
| `src/aspire_so100/so100_kinematics.py` | Exact FK from the MJCF (0.000 mm vs MuJoCo) and a **task-priority IK**: site position (hard) → gripper pointing down in the position nullspace (exact where feasible, minimum tilt otherwise) → closing-axis yaw via the wrist roll |
| `src/aspire_so100/tower_plan.py` | `TowerPlan` schema (block *selectors*, support DAG, offsets, relative yaw), `compile_plan` (selector binding, ordering and static COM/reach checks), `execute_plan` with **level-aware repair**: on an unstable check, find the lowest broken level, park the blocks above it, rebuild |
| `src/aspire_so100/eval_skills.py` | In-process evaluator mirroring the official protocols; `--task stack|tower|plan`, `--randomize`, `--contact-fix` |
| `policies/stack_skills.py`, `policies/tower_skills.py` | Benchmark policies on top of the skill layer |
| `policies/tower_standalone.py` | The tower policy folded into one lint-compliant file (math/numpy only) for `agent_loop`'s bwrap runner |

```bash
uv run python -m aspire_so100.eval_skills --task stack --candidate policies/stack_skills.py --seeds 51-55
uv run python -m aspire_so100.eval_skills --task tower --candidate policies/tower_skills.py --seeds 100-119 --randomize 0.03
uv run python -m aspire_so100.eval_skills --task plan --seeds 1-10          # eight plans on the 5-box scene
```

### Results (frozen policies; held-out seeds evaluated after development on 51-55 / 100-119)

| Task | Split | Success | Notes |
|---|---|---|---|
| Stack small red on blue | held-out 1–10 (official CLI) | **10/10** | centring 2.8 mm mean / 5.7 mm max, height 0.2 mm, all upright (baseline: 4/5 debug, ~15 mm, tipped) |
| 3-block tower | held-out 1–10 + validation 61–62 | **12/12** | |
| 3-block tower, ±3 cm / ±25° layout jitter | seeds 1–10 | **10/10** | 19/20 under `--contact-fix` physics |
| Tower plans, randomized 5-box layouts | 2-block plans (selectors, base relocation), held-out 1–10 | 18/20 | 20/20 on dev 100–109 |
| | 3-block plans (straight, selectors, staircase offsets, crossed), held-out 1–10 | 32/40 | 34/40 on dev; staircase offsets are the weakest (5/10) |
| | 4- and 5-block towers, held-out 1–10 | 1/20 | kinematic limit, see below |

### What limited the original policies

* `wrist_roll=0` ignored the base rotation; the jaw closing-axis yaw is
  `pi/2 + base_rotation + wrist_roll`, so jaws were ~40° off the block —
  hence tipped placements and silent misses that no postcondition caught.
* Position-only IK leaves gripper pitch free; at transit poses it reached
  25°, and long blocks rotated out of the narrow pads.
* mjlab's `Scene` overrides the robot MJCF's `cone="elliptic" impratio="10"`
  with `pyramidal` / `1` (it warns at compile time). `--contact-fix`
  restores them; results above hold under both.

### Known limits

* **Height.** With the wrist pitch at its 1.66 rad limit the gripper cannot
  stay vertical above ~10 cm site height; minimum tilt is ~12–17° at 14 cm
  and 25–30° at 18 cm. Top-down stacking degrades from the 4th 5 cm layer
  on. Taller towers need a different grasp (side grasp) or a taller arm.
* Rendering (`--render`, `get_overhead_rgbd`, the official worker's
  before/after images) currently runs on software EGL on this host — the
  NVIDIA kernel/userspace versions are mismatched until a reboot — so it
  works but slowly. `python -m aspire_so100.agent_loop --worker --candidate
  policies/tower_standalone.py --output <dir> --seed 51` completes with
  `success: true`.

### Environment note

`../github/mjlab` must be `mujocolab/mjlab` at commit `88a7ff8` (the lock
records `rsl-rl-lib==5.5.0` and `wandb<0.29`, which only that commit has),
with its `requires-python` cap relaxed from `<3.14` to `<3.15` — the lock is
resolved for Python 3.14. `uv lock --check` then passes.

### Watching rollouts

* **Interactive (browser):** `uv run python view_scene.py --task tower --candidate policies/tower_skills.py --seed 51`
  simulates the policy, then loops the recorded physical rollout in viser
  (pause / speed / reset) with the wrist camera and overhead RGB-D panels.
  `--scene-only` opens the passive scene.
* **Video (MP4):** `MUJOCO_GL=egl uv run python -m aspire_so100.record_video --task tower --candidate policies/tower_skills.py --seed 107 --randomize 0.03`
  writes `outputs/videos/<task>_<policy>_seed<N>.mp4` — free camera with the
  wrist and overhead cameras alongside, captioned with sim time and the
  running primitive. `--task stack|tower|plan`, `--plan p3_crossed`,
  `--speed` (sim seconds per video second), `--views free,wrist,overhead`.
* **Keyframes:** `uv run aspire-so100 --split debug --seeds 51 --render`
  saves a PNG before/after every primitive under the trial directory; the
  official worker writes `initial.png` / `outcome.png` (side RGB, depth,
  wrist RGB) per trial.
