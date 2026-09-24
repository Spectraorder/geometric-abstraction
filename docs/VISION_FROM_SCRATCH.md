# Mac RGB-D and proprioception: learning a three-level tower from an empty task-skill library

Recorded on 2026-09-21. Original local campaign: `outputs/vision_from_scratch_01/`.

`web/vision/` contains seven recordings, two rounds of candidates/prompts and reduced results. Full sensor arrays, private traces and model event logs remain on the experiment machine. Paths under `outputs/` refer to those local artifacts. See the [deployment guide](VISION_DEPLOYMENT.md) for display options.

**Round 1 failed. Round 2 passed seeds 81/82/83. After freezing that same policy, reserved seeds 91/92 also passed. Every learning, confirmation and validation run, including failures, was recorded.**

The policy received RGB, metric depth, camera calibration and robot joint/tool proprioception. It did not receive true object positions, dimensions, contacts or an online stability oracle. This was RGB-D control, not RGB-only.

## Recordings and attempts

- [All attempts and public evidence](../web/vision/manifest.json)
- [Playable video index](../web/vision/index.html)
- [Round 1 failure](../web/vision/media/round_001_trial_81.mp4)
- [Round 2 success](../web/vision/media/round_002_trial_81.mp4)
- [Independent frozen-policy validation](../web/vision/media/heldout_trial_91.mp4)

Videos restore saved MuJoCo states from every physics step. They are not generated video, and failed segments were not replaced by reruns. Format: 960×600, H.264, 20 FPS, explicitly marked **2× speed**. Side panels show RGB, depth and a display-only wrist view; this policy used side RGB-D and did not read the wrist camera. Recordings include private evaluator waiting after the policy stops.

| Phase | Seed | Outcome | Policy simulation time | RGB-D calls |
|---|---:|---|---:|---:|
| Round 1 learning | 81 | Failed: visual confidence check stopped after red placement | 27.00 s | 9 |
| Round 2 learning | 81 | Passed | 68.38 s | 29 |
| Round 2 confirmation | 82 | Passed | 67.10 s | 28 |
| Round 2 confirmation | 83 | Passed | 68.38 s | 29 |
| Frozen validation | 91 | Passed | 66.00 s | 28 |
| Frozen validation | 92 | Passed | 65.82 s | 27 |
| Sensor blackout | 81 | Failed: no grasp motion | 0.00 s | 1 |

There were four learning/confirmation runs, two independent validations and one control. Round 1 seeds 82/83 were not executed after seed 81 failed. No failed learning attempts are omitted. Installation and isolation smoke tests are stored separately under `infrastructure/`.

## What the model learned

Each round used a new **gpt-6-astra Codex CLI context** to produce a complete Python policy. Round 1 received task/API instructions and the initial RGB image, without repository policies, task skills, user history or this conversation. Round 2 additionally received round 1's code, sensor images, its own perception/decision notes and post-trial success/failure feedback. The root assistant did not manually edit either generated candidate.

Round 1 estimated object positions with color segmentation and depth backprojection. Once the red block hid the blue top, the re-estimated base center shifted and the visual check stopped further motion. Round 2 retained the unobstructed base estimate and re-observed the payload near placement to correct its offset. It compared two camera observations for red-block stability rather than querying simulator status.

**Starting from scratch means an empty task-policy/skill library, not random neural weights.** The model retained pretrained knowledge. Task order, RGB-D conventions, hardware limits and gripper limits were supplied. A generic `move_tool` SDK solved IK from the known robot model with a soft downward-grasp orientation preference. It did not read object state. Robot kinematics and low-level control were not learned from scratch; learning repaired code policies without updating neural weights.

Both rounds retain `prompt.txt`, `invocation.json`, `model.jsonl`, `response.json` and `candidate.py`. Neither formal model call used tools. An earlier context probe stopped before policy execution and remains under `infrastructure/rejected_actor_context`; it did not generate a formal candidate.

The formal CLI calls reported 29,819 input and 5,650 output tokens. This excludes the infrastructure probe and is not a billing statement. Experimental skill-discovery and missing code-mode host warnings are retained in the audit; the model returned policies without tools.

Frozen policy SHA-256:

```text
61957c53bb3578f731607ce3aa60580d071c9245a357931c1136edb9dda17cdf
```

`policies/stack_three_blocks_vision.py` is a byte-for-byte copy. All five successful runs and the blackout control used that hash. Reserved results were not used for repairs.

## Observation boundary

1. Proposal generation used a temporary workspace and separate temporary `CODEX_HOME`, with only existing login credentials copied. User configuration, host skill discovery, shell, browser and apps were disabled; invocation events were audited.
2. The policy executed in a separate Python subprocess over JSON RPC. MuJoCo remained in the parent; the worker had no simulator object.
3. Candidate imports were limited to numerical libraries; file reads, dynamic imports, private attributes and common reflection paths were rejected. On macOS, `sandbox-exec` blocked networking and home-directory reads except numerical runtime paths. This is an access-control description, not a universal malicious-code security proof.
4. Ground truth was used for private scoring and local evidence only. It was not returned through RPC or passed to the next model round. Private `trace.json`/`result.json` were separated from reduced `actor_feedback.json`.
5. A private monitor replaced the old policy-triggered intermediate check: blue/red had to remain stable for two seconds, sampled around every 50 ms with a 3 mm drift check. The policy could not query it. The final original evaluator still required five seconds of whole-tower stability. Thresholds were unchanged; the intermediate check's trigger changed.

The blackout control zeroed RGB and set depth to NaN. The policy reported insufficient tabletop depth and made no grasp motion. This supports visual dependence but is not, by itself, proof of every isolation property. See `audit/verification.json` and `tests/test_vision_boundary.py` alongside source, RPC and OS audits.

## Rerunning the frozen policy on Mac

Use the workspace and its lock file:

```bash
cd /path/to/aspire_so100
unset MUJOCO_GL
uv sync --frozen

# The output directory must be new. This executes physics again.
uv run python scripts/run_vision_trial.py \
  --candidate policies/stack_three_blocks_vision.py \
  --seed 91 --output outputs/my_vision_demo

# Render the saved trajectory without executing the policy again.
uv run python scripts/render_vision_trial.py \
  --trial outputs/my_vision_demo --speed 1 --fps 20
open outputs/my_vision_demo/demo.mp4
```

`result.json` records success and source hash. `camera_*.npz` stores actual policy RGB-D/calibration inputs; `api_calls.json` records actions and sensor requests; `rollout.npz` stores physics states. Evaluator time after policy completion makes the video longer than `policy_sim_seconds`.

This policy uses the restricted sensor SDK and cannot be passed directly to the legacy `view_scene.py --candidate` interface, which lacks the same `camera()`/`proprioception()` API.

## Starting a new empty-skill campaign

The local Codex CLI must already be logged in with access to `gpt-6-astra`. Initialization produces the initial image and protocol, without loading a successful policy.

```bash
uv run python scripts/init_vision_campaign.py --root outputs/my_new_vision_learning
uv run python -u scripts/run_vision_campaign.py \
  --root outputs/my_new_vision_learning --model gpt-6-astra --rounds 0
```

`--rounds 0` continues until one candidate passes all of 81/82/83. Every execution is recorded before the next attempt, including failures. Model calls have a 300-second timeout; policies have 120 seconds and 600 RPC calls. Infrastructure errors stop the campaign rather than silently omitting evidence. Ctrl+C preserves existing artifacts. A new stochastic campaign is not guaranteed to succeed in two rounds.

The frozen candidate is saved as `best_policy.py`. Run seeds 91 and 92 separately and render both; do not use their results for tuning. `--start-round N` resumes complete saved candidates/results/videos. An incomplete proposal requires inspection and a new experiment directory rather than treating partial output as a successful resume.

## Historical comparison and scope

| Experiment | Input classification |
|---|---|
| Original two-block baseline | Privileged state |
| Initial three-level tower | Privileged state and online structure oracle |
| Earlier four-round LLM feedback | Privileged state, ground-truth trajectory feedback and online oracle |
| This two-round visual campaign | RGB-D, calibration and proprioception; object truth restricted to evaluation |

See the [historical observation audit](OBSERVATION_AUDIT.md). RGB/depth panels in older videos do not establish visual control.

This experiment used ideal simulated RGB-D, known calibration/robot model, fixed colors/table and small XY perturbations (±4 mm, initial yaw zero). The Mac renderer reported depth-precision limitations; numerical rendering error remained despite no added sensor noise. Five successful frozen runs apply to this limited setting, not unknown shapes, general scenes or real hardware. This is not a complete reproduction of the ASPIRE paper system.
