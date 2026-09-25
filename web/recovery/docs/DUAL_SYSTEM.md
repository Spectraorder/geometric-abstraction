# Astra planning × Jev decisions

This branch connects real model calls to a bounded **initial-observation gate** in the SO-100 simulation. It does not replace the subsequent motion controller with a model or claim a learned, general-purpose recovery policy.

## Information flow

1. A sandboxed policy derives table visibility and task-color visibility from calibrated RGB-D.
2. It sends a strict state schema over JSON RPC: observation version/time, phase, visible colors, table-fit validity, grasp status, and remaining budgets.
3. The host invokes Astra for a conditional plan and Jev for a typed action. The worker never receives the API key or network access.
4. The host checks the action, state version, plan version, preconditions, confidence, and budget. A revised plan requires reobservation before motion.
5. Once the initial visual preconditions hold, the same frozen RGB-D stacking skill executes in every group. A separate private evaluator determines the final outcome.

Physics pauses while a model call is pending. The gate has at most three observations (two extra observations 200 ms apart), up to two slow repairs, and a four-request host ceiling. Trial wall timeout is 900 seconds to include model startup/wait; the underlying 600-RPC limit remains. These limits differ from historical no-model trials and are reported separately.

## Actual integration findings

The first hybrid development run failed: Astra's plan embedded the transient fact “blue is not visible”; after the target became visible, Jev still proposed reobservation with no budget remaining. The guard stopped without motion. Its trace and the original all-Astra/Jev-only trials are retained.

Protocol `phase-conditional-plan/2` separates current facts from plan rules. The prompt says current observations override stale plan assertions; it also clarifies that a null grasp confirmation is expected before any grasp. Low-confidence decisions may use an already-budgeted reobservation, but cannot execute uncertain motion or increase budgets. All three groups are rerun under this revision; the original protocol is not pooled with it.

See [live integration results](evaluations/dual_system_live.json) for actual counts, timings, token usage, failures, and video references. These are development connection checks on one known seed, not a heldout accuracy or cost-efficiency benchmark. A successful initial gate does not demonstrate online recovery during manipulation. Protocol repair here was an offline harness change; it must not be described as autonomous online self-correction.

## One-scene development results

| Protocol 2 group | Tower outcome | Model calls | Model wait, seconds | Input tokens |
|---|---|---:|---:|---:|
| astra_only | Complete | 4 | 70.03 | 52259 |
| hybrid | Complete | 4 | 24.79 | 15267 |
| jev_only | Stopped before motion | 1 | 1.37 | 647 |

All rows use seed 4001 and the same 0.35-second initial image occlusion. Mixed-system and all-Astra physics duration was 91.65 seconds. The pure-Jev run ended early, so its short wait is not evidence of superior task efficiency. These sequential development runs were not a preregistered performance study; some local rendering/verification ran concurrently. No accuracy, population speedup, or dollar-cost advantage is established. All six trials across both protocol versions are retained.

## Models and costs

Official endpoint: `https://api.typesafe.ai/v1/systemone`, fixed model `jev-1.13.0`. TypeSafe's [model documentation](https://docs.typesafe.ai/models) lists text/JSON input, $0.042 per million input tokens and free output tokens (checked 2026-09-24). The client reports a labeled estimate from returned usage, not a paid invoice. It never falls back to a mock when credentials or the network are unavailable.

The user-provided community article led to a similarly named third-party site whose footer disclaims TypeSafe affiliation. This implementation calls TypeSafe directly.

Astra uses the installed Codex CLI with `gpt-6-astra`: high effort for plans, low effort for decisions, fresh context, tool access disabled, and structured output. Log CLI token counters including reasoning/cache counters without double counting. Subscription billing is not an API invoice: Astra dollar cost is unknown, never zero. CLI startup/system-prompt overhead means these client timings are not pure inference timings.

## Run from a clean checkout

```bash
uv sync
# The file contains only the token. It stays outside Git and is never printed.
uv run python scripts/run_dual_system_trial.py \
  --mode hybrid --fault transient_occlusion --seed 4001 \
  --jev-key-file /absolute/private/path/jev_api.txt \
  --output outputs/hybrid_new

# Other groups use the same scene and sensor/actuator implementation.
uv run python scripts/run_dual_system_trial.py \
  --mode jev_only --fault transient_occlusion --seed 4001 \
  --jev-key-file /absolute/private/path/jev_api.txt \
  --output outputs/jev_new
uv run python scripts/run_dual_system_trial.py \
  --mode astra_only --fault transient_occlusion --seed 4001 \
  --output outputs/astra_new

uv run python research/recovery_lab_20260924/harness/render_fault_trial.py \
  --trial outputs/hybrid_new --speed 2 --fps 20
```

Alternatively set TYPESAFE_API_KEY in the same terminal session and omit the key-file flag. No credentials are included in manifests or public evidence. Do not paste the key into Git, a webpage, or the chat. Every trial requires a new output directory.

The transient proxy obscures the blue target for 0.35 simulation seconds from the first side observation. `--fault persistent_occlusion` masks it throughout the initial gate; `--fault none` is the nominal regression. These are image-space faults, not physical occluders. Fault schedules remain private to the host.

## Offline preparation and tests

```bash
uv run python research/dual_system_20260924/prepare_cases.py
uv run python -m unittest discover -s tests
uv run python -m unittest discover -s research/recovery_lab_20260924/harness -p 'test_*.py'
uv run python scripts/benchmark_policy.py --profile all_supported --output outputs/readiness_new.json
uv run python scripts/serve_vision_demo.py --bind 127.0.0.1 --port 8093 --root web/recovery
```

Nine RGB-D fixture summaries are development probes, not independent accuracy labels. The older `replay.py` checks independent recorded observations; its default test-double responses are explicitly labeled and are not robot rollouts. Use `run_dual_system_trial.py` for physical execution.

## Next stages

- Connect model decisions at grasp verification, placement verification and identity reacquisition, with the same frozen execution primitives.
- Add target support DAGs, temporal uncertainty and progress checks; do not pass simulator support/contact truth to a model.
- Calibrate routing thresholds on development data, then freeze a fresh paired scene/fault manifest.
- Demonstrate an online abnormal observation → useful Astra plan revision → fresh Jev decision → recovery; compare against repair-disabled hybrid and deterministic rules.
- Run the benchmark profiles and report success–cost–latency tradeoffs with uncertainty. No advantage is assumed in advance.
