# Luna decides; Astra plans

This experiment asks a narrow, practical question: **how much reasoning effort does each part of a fast/slow robot policy need?** It replaces Jev in the new experiments with Luna. Historical Jev results are retained as historical evidence, not pooled into this comparison.

The robot must build a blue–red–green tower. Before it moves, a model gate checks what the RGB-D camera can see. Astra writes a conditional plan. Luna decides whether to start, observe again, ask Astra for a revision, or stop. After the gate permits motion, every configuration runs the same previously validated visual stacking skill.

The model gate is real, the physics is simulated, and the videos replay recorded simulation states. This is **not** a new robot skill learned from scratch, and models do **not** yet control recovery during a grasp or placement.

## Open the experiment

- [Public English page](https://aspire-so100-recovery.vercel.app/luna-astra/): effort grid, single-model controls, every attempt, videos and readable decision timelines.
- [Machine-readable results](results.json).
- [Frozen experiment manifest](campaign.json).
- [Task catalog](../docs/TASK_CATALOG.md) and [larger benchmark design](../docs/POLICY_BENCHMARK.md).

<!-- measured-results:start -->
## Measured pilot results

All **57 scheduled attempts** are retained. **35/38** tower-completion checks and **19/19** correct-stop checks passed. There were **198 model calls**, **0 failed provider calls**, and **20 guard interventions**. Those intervention counts matter even when the final endpoint passed.

| Configuration | Tower | Correct stop | Total model wait | Input tokens | Slow repairs | Rule agreement |
|---|---:|---:|---:|---:|---:|---:|
| Luna low / Astra low | 2/2 | 1/1 | 144.11 s | 139,687 | 1 | 7/7 |
| Luna low / Astra medium | 1/2 | 1/1 | 180.97 s | 152,843 | 2 | 5/7 |
| Luna low / Astra high | 2/2 | 1/1 | 165.64 s | 139,880 | 1 | 7/7 |
| Luna low / Astra xhigh | 2/2 | 1/1 | 171.50 s | 126,707 | 0 | 7/7 |
| Luna medium / Astra low | 2/2 | 1/1 | 147.18 s | 139,660 | 1 | 7/7 |
| Luna medium / Astra medium | 2/2 | 1/1 | 164.27 s | 139,709 | 1 | 7/7 |
| Luna medium / Astra high | 1/2 | 1/1 | 165.62 s | 139,714 | 1 | 6/7 |
| Luna medium / Astra xhigh | 2/2 | 1/1 | 193.85 s | 139,821 | 1 | 7/7 |
| Luna high / Astra low | 2/2 | 1/1 | 151.96 s | 139,683 | 1 | 7/7 |
| Luna high / Astra medium | 2/2 | 1/1 | 137.60 s | 139,602 | 1 | 7/7 |
| Luna high / Astra high | 2/2 | 1/1 | 152.04 s | 139,646 | 1 | 7/7 |
| Luna high / Astra xhigh | 1/2 | 1/1 | 231.98 s | 152,987 | 2 | 6/7 |
| Luna only low | 2/2 | 1/1 | 87.51 s | 86,572 | 0 | 7/7 |
| Luna only medium | 2/2 | 1/1 | 88.78 s | 86,563 | 0 | 7/7 |
| Luna only high | 2/2 | 1/1 | 94.79 s | 86,571 | 0 | 7/7 |
| Astra only low | 2/2 | 1/1 | 167.39 s | 144,166 | 1 | 7/7 |
| Astra only medium | 2/2 | 1/1 | 146.37 s | 144,072 | 1 | 7/7 |
| Astra only high | 2/2 | 1/1 | 162.44 s | 144,151 | 1 | 7/7 |
| Astra only xhigh | 2/2 | 1/1 | 227.40 s | 144,208 | 1 | 7/7 |

Each row covers the same three cases. Raw proposal errors can be hidden by successful guards; endpoint success is not proof that every model decision was correct. All call traces remain inspectable on the public page.

The measured call totals do not establish dollar-cost efficiency. They include startup and subscription-client overhead, and the cases are too few for a stable effort ranking. There is no demonstrated online recovery during manipulation in this experiment.

<!-- measured-results:end -->

## What the pilot suggests

All three pure-Luna controls passed all three endpoints, using 87.51–94.79 seconds of total model wait per configuration. Hybrid configurations used 137.60–231.98 seconds and three configurations failed the transient-occlusion case. The pure-Astra controls also passed all three endpoints.

For this small, simple observation gate, the added planner did not establish a quality benefit. Plan ambiguity and the lack of an observation reserve after final-step escalation are concrete issues to address before claiming fast/slow self-correction. This result does not imply Luna will be sufficient for longer-horizon manipulation tasks.

At the requested Astra-high setting, Luna low / medium / high passed 3/3, 2/3 and 3/3 respectively. Their total model waits were 165.64, 165.62 and 152.04 seconds. These are single matched pilot runs, not a statistically reliable ordering of effort levels.

## What is compared?

The main grid has **12 configurations**: Luna at low / medium / high crossed with Astra at low / medium / high / xhigh. Astra high is the requested default slow configuration; its column lets you compare all three Luna settings directly.

There are **7 additional controls**: pure Luna at each of its three settings, and pure Astra at each of its four settings. Pure Luna gets the same fixed initial task description without a generated plan. Pure Astra uses its chosen effort for both planning and decisions. This is a system-level comparison; generated plans are allowed to differ. It does not isolate decision quality under one identical generated plan.

Every configuration runs the same three cases:

| Case | Scene seed | Expected endpoint |
|---|---:|---|
| Normal visibility | 4601 | Complete the tower |
| Blue target hidden for 0.35 simulation seconds | 4602 | Reobserve, then complete the tower |
| Blue target remains hidden | 4603 | Explicitly stop without arm or gripper motion |

That makes **57 scheduled physical rollouts**. These are three matched case/seed combinations, one per condition. The seed differs between conditions, so condition effects and layout effects cannot be separated. The comparisons between model configurations remain paired on each exact case.

A separate low-effort Luna probe used the collaboration subagent interface with no inherited conversation or tools. A separate Luna-low / Astra-high smoke run tested the live adapter before the manifest was frozen. Neither is counted among the 57 trials.

## What does “subagent” mean here?

The probe explicitly used `collaboration.spawn_agent` with `gpt-6-luna`, low effort and no forked history. For the repeatable batch, each model call launches a fresh, isolated `codex exec` child session with an explicit model and effort, strict JSON output, tools disabled, a temporary working directory and no repository context. It does not reuse the root assistant's conversation or judgments. No Jev endpoint or credential is used.

The distinction matters: batch timing includes the **Codex child-session startup and prompt overhead**, not just model inference. The requested models are `gpt-6-luna` and `gpt-6-astra`; “fast” and “slow” describe their roles, not a guaranteed latency ordering.

## What information can the actor use?

The actor receives calibrated RGB-D and robot proprioception. The model receives only a strict summary: observation version and time, current phase, visible colors, table fit, grasp status and remaining budgets. It never receives simulator object poses, contact truth, fault schedules, success labels, or the private evaluator's result.

The model cannot change motion code, increase budgets or lower the confidence threshold. There are at most two extra observations, 200 ms apart, and two slow repairs. A repaired plan requires a fresh observation before movement. The 0.8 confidence gate is a fixed, uncalibrated development setting.

## Read the scores carefully

**Tower completion** and **correct abstention** are separate. Persistent occlusion should end without a tower. A static video can therefore be a successful stop test; it should not be counted as a successful tower build.

**Endpoint passes / 3** combines two completion checks and one stop check. It is useful for this pilot but is not a general success probability. With only one case per condition, do not infer a statistically reliable ranking.

**Rule agreement** compares a proposed action with fixed visibility rules. It is not independently labeled decision accuracy. **Slow repairs** counts actual replanning calls, not successful recoveries. **Guard interventions** includes legitimate escalation as well as rejected proposals, and records when the model's proposal is changed or escalated by the harness.

**Model wait** sums measured call durations. **Trial wall time** also includes simulation and local orchestration. **Simulation time** excludes model waiting. Two trials run concurrently on the same Mac in a fixed randomized order; lightweight tests and UI verification also ran during the campaign. Video rendering starts only after all model trials finish. Model load, client startup and machine contention can affect timings. There is no claim of a clean inference-speed benchmark.

**Input/output/cache tokens** come from returned CLI usage. Cached input is a subset, not extra tokens to add again. Dollar costs remain unknown under the subscription client; unknown must never be shown as zero. Failed provider calls are retained with unknown usage if no usage was returned. The harness does not automatically retry them.

## Reproduce one case

Use the installed Codex CLI with access to the requested models, Python 3.14 and the project environment. On macOS, use its native OpenGL backend; do not force Linux EGL.

```bash
uv sync
uv run python scripts/run_luna_astra_trial.py \
  --mode luna_hybrid --fast-effort medium --slow-effort high \
  --fault transient_occlusion --seed 4602 \
  --output outputs/luna_astra_example

uv run python scripts/render_luna_astra_trial.py \
  --trial outputs/luna_astra_example --endpoint tower_completion --speed 3 --fps 12
```

Use `--mode luna_only` for the fixed-plan control. For pure Astra, use `--mode astra_only` and set both efforts to the same value. Use `--fault none` or `--fault persistent_occlusion` for the other cases. Every trial needs a new output directory.

## Run the full frozen batch

```bash
uv run python scripts/run_luna_astra_batch.py --output outputs/luna_astra_batch --workers 2
uv run python scripts/render_luna_astra_batch.py --root outputs/luna_astra_batch --workers 2
uv run python scripts/export_luna_astra_results.py --root outputs/luna_astra_batch
uv run python scripts/serve_vision_demo.py --bind 127.0.0.1 --port 8093 --root web/recovery
```

Open `http://127.0.0.1:8093/luna-astra/index.html`. Resume skips terminal attempts, including failures; it refuses to silently rerun an unfinished directory. Source hashes are frozen, and source snapshots are retained locally. Full rollouts and sensor frames stay in the local output directory; the public page contains rendered video and whitelisted evidence only.

## A failure worth inspecting

In `Luna low / Astra medium`, transient occlusion on seed 4602 failed. The first two decisions correctly requested new observations. On the third, all colors were visible, but Luna requested escalation with confidence 0.99. Astra revised the plan; because no observations remained and a revision requires fresh evidence, the guard stopped before any motion.

The initial plan still contained “Currently blue is missing, so reobserve.” That stale assertion was present even though the plan also said to prioritize current evidence. This is an observable plan/state inconsistency, not proof of the model's hidden reasoning. The failure also exposes a harness constraint: **an escalation at the final observation has no remaining recovery path**. The experiment keeps this behavior frozen for all configurations rather than repairing it halfway through.

[Inspect this failed attempt](https://aspire-so100-recovery.vercel.app/luna-astra/?group=hybrid__luna%3Dlow__astra%3Dmedium&case=transient#demos). Its original proposal, revised plan and stationary recording remain available. High self-reported confidence did not prevent the failure. The same visible-target → escalation → stop pattern also occurred for Luna high / Astra xhigh on this case. These examples show that increasing effort alone did not resolve this protocol ambiguity in the observed attempts; they do not establish a general ranking of the models.

## What would make the next experiment stronger?

First separate transient scene facts from the plan using typed conditional rules, and specify a bounded observation reserve for applying repaired plans. Treat that as a new protocol and rerun all comparison groups; do not mix its results into this frozen batch. Then increase the number of independent scenes per condition and repeat each model configuration, without selecting only successful attempts. Then connect model decisions after grasp and placement verification and test recovery from a slipped block. To establish a fast/slow self-correction claim, record the complete chain: unexpected observation → useful Astra revision → fresh Luna decision → successful recovery. Compare this with fixed rules and a hybrid with replanning disabled.

## Verification

The main test suite ran 91 tests: 89 passed and two Linux-only checks were skipped on macOS. All 14 legacy recovery-contract checks passed. The [campaign audit](audit.json) verified all 57 model/effort routes, identical frozen execution sources, one shared policy candidate, and each exported video hash. All 58 videos (57 campaign plus one setup run) decoded without errors.

The English page was checked in the Codex in-app browser: configuration and case selection, direct case URLs, sorting, readable decision history, no captured console errors, and a 390-pixel mobile viewport without horizontal page overflow. A successful replay was observed playing, with its time advancing. These checks do not constitute cross-browser or real-robot validation.
