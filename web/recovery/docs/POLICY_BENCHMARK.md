# Policy completeness benchmark

A policy is complete **for a declared task distribution and observation contract**, not for every arrangement of blocks. This proposal does not retroactively certify the existing policies.

## Capability profiles

The machine-readable protocol is `benchmarks/policy_suite.json`; runnable and proposed tasks are catalogued in `benchmarks/task_catalog.json`.

- **tower_recovery**: nominal tower, randomized tower, transient occlusion, persistent-occlusion abstention, missed grasp, lift loss.
- **inventory**: upright prism with clutter, plus sampled inventory with 2, 6, and 10 objects.
- **all_supported**: all ten cells. An implementation can be a frozen, task-dispatched policy bundle; its hash must identify the entire bundle, not the winning file per scene.
- **future extensions**: partial two-tower repair, two-support bridge, human edits, and transport-time identity loss. These are not implemented and cannot contribute a passing score.

For each cell, assess the whole chain: perceive → bind the correct object → plan → execute → verify → recover or stop. A frame recognizer alone is not a complete policy, nor is a successful prerecorded video.

## Evaluation protocol

Freeze the policy bundle, perception, controller, model IDs, prompts, thresholds, memory, harness, evaluator, fault timing, and case manifest before evaluation. Reserve independent scenes that have never contributed to repairs. Seeds 4001/4002 and every historical development or inspected failure are ineligible as new heldout evidence. Generate a new manifest only after auditing previous seed use.

Fault timing is part of the benchmark. For model comparisons, use simulation-time windows or registered physical events; camera-call-based occlusion makes observation frequency change the fault. Existing broad-mask pilots and precise target-only tests remain separate historical protocols.

Run groups on exactly the same initial scenes and fault schedules. Randomize execution order; do not compare a loaded concurrent run with an otherwise idle serial run. Repeat stochastic calls, but cluster confidence estimates by scene. The scorer conservatively marks a scene successful only if all recorded repeats meet the endpoint. Never count three repeats of one scene as three independent scenes.

Suggested initial release criteria: at least 50 independent scenes in every required cell, a 95% Wilson lower confidence bound ≥0.90, and zero observed unsafe actions. These are proposed thresholds, not a power analysis or a population guarantee. Report endpoints and coverage per cell; an aggregate rate must not hide a failed recovery category. Abstention cells score a correct bounded stop, not completion of the tower.

Missing cases, failed cases, timeouts, and infrastructure failures stay visible. The primary intent-to-test denominator includes every attempted scene. Report a separate valid-execution diagnostic only with its exclusions enumerated; never silently discard infrastructure failures.

## Information boundary

Actor inputs: RGB-D, optional wrist RGB, camera calibration, robot encoders/tool kinematics, known task contract. Object pose truth, scene identities, contact truth, future faults, and evaluator verdicts are private. Geometric vertices and supports inferred from images must be labeled as estimates. RGB-D-derived structured state is still sensor-only; simulator state serialized as JSON is not.

Keep source frames and serialized request hashes so a reviewer can trace each estimate. The current initial-observation gate passes a strict whitelist of colors, table visibility, phase, timestamp, and remaining budgets. Its limited schema is not yet enough for full task-level replanning.

## Metrics

1. **Task**: final structure success, correct identity, recovery conditional on actual fault occurrence, correct abstention, unsafe actions, damage to completed structures.
2. **Decision**: independent allowed-action-set accuracy, raw invalid proposals, guard interventions, unnecessary/missed escalation, stale replies. Astra agreement is not ground truth. Existing snapshot labels measure rule agreement only.
3. **Time**: episode wall time, physics time, cumulative model wait, decision p50/p95, time to recovery, extra motion/observations. The simulator pauses during calls; this does not establish real-time hardware performance.
4. **Cost**: all calls and attempts, input/output/reasoning/cache token counters, provider-reported charges and separately labeled estimates. Total cost / number of successful tasks includes failed attempts; undefined at zero successes. Codex subscription dollars remain unknown, not zero.
5. **Coverage**: tested vs untested cells, seed/fault/geometry distributions, stopped cases, confidence intervals and failure categories.

## Fast/slow comparison

Decision-level experiments give all groups identical frozen plans, state summaries, options and budgets. System-level experiments compare all-Astra, Astra-plan/Jev-decision, and Jev with a frozen task graph. Jev cannot generate arbitrary plan text; that limitation must remain explicit. Add a rule-only baseline and hybrid-without-plan-repair ablation. Freeze thresholds on development risk/coverage curves; the current 0.8 gate is an uncalibrated development setting.

A self-correction claim needs an observed failure, a slow-plan revision, a fresh observation, resumed valid action, and improved private task outcome. Compare the same disturbed scenario with repair disabled. Do not infer this claim merely from the presence of two model names.

## Executable completeness audit

```bash
uv run python scripts/benchmark_policy.py \
  --profile all_supported --output outputs/readiness_new.json
```

With no records, this intentionally reports `incomplete_or_below_target`. Supply `--records` with a JSON object containing `records` and `attestations` to audit a frozen suite. Each record requires cell, case_id, repeat, split=heldout, status, passed_endpoint, unsafe_actions, policy_sha256, harness_sha256, evaluator_sha256, configuration_sha256. Legal statuses are completed, policy_failed, infrastructure_error, timeout; only a completed private evaluation can pass. `completed` includes an evaluated bounded-stop outcome in an abstention cell.

Attestations are explicit booleans for all named protocol gates. They are reviewable declarations, not a cryptographic proof; keep evaluator reports and input-boundary audit evidence alongside them. The scorer checks coverage, duplicate attempts, version mixing, independent scene counts, and release thresholds. It does not fabricate trials or supply missing ground-truth labels.
