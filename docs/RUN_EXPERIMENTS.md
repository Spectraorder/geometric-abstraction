# Running a new geometry-abstraction experiment

This guide describes the earlier capable-starter comparison. For the completed
paper studies and current reporting commands, use the [documentation index](README.md)
and [script guide](../scripts/README.md). Study-specific protocols supersede the
defaults below.

Run commands from the `aspire_so100/` repository root, using Bash. This guide is
for a **new** Visual Base versus Full geometry comparison on the cleaned code.
It does not resume the budget-stopped historical study. Writing or reading this
guide does not launch a campaign.

The current comparison removes the supplied task-solving starter and recipe
advice. For new runs with explicit shared API guidance, use the
[interface-contract experiment guide](INTERFACE_CONTRACT_EXPERIMENT.md) and
`scripts/run_interface_contract_geometry_experiment.py`. The completed
[neutral-start variant](UNSEEDED_GEOMETRY_EXPERIMENT.md) remains separately
reproducible. Commands below describe the earlier capable-starter variant and
its historical protocol; recorded results remain unchanged.

## 1. Choose the comparison and budget

Use `scripts/run_constraint_geometry_experiment.py`. It supports explicit seeds,
revision count and separate execution phases.

- `basic`: raw side/wrist RGB-D, calibration, robot proprioception/CAD and action
  status. The policy may compute geometry from pixels, but receives no supplied
  object coordinates.
- `symbolic`: the same actuator plus supplied visual geometry, geometric operators
  and compact target checkpoints, including conditional partial constraints.
- Both start from `policies/support_graph_visual_starter.py` and use the same
  hidden RGB-D PnP actuator. Current task: `thin_beam_bridge`, four unique-color
  objects. Conditional motion prediction remains a separate research capability;
  its presence in the repository does not enable it in this comparison.

For `R` generated revisions per arm, `T` training layouts and `H` reserved layouts:

```text
Training slots = 2 × (R + 1) × T
Reserved slots = 2 × (R + 1) × H
Model generations = 2 × R
```

R0 is the common starter and is included in these counts. Every revision is tested;
training-selected versions reuse their reserved batch rather than adding a second
batch. Invalid candidates remain accounted slots and may not execute a rollout.

The current budget for new experiments is **three generated revisions per arm**,
plus the R0 baseline. The CLI already defaults to `--rounds 3`. The example below
uses **R=3, T=5, H=20**: 40 training slots, 160 reserved slots, 200 total slots and
six model generations when evaluating every revision. Evaluating only the two
training-selected policies instead requires 40 reserved slots under a separately
registered selected-policy evaluation; the standard evaluation command below
still evaluates all revisions. The five training layouts are
fixed across revisions and shared between arms. The 20 reserved layouts are also
shared; they are not a limit of 20 total rollouts. This is one independent learning
chain per arm. Reserved evaluation uses four parallel workers after global freeze.

### What R0 already knows

R0 is the shared prewritten starter, before any actor proposal. It is not an empty
or randomly initialized policy. `policies/support_graph_visual_starter.py` already
segments objects by color, back-projects calibrated RGB-D into world coordinates,
fits the tabletop, estimates visible top surfaces and follows the public support
graph to choose placement targets. Both arms call the same hidden RGB-D pick/place
actuator, which provides existing grasping and motion-control capability.

In `geometry_diagnostic_01`, this starter completed seeds 2026092601 and 2026092603
with all four required placements and a stable final structure: 2/5 successes in
both arms. `starter_parity.json` verifies matching physical trajectories and action
boundaries on all five training layouts. No model call generated R0. Its success
measures the supplied baseline's capability, not learning during this experiment.
The comparison therefore measures policy improvement from a capable shared
starter with different supplied observation interfaces, not learning manipulation
from scratch. The initial 40% is an observed five-layout result, not a preset rate.

### Camera framing and scene distribution

The side camera moves 24 percent closer along its existing optical axis, from
world `(0.48, -0.60, 1.10)` to `(0.3648, -0.528, 0.944)` meters. Vertical FOV remains
60 degrees and RGB-D resolution remains 320 by 240. Calibration is derived from
the actual camera pose; the physical camera mast and mount move with the camera.
At the original look-at point, projected linear size increases by about 32 percent.

Keep the existing generator for this comparison. Task-critical object centers
already lie within X `[-0.106, 0.106]`, Y `[-0.366, -0.194]` meters, including jitter.
Object footprints extend beyond these center bounds. The four-object bridge has
no extra distractors. Shrinking the area further could increase crowding, change
transfer distances and reduce layout diversity; it is not needed just for framing.

The offline diagnostic checks five development resets and projects conservative
workspace envelopes including object extents and building heights up to 0.60 m.
The closer view retains at least 38 pixels around the task workspace and 13 pixels
around the larger inventory envelope. These checks establish static framing only;
they do not prove visibility behind the robot, reachable grasps, or rollout success.

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/check_side_camera_framing.py --output outputs/camera_framing_check
```

Use a new experiment registration for the camera and iteration changes. Old camera
results remain historical evidence, not a controlled comparison with this protocol.

### Persistent actor sessions and selective visual feedback

New registrations default to `--actor-context persistent --actor-images on_failure`.
Each experimental arm has its own isolated Codex home, working directory and
session ID. R1 starts that session; R2 and later use `codex exec resume <SESSION_ID>`.
The runner never uses `--last` and never shares sessions between arms or studies.
This is a new protocol version, not a change to a registered historical campaign.

- **R1:** send the interface, frozen executor, common starter, R0 training feedback
  and initial visual grounding. The common selector uses at most two training
  layouts, with first/last side RGB and depth previews: up to eight attachments.
  Both arms use the same selection rule. R0 itself makes no model call.
- **R2 onward:** append latest training outcomes, event-centered checkpoints,
  training-only retention decisions and the selected parent's hash. Rejected
  proposals remain in history; the exact incumbent is resent when restoration is
  needed. Earlier images remain historical evidence in the persistent session.
- **Supplemental images:** with `on_failure`, use only common action errors/notes
  and success flags to identify object/tool mismatch, insufficient current views,
  or unexplained failure. In registered seed order, take the first eligible layout
  with images. Attach at most two existing RGB frames (one side, one wrist) nearest
  and no later than its first recorded exception, or the last acquisition if no
  exception was recorded. Log trigger, seed, timestamps and calibration in the
  audited prompt. The same selector and cap apply to both arms, although actual
  failures and attachment counts may differ. There is no extra model invocation.
- **Text evidence:** every layout contributes its first explicit failed action,
  nearby same-object observations and, when available, a same-arm successful
  reference. References are historical examples, not measurements of another
  scene. A 16,000-character cap drops optional references first, then shortens
  each layout equally; it never replaces missing current records with old ones.
- **Visual memory:** the session and original image assets are persisted. The actor
  is also asked to retain useful visual observations and unresolved uncertainty in
  its hypothesis. Old images, positions and relations remain historical evidence,
  never current measurements. Context compaction may summarize earlier content;
  session persistence does not guarantee permanent access to every original pixel.
- **Accounting and audit:** each invocation retains its new prompt, raw events,
  attachment hashes and count, raw token usage, elapsed time, session ID and an
  archived session transcript. Before resuming, the live transcript must match the
  preceding completed turn. The auditor verifies same-arm history continuity.
  The prompt character cap applies to the newly submitted text, not the accumulated
  model context. Compaction records are counted in archived session history.
- **Interruptions:** a failed or partially completed call is recorded and cannot be
  silently retried. Investigate the attempt and session state before continuing.
  Authentication is copied only for the invocation and removed afterwards; it is
  never part of the archived transcript or audit metadata.

`--actor-context fresh` retains independent calls as an explicit comparison option.
`--actor-images initial_only` disables supplemental images;
`--actor-images every_revision` retains the repeated-image schedule. These choices
are independent and must be registered before execution. In fresh/initial_only
mode, later calls receive historical hypotheses but do not inherit the first turn's
images. Keep both settings matched across arms for the geometry comparison.

Fewer new attachments do not imply zero historical-image processing or zero input
cost: the resumed history still contributes to model context, and cache behavior
must be measured from reported usage. Compare input, cached input, output/reasoning
usage, image counts and elapsed time alongside success. The runner still modifies
Python policies between rollouts; it does not measure an online LLM decision at
each robot action. Local RGB-D acquisition continues with either image schedule.

Supplemental images are a deterministic host-side escalation, not an interactive
actor image-request API. Report their actual counts and token costs. A separate
matched-image ablation is needed to isolate the effect of fewer images.

Keep the five shared adaptation cases separate from both the five training
layouts and the 20 reserved layouts. If their feedback is used to modify policies,
they are adaptation development cases, not held-out tests. A proposed two-revision
adaptation pilot would add 30 rollout slots across both arms, including the
pre-adaptation baseline. Its orchestration, image schedule and budget need separate
registration; the commands in this guide do not implement that phase. Unseen
target-distribution layouts are needed to assess adaptation generalization.

### What one iteration means

One iteration is one policy-improvement round **per arm**:

1. Start or resume that arm's own registered actor session. Supply new training
   feedback and allowed checkpoints, with images according to the registered schedule.
2. Invoke the actor once to generate a complete revised policy.
3. Evaluate that policy on the same registered training layouts.
4. Accept the candidate only if it preserves every incumbent training success
   and repairs at least one failure. Retain rejected candidates and their outcomes
   as evidence; the next proposal revises the incumbent.

It is not a robot action, an action-boundary checkpoint, a rollout or a gradient
update. With five training layouts, advancing both arms by one revision requires
two model generations and ten training slots. R0 evaluation is additional;
reserved evaluation happens after freezing and never supplies actor feedback.
Infrastructure failures can create extra invocation attempts, which must be
reported separately from completed generations. Repeated training layouts are
additional interactions, not additional distinct scenes.

### Is iteration count a useful metric?

Iteration count is a useful controlled budget axis, but not a standalone measure
of computational efficiency. Prompt histories grow, output/reasoning token counts
vary, and one revision can range from a small parameter change to a policy rewrite.
Equal call counts therefore do not guarantee equal computation. Revisions can also
regress: in the latest tested chain, Full R1 scored 75% on reserved layouts while
the training-selected Full R3 scored 65%.

The primary research question should be:

> At matched model-call and training-rollout budgets, does supplied geometric
> abstraction produce higher held-out success than Visual Base?

For a new study, register the following reporting choices before execution:

| Measure | Interpretation |
| --- | --- |
| Held-out SR at fixed proposal budgets, such as R0, R1 and R2 | Primary comparison of policy improvement under matched opportunities |
| First registered budget reaching a predeclared SR threshold | Secondary description of improvement speed; report runs that never reach it |
| Actual input, output and reasoning tokens, where reported | Resource accounting; equal iteration counts are not equal token costs |
| Training slots, executed rollouts, invalid slots and distinct layouts | Distinguish interaction budget from dataset size and execution failures |
| Completed generations and all invocation attempts | Keep service interruptions visible without treating them as learned revisions |

Keep the model, reasoning settings, configured limits, training layouts, feedback
selection rules and actuator matched between arms; the supplied geometry is the
intended treatment difference. Log actual resource use rather than assuming these
controls equalize it. Provider token categories may overlap; do not add reasoning
tokens to output tokens unless the accounting explicitly treats them separately.

At each proposal budget, report held-out SR of the **guarded incumbent** and the
raw candidate separately. Start with R0. Accept a candidate only if its successful
training seeds strictly contain the incumbent's successful seeds. Ties and any
lost success reject the proposal, even if total success increases. Record repaired
and regressed seeds relative to the incumbent. Selection, proposal ancestry and
final freezing use the same rule; audit reconstructs it from training outcomes.

This deliberately conservative rule can block useful tradeoffs and plateau on
five layouts. It prevents observed training regressions, not unseen-test regressions.
Scores range from 0/5 to 5/5. Never select versions or stop training using reserved
results. All candidates, including rejected ones, retain their registered test slots.

Ask the actor for one falsifiable failure hypothesis and identify the affected
phase. Perception, worksite selection, action order and justified recovery may
change; the shared actuator and its acceptance checks remain fixed within a run.
Acceptance still uses the same strict training-success retention rule.

Require `policy_probe` records for both activation and nonactivation of the
proposed branch, including its hypothesis, step, object, trigger reason and
available before/after error. Own ad hoc diagnostics also return in feedback, with
explicit unverified-policy provenance. At most four custom records per layout are
selected, each limited to 1,600 characters; omitted counts and truncation are
visible. Under the overall 16,000-character cap, optional successful references
are dropped first, custom diagnostics reduce to one per layout, then checkpoint
history shortens equally. Invalid candidates never borrow earlier diagnostics.

For each revision, `hypothesis_review.json` compares the issued motion, gripper
and wait commands against that proposal's incumbent parent, per training layout.
Sensor-only additions do not count as changed actions. The next actor sees command
counts, equality and first changed index. Branch reports are policy claims, while
command comparisons are reconstructed from host RPC logs. Neither certifies
physical trajectories, grasp security or task success. The variant audit checks
these reviews against the recorded calls, including the final proposal.

### Interpreting observation errors

- `Observed object is too far from the expected tool neighborhood`: the measured
  center differs from the expected tool neighborhood by over 0.12 m. Slippage,
  association error, or an inconsistent estimate are possible; the message alone
  does not distinguish them. Keep the threshold. Never infer a secure grasp from
  the executor's `held` flag alone or continue placement using an old pose.
- `No usable current view`: neither current side/wrist evidence nor joint cuboid
  correspondence supports an accepted pose. Occlusion and insufficient visible
  faces can cause this without object loss. Preserve uncertainty. Learn visual
  dimensions before occlusion when possible; recover with fresh permitted evidence
  and at most one justified retry, then stop if still unresolved.

The v4 shared executor catches both `ValueError` and `RuntimeError` during its
optional post-release observation. It returns `released`, `held=False` with
`observation_unavailable` in the rich result when that fit fails. Base retains its
status/held-only projection. Pre-release failures still propagate, and release
never certifies stable placement or task success. This changes failure reporting,
not pose acceptance thresholds or physical release conditions.

The common starter records a pick/place exception checkpoint and diagnostic
note before propagating the error. It does not silently convert failure to success
or insert a new physical recovery action. Generated Full policies are instructed
to refresh after waiting/moving before logging settled geometry; Base uses fresh
pixels and its own measurements. Partial bounds and cached poses never replace a
current executable observation. A closer camera helps pixel coverage but cannot
by itself fix slippage or guarantee an unoccluded view.

A result can support policy-improvement efficiency in model proposals and training
interactions without demonstrating faster individual reasoning. Token use and
reasoning latency need separate measurements. Rollout elapsed time is not the
current optimization objective. A positive single-chain result remains preliminary;
independent learning chains are needed to assess repeatability.

## 2. Check the environment

```bash
uv sync --locked
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MUJOCO_GL=egl
command -v bwrap
command -v codex
df -h .

.venv/bin/python -m unittest discover -s tests -p 'test_visual_input_session.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_partial_geometry.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_constraint_feedback.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_geometry_actor_session.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_geometry_iteration.py' -v
```

Linux policy isolation requires `bwrap`; headless rendering requires a working EGL
setup. Policy/perception computation stays on CPU. The registered minimum free disk
check is 3 GiB, but allow substantially more for RGB-D frames and rollout logs.

The actor currently invokes the installed `codex` CLI, with model `gpt-6-astra`
and high reasoning effort. It copies the existing local CLI `auth.json` into an
isolated per-arm environment for each invocation when present, then removes the
copy. Ensure this account can access the configured model. This experiment path does **not** read `.apikey` or use the
old OpenRouter/Molmo pointing route. Actor calls transmit prompts and selected
training images; they are model calls, unlike the local preparation/audit steps.

Model, task and interface defaults are set in the experiment's Python source,
not exposed as CLI options here. Make such changes **before registration**, then
use a new output directory. Do not edit the registered manifest to change a run.

## 3. Register a fresh experiment — no rollout or model call

Choose an unused output directory and disjoint seed sets. The seeds below are
examples; check against your previous used/inspected layouts before treating the
reserved set as fresh. Training feedback must never include reserved outcomes.

```bash
EXPERIMENT_DIR=outputs/geometry_three_round_01

.venv/bin/python scripts/run_constraint_geometry_experiment.py \
  --output "$EXPERIMENT_DIR" \
  --phase prepare \
  --rounds 3 \
  --actor-context persistent \
  --actor-images on_failure \
  --eval-workers 4 \
  --train-seeds {2026092601..2026092605} \
  --reserved-seeds {2026122701..2026122720}
```

Preparation writes `manifest.json`, the common starter, shared executor, both
geometry libraries and source snapshots. Check the manifest's `rounds`,
`train_seeds`, `heldout_seeds`, `arms`, `model`, `recipe`, `actor_context_mode` and
`actor_images` before execution.
The output directory must not already exist. The example registers five training layouts and 20 fresh reserved layouts.
The previously inspected 20 test layouts are development evidence for subsequent
changes and must not be presented as a new untouched test set.

## 4. Check the shared starter, then train

These commands execute simulation. Training also makes the model calls.

```bash
.venv/bin/python scripts/run_constraint_geometry_experiment.py \
  --output "$EXPERIMENT_DIR" --resume --phase bootstrap

.venv/bin/python scripts/run_constraint_geometry_experiment.py \
  --output "$EXPERIMENT_DIR" --resume --phase train

.venv/bin/python scripts/run_constraint_geometry_experiment.py \
  --output "$EXPERIMENT_DIR" --resume --phase audit
```

Bootstrap runs the two R0 arms on training layouts and checks matching physical
trajectories/action boundaries while verifying Base has no supplied geometry.
Training reuses complete R0 results; bootstrap does not add slots to the budget.

Training then generates each arm's revisions using only its own training history.
On completion, expect `training_summary.json` and `all_candidates_frozen.json`.
Selection uses the guarded incumbent rule described above, with ties retaining
the incumbent. `training_summary.json` retains every raw proposal.
Do not manually repair an actor candidate or select a version using reserved SR.
The phase audit writes `audit.json`; during training it is intentionally partial.

## 5. Evaluate frozen versions

Only proceed after training, the freeze file and the training audit are complete.
This phase executes the registered reserved slots with the registered
`evaluation_workers` (default four) and makes no model calls. Results retain seed
order regardless of completion order. Parallel rollout times are not a controlled
policy-computation latency metric.

```bash
.venv/bin/python scripts/run_constraint_geometry_experiment.py \
  --output "$EXPERIMENT_DIR" --resume --phase evaluate
```

Both arms and all revisions must be frozen first. Keep all outcomes, including
regressions. Changing the code, libraries or seeds requires a new registration.
The old `outputs/constraint_geometry_learning_replications_01` remains stopped;
none of these commands should point to it.

## 6. Audit the complete result

Use the constraint variant's auditor, rather than a historical variant's CLI.
This strict check rejects an incomplete result instead of presenting it as final.

```bash
PYTHONPATH=scripts .venv/bin/python - "$EXPERIMENT_DIR" <<'PY'
import json
import sys
from pathlib import Path
from run_constraint_geometry_experiment import audit

root = Path(sys.argv[1]).resolve()
report = audit(root, allow_partial=False)
if report['status'] != 'completed':
    raise RuntimeError('The study is not complete')
(root / 'audit_final.json').write_text(json.dumps(report, indent=2) + '\n')
for arm, curve in report['curves'].items():
    print(arm, 'selected revision:', curve['selected_revision'])
    print('Reserved curve:', [(row['revision'], row['sr']) for row in curve['reserved']])
PY
```

Read `summary.json` for the execution summary and `audit_final.json` for verified
curves/provenance. Report R0 and every revised policy, the training-selected
versions, invalid slots and model-attempt counts. Do not replace the selected
version with whichever later looks best on the reserved set.

## Progress, interruptions and larger studies

`progress.json` records the current phase. Each arm has `revision_000/`,
`round_001/`, etc., including submitted prompts, model logs, candidates and trial
records. Persistent actor state is in `<arm>/.actor_session/`; keep it for
continuation and never copy it between arms. Per-turn `session.jsonl` snapshots
support offline auditing. The selected reserved batch is in `<arm>/heldout/`; other revisions are
in `<arm>/reserved_revisions/revision_NNN/heldout/`.

Resume the same phase with `--resume` only after confirming the prior process has
ended. Complete trials/generations are reused. A directory left by an incomplete
model call or rollout may need explicit investigation; do not delete it blindly,
start a second coordinator, or change candidate bytes to make an audit pass.
An early-stopped run must be reported as incomplete, retaining all finished trials.

For independent replication, use isolated chains with disjoint training layouts,
register the common fresh reserved set in advance, and finish/freeze **all** chains
before evaluating any of them. Report per-chain differences; repeated layouts and
versions are not independent learning replications.

Do not use `run_constraint_geometry_replications.py` as a generic small-run command:
it still registers historical fixed seeds, three chains and 480 reserved slots,
and expects a local authorization receipt from the previous study. A new automated
multi-chain schedule needs a deliberately updated registration/budget first.


## Reduced budget and parallel evaluation for the active campaign

The user reduced `outputs/geometry_persistent_01` from 50 to 20 shared reserved
layouts after evaluation had begun. Serial dispatch was stopped, its in-flight
rollout was allowed to finish, and 182 completed reserved rollouts were retained.
The original manifest, trained policies and freeze gate remain unchanged.

`evaluation_amendment.json` registers the first 20 seeds in the original order,
independent of outcomes, and four concurrent rollout workers. Existing results
are reused. Completing the reduced comparison requires only 40 additional R2
rollouts; no further model calls or training are required. The balanced comparison
contains 120 slots (two arms, three versions, 20 layouts). Previously completed
results outside this subset remain separate evidence and count toward actual cost.

The continuation uses `scripts/run_budget_limited_geometry_evaluation.py` and a
coordinator lock. Do not launch another coordinator while it is running. Its log
is `parallel_eval.log`, and `lifecycle.json` records progress. It writes
`summary_budget_limited.json` and `audit_budget_limited.json` after all reduced-scope
slots pass audit. This is completion of the amended scope, not completion of the
original 300-slot reserved protocol. Parallel evaluation changes resource
contention; do not interpret its per-rollout elapsed times as a controlled latency
comparison with the earlier serial batches.

## Diagnostic-feedback continuation

`outputs/geometry_guarded_01` completed 30 training rollouts and four model
generations with 2/5 success in both arms at every revision. Its candidates,
contexts and results remain unchanged. `outputs/geometry_diagnostic_01` is a new
v4 registration using the same five development layouts, closer camera and spawn
distribution, with two generated revisions per arm. The new run makes another
30 training slots and four model calls; these are additional costs, not a rewrite
of the first run. It starts fresh actor sessions because the executor contract
and feedback protocol changed. It does not ingest prior candidates or reserved
outcomes. The previously registered 20 reserved layouts remain unused; this
continuation launches training only.

The v4 continuation completed all 30 training rollouts and four model generations.
Training SR was Base 2/5 at R0, R1 and R2; Full was 2/5, 2/5 and 4/5. The guard
retained Base R0 and accepted Full R2, which repaired development seeds 2026092604
and 2026092605 without losing either initial success. Seed 2026092602 still failed.
Four exact actor contexts and four hypothesis reviews passed audit. No reserved
rollout was run, so this is a training improvement, not evidence of held-out
generalization or an isolated causal effect of any one protocol change.

Both arms attached eight images at R1 and two at R2. Across two generations, Base
used 57,337 input tokens (22,144 cached) and 14,986 output tokens; Full used 62,764
input tokens (35,328 cached) and 13,009 output tokens. These corrected figures use
per-turn differences of cumulative session counters; the earlier sum double-counted
R1. Raw generation records and the original review remain unchanged. Reasoning tokens are included
in output and must not be added again. Summed generation wall time was 456.6 s
for Base and 398.4 s for Full; calls ran concurrently and latency remains a
single-run observation. See `outputs/geometry_diagnostic_01/training_review.json`
for the original selection review and `audit.json` for provenance checks. Its
resource sums are superseded by the counter correction above.

The user subsequently authorized the complete reserved evaluation. All 120
reserved rollouts finished, and `audit_final.json` reports 150 total valid
rollouts, zero invalid slots and completed provenance checks. Training-selected
Base R0 scored 6/20 (30%); Full R2 scored 9/20 (45%). See
[the completed result](history/early_geometry/GEOMETRY_DIAGNOSTIC_RESULTS.md) for full curves, paired
regressions, resources and limits. The prior statements about an unused reserved
set describe the earlier training-only phase.

## Extending the completed chain to five iterations

The user authorized **five total generated revisions per arm**, continuing the
existing R1/R2 history with R3, R4 and R5. The registration is
`outputs/geometry_diagnostic_01/extension_05/amendment.json`. This adds 30 training
rollouts and six model generations on the same five training layouts. Both actors
resume their own existing sessions; Base initially revises retained R0 and Full
initially revises retained R2. Original manifests, summaries, frozen policies,
model turns and reserved results remain unchanged. Only live session state advances.

`scripts/extend_geometry_training.py` checks the original completed audit, hashes
the extension runner and retained evidence, and uses an effective in-memory round
budget without editing the original manifest. New round directories are added
to the existing arm chains. Extension progress, selected policies, freeze gate,
training summary and audit are separate under `extension_05/`. Candidate acceptance
still uses training-only strict success-set improvement. The final extension
audit reconstructs prompts, images, session lineage, model usage, training outcomes,
hypothesis reviews and policy selection, and rechecks the original completed study.

```bash
.venv/bin/python scripts/extend_geometry_training.py \
  --root outputs/geometry_diagnostic_01 --rounds 5 --phase audit
```

This continuation was requested after the original reserved results were inspected,
so it is exploratory and must not be described as a preregistered five-round
comparison. No reserved outcomes or human diagnosis enter the actor prompts. The
extension runs training only; future confirmation needs a separately registered
fresh reserved split. Its new costs must be added to the previous study's costs.
Do not run the original two-round training command against this extended live
session, start a second coordinator, or silently retry an incomplete model turn.

The five-round continuation completed and passed audit: 30 additional valid
training rollouts, zero invalid slots and six additional actor turns. Base R5
reached 4/5, matching Full's retained R2; R3/R4 did not improve either arm and
Full R5 also remained at 4/5. No new reserved evaluation was run. See
[the five-round report](history/early_geometry/GEOMETRY_FIVE_ROUND_TRAINING.md) for complete curves,
remaining failure diagnostics and corrected token accounting. Persistent-session
CLI usage is cumulative: use verified per-turn counter differences, not sums of
`generation.json` usage fields. `extension_05/resource_accounting.json` provides
those differences without modifying the raw records.

## Fresh evaluation of the five-round selected policies

The user subsequently authorized test evaluation. A new 20-layout shared split
(seeds 2026112701–2026112720) was registered before execution under
`extension_05/fresh_evaluation_20/registration.json`. Only training-selected Base R5
and Full R2 were evaluated: 40 rollouts, four workers, no actor calls or policy
changes. All rollouts and provenance checks passed. Base scored 15/20 (75%);
Full scored 14/20 (70%). The paired outcomes were 13 shared successes, two Base-only
successes, one Full-only success and four shared failures (exact McNemar p=1.0).
See [fresh test results](history/early_geometry/GEOMETRY_FRESH_TEST_RESULTS.md) for runtime metrics,
limitations and reproducible audit instructions. Earlier statements about no new
reserved evaluation describe the training phase, before this authorization.

## Current iteration budget after the five-round pilot

The user set the budget for subsequent experiments to three generated revisions
per arm (R1–R3), with R0 evaluated separately as the shared baseline. The existing
five-round continuation and fresh test results remain complete historical evidence;
they are not truncated or reclassified as a three-round experiment. The budget
choice followed inspection of the pilot results and should be described as a
prospective cost decision, not independent evidence of a three-round advantage.
Updating this guide does not start another training campaign.

## Completed neutral-start training

`outputs/geometry_unseeded_01` completed three model turns per arm. Training
selected Base R3 (2/5) and Full R2 (3/5), with 35 executed training rollouts and
five invalid candidate slots. All six actor contexts and hypothesis reviews passed
audit. Reserved evaluation remains unstarted. See the
[neutral-start training results](history/early_geometry/UNSEEDED_GEOMETRY_TRAINING_RESULTS.md) for the
generated-code interface errors, resource totals and interpretation limits.

The user subsequently authorized the registered neutral-start test evaluation.
All 160 reserved slots are complete: 140 executed rollouts and 20 invalid Base R1
slots. Final selected Base R3 scored 6/20 (30%); Full R2 scored 12/20 (60%). Full R3
scored 10/20 and remains rejected under the training-only guard. The paired selected
comparison has five shared successes, seven Full-only successes, one Base-only
success and seven shared failures (exact McNemar p=0.0703125). Final audit passed.
See [the neutral-start test results](history/early_geometry/UNSEEDED_GEOMETRY_TEST_RESULTS.md). Earlier
statements about unstarted reserved evaluation refer to the training phase.

## Completed five-round shared-interface training

`outputs/geometry_interface_contract_01` completed R1–R5 with the explicit shared
interface contract. Base selected R4 at 3/5 and Full selected R4 at 4/5. The previous
module-assignment and action-status-type errors did not recur. Both R5 candidates
exceeded the source budget and remain invalid slots; the retained R4 policies are
unchanged. The audit verifies 50 executed training rollouts, ten invalid slots and
ten actor turns. Reserved evaluation remains unstarted. See
[the training report](history/early_geometry/INTERFACE_CONTRACT_TRAINING_RESULTS.md) for resource totals
and the cost of extending beyond three rounds.

## Expanded source limit for new interface-guided runs

The policy source limit for subsequent interface-guided experiments is now
**50000 characters**. Use `scripts/run_expanded_budget_geometry_experiment.py`
for preparation, training, evaluation and audit. Both arms receive the same new
limit, which is enforced by the validator and recorded in the manifest. See the
[configuration and launch instructions](INTERFACE_CONTRACT_EXPERIMENT.md#expanded-source-budget-for-subsequent-runs).
The earlier 30000-character entry point and completed results remain frozen;
this change does not retry R5 or start another campaign.

## Completed shared-interface selected-policy test

The user subsequently authorized reserved testing of the frozen selected policies.
The separate `geometry_interface_contract_01/selected_evaluation_20` registration
completed 40 rollouts on 20 shared layouts with four workers: Base R4 scored
12/20 (60%), Full R4 scored 8/20 (40%). Paired outcomes were eight shared successes,
four Base-only successes and eight shared failures (exact McNemar p=0.125).
Audit passed; evaluation took 84.08 seconds and made no model calls. See
[the test report](history/early_geometry/INTERFACE_CONTRACT_TEST_RESULTS.md) for efficiency measurements,
failure accounting and limitations. The original full-revision test protocol
remains incomplete; the newer 50000-character configuration was not evaluated.

## Parallel five-replica study

The user authorized `outputs/geometry_parallel_01`: five independent chains per
arm, three generated rounds, five paired training layouts per replica and 100
common fresh test layouts. Its 50000-character variant uses global limits of four
model requests and four simulator rollouts. All training must freeze before any
reserved tests. The budget is 30 model calls, 200 training slots and 1000 selected
policy test rollouts. See [parallel study operations](PARALLEL_GEOMETRY_STUDY.md)
for launch, monitoring, failure handling, audit and analysis rules.

## Per-iteration curves for the parallel study

The user authorized a post-hoc R0-R3 curve extension under
`geometry_parallel_01/learning_curves`: reuse 1000 selected-policy slots and add
3000 rollouts on the same 100 layouts, with no model calls. The global rollout
limit is up to 16, subject to a 32-rollout hardware pilot. New data is stored on
`/mnt/ssd`, leaving original evidence unchanged. See
[the full curve protocol](GEOMETRY_ITERATION_CURVES.md) for monitoring, raw versus
training-retained curves, threshold definitions, token plots and interpretation.
