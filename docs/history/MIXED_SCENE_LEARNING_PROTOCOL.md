> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Mixed-shape geometry learning pilot

**Current status: completed.** The final corrected mixed-shape pilot has 300 valid
rollout records and zero invalid slots; the full source, pairing, and freeze audit
passes. Training-selected reserved results are basic 0/20 (R0), numeric 8/20 (R3),
and symbolic 12/20 (R2). This is one actor chain per arm; numeric versus symbolic
has paired p = 0.125 and still requires independent replication. See the
[completed results and full learning curves](MIXED_SCENE_LEARNING_RESULTS.md).
The chronological checkpoints below preserve the development and repair history;
their earlier pending/running descriptions are no longer the current status.

This pilot tests a symbolic/metric geometry interface in an agent that revises a
high-level robot policy. It does not assume the abstraction will improve learning.
The earlier replicated three-cube study did not demonstrate an iteration benefit.
Manual mixed-shape development results are retained separately and are not actor
training examples or evidence from this pilot.

## Contrasts

| Arm | Supplied observation and state | Additional reasoning interface |
|---|---|---|
| basic | Historical RGB-D visible-top PnP estimates and a freshness-aware surface packet | None |
| numeric | Current mixed rigid RGB-D estimates, measured dimensions, vertices/axes, availability, last-two-estimate history and separate desired poses | Metric values only |
| symbolic | Exactly the numeric observer, state and PnP executor | Typed face/axis/point references, three-valued geometric relations, generic desired alignment and text serialization |

Basic versus numeric tests the combined perception/state package. Numeric versus
symbolic isolates the supplied reasoning interface and its feedback serialization.
This is not a language-versus-vision contest: all actors receive side RGB/depth
examples under the same selection rule, and all runtime policies can query RGB-D.
The basic arm already receives some metric abstraction. Neither control is called
"no 3D information."

All three arms start with the identical `support_graph_starter.py`. It anchors the
worksite at the first visually observed foundation, builds the remaining public
support graph once, and stops after an unconfirmed release. No previously developed
workspace planner or successful completed policy is supplied. Shared PnP motion,
robot-only IK and action methods are frozen. The observer is loaded in a separate
namespace; actor functions cannot override its implementation. Public observations
are copied to avoid mutation of cached dimension models.

## Registered distribution and budgets

- One four-object `thin_beam_bridge`: two cubes, a flat thin-prism beam and a cylinder.
  This tests the requested low-rise multi-support structure, not increased height.
- Unique colors with a deterministic cyclic permutation based on seed modulo four.
  Shape/color goals are public; metric dimensions must be measured on the table.
- Five training seeds: **76001–76005**, identical across arms and revisions.
- Twenty reserved seeds: **77001–77020**, identical across all frozen revisions.
  An exact result/manifest search found no earlier uses before registration.
- Three model revisions after R0, one independent generation chain per arm.
- Model `gpt-6-astra`, high reasoning, 900-second generation timeout. Tools, skills,
  apps, repository context and user configuration are disabled in model subjects.
- Maximum 30,000 planner characters, target 14,000; maximum 120,000 prompt characters.
  Actual token usage is recorded rather than forced equal. Invalid submissions
  consume the revision and its rollout slots, with no manual repair/free rewrite.
- Fixed simulation physics (`noslip_iterations=0`), ideal calibrated RGB-D with
  pixel-center depth and side/wrist sensors, CPU policy computation, GPU simulation.
  Runtime limit 120 wall seconds and 1,200 API calls. Duration is not an outcome.

Actors see only their own training binary outcomes, errors, bounded notes and
hypotheses, latest exact source, and first/last side RGB/depth for up to two training
examples (first failure, first success, then seed order). For the same examples,
initial/final own `planner_state` packets are supplied. Invalid latest revisions
fall back to the most recent actual rollout per training seed. No private poses,
contacts, dimensions, evaluator reasons or reserved results enter actor feedback.
The full executor source is common context; each arm's public state API is documented.

## Freeze and evaluation

The source manifest and complete runtime source snapshot are written before any
new training rollout. Every trial checks source, library, candidate, seed and sensor
settings. All twelve policies (R0–R3 for each arm) are archived and hashed before
any reserved rollout. Each arm's final selection is the highest training SR,
breaking ties toward the earliest revision. This choice is made before reserved
results exist and is never revised based on them.

After that global freeze, every revision runs on all twenty reserved layouts.
The selected revision's results are reused for its learning-curve entry, not rerun.
The planned budget is **60 training slots plus 240 reserved slots**, at most 300
physical rollouts. Candidate validation failures remain zero-success slots with
explicitly zero physical execution. Completed trials are preserved for resumption;
an incomplete process is inspected rather than silently rerun. Low disk space
stops before starting another trial, without deleting previous evidence.

Primary descriptive outcomes are reserved SR by revision, first reserved revision
reaching 80%, mean reserved SR over R0–R3, and reserved SR of the training-selected
candidate. Also report training curves, invalid slots, execution errors, actor
usage and paired outcome counts. Conditional-on-policy paired tests may describe
seed-level differences but cannot replace replication over independent model chains.
All revisions use the same layouts; they are correlated, not independent samples.
One model chain per arm and twenty layouts make this an exploratory pilot, not a
conclusive test of agent learning. A favorable result still requires replication.

No further model improvement may use this reserved split after inspection. Success
is the unchanged private physical evaluator; geometric support candidates do not
certify physical contact or stability. Current observations, desired poses and
history remain distinct; temporal pose fusion and conditional prediction are off.

## Validation and execution

Fifteen state/interface/protocol tests passed before registration. A technical
smoke on previously inspected development seed 75103 with permuted colors ran all
three common starters: basic failed without an exception; numeric and symbolic
succeeded with exactly identical physical trajectories. Each real feedback context
fit the prompt budget. These are integration checks, not held-out pilot results.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/run_mixed_scene_experiment.py \
  --output outputs/mixed_scene_learning_pilot_01
```

For a completed training freeze, use `--resume --phase evaluate`; `--resume` also
recognizes an existing training freeze. Inspect `progress.json` and the launcher
log for the current phase. Do not edit frozen runtime files during the campaign.

## Current measured status

The 15 local R0 trials are complete. Basic succeeds on 2/5 training layouts;
numeric and symbolic each succeed on 3/5. All five numeric/symbolic physical
trajectories are exactly identical. Both geometry arms fail on the same two
layouts, including a cylinder-center observation error on 76005. These are
training results from a fixed starter, not a symbolic learning benefit.

Source/result hashes and matched pre-policy RGB scenes pass the local audit.
First-revision feedback is ready locally: 29,696 / 40,581 / 77,969 characters for
basic / numeric / symbolic, each with eight sensor images. None was transmitted.
There are zero actor generations and zero reserved rollouts. Model generation is
pending the user's explicit permission following the automatic approval rejection.

The local report is [here](../evaluations/mixed_scene_learning_pilot_01.json).
Reproduce its audit without model calls or simulation:

```bash
.venv/bin/python scripts/summarize_mixed_scene_learning.py \
  --root outputs/mixed_scene_learning_pilot_01 \
  --output docs/evaluations/mixed_scene_learning_pilot_01.json --allow-partial
```

When the campaign is complete, omit `--allow-partial` to require complete evidence;
optionally pass `--figure docs/media/mixed_scene_learning_pilot_01.png` for curves.

### Evidence-auditor validation

The offline auditor now checks per-seed outcomes rather than aggregate counts
alone, checks the selected source against its actual training revision, verifies
model identity and returned candidate source, and requires the first reserved
artifact to follow the global freeze. A result completed after the freeze is not
sufficient by itself. It rejects non-boolean successes and duplicate paired seeds.

Six local tests passed using temporary synthetic records only. They cover valid
complete accounting including invalid-candidate slots, altered training outcomes,
early reserved starts, mismatched model identities, non-boolean outcomes, changed
selected sources and duplicate seed pairing. Synthetic fixtures are not measured
rollouts and are never added to the pilot's result directory. The actual 15 R0
trials pass the strengthened partial audit; all frozen runtime hashes remain
unchanged. No model or reserved run occurred during this validation.

### Transmission authorized and campaign resumed

The user explicitly approved running the experiments after the request specifying
the OpenAI Codex destination and source/training-sensor payload. The frozen campaign
resumed with the existing 15 R0 results; no prior trial was rerun. The previous
approval-related status above is historical. See `progress.json` and
`authorized_campaign.log` in the pilot directory for authoritative live status.

### Validator correction before second-revision generation

The first basic-arm candidate imported `ndimage` from SciPy, which the actor
contract explicitly permits. The host validator nevertheless listed `ndimage`
as a protected implementation name from an earlier, shared-namespace design.
The runtime library now has its own namespace, so the actor import cannot mutate
that library. Rejecting this candidate was an infrastructure defect, not a policy
failure. The five erroneous validation slots remain intact in the original pilot.

The coordinator was suspended while its last R1 physical rollout completed, then
terminated before any R2 model invocation. Exactly three model calls had completed.
The correction removes only `ndimage` from the protected-name set. Twenty-two
state/protocol/auditor tests pass, including direct and aliased SciPy import cases.
No actor source, initial policy, observer, executor, scene, feedback contract or
reserved split changed.

The corrected continuation is `outputs/mixed_scene_learning_pilot_01_corrected`.
Its manifest records the parent hash and exact source difference. It reuses all
15 R0 rollouts, the three original R1 model outputs, and the ten valid numeric and
symbolic R1 rollouts byte-for-byte. Only the basic R1 candidate needs its five
physical trials. The original host-validation failures are excluded from the
corrected learning curve and explicitly retained as the reason for the correction.
This is the same pilot with an infrastructure correction, not a replication or
an additional model attempt. Six authorized model calls remain for R2/R3.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/run_mixed_scene_experiment.py \
  --output outputs/mixed_scene_learning_pilot_01_corrected --resume
```

### Public observation schema defect and feedback-equivalence repair

R2 completed with 2/5 basic, 0/5 numeric and 0/5 symbolic. All five symbolic trials
raised `KeyError: axis_dimensions`. The public contract promised this field from
both `observe` and `get`, but shape-specific fits did not always supply it. A
recorded RGB-D fixture reproduces the omission on the cylinder; an initial
cube-specific diagnosis was not established. The existing `MetricScene` ingestion
already normalizes the field, so `get` and state packets were consistent while
`observe` could violate the contract. This is an abstraction-interface defect,
not evidence against symbolic reasoning or an error in the candidate's field access.

All nine already-launched model outputs were preserved. The coordinator stopped
before any R3 policy execution or reserved rollout. The interrupted continuation
retains 45 physical trial results and is marked withdrawn from learning-effect
claims. The earlier validator-only correction and its original artifacts also
remain intact. See [the archived partial audit](../evaluations/mixed_scene_learning_pilot_01_schema_defect.json).

The fix returns the existing validated ledger observation from the common sensor
callback. It adds canonical schema fields without a new estimator, controller,
pose prior, dimension source or action. Twenty-three related tests pass. On the
already inspected training seed 76001, numeric and symbolic R0 retain exactly the
same complete physics trajectories and state packets. The unchanged R2 symbolic
candidate succeeds after the repair, without the missing-field exception. This
single regression success is not a new generalization estimate.

The continuation `outputs/mixed_scene_learning_pilot_01_schema_corrected` has a
new source snapshot and explicit parent lineage. Before reusing the six R1/R2
model outputs, a local gate replays every R0/R1 geometry-arm trial and requires
exact prior physical trajectories and projected feedback. It also reconstructs
all six retained model inputs and requires byte-identical prompts and attached
images. Basic-arm results can be reused because its library and policy hashes
are unchanged. Any discrepancy stops this continuation before further model work.
The gate itself makes no model calls. It is running; its pass is not assumed.

If the gate passes, the numeric/symbolic R2 policies run under the repaired schema.
The previously generated R3 outputs cannot be reused, because their feedback
included the interface failure. R3 must be regenerated from the corrected results.
This requires three replacement model calls; the original discarded calls remain
counted in total service usage. It does not create extra policy-improvement
iterations or an independent replication. All policies must still freeze before
any reserved rollout. No policy is manually edited or selected using reserved data.

The feedback-equivalence gate has now **passed**: all 30 R0/R1 paired trial
records (including ten unchanged basic-arm records) have identical physical
trajectories and projected feedback, and all six retained generation contexts
have byte-identical prompts and attached images. The original candidates are
unchanged. The schema-corrected continuation has resumed to execute corrected
R2 outcomes and replace the three R3 generations under the same approved service
and payload scope. The three discarded calls remain counted, so completing this
repaired pilot will consume twelve actual model calls for nine retained revisions.
This replacement was decided before any original R3 policy execution or reserved
result. It is not selection among competing R3 outcomes.

Corrected R2 training has completed: basic **2/5**, numeric **0/5**, symbolic
**5/5**, with no execution errors in those fifteen trials. R0/R1 were basic 2/5
then 0/5; numeric and symbolic each 3/5 then 3/5. These corrected results preserve
all failures and the unchanged R2 candidates. Symbolic first reaches the 80%
training threshold at R2; this is one actor chain and five training layouts, not
held-out evidence. All three replacement R3 generations are now running. The
largest new context is 113,927 characters, below the frozen 120,000-character cap.
Reserved evaluation remains unstarted.

### Training freeze and selected-policy checkpoint

All training is complete. Success counts from R0 through R3 are basic
`[2, 0, 2, 1]`, numeric `[3, 3, 0, 4]`, symbolic `[3, 3, 5, 5]`, each out of five.
The predeclared training-only rule selects basic R0, numeric R3 and symbolic R2.
All twelve revisions were frozen before reserved evaluation; a checkpoint audit
verified their hashes and the ordering of every reserved start observed at that
checkpoint. The final full audit is still required.

The selected candidates have now completed their twenty reserved layouts:
basic **0/20**, numeric **8/20**, symbolic **12/20**. Symbolic succeeds on all eight
numeric-success layouts and four more. Their paired exact McNemar test is
`p=0.125` (unadjusted, conditional on these policies); this is an exploratory
positive difference, not demonstrated reliability across independent actor chains.
Basic versus numeric has eight numeric-only successes and no basic-only successes
(`p=0.0078125`), but that contrast combines perception and state interfaces.
It does not isolate a text-reasoning effect.

The source-checked checkpoint is stored in
`outputs/mixed_scene_learning_pilot_01_schema_corrected/selected_policy_checkpoint.json`.
The remaining all-revision reserved evaluation is running. Do not use the selected
candidate checkpoint as a complete learning-curve result or claim the goal achieved.
