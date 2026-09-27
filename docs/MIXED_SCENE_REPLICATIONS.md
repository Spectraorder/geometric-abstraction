# Independent mixed-shape representation replications

**Stopped at the user's scope change.** Only base versus full geometry abstraction
remains in scope. The coordinator exited with code 143 after 120 completed training
and 195 completed reserved records; one interrupted reserved directory is retained.
No result was selected as the stopping criterion. This is not a completed replicated
comparison. See `outputs/mixed_scene_replications_01_corrected/administrative_status.json`
and the [active two-arm experiment](SHARED_OBSERVER_LEARNING_PROTOCOL.md).

The protocol and earlier checkpoints below are historical records.

**Current run:** `outputs/mixed_scene_replications_01_corrected`. The original
run was withdrawn after a hidden validator name restriction rejected a permitted
numerical helper. The corrected continuation preserves 55 valid trial records and
eight model generations with byte-identical inputs. Two unexecuted R2 generations
were discarded and remain counted. The full study will contain eighteen retained
calls and twenty actual calls. No reserved rollout preceded the repair.

The completed one-chain pilot selected symbolic R2 at 12/20 reserved successes and
numeric R3 at 8/20. The paired comparison was inconclusive (p = 0.125). This study
replicates the representation contrast without changing its runtime method.

Three independent numeric/symbolic chains each use five fresh training layouts,
three model-generated policy revisions, and twenty common fresh reserved layouts.
Training seeds are 78001–78005, 78101–78105, and 78201–78205; reserved seeds are
79001–79020. These splits had no previous trial directories at registration.
Budgets are 18 model generations, 120 training slots, and 480 reserved slots.

The source snapshot allows only orchestration changes to select two arms. Library,
starter, executor, model, feedback, candidate budget, simulation, and sensor hashes
or settings match the corrected pilot. All actor contexts remain isolated and
receive only their own training feedback. No earlier candidate or manual planner
is supplied. Checkpoint memory and conditional prediction remain outside this study.

All three chains train before any reserved rollout. Every R0–R3 candidate freezes,
not just the selected candidate. Selection remains highest training SR with the
earliest revision breaking ties. Resume preserves the original global freeze gate
and validates every archived policy hash before reserved execution.

Report each chain's reserved learning curve, mean SR over R0–R3, first reserved
revision reaching 80%, and training-selected SR. Preserve regressions and chains
that never reach the threshold. Compare numeric and symbolic within each chain.
Common reserved layouts and repeated policies are dependent; do not pool the 600
slots as independent trials or treat three actor chains as proof of broad robustness.
Earlier cube and mixed-shape results remain separate evidence.

The registered experiment is `outputs/mixed_scene_replications_01`. Training was
launched with `--phase train` while storage capacity was checked. The user then
authorized legacy artifact cleanup, retaining one sample per task. Cleanup excludes
this experiment, the prior geometry learning campaigns, and correction lineage.
Full trajectory and sensor recording remains unchanged for the new study.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/run_mixed_scene_replications.py \
  --output outputs/mixed_scene_replications_01_corrected --resume --phase all
```

Only resume after confirming the earlier coordinator is terminal; an observation
timeout is not a stopped process. The study auditor accepts partial progress without
calling it a completed experiment:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/summarize_mixed_scene_replications.py \
  --root outputs/mixed_scene_replications_01_corrected \
  --output docs/evaluations/mixed_scene_replications_01_corrected.json --allow-partial
```

## First-chain starter parity checkpoint

All five R0 training pairs in replica 1 have bit-identical saved physical arrays,
identical success/error outcomes and camera-call counts, and identical numeric
checkpoint packets after removing only symbolic schema/facts/reference fields.
The evidence hashes are in
`outputs/mixed_scene_replications_01/replica_01/starter_parity.json`.
This is an offline interface-equivalence check. Physical arrays never enter actor
feedback, and the check is not evidence of a learning benefit. No reserved layout
was read. Replicas 2 and 3 are not covered by this checkpoint.


## Isolated helper-name validator repair

Replica 2 numeric R1 defined `geometry(f)`, an ordinary numerical helper permitted
by the actor contract. The host validator reserved the hidden injected name
`geometry`, although the trusted Session constructor is bound before actor functions
execute and the backend has its own namespace. Its five rejected slots were a host
interface defect, not a demonstrated reasoning failure. The original run retains
55 valid physical records, five rejected slots, and ten completed model outputs.
It stopped at its source-freeze gate before either already-launched R2 candidate
executed. All reserved layouts were untouched.

The validator now reserves only the documented `Session` constructor. Direct access
to the injected factory's `.session` is forbidden; public construction remains
`Session(api, objects)`. Numerical helper names cannot overwrite library functions
in the separate backend namespace. A regression covers a helper named `geometry`,
unchanged observer behavior, and rejection of direct or aliased factory access.
The original rejected candidate passes without any human code edits.

The corrected continuation changes only this pre-execution host validator. Observer,
executor, actor bundles, task/settings, and all retained trial files are identical.
Eight retained actor prompts and their attached images are byte-identical. Both
replica 2 R2 outputs are archived and discarded before execution; the replacement
pair receives corrected feedback. This adds two service calls, not two improvement
iterations or a new independent replication. The repair gate is
`outputs/mixed_scene_replications_01_corrected/repair_equivalence.json`; the study
auditor independently checks its lineage, source delta, reused artifacts, inputs,
and discarded-call accounting. The invalid raw slots remain in the withdrawn root.

## All-chain starter parity

The corrected continuation now passes the same check across all fifteen paired R0
training layouts. Numeric and symbolic have bit-identical saved physical arrays,
identical initial images, camera-call counts and success/error outcomes, and exactly
matching numerical checkpoint states. Only symbolic schema/facts/reference fields
are removed for the state comparison. The check covers all three registered training
splits and no reserved scene or evolved candidate.

[Full parity audit](evaluations/mixed_scene_replication_starter_parity.json):

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_mixed_replication_starters.py \
  --root outputs/mixed_scene_replications_01_corrected \
  --output docs/evaluations/mixed_scene_replication_starter_parity.json
```

This establishes a matched starting execution for the representation comparison;
it does not establish an improvement after policy evolution. Saved physical arrays
are used only for this offline check and are never supplied to the policy actors.

## Secondary stopping-budget analysis

A secondary analysis was declared on 2026-09-26 at 17:48:45 UTC, after training
progress had been observed but before the study-wide freeze or any reserved
rollout. It does not replace the primary fixed-revision curves or the final
training-selected comparison. The declaration is
`outputs/mixed_scene_replications_01_corrected/secondary_analysis_plan.json`;
its analyzer hash matches `secondary_analyzer_snapshot.py` in that directory.

At each stopping budget R0 through R3, retain the candidate with the highest
training success rate among revisions already available; ties retain the earliest
revision. Report that candidate's reserved success rate separately for each chain.
Reserved outcomes never select a candidate. This describes when to stop the
existing chains, not a counterfactual evolution process that changes the parent
policy between revisions. Training retention does not guarantee monotonic reserved
success. Shared reserved layouts and reused candidates remain dependent.

Run the frozen analyzer only after the complete study passes its audit:

```bash
.venv/bin/python outputs/mixed_scene_replications_01_corrected/secondary_analyzer_snapshot.py \
  --report docs/evaluations/mixed_scene_replications_01_corrected.json \
  --output docs/evaluations/mixed_scene_replications_01_retained_budgets.json
```

## Training-only interpretation caveat

After all candidates froze, a training trace audit found that evolved candidates
can drop an optional initial-orientation descriptor that the starter preserves.
A paired sensor replay confirms that this changes dimension acquisition on some
identical training frames. The run continues unchanged, but its curves also reflect
this constructor ambiguity; they do not alone isolate symbolic-operator benefits.
This caveat was recorded before inspecting reserved outcomes. See the
[training failure diagnostic](MIXED_TRAINING_FAILURE_DIAGNOSTIC.md).
