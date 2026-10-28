# Shared interface guidance for the neutral-start experiment

Use `scripts/run_interface_contract_geometry_experiment.py` for a new experiment
with explicit API types, valid source examples and a shared checkpoint adapter.
This is a successor to the completed neutral-start study, not a modification of
its frozen prompts, candidates or results. No new campaign has been launched.

## Why the interface needs clarification

The previous common prompt did not explicitly say that module-level assignments
are rejected. Base R1 consequently put a diagnostic constant at module scope and
consumed five invalid training slots. The prompt also showed
`action_status=returned_status` without defining that variable's type. Both actors
passed an entire action-result dictionary instead of its status string and lost
training revisions to that interface error.

These are task-independent integration issues. Making their contracts explicit
improves the experimental interface without supplying a construction sequence,
perception algorithm, worksite, placement target or recovery solution. It does not
make generated programs automatically correct.

## Changes shared by both arms

The [policy interface contract](POLICY_INTERFACE_CONTRACT.md) is included verbatim
in both first-turn prompts and hashed in the source snapshot. It specifies allowed
module-level declarations, complete-source submission, action return fields,
checkpoint argument types, coordinate units and the difference between action
status and observation availability. Constants belong inside functions; the
entire result dictionary must not be passed or stringified as an action status.

`policies/interface_only_starter.py` adds this ordinary policy helper:

```python
def checkpoint_action(session, color, step, result):
    status = result['status']
    if not isinstance(status, str):
        raise ValueError('Action result status must be a string')
    return session.checkpoint(color, step, action_status=status)
```

The helper forwards the existing status unchanged. It performs no motion or
observation refresh, chooses no object or target, certifies no success and does
not swallow action errors. It does not expose Full's geometry to Base. Generated
policies must include the helper if they use it; it is not an injected global.
The starter's `run(api)` still only captures side/wrist images and writes a note.

Both arms keep their previous geometry libraries, executor, goal presentation,
three model turns, five training layouts, 20 reserved layouts, persistent sessions,
image schedule and retention rule. R1 remains the first model-generated policy.
The new experiment changes the shared interface guidance and starter adapter;
report this protocol difference when comparing it to earlier studies.

## Validation and repair budgets

Five focused tests passed. They exercise the real Base and Full checkpoint
implementations for all four current action statuses, reject malformed status
values without coercion or logging, validate the starter and code examples with
the actual source validator, confirm non-acting R0 behavior, and reconstruct both
first-turn prompts with the same shared contract. The tests also verify that
scoped variant configuration restores the historical modules afterward.

These are developer checks of the supplied interface scaffold, not tests of future
model-generated candidates or evidence of improved training performance. Existing
candidate validation still runs before simulation. There is no free actor repair
turn or manual candidate edit. Interface failures and every model call remain
counted. Any future automatic repair allowance must be bounded, identical across
arms, separately registered and included in token/call accounting rather than
silently removed from the learning budget.

## Registering a future run

Choose a new output directory and an unused reserved split. The following seeds
are examples and must be checked against any subsequently inspected data before
registration. Preparation does not invoke a model or simulator.

```bash
EXPERIMENT_DIR=outputs/geometry_interface_contract_01

.venv/bin/python scripts/run_interface_contract_geometry_experiment.py \
  --output "$EXPERIMENT_DIR" --phase prepare --rounds 3 \
  --train-seeds {2026092601..2026092605} \
  --reserved-seeds {2027012701..2027012720}

.venv/bin/python scripts/run_interface_contract_geometry_experiment.py \
  --output "$EXPERIMENT_DIR" --resume --phase bootstrap

.venv/bin/python scripts/run_interface_contract_geometry_experiment.py \
  --output "$EXPERIMENT_DIR" --resume --phase train

.venv/bin/python scripts/run_interface_contract_geometry_experiment.py \
  --output "$EXPERIMENT_DIR" --resume --phase audit
```

Use this entry point for every phase, including `evaluate`. The standard full-curve
evaluation still accounts for 160 reserved slots; training-selected-only testing
requires its own 40-slot registration. No prior actor session may be resumed with
the changed prompt or starter. Preserve the completed
[neutral-start test results](history/early_geometry/UNSEEDED_GEOMETRY_TEST_RESULTS.md) as historical evidence.

## Authorized five-round run

The user subsequently authorized five rounds for
`outputs/geometry_interface_contract_01`. Its immutable manifest registers R1–R5
per arm, five shared training seeds 2026092601–2026092605 and 20 shared reserved
seeds 2027012701–2027012720. This run overrides the three-round example above.
The training budget is 60 slots including R0 and ten model generations. Both
actors start new persistent sessions with the same shared interface contract and
adapter; historical candidates and diagnoses are not inserted into their prompts.
The launch runs training and its final audit only, with no reserved evaluation.
The full-curve reserved phase would account for 240 slots across six revisions
including R0; a separate selected-policy-only protocol would require 40 slots.

Progress and logs are in `training_lifecycle.json`, `progress.json` and
`training.log` under the new root. On successful completion, the coordinator writes
`audit_training.json` and corrected `resource_accounting.json`. Use the project
virtualenv interpreter without resolving its symlink to the system interpreter.

The five-round run has completed and passed audit. Both arms selected R4: Base
3/5 and Full 4/5. The previous assignment/status-type errors did not recur, but
both R5 sources exceeded the 30000-character limit. Accounting retains 50 executed
training rollouts and ten invalid candidate slots; no reserved evaluation was run.
See [the completed five-round training report](history/early_geometry/INTERFACE_CONTRACT_TRAINING_RESULTS.md)
for complete curves, source-length failures and verified resource use. The earlier
statement that no campaign had launched describes the configuration-development phase.

## Expanded source budget for subsequent runs

After inspecting the two over-budget R5 submissions, the user increased the policy
source limit from 30000 to **50000 characters**, including comments and helpers.
Use `scripts/run_expanded_budget_geometry_experiment.py` for subsequent runs and
for every phase of those runs. It registers a distinct variant with the same
50000-character limit in the manifest, both actor prompts and the actual source
validator. Its [shared interface contract](POLICY_INTERFACE_CONTRACT_50K.md) differs
from the earlier contract only in the source limit. All syntax and interface
restrictions, retention rules, token accounting and rollout budgets still apply.
Larger permitted sources can increase generation and context costs; this limit
change alone is not evidence of improved task performance.

The earlier entry point remains at 30000 characters to reproduce its frozen study.
The new entry point rejects its manifests rather than reclassifying the old R5
failures. A new run requires a new output directory and separately registered
seeds; no training or evaluation is started by this configuration update.

For example, after choosing an unused output directory and reserved split:

```bash
.venv/bin/python scripts/run_expanded_budget_geometry_experiment.py \
  --output "$EXPERIMENT_DIR" --phase prepare --rounds 3 \
  --train-seeds "${TRAIN_SEEDS[@]}" --reserved-seeds "${RESERVED_SEEDS[@]}"

.venv/bin/python scripts/run_expanded_budget_geometry_experiment.py \
  --output "$EXPERIMENT_DIR" --resume --phase train
```

The usual default remains three rounds; pass `--rounds 5` at preparation only for
an explicitly authorized five-round campaign. Boundary tests exercise acceptance
at exactly 50000 characters, rejection at 50001, continued syntax restrictions,
registration/prompt consistency, legacy-run rejection and restoration of the old
budget after leaving the new configuration.

## Selected-policy test evaluation

The user subsequently requested test evaluation of the completed five-round run.
The separate registration at
`outputs/geometry_interface_contract_01/selected_evaluation_20/registration.json`
fixes both training-selected R4 policies, the original 20 reserved seeds
2027012701–2027012720, four workers and 40 rollouts. It preserves the original
manifest, freeze gate, training summary, audit, selected policies and actor state
by hash. No actor calls, policy edits or test-based reselection are allowed.
The original full-revision evaluation remains a separate, uncompleted protocol.

```bash
.venv/bin/python scripts/evaluate_selected_interface_policies.py \
  --root outputs/geometry_interface_contract_01 \
  --output outputs/geometry_interface_contract_01/selected_evaluation_20 \
  --phase audit
```

The evaluation runner records progress in `selected_evaluation_20/progress.json`
and logs in `selected_evaluation_20/evaluation.log`. Its final audit checks policy
and observer hashes, runtime settings, sandbox isolation, public goal receipts,
paired initial scenes/images and the unchanged training/actor history.

The selected-policy evaluation completed and passed audit: Base R4 scored 12/20,
Full R4 scored 8/20. All 40 rollouts executed in 84.08 seconds with no new model
calls. See [the test report](history/early_geometry/INTERFACE_CONTRACT_TEST_RESULTS.md). Earlier statements
about unstarted reserved testing describe the training-only phase.
