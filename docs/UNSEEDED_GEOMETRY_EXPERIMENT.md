# Geometry comparison without a supplied task-solving policy

Use `scripts/run_unseeded_geometry_experiment.py` for the new comparison. Both
arms start with `policies/observation_only_starter.py`, which captures one side
and one wrist image and returns without motion. The experiment keeps three model
generations per arm, five shared training layouts, 20 shared reserved layouts,
persistent isolated actor sessions and training-only incumbent retention.

## Removed planning priors

- The supplied visual starter's segmentation and object-location helpers, table
  fitting, support traversal, workspace anchor and placement-target calculations.
- Prompt descriptions of a particular bridge composition or a successful action
  sequence, and recipe-specific failure/recovery advice from the previous pilot.
- Public goal list order as a construction sequence. The new trial driver sorts
  nodes by color, replaces role names with color identifiers and consistently
  relabels support references and the visual length reference. Desired geometry,
  support relationships and orientations remain unchanged.

The original source files and frozen studies remain unchanged. A separate driver
changes only presentation of the public goal. Physics and the private success
criterion still use the same internal recipe. The driver records a goal-contract
receipt for each rollout; the experiment audit checks that receipt and the exact
public goal against the registered color map.

## Retained capabilities and assumptions

The desired task must still be specified. `api.task()` supplies final-state
constraints, including object identities, support relationships and relative
locations; it supplies no execution program or current object poses. The actor
must interpret these constraints and generate its own policy.

Both arms retain calibrated RGB-D, robot proprioception/CAD, robot IK queries and
the same CPU pick/place primitives. Their shared generic actuator reference source
is still provided; this is not learning low-level control from scratch. Shared
perception/grasping also retains the existing unique-color interface, known shape
labels and initial table/orientation assumptions. Duplicate-color tasks are
explicitly unsupported by this variant rather than silently merging identities.

Base must calculate any high-level object geometry from pixels. Full receives the
existing measured geometry, references, relations and alignment operators. Those
operators do not choose the construction sequence or placement targets. Neither
arm receives simulator object truth. These retained assumptions mean the change
removes the supplied task solution, not every domain prior or dependence on a
particular task interface. One task family still cannot establish cross-task
transfer.

## Meaning of revisions and budgets

R0 is a non-acting observation control. Its success must be measured rather than
hardcoded, but it is expected to fail tasks requiring assembly. It is not the
model's zero-shot performance. R1 is the first model-generated solution and already
receives R0 images and outcome feedback. R2 and R3 are further improvement attempts.
Report R1 quality, improvement from R1, selected reserved success and cumulative
resource use; improvement over a deliberately inactive R0 alone is weak evidence.

The same retention guard applies: preserve every incumbent training success and
repair at least one failure; ties retain the incumbent. If no candidate succeeds,
the selected policy can remain the non-acting R0. Record unsuccessful proposals
and their costs rather than treating them as free retries.

Three generated revisions plus R0 require 40 training slots across both arms and
six model generations. The standard evaluation phase evaluates all four revisions
on 20 layouts per arm, or 160 reserved slots. A selected-policy-only evaluation is
a separate 40-slot protocol and must not silently replace the full-curve protocol.
Initial R0 feedback attaches side RGB/depth from two layouts (four images per arm);
the wrist capture is retained as runtime evidence. Later supplemental images use
the existing failure-triggered cap of two per turn. Token reporting must use
per-turn differences of cumulative session counters.

## Registered configuration and validation

`outputs/geometry_unseeded_01` registers three generations per arm, training seeds
2026092601–2026092605 and reserved seeds 2026122701–2026122720. The training layouts
are reused development cases; the reserved layouts were unexecuted at registration and were subsequently
evaluated after training was frozen. No historical policy or diagnosis is included in actor prompts.

Preparation and R0 bootstrap are local operations; they do not invoke a model.
Run subsequent phases from the repository root:

```bash
EXPERIMENT_DIR=outputs/geometry_unseeded_01

.venv/bin/python scripts/run_unseeded_geometry_experiment.py \
  --output "$EXPERIMENT_DIR" --resume --phase audit

# This phase starts the six registered model calls and remaining training rollouts.
.venv/bin/python scripts/run_unseeded_geometry_experiment.py \
  --output "$EXPERIMENT_DIR" --resume --phase train

# This phase executes all 160 registered reserved slots after freezing policies.
.venv/bin/python scripts/run_unseeded_geometry_experiment.py \
  --output "$EXPERIMENT_DIR" --resume --phase evaluate
```

Use this entry point for every phase of the new variant, including audit. The
historical constraint experiment runner rejects the new variant by design. Do not
resume any old actor session with the changed starter or prompt.

Focused tests cover preservation of goal geometry and support relationships,
removal of construction-order hints, observation-only starter calls, first-turn
prompt isolation and restoration of the historical module configuration.

The registered R0 bootstrap completed: Base 0/5 and Full 0/5, ten valid rollouts,
zero invalid slots and no policy errors. API traces contain no motion commands.
Paired baseline trajectories, public-goal contracts and frozen sources passed audit.
Actual first-turn contexts were reconstructed locally with four image attachments
per arm; no actor was called. Both shared geometry libraries exactly match the
prior study, and its frozen source checks still pass. Evidence is in the new root's
`audit.json`, `starter_parity.json` and `bootstrap_review.json`. At that bootstrap checkpoint, training and reserved evaluation had not started.

The user subsequently authorized training. All three turns per arm completed,
selecting Base R3 at 2/5 and Full R2 at 3/5. The audit verified 35 executed training
rollouts and five invalid candidate slots, with no reserved evaluation. See
[the completed training report](history/early_geometry/UNSEEDED_GEOMETRY_TRAINING_RESULTS.md) for interface
failures, corrected token accounting and the postprocessing repair.

The later authorized reserved evaluation is now complete: Base R3 scored 6/20
and Full R2 scored 12/20. All 160 reserved slots were accounted for (140 executed
rollouts and 20 invalid-candidate slots), with final audit passed and no model
calls. See [the completed test report](history/early_geometry/UNSEEDED_GEOMETRY_TEST_RESULTS.md) for full
curves, paired statistics, runtime metrics and interpretation limits.

For subsequent runs, a separate [shared interface contract variant](INTERFACE_CONTRACT_EXPERIMENT.md)
adds explicit syntax/type documentation and a task-independent checkpoint adapter
to both arms. The completed study's prompts, starter, source hashes and results
remain unchanged. No successor training run has been started.
