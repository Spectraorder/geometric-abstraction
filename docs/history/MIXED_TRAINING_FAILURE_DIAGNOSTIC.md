> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Training failure stages and initial-orientation contract

This diagnostic covers the 120 completed training slots in
`mixed_scene_replications_01_corrected`. It was performed while frozen reserved
execution was running, without inspecting reserved outcomes. No live policy,
observer, prompt, or executor was changed.

## Command-stage evidence

| Recorded ending | Numeric | Symbolic |
| --- | ---: | ---: |
| Successful task | 15 | 22 |
| No robot actuation | 15 | 12 |
| Last jaw command closed; placement descent recorded | 23 | 19 |
| Last jaw command was release opening | 6 | 7 |
| Last jaw command reopened after unconfirmed grasp | 1 | 0 |

These are descriptive training counts, not independent trials or a held-out
learning comparison. The trace classifier uses commands from the frozen shared
executor. A closed command does not establish attachment, and a release command
does not establish correct placement. The 42 failures after placement descent
include perception exceptions and normal returns; they do not all have one cause.

Of the 27 trials with no actuation, six contain an uncaught worker observation
failure, twelve log unavailable initial observation/session acquisition, and nine
log rejection of all candidate worksites by clearance or nominal IK checks.
The earlier [R0 startup diagnostic](MIXED_SCENE_STARTUP_DIAGNOSTIC.md) remains true:
all fifteen numeric starters acted. Startup failures appear in evolved candidates.

## A constructor ambiguity changes the observer

The actor contract says to construct `Session(api, objects)` with public
shape/color descriptors. The starter passes complete task nodes. The observer also
reads the optional `orientation` field, defaulting to `upright`; for a thin prism,
`flat_x` dispatches to the flat-beam estimator before the column estimator.
Several evolved candidates copy only shape/color, dropping this field. Although
the prompt describes the beam's initial flat orientation, it does not explain
that preserving this descriptor changes dimension acquisition. Desired orientation
and initial orientation are also represented by the same field in this scene.

An isolated replay compares both descriptors on the same fifteen saved initial
training side RGB-D frames, using the frozen numeric library and a table plane
fitted from those frames:

- Complete public orientation descriptor: 14/15 usable estimates.
- Shape/color only: 10/15 usable estimates.
- Four paired frames change from usable to unavailable; one is unavailable in both.

This replay uses unmasked initial side frames, not the complete robot-mask and
wrist-fallback session. It establishes an orientation-dispatch effect on identical
sensor data, not that all eighteen recorded acquisition failures have this cause,
and not a physical success-rate improvement. No simulator object poses or sizes
are used. The public scene description supplies the initial flat-beam hint.

## Implication for the representation comparison

Both arms had the same constructor ambiguity, but learned policies preserve its
optional metadata differently. Consequently, aggregate learning curves also reflect
handling this interface trap. The frozen study will finish unchanged, including
all failures; it cannot alone isolate the benefit of symbolic reasoning operators
from descriptor handling. This limitation is recorded before inspecting its
reserved outcomes. No candidate is repaired or selected using those outcomes.

Before a subsequent confirmatory comparison, make the common constructor contract
unambiguous and test invariance to irrelevant task-node fields. Keep initial visual
assumptions separate from desired poses. Either preserve explicitly declared initial
hints in the common adapter, or select supported geometric hypotheses from RGB-D;
do not silently infer initial orientation from a desired final state. Validate that
change on training/development sensors, then freeze a fresh matched experiment.

The other priority is explaining unconfirmed placement using action checkpoint
geometry and explicitly conditional predictions. The command counts alone do not
justify temporal pose fusion, automatic release, or treating a predicted pose as
an observation.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_training_action_stages.py \
  --root outputs/mixed_scene_replications_01_corrected \
  --output docs/evaluations/mixed_training_action_stages.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_initial_orientation_contract.py \
  --root outputs/mixed_scene_replications_01_corrected \
  --output docs/evaluations/initial_orientation_contract_replay.json
```

Reports retain per-trial provenance and input hashes:
[action stages](../evaluations/mixed_training_action_stages.json),
[paired sensor replay](../evaluations/initial_orientation_contract_replay.json).
