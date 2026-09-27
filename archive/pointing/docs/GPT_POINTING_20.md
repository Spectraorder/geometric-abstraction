# Twenty RGB-only subagent pointing trials

The frozen batch scored **20/20 correct decisions**: **18/18 visible targets hit**
and **2/2 absent targets correctly rejected**. Every proposed pixel passed the
local color/depth gate. Of eight flat beams, seven passed the existing top-face
geometry fit and one was rejected. These are localization results, not tower
assembly results.

| Requested target | Trials | Exact target hits | Correct abstentions |
| --- | ---: | ---: | ---: |
| Flat thin beam | 8 | 8 | 0 |
| Cube | 6 | 6 | 0 |
| Cylinder | 4 | 4 | 0 |
| Absent cylinder | 2 | 0 | 2 |
| Total | 20 | 18 | 2 |

[All 20 proposals on real camera RGB](media/gpt_pointing_20_overview.jpg) ·
[Machine-readable evidence](evaluations/gpt_pointing_20.json) ·
[Offline target masks](evaluations/gpt_pointing_20_target_masks.npz)

## Protocol and input boundary

Before requesting predictions, the host fixed a manifest of 20 distinct archived
320 x 240 images: seeds 1601–1610 with zero and two distractors. A fixed target
schedule selected eight beams, six cubes, four cylinders and two absent-target
controls; the ordering was shuffled with RNG seed 925. Eighteen images show the
initial view and two show later cylinder scenes. Four images overlap the earlier
six-query pilot; sixteen were new to this pointing experiment.

Three fresh-context subagents processed batches of seven, seven and six queries,
using the inherited parent model without overrides. Each saw only its assigned
RGB PNGs and public target descriptions. Agents viewed the originals and used
RGB-only zooms/crops, with transforms recorded in the report. They received no
prior predictions, depth, calibration, scene metadata, or evaluator labels.
All 20 outputs were frozen before batch validation, with no validator feedback,
retries, or coordinate corrections. These are 20 one-shot image queries in three
agent contexts, not 20 separate fresh-context calls.

Predictions SHA-256:
`76ebdf19dff7a42ccaa9f88db572a964c273c321ea4886160687102849f58db3`.

The agents share the workspace; their input restriction is behavioral rather
than an enforced filesystem allowlist. No OpenRouter request or API credential
was used. Hosted subagent inference is not CPU-only local perception. The RGB-D
validator runs locally on CPU; the frozen assembly policy was not changed.

## Independent offline identity scoring

The runtime validator still uses only RGB-D and camera calibration. To establish
whether a point hits the requested object rather than another same-color region,
a **separate offline evaluator** restores the recorded simulator state and
renders a per-geometry segmentation mask. Privileged state is used **only in
this scoring step**, after predictions are frozen; it is never provided to the
agents or runtime locator.

A point is a hit only if `(floor(u), floor(v))` lies on a visible geometry of the
requested object. There is no nearest-pixel adjustment or tolerance radius. An
abstention is correct if the target has no visible pixels. Both controls were
absent from their scene inventories, not merely occluded. All 18 positive targets
had visible masks of 68–120 pixels.

All 20 restored frames matched the archived timestamp, camera calibration,
valid-depth mask, and depth exactly (maximum depth difference 0 m). Replayed RGB
was not bit-identical and was never substituted for actor input; segmentation
alignment was established by camera/depth correspondence. The host also checked
that each PNG seen by the agent exactly matched its archived RGB array.

## Remaining failure and limitations

T18 correctly points to the red thin beam at `(179,117)`, but the existing fitter
rejects its top boundary as `Beam top boundary is clipped or rotated`.
[View this geometry failure](media/gpt_pointing_20_geometry_failure.png).
Pointing does not recover hidden corners or supply a complete grasp model. The
seven accepted one-face fits retain their existing inferred-geometry flags.

The two configurations share ten seeds, so samples are paired and correlated.
Targets and scenes were selected for development, not randomly sampled from all
possible tasks. Color/shape coverage is narrow: cylinders are green, absent
controls are yellow cylinders, and beams are red or green. There are no full
occlusion controls, triangular-prism targets, or upright thick-prism targets.
Consequently, 20/20 does not establish 90% generalization or end-to-end stacking
success. Standalone API latency and cost were not measured.

## Reproduction

From the repository root, with the archived local trial files available:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/evaluate_pointing_agent.py \
  --tasks outputs/subagent_pointing_20/tasks.json \
  --frames outputs/subagent_pointing_20/validation_frames.json \
  --predictions outputs/subagent_pointing_20/predictions.json \
  --output outputs/subagent_pointing_20/validation

MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/score_pointing_replays.py \
  --frames outputs/subagent_pointing_20/validation_frames.json \
  --predictions outputs/subagent_pointing_20/predictions.json \
  --output outputs/subagent_pointing_20/offline_scoring

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s tests -p 'test_pointing*.py' -v
```

Twelve local tests pass, including exact target-mask scoring, adjacent misses,
absent-target false positives, and malformed predictions. The tests make no model
calls. The saved masks and report permit rechecking the hit decisions without
rerendering; masks are evaluator artifacts and must never become policy inputs.
