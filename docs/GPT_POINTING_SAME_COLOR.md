# Same-color RGB pointing experiment

The twenty frozen queries produced **16 exact target hits (80%)**, two selections
of the wrong same-color shape, and two abstentions despite visible target pixels.
Among the eighteen non-abstained points, identity precision was **16/18 (88.9%)**.
Thus the near-perfect result on earlier, simpler examples does not carry over to
this same-color stress test.

| Requested shape | Exact hits / four queries | Remaining outcomes |
| --- | ---: | --- |
| Cube | 3/4 | Green thick prism selected instead |
| Slender rectangular prism | 3/4 | Yellow target: abstention |
| Thick rectangular prism | 3/4 | Blue slender prism selected instead |
| Cylinder | 4/4 | None |
| Triangular prism | 3/4 | Yellow target: abstention |

[All twenty proposals](media/gpt_pointing_same_color_overview.jpg) ·
[Four failure cases with offline target outlines](media/gpt_pointing_same_color_failures.png) ·
[Machine-readable evidence](evaluations/gpt_pointing_same_color.json) ·
[Offline target masks](evaluations/gpt_pointing_same_color_target_masks.npz)

All five same-color shapes had visible pixels in every scene. Both wrong points
landed inside a different same-color object; all eighteen proposals, including
these two errors, passed the CPU color/depth gate. The single requested flat
thin-beam fit also passed. A point therefore needs independent shape verification
before reassociating a policy track. This experiment does not test whether the
full policy's later geometry checks would reject either mistaken identity.

The two abstentions cited robot occlusion or ambiguous visible shape. Their
target masks contain 37 and 86 pixels respectively. They are scored as missed
visible targets, while recognizing that refusing uncertain evidence can be safer
than choosing another object. No coordinate correction or second attempt was
allowed. Seventeen pointing regression tests passed.

This development test uses twenty new simulated camera images, with seeds
2501–2520. Each scene contains all five inventory shapes in the requested color,
plus three objects of other colors. The target schedule covers each of the four
colors and five shapes exactly once. Target order was shuffled with RNG seed
926 before rendering. Long rectangular blocks alternate between flat and
upright initialization across scenes. All requested objects exist in the scene;
visibility is measured separately after predictions are frozen.

## Input boundary and protocol

Three fresh-context subagents process fixed batches of seven, seven and six
queries, using the inherited parent model without overrides. The RGB-only
instruction follows [the existing protocol](prompts/RGB_POINTING_AGENT.md).
Only the assigned task manifest, target color/shape description and 320 × 240
RGB PNGs are provided. RGB-only zooms are allowed. The pointing agents receive
no depth, calibration, simulator state, object masks or scoring feedback. There
are no retries or corrections after validation. This shared-workspace boundary
is enforced by instructions, not a filesystem allowlist. Hosted model inference
is not CPU-only local perception. No OpenRouter calls or credentials are used.

The host reconstructs each scene after all predictions are saved. Exact RGB,
depth, valid-depth mask, timestamp and camera calibration correspondence are
checked before offline segmentation scoring. A hit requires the original
proposed pixel, floored to integer coordinates, to lie on the requested object's
visible geometry. No snapping or tolerance radius is used. A refusal with any
visible target pixels is scored as a false abstention; this deliberately strict
rule does not establish that a partly occluded shape is visually identifiable.
Privileged segmentation is solely an offline label, never an actor input.

The existing CPU RGB-D gate is evaluated separately. Passing color/depth support
does not certify shape identity. These are localization queries, not actual
robot actions or an end-to-end policy evaluation. The frozen CPU policy and
optional pointing policy are unchanged by this experiment.

## Reproduction

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/evaluate_same_color_pointing.py prepare \
  --output outputs/same_color_pointing_new
```

Give only the generated `batch_1.json`, `batch_2.json`, `batch_3.json` manifests and
referenced PNGs to fresh RGB-only subagents using the existing instruction. Save
one prediction per task, concatenate the three batches into `predictions.json`,
then score without showing the results to those agents:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/evaluate_same_color_pointing.py score \
  --output outputs/same_color_pointing_new

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m unittest discover -s tests -p 'test_pointing*.py' -v
```

Preparation refuses to overwrite an existing host manifest. Source frame and
PNG hashes are validated before scoring. Saved offline masks permit rechecking
identity decisions without rerendering. This test does not cover duplicated
objects of identical color and shape, physically touching piles, new textures,
real cameras or alternate camera viewpoints.

## Viewing transforms and limitations

Batch 1 viewed original PNGs and a 5× nearest-neighbor crop `(100,80)-(220,180)`.
Batch 2 used RGB-only JPEG previews and a 5× crop `(100,80)-(220,185)`. Batch 3
viewed original RGBs and 4×–6× crops `(110,70)-(215,175)`. All returned coordinates
refer to the original image. JPEG viewing compression in batch 2 is an additional
protocol variation. Filesystem `view_image` failed with a sandbox mount error, so
agents displayed RGB bytes directly. No scene was replaced or selected based on
prediction quality. Twenty queries across three contexts remain a small
development sample; the measured 80% is not a broad generalization estimate.
