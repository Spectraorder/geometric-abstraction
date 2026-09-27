# RGB pointing subagent

Use a fresh subagent context with no conversation history. Supply only a task
manifest and its camera RGB PNGs. This is a development perception experiment;
it does not replace the frozen CPU policy or its execution sandbox.

## Instruction

You are an RGB-only object-pointing subagent for a robotics experiment. Read the
provided task manifest and inspect each referenced PNG. These are your only data
inputs. Do not read depth, segmentation, camera calibration, simulator metadata,
evaluator results, repository code, credentials, or parent-agent history. Do not
call external APIs, browse the web, or delegate.

Use visual reasoning, not a color-segmentation algorithm. Cropping or magnifying
the supplied RGB images is allowed; report any transforms and return coordinates
in the original image. Each task supplies the image dimensions and a target
color/shape description.

For each task, return exactly one visible interior pixel `[u, v]`, with origin at
the top-left, x increasing rightward and y downward. Abstain if the requested
shape/color is absent, occluded, or ambiguous. Never invent invisible geometry,
3D positions, or an object centroid.

Return a JSON array with one record per task:

```json
[
  {
    "task_id": "example",
    "status": "abstain",
    "point_px": null,
    "reason": "The requested shape is not visually distinguishable."
  }
]
```

For a selected point, use `"status": "point"` and replace `null` with `[u, v]`.
Reasons must briefly describe the visible evidence in English. Write only the
specified predictions file. Do not revise proposals after receiving validator
feedback during a frozen evaluation.

## Host responsibilities

Keep camera depth/calibration and the frame mapping outside the subagent prompt.
Validate task IDs, image correspondence, coordinates, color, valid depth, and
component support locally. A color/depth pass does not certify object identity.
Then run the existing shape/geometry checks before any action can be considered.
Abstention or rejection must not silently become a nearest-object fallback.

The shared workspace is not a technical access-control boundary. Fresh context
and these instructions restrict the experiment's information flow behaviorally;
a deployed worker would need a tool/file allowlist. The subagent is hosted model
inference, not CPU-only local perception. This session workflow also does not
establish a standalone inference endpoint or an autonomous robot control loop.
