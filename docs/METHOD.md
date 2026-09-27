# Target-focused geometric abstraction

The system gives an agent structured geometric evidence derived from RGB-D.
Typed fields carry metric values, identities and evidence status; JSON is the
text serialization, not a second independently maintained world model.

## Observation to text

1. Calibrated side/wrist RGB-D provides world-frame surface points. Public robot
   geometry supports robot masking. Unique colors provide object association.
2. Simple rigid-shape fitting estimates geometry; dimensions are learned from
   initial on-table observations. Known shape families and orientation assumptions
   remain part of the method.
3. A compact record describes the active target after a semantic action boundary:
   identity, time, frame, current evidence, action status, optional intent and
   residual. Detailed geometric references are available on request.
4. The policy reasons over the packet and uses the same visual pick/place actuator
   available to Base. Offline scoring never becomes a runtime object observation.

The core implementations are [target records](../policies/geometry_step_state.py),
[partial constraints](../policies/partial_geometry_state.py),
[Base interface](../policies/visual_input_session.py), and
[Full interface builder](../scripts/build_partial_geometry_session.py).

## Minimal state

`GeometryStepState` retains one target, one optional goal and one preceding center.
It records an estimated center change only across available observations. Missing
or stale evidence clears the previous observation and leaves residuals unknown.
It does not maintain a growing scene history or infer contact from proximity.

When a current cuboid fit has unique correspondence but no accepted point pose,
Full can expose conditional center bounds:

```text
lower_m <= basis_world @ center_world <= upper_m
pose_available = false
```

These bounds depend on fitted orientation, visual dimensions and association.
They are not calibrated confidence intervals. The executor's pose-acceptance rule
is unchanged; a region does not silently become an executable center. The packet
is valid only for the same target and observation time.

Keep three concepts separate:

| Concept | Meaning | Status |
| --- | --- | --- |
| Current evidence | Visual estimate or conditional constraints at this observation time | In the frozen Full interface |
| Intent | Desired geometric relation or placement point | Available; not proof of feasibility or stability |
| Conditional prediction | Motion under an explicit rigid-attachment assumption | Offline diagnostic only |

Point and region transport replays expose contradictions between predicted motion
and later visual evidence. They neither recover a certified pose nor identify slip
separately from visual-model error. No policy success benefit from prediction memory
has been demonstrated. See [the diagnostic evidence](CONDITIONAL_ATTACHMENT_EVIDENCE.md).

## What the experiment can establish

Base receives pixels, calibration, public labels, robot proprioception/CAD and
qualitative action feedback. It may write its own visual estimator. Full additionally
receives supplied visual geometry and bounded target feedback. Both use the same
hidden actuator. The contrast concerns the whole perception/geometry package;
it does not isolate symbolic operators, memory or any single fitting improvement.

In completed training, Full checkpoint length has median 366 characters and maximum
1,074. This is compact text, but it is not evidence of lower token use or faster
model reasoning. A difficult-frame CPU replay measures 134.80 ms median, excluding
sensing, initial model acquisition, simulation, actions and model reasoning.
No end-to-end realtime or VLM speedup claim follows from that measurement.

The latest single-chain reserved result is promising; earlier independent-chain
results are mixed. [Results and limitations](PARTIAL_GEOMETRY_BUDGET_LIMITED_RESULTS.md)
remain separate from design claims. [Research notes and primary papers](history/GEOMETRIC_STATE_RESEARCH.md)
record the rationale and previous alternatives without expanding the active method.
