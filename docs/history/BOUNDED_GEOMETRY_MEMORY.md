> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Conditional memory checked against current geometric bounds

The earlier temporal diagnostic showed that keeping old hidden coordinates can
create large errors when an object moves. This separate prototype tests a narrower
hypothesis: a previously fully observed center can provide a static prior only
while the current RGB-D containment interval remains compatible with it.

At each checkpoint, current face constraints update observed translation
directions. A previous center supplies only the remaining directions. If the
candidate falls outside the current model-conditioned containment interval, the
static hypothesis is rejected and its full-observation anchor is revoked. A new
full observation can establish another anchor. The prototype never changes the
current observed rank or labels remembered coordinates as newly observed.

The representation preserves the condition, the time of the full anchor, and
whether the current bounds rejected memory. An unavailable current pose remains
unavailable; this diagnostic does not assess fully occluded object tracking or
conditional gripper transport.

## Paired development replay

All three methods receive identical frozen RGB-D fits at identical checkpoints.
Sensor-only estimates are saved before private physics is read for scoring.
The rule was evaluated unchanged on the second set of failed rollouts.

| Set / method | Partial checkpoints | Median error | p95 error | Maximum error |
| --- | ---: | ---: | ---: | ---: |
| Five successful rollouts: instantaneous | 48 | 1.058 mm | 3.274 mm | 80.091 mm |
| Same: static memory | 48 | 0.247 mm | 15.188 mm | 80.090 mm |
| Same: memory checked against bounds | 48 | 0.217 mm | 0.570 mm | 80.091 mm |
| Five failed rollouts: instantaneous | 63 | 0.559 mm | 2.981 mm | 155.002 mm |
| Same: static memory | 63 | 0.303 mm | 149.177 mm | 149.663 mm |
| Same: memory checked against bounds | 63 | 0.164 mm | 2.411 mm | 155.002 mm |

Bounded memory was used 31 and 28 times; current evidence rejected 10 and 17
static-prior candidates. It improves the reported percentiles on these paired
sets, but does not fix the worst current perception errors. Those large-error
states lacked sufficient orientation evidence for the containment prototype to
emit bounds. They are retained in the report, not removed from the denominator.

This supports testing evidence-checked memory as part of the future interface.
It does not establish improved policy success, robustness to real depth noise,
or a generally accurate tracker. Compatibility does not prove stationarity:
small hidden motion may remain inside the feasible interval. The interval itself
assumes correct object association and sufficiently accurate measured shape and
orientation.

## Artifacts and experiment separation

- `scripts/audit_bounded_memory.py` implements the three replay modes; all source
  snapshots and sensor-only/scored outputs are under
  `outputs/bounded_memory_audit_01` and `_02`.
- Two tests exercise hidden-axis motion rejection and ensure that retained memory
  keeps its condition, age and partial observed rank.
- [Complete diagnostic](../evaluations/bounded_memory_diagnostic.json) includes all
  checkpoint counts, extrema, rejection counts and remaining large-error cases.

The ongoing v2 policy repetitions still use instantaneous geometry. No current
candidate, geometry library, training feedback or held-out execution was changed
by these offline diagnostics. The broader [interval assumptions](RIGID_CONTAINMENT_DIAGNOSTIC.md)
remain in force.


A separate [robot visual-geometry diagnostic](ROBOT_VISUAL_GEOMETRY_MASK.md)
subsequently identified the remaining large perception errors as tool-marker
confusion. Its paired report distinguishes corrected poses from two cases that
now abstain. The original results above remain unchanged.
