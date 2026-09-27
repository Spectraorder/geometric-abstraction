> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Desired rigid geometry from symbolic point and axis constraints

The prototype closes a missing interface step: the agent can name geometric
relations for the desired state, while numerical code computes the corresponding
rigid transform. It is separate from the frozen v2 experiments and has not yet
improved a policy's measured success rate.

A point-coincidence constraint and a directed-axis constraint determine five
rigid-pose degrees of freedom. Rotation around that axis remains free. The solver
returns a minimum-angle representative and explicitly reports
`unconstrained_twist=true`. A second perpendicular axis fixes twist when edge
alignment matters. The transform preserves every pairwise landmark distance.

For example, an agent can specify these geometric intentions:

```json
[
  {"predicate": "coincident", "source": "red.bottom_face.center", "target": "blue.top_face.center"},
  {"predicate": "opposed", "source": "red.bottom_face.normal", "target": "blue.top_face.normal"}
]
```

The numerical solver itself knows no colors, tower order, cube dimensions, grasp
primitive or robot API. It accepts world points and axes and returns a desired
SE(3) transform. The example adapter resolves the named references from sensor
geometry; the core operation also applies to other rigid shapes and geometric
mating tasks. The current prototype is not a general symbolic-language parser.

## Real sensor example

[The saved example](../evaluations/rigid_goal_example.json) uses the actual RGB-D
checkpoint at 31.30 simulation seconds from v2 repetition 2, symbolic revision 1,
seed 71101. Source centers reproduce the recorded runtime geometry at its saved
precision. The target red center is approximately
`[-3.986, -275.108, 444.841] mm`, derived from the measured references rather than
catalog dimensions or simulator poses.

The computed point residual is numerically zero and the axis residual is about
1.1e-16. Those values only verify the algebraic constraints on uncertain input
geometry. This is a counterfactual goal constructed after the rollout, not a
claim that the policy used it or physically reached it. Reachability, collision,
contact and stability are not certified. The retained example reads no private
result or physics trajectory.

## Relation to prior work

[ReKep](https://arxiv.org/abs/2409.01652) expresses manipulation constraints over
keypoints and combines them with numerical optimization. Our limited prototype
uses analytic rigid transformations and does not reproduce its perception,
tracking, planning stack or results.

[kPAM](https://arxiv.org/html/1903.06684v2) formulates manipulation goals through
semantic keypoints and geometric costs/constraints. This motivates separating
intent from numeric target construction; our block references come from rigid
RGB-D fits, not its learned category-level keypoint detector.

[PARL, including Mao and Wu](https://arxiv.org/abs/2405.03864), learns state,
action feasibility and transition abstractions from language-annotated
experience. The relevant design lesson is that a state description and a useful
planning interface are different pieces. Our analytic interface is not its
learned abstraction method, and its results do not establish our benefit.

## Boundaries and verification

- `scripts/rigid_goal_prototype.py` contains the action-free solver and landmark
  transformation. Degenerate axes and inconsistent secondary-axis specifications
  raise errors rather than silently changing the requested constraint.
- `scripts/record_rigid_goal_example.py` grounds the example in a frozen visual
  library and checks agreement with the actual runtime record.
- Four tests cover arbitrary 3D frame transformations, shape preservation,
  antiparallel-axis ambiguity, scale-independent face matching and invalid axes.

A desired transform must never be applied as an observed state update. Conditional
motion prediction requires separate evidence, such as a fallible attachment
hypothesis plus measured encoder displacement; a requested destination is not
that evidence. Input-reference uncertainty must remain visible to the agent.

The next policy comparison must label any supplied goal solver as part of the
treatment. It must keep the physical executor identical and separate the benefit
of metric estimates from that of relational goal construction. The ongoing v2
comparison supplies neither this goal solver nor the interval prototype.
