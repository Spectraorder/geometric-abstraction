# Ring policy starter: geometry and API interpretation

This note accompanies the Full API scaffold. It supplies geometric semantics and
qualitative manipulation knowledge, not a controller, grasp ranking, trajectory,
object dimensions, successful policy, or layout-specific target. The agent chooses
the entire procedure. R0 still collects observations without acting.

## References describe different physical things

- `hole.center_m` lies in the opening's top plane. It is a reference for reasoning
  about the opening, not a material contact point. Likewise, the ring's geometric
  center is not necessarily a graspable point.
- `outer_boundary_m` and `hole.inner_boundary_m` bound the material. The entries in
  `walls` describe outer surfaces; they are geometric references, not prevalidated
  grasps. A proposed finger position must be checked against both boundaries.
- `height_m` is the initialized thickness along the cap normal. For a tilted ring,
  it is not the world-z extent. Use current boundary points and their evidence.
- The peg cap position may be retained because the public task declares the peg
  fixed. The ring can slip or rotate; an earlier ring pose is historical.

## Robot previews have precise, limited meanings

- `fixed_pad_m` targets the center of the fixed finger pad. It is different from
  the tool marker, the midpoint between fingers, and any object reference.
- `closing_axis` points from the fixed pad to the moving pad. For example, fixed
  pad `(0, 0, 0)` and moving pad `(0.02, 0, 0)` define direction `(1, 0, 0)`.
  These numbers illustrate a vector convention; they are not scene targets.
- A requested closing direction is a three-dimensional direction, not a yaw
  angle. Wrist roll, arm posture and jaw opening jointly affect the achievable
  direction. Rejection of one requested direction does not reject all candidate
  grasps. Inspect the returned geometry at the actual proposed point and opening.
- Jaw angles are radians, not metric gaps. Obtain pad separation and finger
  geometry from previews. A reachable preview does not establish a collision-free
  path, a successful grasp, or stable attachment.

## Compare geometry where it can physically overlap

Projecting an entire tilted finger onto a horizontal plane can make an upper
finger part look as though it intersects material far below it. Assess geometric
clearance in the relevant three-dimensional region, including heights. Conversely,
checking only pad centers can miss collisions involving finger thickness or housing.

A hole-center residual alone does not certify insertion clearance: the hole and
peg axes, ring thickness, observed inner boundary, peg radius, and fingers all
matter. Retain explicit measurement margins consistent with boundary resolution;
printed decimal precision is not measurement accuracy. Do not suppress ambiguous
observations or silently turn inferred geometry into directly observed evidence.

## Commands, observations and feedback

Keep commanded robot targets separate from measured robot positions. The scaffold's
`move_increment` retains a caller-owned command target; this can prevent tracking
error from accumulating across successive commands. Neither target bookkeeping
nor a jaw-close command establishes the ring's current position or a successful
lift. Obtain fresh visual evidence after physical motion.

When an operation fails, record the failed check and its relevant public values:
for example, requested versus previewed pad position, closing-direction alignment,
measured clearance versus its required margin, or unavailable observation. This
helps distinguish unreachable requests, geometric rejection, lost observation,
and physical execution failure. No error code alone proves that the task is
physically impossible. Keep diagnostics concise and do not report success from
commands alone; the private evaluator provides the training outcome.

For a future Base/Full comparison, provide the same qualitative knowledge and
robot-interface explanations to both arms. Full-specific field examples explain
its additional observation interface; Base must derive geometric values from its
allowed sensor inputs. Record this guidance as supplied prior knowledge.
