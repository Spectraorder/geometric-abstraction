# Stacking skill

## Task pattern

Place one tabletop object stably on the upper surface of another object.

## Initial strategy

1. Localize the movable object and the supporting object on every reset.
2. Use a top-down grasp through a collision-free pre-grasp waypoint.
3. Lift vertically before translating laterally.
4. Compute the release height from the current support-object height, not from the table.
5. Release slightly above the desired final center and wait for physics to settle.
6. Retreat vertically so the gripper does not knock over the stack.

## Failure signatures and repair directions

- **IK failure before grasp:** lower the pre-grasp height or introduce an intermediate joint-space waypoint.
- **Object moves but is not lifted:** adjust the grasp-center height or gripper closing position.
- **Object falls during transit:** reduce acceleration, close farther, or rotate the wrist so the fingers span the short object axis.
- **Object lands beside the support:** compensate for grasp offset and release only after horizontal alignment.
- **Object touches the support but tips:** slow the descent, release closer to the surface, and retreat vertically.

Promote parameter changes into this file only after they improve multiple debug seeds and then pass the held-out split unchanged.

## Debug evidence

- A fixed `wrist_roll=0` grasp succeeded on 4/5 debug seeds (51–55).
- Aligning wrist roll directly to the observed object yaw dropped success to 3/5,
  so raw object yaw is not a valid wrist-joint mapping for this embodiment.
- Keep the fixed-roll baseline until a closed-loop retry or calibrated frame
  transform improves the full debug split.
