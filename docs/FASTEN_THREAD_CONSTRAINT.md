# Thread deflection and rebound correction

Current learning scene: `fasten_robodojo_v2`. The imported visual meshes are
unchanged. `fasten_robodojo_v1` and the procedural versions remain available for
historical state playback. A campaign cannot mix physics versions; new learning
must use a fresh output root.

## Reproduced problem

The best recorded single-nut V1 trial (round 3, seed 5101) had approximately
5.10 mm of axial spring deflection, 12.81 degrees of nut tilt, and 1.06 mm of
nut/bolt proxy penetration during manipulation. These are private diagnostics
of saved simulator states, not quantities exposed to the policy. The old
1200 N/m spring and 0.15 N m/rad upright torque were too compliant under jaw
loads. Smooth contact proxies also let an unengaged nut slide down the bolt.
The visual thread triangles themselves do not provide collision constraints.

## Corrected reduced-order model

An engaged nut is coupled to a collisionless guide with axial slide and spin
joints. A solver joint equality couples slide to spin at the unchanged 3 mm
pitch, and a weld equality couples the nut to that guide. The guide is based
at the fixed bolt, never at the gripper. This follows MuJoCo's
[joint and weld equality semantics](https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-joint).
The jaw contacts still determine whether the robot holds or slips on the nut.
No physical nut or robot position/velocity is overwritten at capture. Only the
auxiliary guide's initial state is initialized to the nut's current pose.

The guide has 1 g mass and diagonal inertia 1e-7 kg m^2 for numerical
conditioning; its weight is compensated. Its small inertia remains in the
coupled dynamics. Constraint solref is (0.004, 1), impedance is 0.9999, and weld
torque scale is 1 m. These are simulation surrogate parameters, not calibrated
material stiffnesses. Existing static spin friction and damping remain active.
Unthreading removes the guide constraints; reset removes all engagements.

A hidden 14.5 mm radius entry shoulder collides only with nut proxies. It stops
an unengaged nut near the existing entry height and is disabled for an occupied
bolt after valid grasp/alignment capture. It is restored on disengagement/reset.
This is an explicit approximation of thread entry, not an extra visible washer
or a claim that the original thread faces have become contact-resolved.

The model still omits detailed thread-face collisions, cross-threading,
deformation, and tightening torque. Small numerical compliance remains.

## Validation

A repeatable fixture applies 3 N lateral force, 6 N downward force, and
0.025 N m tilting torque to an engaged nut for one second, then releases the
load for two seconds. It uses identical conditions for both versions.

| Measurement | V1 spring | V2 solver |
| --- | ---: | ---: |
| Peak axial constraint error | 9.319 mm | 0.186 mm |
| Peak lateral error | 3.267 mm | 0.0034 mm |
| Peak tilt | 14.50 degrees | 0.0037 degrees |
| Rebound after load removal | 4.877 mm | 0.177 mm |

Raw results: `outputs/fasten_single_nut_constraint_check/final_audit/load_comparison.json`.
Reproduce on a compute node:

```bash
uv run --no-sync python scripts/audit_fasten_thread_constraint.py \
  --output outputs/fasten_thread_audit_new
```

Regression coverage includes loaded deflection/release, unengaged entry stop,
constraint reset/unthreading, no pose overwrite at capture, net rotation and
depth, reversal, long gravity holds, near-complete no-free-credit checks,
one-pair scoring, reference isolation and historical replay.

All 40 relevant regression tests passed.

The old best policy was rerun unchanged as a development diagnostic, without
any new model call. It ended at 0.293 net turns, score 0, with no policy-process
error. Its own IK check rejected a move, and its cleanup tried to pull a nut
that had already engaged during alignment. The policy still needs adaptation.
The recorded-state audit (10 ms sampling, excluding the first 100 ms after
capture) measured at most 1.049 mm axial error, 0.049 mm lateral error and
0.181 degrees tilt while engaged. This residual under robot contact is larger
than in the controlled load fixture; the new solver is not perfectly rigid.
The trajectories differ, so these are not a matched-policy success comparison.
Raw audit: `outputs/fasten_single_nut_constraint_check/final_audit/policy_contact_audit.json`.

The final replay is:

```bash
uv run --no-sync python view_scene.py \
  --vision-trial outputs/fasten_single_nut_constraint_check/final_seed_5101
```

This is a physics-check replay, not a successfully learned screwing policy.
The former `seed_5101` check and `prototype_sources/` retain an intermediate
entry-shoulder/angular-constraint experiment and are not final V2 evidence.
The original campaign and its best-trial pointer remain historical V1 evidence.
