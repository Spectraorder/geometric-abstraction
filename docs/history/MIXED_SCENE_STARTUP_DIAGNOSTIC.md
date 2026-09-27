> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Training-only startup diagnostic

The fifteen numeric R0 trials across the three registered training splits all
executed robot actions and recorded planner checkpoints. Five succeeded; none
failed before the first recorded `move_tool`, `move_joints`, or `gripper` call.
This contradicts attributing their failures to an inability to start the session.
The observation-error messages alone did not identify the failure stage.

The current `MixedSession` constructor does attempt initial model acquisition for
all public objects and propagates observation errors. The starter then reobserves
all objects before recording its first checkpoint. This is a real interface
constraint, but it is not an observed startup blocker in these fifteen trials.
The earlier manual development planner also observed every object before planning;
its higher inspected development SR cannot be explained merely by avoiding that
requirement. Different scene splits and high-level planners remain confounded.

This diagnostic reads only completed training result JSON, API call logs and
policy geometry records. It does not use simulator object poses, private contacts
or reserved scenes, and it does not change the frozen method. The zero startup
failure count does not establish when or why later perception failed, nor cover
all evolved candidates. Those require separate action-stage evidence.

See [the per-trial audit](../evaluations/mixed_replication_startup_diagnostic.json).
