"""API scaffolding for policy learning, without a manipulation procedure.

All targets, object choices and action order must be supplied by the learned
policy. The initial run only collects public observations. Motion helpers are
examples of explicit target handling, not object tracking or collision checking.
"""
import numpy as np


def observe_current(api, object_id):
    """Return a current sensor estimate or explicitly report missing evidence."""
    try:
        return api.observe(object_id)
    except (RuntimeError, ValueError) as error:
        api.note('Observation unavailable for '+str(object_id)+': '+str(error)[:180])
        return None


def preview_target(api, point, roll, jaw):
    """Preview a caller-chosen pad target without moving or checking objects."""
    target = np.asarray(point, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('Expected a finite three-dimensional world target')
    try:
        return api.plan_pad(target.tolist(), float(roll), float(jaw))
    except (RuntimeError, ValueError) as error:
        api.note('Target preview unavailable: '+str(error)[:180])
        return None


def move_target(api, point, roll, seconds=1.5):
    """Execute an explicit pad target after a robot-only reachability preview."""
    target = np.asarray(point, dtype=float)
    robot = api.robot()
    if preview_target(api, target, roll, robot['jaw']) is None:
        return None
    try:
        measured = api.move_pad(target.tolist(), float(roll), seconds=float(seconds))
    except (RuntimeError, ValueError) as error:
        api.note('Motion unavailable: '+str(error)[:180])
        return None
    return {'commanded_pad_m': target.tolist(), 'robot': measured}


def move_increment(api, commanded_pad_m, delta_m, roll, seconds=1.5):
    """Update a caller-owned command target rather than accumulating tracking error.

The caller chooses the initial target and all increments. Retaining this command
can preserve commanded height during lateral moves despite tracking error. It is
not a measured object pose; fresh object observations are still required.
"""
    command = np.asarray(commanded_pad_m, dtype=float)
    delta = np.asarray(delta_m, dtype=float)
    if command.shape != (3,) or delta.shape != (3,):
        raise ValueError('Expected command and increment vectors of length three')
    return move_target(api, command+delta, roll, seconds)


def command_jaw(api, angle, seconds=0.6):
    """Forward a caller-selected jaw command; no grasp success is inferred."""
    try:
        return api.gripper(float(angle), seconds=float(seconds))
    except (RuntimeError, ValueError) as error:
        api.note('Jaw command unavailable: '+str(error)[:180])
        return None


def run(api):
    task = api.task()
    api.robot()
    frame = api.camera('side')
    api.note('Calibrated RGB-D acquired: '+str(frame['rgb'].shape))
    if task['geometry_available']:
        for item in task['objects']:
            observe_current(api, item['id'])
    api.note('API scaffold initialized; no manipulation procedure is supplied.')
