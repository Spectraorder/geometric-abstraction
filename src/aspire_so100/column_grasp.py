"""Bounded CPU orientation planning from observed payload axes and robot IK.

An upright payload allows free world yaw. Matching that constraint can retain
upright objects even when a tall destination requires a pitched wrist. These
previews establish kinematic feasibility only, not grasp or collision safety.
"""
import numpy as np


def _grasp_unit(vector):
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-9:
        raise ValueError('Expected a finite nonzero three-vector')
    return vector / np.linalg.norm(vector)


def _grasp_rotation(rotation):
    rotation = np.asarray(rotation, dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or np.linalg.det(rotation) < .999):
        raise ValueError('Expected a finite proper rotation matrix')
    return rotation


def _grasp_align(source, target):
    """Shortest proper rotation between unit vectors, including opposite axes."""
    source, target = _grasp_unit(source), _grasp_unit(target)
    cross = np.cross(source, target)
    cosine = float(np.clip(source @ target, -1., 1.))
    if cosine < -1.+1e-10:
        reference = np.eye(3)[int(np.argmin(abs(source)))]
        axis = _grasp_unit(np.cross(source, reference))
        return 2*np.outer(axis, axis)-np.eye(3)
    skew = np.array([[0., -cross[2], cross[1]], [cross[2], 0., -cross[0]],
                     [-cross[1], cross[0], 0.]])
    return np.eye(3)+skew+skew @ skew/(1.+cosine)


def _grasp_yaw(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])


def orientation_for_payload(api, position, local_up, seed_rotation, *, max_calls=12,
                            position_tolerance=.003):
    """Find a reachable orientation with low predicted payload tilt.

    ``local_up`` is ``R_tool.T @ observed_payload_normal`` after a confirmed
    grasp. ``position`` is the desired tool position, not the object center.
    The caller must account for the rotated observed grasp offset. The returned
    ``orientation`` is the exact command passed to the winning IK preview;
    ``tool_rotation`` is the rotation that preview actually achieved.

    Eight coarse world-yaw choices, two yaw refinements, and two corrections
    for the SDK's soft orientation residual use at most 12 RPCs.
    Unreachable results remain explicit; the helper never commands motion.
    """
    if type(max_calls) is not int or not 1 <= max_calls <= 12:
        raise ValueError('max_calls must be an integer from 1 through 12')
    if not np.isfinite(position_tolerance) or position_tolerance <= 0:
        raise ValueError('position_tolerance must be positive and finite')
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Expected a finite tool position')
    local_up = _grasp_unit(local_up)
    seed_rotation = _grasp_rotation(seed_rotation)
    world_up = np.array([0., 0., 1.])
    upright = _grasp_align(seed_rotation @ local_up, world_up) @ seed_rotation
    best = None
    calls = 0

    def query(yaw, requested=None):
        nonlocal best, calls
        requested = _grasp_yaw(yaw) @ upright if requested is None else requested
        preview = api.plan_tool(position.tolist(), orientation=requested.tolist())
        calls += 1
        actual = _grasp_rotation(preview['tool_rotation'])
        error = float(preview['position_error'])
        if not np.isfinite(error) or error < 0:
            raise ValueError('Robot IK returned an invalid position error')
        predicted_up = actual @ local_up
        up_error = float(np.arccos(np.clip(predicted_up[2], -1., 1.)))
        reachable = error <= position_tolerance
        # Reachable candidates are ranked by uprightness. When no candidate
        # reaches the point, return the smallest error with reachable=False.
        score = (not reachable, up_error if reachable else error, error, abs(yaw))
        item = {'orientation': requested, 'tool_rotation': actual,
                'tool_position': np.asarray(preview['tool_position'], dtype=float),
                'position_error': error, 'up_error': up_error,
                'predicted_up': predicted_up, 'reachable': reachable, 'yaw': float(yaw)}
        if best is None or score < best[0]:
            best = score, item
        return reachable and up_error < 1e-5 and error < .0001

    for degrees in (0., 45., -45., 90., -90., 135., -135., 180.):
        if calls >= max_calls or query(np.deg2rad(degrees)):
            break
    for step in (np.pi/8,):
        if calls >= max_calls or (best[1]['reachable'] and best[1]['up_error'] < 1e-5
                                   and best[1]['position_error'] < .0001):
            break
        center = best[1]['yaw']
        for yaw in (center-step, center+step):
            if calls >= max_calls or query(yaw):
                break
    # The SDK trades orientation against Cartesian error. Compensating its
    # observed angular residual can improve uprightness while the explicit
    # position tolerance still rejects excessive Cartesian compromise.
    for _ in range(2):
        if calls >= max_calls or best[1]['up_error'] < 1e-5:
            break
        current = best[1]
        correction = _grasp_align(current['predicted_up'], world_up)
        if query(current['yaw'], correction @ current['orientation']):
            break
    return best[1] | {'calls': calls}


def grasp_orientation_for_target(api, source_position, target_rotation, *, max_calls=12,
                                 position_tolerance=.003):
    """Match source grasp pitch to an achievable destination tool rotation.

    The initially upright object's tool-frame normal after grasp is
    R_source.T @ world_up. Requiring it to equal R_target.T @ world_up permits
    an upright transfer with different world yaw at source and destination.
    The source position must already include a visually measured grasp offset.
    After lifting, use the actual observed payload normal to refine the target
    through orientation_for_payload; jaw closure can rotate the object.
    """
    target_rotation = _grasp_rotation(target_rotation)
    result = orientation_for_payload(api, source_position,
                                     target_rotation.T @ np.array([0., 0., 1.]),
                                     target_rotation, max_calls=max_calls,
                                     position_tolerance=position_tolerance)
    predicted_up = target_rotation @ result['tool_rotation'].T @ np.array([0., 0., 1.])
    return result | {'transfer_up': predicted_up,
                     'transfer_up_error': float(np.arccos(np.clip(predicted_up[2], -1., 1.)))}
