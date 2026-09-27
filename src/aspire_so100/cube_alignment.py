"""Plan-view cube edge alignment from measured visual geometry only."""
import numpy as np


def cube_plan_corners(model):
    corners = np.asarray(model['top_vertices'], dtype=float)
    center = np.asarray(model['center'], dtype=float)
    if corners.shape != (4, 3) or center.shape != (3,) or not np.isfinite(corners).all() or not np.isfinite(center).all():
        raise ValueError('Four finite measured cube-face corners are required')
    # Remove the top face's tilt-induced center displacement: compare body axes.
    xy = corners[:, :2] - corners[:, :2].mean(axis=0) + center[:2]
    order = np.argsort(np.arctan2(xy[:, 1]-center[1], xy[:, 0]-center[0]))
    return xy[order]


def cube_edge_alignment(upper, lower):
    """Match square edges modulo quarter turns, without using semantic axes."""
    a, b = cube_plan_corners(upper), cube_plan_corners(lower)
    variants = [np.roll(a, shift, axis=0)-b for shift in range(4)]
    distances = [np.linalg.norm(v, axis=1) for v in variants]
    best = distances[int(np.argmin([np.mean(v*v) for v in distances]))]
    ea, eb = a[1]-a[0], b[1]-b[0]
    yaw = np.arctan2(ea[1], ea[0])-np.arctan2(eb[1], eb[0])
    yaw = (yaw+np.pi/4) % (np.pi/2)-np.pi/4
    xy = np.asarray(upper['center'])[:2]-np.asarray(lower['center'])[:2]
    return {'xy_error_m': float(np.linalg.norm(xy)), 'xy_offset_m': xy,
            'yaw_error_rad': float(yaw), 'corner_max_m': float(np.max(best)),
            'corner_rms_m': float(np.sqrt(np.mean(best*best)))}


def cube_alignment_observable(model):
    if (model.get('shape') != 'cube' or 'top_vertices' not in model
            or np.asarray(model['normal'])[2] <= .99):
        return False
    if model.get('observed_face_count', 0) >= 2 and not model.get('inferred_orientation', False):
        return True
    # One visible vertical side plane observes horizontal heading; an earlier
    # upright pose supplies the hidden tilt. Missing translation axes must have
    # narrow bounds from the currently visible finite surface, not the prior.
    if 'rotation' not in model or 'finite_center_width_m' not in model:
        return False
    rotation = np.asarray(model['rotation'])
    constrained = np.asarray(model['constrained_axes'], dtype=bool)
    horizontal_normal = np.any(np.linalg.norm(rotation[:2, constrained], axis=0) > .99)
    widths = np.where(constrained, 0., np.asarray(model['finite_center_width_m']))
    uncertainty = abs(rotation[:2]) @ widths / 2
    return bool(horizontal_normal and np.max(uncertainty) <= .001)
