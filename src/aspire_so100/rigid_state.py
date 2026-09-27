"""CPU rigid cube targets and visual state errors; no simulator or action access."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def cube_vertices(center, rotation, side):
    center, rotation = np.asarray(center, float), np.asarray(rotation, float)
    if (center.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(center).all()
            or not np.isfinite(rotation).all() or not np.isfinite(side) or side <= 0
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or np.linalg.det(rotation) < .999):
        raise ValueError('A finite rigid pose and positive measured cube size are required')
    signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    return center + signs * side / 2 @ rotation.T


def upright_cube_rotation(observed_rotation, up):
    """Preserve an observed horizontal edge heading while making the goal upright."""
    rotation = np.asarray(observed_rotation, float)
    z = np.asarray(up, float)
    z = z / np.linalg.norm(z)
    i = int(np.argmin(np.abs(z @ rotation)))
    x = rotation[:, i] - z * (z @ rotation[:, i])
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def stack_targets(observations, order, table_plane, *, anchor=None):
    """Construct a desired column from measured sizes and a visual base anchor.

    anchor is None or an observation dictionary containing center, rotation,
    position_fully_observed and inferred_orientation; an XYZ array is not valid.
    The anchor can be frozen after a controller relocates the base. Targets are
    constructed goals, never measurements or a statement that stacking succeeded.
    """
    if len(order) < 2 or len(set(order)) != len(order):
        raise ValueError('At least two unique cube identities are required')
    base = observations[order[0]] if anchor is None else anchor
    required = ('center', 'rotation', 'position_fully_observed', 'inferred_orientation')
    if not isinstance(base, dict) or any(key not in base for key in required):
        raise ValueError('anchor must be an observation dict with center, rotation, position_fully_observed and inferred_orientation; XYZ arrays are not anchors')
    cube_vertices(base['center'], base['rotation'], float(observations[order[0]]['height']))
    plane = np.asarray(table_plane, float)
    if plane.shape != (3,) or not np.isfinite(plane).all():
        raise ValueError('A measured tabletop plane is required')
    normal = np.r_[-plane[:2], 1.]
    normal /= np.linalg.norm(normal)
    rotation = upright_cube_rotation(base['rotation'], normal)
    center = np.asarray(base['center'], float).copy()
    targets = {}
    previous_side = None
    for index, name in enumerate(order):
        side = float(observations[name]['height'])
        if previous_side is not None:
            center = center + normal * (previous_side + side) / 2
        vertices = cube_vertices(center, rotation, side)
        targets[name] = {'center': center.copy(), 'rotation': rotation.copy(),
                         'vertices': vertices, 'height': side,
                         'source': 'constructed_task_goal_from_visual_geometry',
                         'support': 'table' if index == 0 else order[index-1],
                         'anchor_position_fully_observed': bool(base['position_fully_observed']),
                         'anchor_orientation_inferred': bool(base['inferred_orientation'])}
        previous_side = side
    return targets


def cube_rotation_error(rotation, target):
    """Smallest angle over the 24 proper cube symmetries, in radians."""
    relative = np.asarray(target).T @ np.asarray(rotation)
    best = np.pi
    for permutation in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for x in (-1., 1.):
            for y in (-1., 1.):
                for z in (-1., 1.):
                    symmetry = np.eye(3)[:, permutation] @ np.diag([x, y, z])
                    if np.linalg.det(symmetry) < 0:
                        continue
                    angle = np.arccos(np.clip((np.trace(relative @ symmetry)-1)/2, -1., 1.))
                    best = min(best, float(angle))
    return best


def state_error(observed, target):
    """Report estimates and observability separately; never certify contact/stability."""
    delta = np.asarray(target['center']) - np.asarray(observed['center'])
    vertices = np.asarray(observed['vertices'])
    desired = np.asarray(target['vertices'])
    distances = np.linalg.norm(vertices[:, None] - desired[None, :], axis=2)
    rows, cols = linear_sum_assignment(distances ** 2)
    return {'translation_to_goal_m': delta, 'center_error_m': float(np.linalg.norm(delta)),
            'xy_error_m': float(np.linalg.norm(delta[:2])),
            'rotation_error_rad': cube_rotation_error(observed['rotation'], target['rotation']),
            'vertex_rms_m': float(np.sqrt(np.mean(distances[rows, cols] ** 2))),
            'pose_fully_observed': bool(observed['geometry_reliable']
                                        and not observed['inferred_orientation']),
            'interpretation': 'visual_estimate_not_contact_or_stability_truth'}


def support_relation(upper, lower):
    """Signed vertex-to-support-plane gap and projected center margin estimates."""
    rotation = np.asarray(lower['rotation'])
    axis = int(np.argmax(abs(rotation[2])))
    normal = rotation[:, axis] * np.sign(rotation[2, axis])
    face = np.asarray(lower['center']) + normal * lower['height']/2
    vertices = np.asarray(upper['vertices'])
    gap = float(np.min((vertices-face) @ normal))
    horizontal = [i for i in range(3) if i != axis]
    offset = rotation[:, horizontal].T @ (np.asarray(upper['center']) - face)
    margin = float(lower['height']/2 - np.max(abs(offset)))
    return {'lowest_vertex_gap_m': gap, 'projected_center_margin_m': margin,
            'both_poses_fully_observed': bool(upper['geometry_reliable'] and lower['geometry_reliable']),
            'contact_or_stability_certified': False}


def geometry_json(value):
    """Convert numerical observations to finite JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return geometry_json(value.tolist())
    if isinstance(value, np.generic):
        return geometry_json(value.item())
    if isinstance(value, dict):
        return {str(k): geometry_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [geometry_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
