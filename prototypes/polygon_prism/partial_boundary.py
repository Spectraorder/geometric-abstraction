"""Current visible-edge registration against an RGB-D initialized convex footprint.

No previous pose, tool command or object catalog is used. Hidden vertices are
model-derived, and acceptance requires independent edges plus current visibility.
"""
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull


def prism_clear_boundary(frame, evidence):
    """Keep half-pixel transitions backed by a current drop below the cap plane."""
    cap, valid, points = evidence['cap'], evidence['valid'], evidence['points']
    level, normal = evidence['level'], evidence['normal']
    below = valid & ((points@normal) < level-.0008)
    pixels = []
    for axis in (0, 1):
        first, second = ((np.s_[:-1, :], np.s_[1:, :]) if axis == 0 else
                         (np.s_[:, :-1], np.s_[:, 1:]))
        boundary = (cap[first] & below[second]) | (below[first] & cap[second])
        row, col = np.where(boundary)
        pixels.append(np.column_stack((col+(.5 if axis == 1 else 0),
                                        row+(.5 if axis == 0 else 0), np.ones(len(row)))))
    pixels = np.concatenate(pixels)
    if len(pixels) < 12:
        raise ValueError('Partial cap boundary has too few unoccluded samples')
    pose = np.asarray(frame['camera_to_world'])
    rays = pixels@np.linalg.inv(frame['intrinsics']).T@pose[:3, :3].T
    scale = (level-pose[:3, 3]@normal)/(rays@normal)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Unresolved partial boundary projection')
    world = pose[:3, 3]+scale[:, None]*rays
    return (world-evidence['origin'])@evidence['basis']


def prism_visible_lines(boundary, pixel_size):
    """Extract supporting straight lines from the remaining convex boundary."""
    singular = np.linalg.svd(boundary-boundary.mean(axis=0), compute_uv=False)
    if singular[-1] < 1e-6*singular[0]:
        raise ValueError('Partial cap boundary does not establish two independent edges')
    hull = ConvexHull(boundary)
    lines = []
    for normal, level in zip(hull.equations[:, :2], -hull.equations[:, 2]):
        for repeat in range(2):
            selected = abs(boundary@normal-level) <= .7*pixel_size
            if np.count_nonzero(selected) < 5:
                break
            samples = boundary[selected]
            center = samples.mean(axis=0)
            _, _, vh = np.linalg.svd(samples-center, full_matrices=False)
            fitted = vh[-1]
            fitted *= 1. if fitted@normal > 0 else -1.
            normal, level = fitted, float(fitted@center)
        else:
            span = float(np.ptp(samples@np.array([-normal[1], normal[0]])))
            if span < max(.004, 5*pixel_size):
                continue
            if np.max(boundary@normal-level) > pixel_size:
                continue
            if any(normal@row['normal'] > np.cos(np.deg2rad(5)) and abs(level-row['level']) < pixel_size for row in lines):
                continue
            lines.append({'normal': normal, 'level': level, 'span': span})
    if len(lines) < 2:
        raise ValueError('Partial cap boundary does not establish two independent edges')
    return lines


def prism_segment_residual(points, polygon):
    edge = np.roll(polygon, -1, axis=0)-polygon
    delta = points[:, None]-polygon[None]
    fraction = np.sum(delta*edge[None], axis=2)/np.sum(edge**2, axis=1)
    closest = polygon[None]+np.clip(fraction, 0, 1)[..., None]*edge[None]
    residual = points[:, None]-closest
    index = np.argmin(np.sum(residual**2, axis=2), axis=1)
    return residual[np.arange(len(points)), index], index


def prism_registration_residual(parameters, original, boundary, sides):
    polygon = prism_transform_polygon(original, parameters)
    values = [prism_segment_residual(boundary, polygon)[0].ravel()/np.sqrt(len(boundary))]
    if len(sides):
        values.append(prism_segment_residual(sides, polygon)[0].ravel()/np.sqrt(len(sides)))
    return np.concatenate(values)


def prism_transform_polygon(polygon, parameters):
    angle, x, y = parameters
    cosine, sine = np.cos(angle), np.sin(angle)
    return polygon@np.array([[cosine, sine], [-sine, cosine]])+[x, y]


def prism_registration_visible(frame, evidence, polygon):
    """Every predicted interior cap pixel must be observed or explicitly occluded."""
    normal, level = evidence['normal'], evidence['level']
    world = evidence['origin']+polygon@evidence['basis'].T
    pose = np.asarray(frame['camera_to_world'])
    camera = (world-pose[:3, 3])@pose[:3, :3]
    if np.any(camera[:, 2] <= 0):
        return False
    projected = camera@np.asarray(frame['intrinsics']).T
    projected = projected[:, :2]/projected[:, 2:3]
    height, width = evidence['valid'].shape
    lo, hi = np.floor(projected.min(axis=0)).astype(int), np.ceil(projected.max(axis=0)).astype(int)
    if np.any(lo < 1) or np.any(hi >= [width-1, height-1]):
        return False
    row, col = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
    pixels = np.column_stack((col.ravel(), row.ravel(), np.ones(row.size)))
    rays = pixels@np.linalg.inv(frame['intrinsics']).T@pose[:3, :3].T
    depth = (level-pose[:3, 3]@normal)/(rays@normal)
    intersections = pose[:3, 3]+depth[:, None]*rays
    local = (intersections-evidence['origin'])@evidence['basis']
    hull = ConvexHull(polygon)
    # Ignore a single-pixel border whose occupancy is limited by raster sampling.
    inside = np.max(local@hull.equations[:, :2].T+hull.equations[:, 2], axis=1) < -evidence['pixel_size']
    if np.count_nonzero(inside) < 20:
        return False
    rows, cols = row.ravel()[inside], col.ravel()[inside]
    expected = depth[inside]
    actual = np.asarray(frame['depth_m'])[rows, cols]
    excluded = np.asarray(frame.get('robot_excluded_depth', np.zeros((height, width), bool)))[rows, cols]
    known = (evidence['valid'][rows, cols] | excluded) & np.isfinite(actual) & (actual > 0)
    if not np.all(known):
        return False
    visible = evidence['cap'][rows, cols] & (abs(actual-expected) < .0015)
    hidden = actual < expected-.0015
    return bool(np.mean(visible) >= .35 and np.mean(visible | hidden) >= .98)


def prism_register_partial_boundary(frame, model, evidence, candidates_only=False):
    if model.get('kind') != 'visual_polygon_prism_v1':
        raise ValueError('Partial boundary requires an initialized polygon model')
    original = np.asarray(model['footprint_local_m'], float)
    if original.ndim != 2 or original.shape[1] != 2 or not 3 <= len(original) <= 12 or not np.isfinite(original).all():
        raise ValueError('Invalid initialized polygon boundary')
    pixel = evidence['pixel_size']
    boundary = prism_clear_boundary(frame, evidence)
    lines = prism_visible_lines(boundary, pixel)
    edges = np.roll(original, -1, axis=0)-original
    normals = np.column_stack((edges[:, 1], -edges[:, 0]))/np.linalg.norm(edges, axis=1)[:, None]
    levels = np.sum(original*normals, axis=1)
    angles = np.arctan2(normals[:, 1], normals[:, 0])
    cap_points = (evidence['points'][evidence['cap']]-evidence['origin'])@evidence['basis']
    cloud = evidence['points'][evidence['mask']]
    axial = cloud@evidence['normal']-evidence['level']
    sides = (cloud[(axial < -.002) & (axial > -float(model['height'])+.002)]-evidence['origin'])@evidence['basis']
    sides = sides[::max(1, int(np.ceil(len(sides)/160)))]
    candidates, starts = [], []
    for i, first in enumerate(lines):
        for second in lines[i+1:]:
            matrix = np.array([first['normal'], second['normal']])
            if abs(np.linalg.det(matrix)) < .3:
                continue
            observed_angles = np.arctan2(matrix[:, 1], matrix[:, 0])
            for a in range(len(original)):
                for b in range(len(original)):
                    if a == b:
                        continue
                    changes = observed_angles-angles[[a, b]]
                    mismatch = np.arctan2(np.sin(changes[0]-changes[1]), np.cos(changes[0]-changes[1]))
                    if abs(mismatch) > np.deg2rad(12):
                        continue
                    angle = np.arctan2(np.sin(changes).sum(), np.cos(changes).sum())
                    translation = np.linalg.solve(matrix, np.array([first['level'], second['level']])-levels[[a, b]])
                    start = np.r_[angle, translation]
                    if any(np.linalg.norm(prism_transform_polygon(original, start)-old) < pixel for old in starts):
                        continue
                    polygon = prism_transform_polygon(original, start)
                    starts.append(polygon)
                    residual, _ = prism_segment_residual(boundary, polygon)
                    if np.sqrt(np.mean(np.sum(residual**2, axis=1))) > 2.5*pixel:
                        continue
                    result = least_squares(prism_registration_residual, start, args=(original, boundary, sides),
                                           max_nfev=40, x_scale=[1., .03, .03], ftol=1e-8, xtol=1e-8, gtol=1e-9)
                    polygon = prism_transform_polygon(original, result.x)
                    residual, index = prism_segment_residual(boundary, polygon)
                    rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
                    if rms > .85*pixel or np.max(np.linalg.norm(residual, axis=1)) > 2*pixel:
                        continue
                    hull = ConvexHull(polygon)
                    if np.max(cap_points@hull.equations[:, :2].T+hull.equations[:, 2]) > 1.5*pixel:
                        continue
                    supported = []
                    transformed_edges = np.roll(polygon, -1, axis=0)-polygon
                    for k in np.unique(index):
                        selected = boundary[index == k]
                        direction = transformed_edges[k]/np.linalg.norm(transformed_edges[k])
                        if len(selected) >= 5 and np.ptp(selected@direction) >= max(.004, 5*pixel):
                            supported.append(direction)
                    if len(supported) < 2 or max(abs(np.linalg.det(np.array([u, v]))) for u in supported for v in supported) < .3:
                        continue
                    # Scale yaw by object radius so rank tests use meter displacements.
                    radius = np.max(np.linalg.norm(original, axis=1))
                    singular = np.linalg.svd(result.jac/np.array([radius, 1., 1.]), compute_uv=False)
                    if singular[-1] < .12*singular[0]:
                        continue
                    if not prism_registration_visible(frame, evidence, polygon):
                        continue
                    try:
                        prism_record(frame, model, evidence, polygon, {})
                    except ValueError:
                        continue
                    candidates.append({'polygon': polygon, 'rms': rms, 'edges': len(supported),
                                       'condition': float(singular[0]/singular[-1])})
    if not candidates:
        raise ValueError('Partial cap boundary has no sufficiently constrained depth-consistent registration')
    if candidates_only:
        return candidates
    return prism_select_registration(candidates, pixel)


def prism_select_registration(candidates, pixel):
    """Equivalent symmetries may remain; different geometric completions may not."""
    candidates.sort(key=lambda item: item['rms'])
    best = candidates[0]
    center = point_polygon_moments(best['polygon'])[1]
    spread = 0.
    for candidate in candidates[1:]:
        spread = max(spread, float(np.linalg.norm(point_polygon_moments(candidate['polygon'])[1]-center)))
        distances = np.linalg.norm(best['polygon'][:, None]-candidate['polygon'][None], axis=2)
        if max(np.min(distances, axis=0).max(), np.min(distances, axis=1).max()) > max(.003, 2.5*pixel):
            raise ValueError('Partial cap boundary admits ambiguous shape registrations')
    if spread > max(.002, 1.5*pixel):
        raise ValueError('Partial cap boundary does not uniquely constrain the object center')
    return best['polygon'], {'registration_supported_edges': best['edges'],
                             'registration_edge_rms_m': best['rms'], 'registration_condition': best['condition'],
                             'registration_candidate_count': len(candidates), 'registration_center_spread_m': spread}


def prism_secondary_surface_consistent(primary, polygon, secondary, model):
    """Use another current view's measured side surfaces to reject correspondences."""
    cloud = secondary['points'][secondary['mask']]
    axial = cloud@primary['normal']-primary['level']
    if np.any(axial > .002) or np.any(axial < -float(model['height'])-.002):
        return False
    local = (cloud-primary['origin'])@primary['basis']
    hull = ConvexHull(polygon)
    distances = local@hull.equations[:, :2].T+hull.equations[:, 2]
    tolerance = max(.001, 1.5*secondary['pixel_size'])+3*secondary['rms']
    if np.any(np.max(distances, axis=1) > tolerance):
        return False
    side = axial < -.002
    if np.count_nonzero(side) < 20:
        return False
    return bool(np.mean(np.min(abs(distances[side]), axis=1) > tolerance) <= .1)


def observe_partial_polygon_views(frames, color, model, table_plane):
    """Disambiguate a constrained current fit using already captured same-time views."""
    stamps = np.asarray([float(frame['time']) for _, frame in frames])
    if len(frames) < 2 or not np.isfinite(stamps).all() or np.ptp(stamps) > 1e-6:
        raise ValueError('Polygon registration requires at least two same-time views')
    if len({view for view, _ in frames}) != len(frames):
        raise ValueError('Polygon registration requires distinct current views')
    poses = [np.asarray(frame['camera_to_world']) for _, frame in frames]
    if all(np.allclose(pose, poses[0], atol=1e-8, rtol=0) for pose in poses[1:]):
        raise ValueError('Repeated camera geometry is not another current view')
    observations = []
    for view, frame in frames:
        try:
            observations.append((view, frame, prism_cap_evidence(frame, color, table_plane)))
        except ValueError:
            continue
    if len(observations) < 2:
        raise ValueError('Insufficient current polygon evidence across views')
    for view, frame, primary in observations:
        try:
            candidates = prism_register_partial_boundary(frame, model, primary, candidates_only=True)
        except ValueError:
            continue
        other = [(name, evidence) for name, _, evidence in observations if name != view]
        accepted = [candidate for candidate in candidates if all(
            prism_secondary_surface_consistent(primary, candidate['polygon'], evidence, model) for _, evidence in other)]
        if not accepted:
            continue
        try:
            polygon, registration = prism_select_registration(accepted, primary['pixel_size'])
        except ValueError:
            continue
        registration['registration_views'] = [view]+[name for name, _ in other]
        measured = prism_record(frame, model, primary, polygon, registration)
        return dict(measured, view='joint_current_polygon_edges', source='current_multiview_partial_polygon_registration')
    raise ValueError('Current polygon views do not resolve the remaining registration ambiguity')
