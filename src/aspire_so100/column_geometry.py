"""CPU surface models for the column curriculum, using only calibrated RGB-D.

A model stores dimensions learned while an object is visible on the tabletop.
Later fits may infer hidden surfaces from that model and a previously observed
pose propagated by robot proprioception. Such predictions are explicitly marked.
"""
import numpy as np
from scipy.optimize import least_squares

from aspire_so100.instance_geometry import inventory_points, inventory_table, inventory_color_mask
from aspire_so100.cuboid_geometry import estimate_cuboid_geometry


class ColumnGeometryConflict(ValueError):
    """Visible finite surfaces contradict a learned object model or pose."""
    model = None


def _surface_samples(frame, color):
    points, valid = inventory_points(frame)
    mask = valid & inventory_color_mask(frame['rgb'], color)
    cloud = points[mask]
    if len(cloud) < 8:
        raise ValueError('Insufficient visible object depth: ' + color)
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    normals = np.cross(dx, dy)
    lengths = np.linalg.norm(normals, axis=2)
    interior = (mask[1:-1, 1:-1] & mask[1:-1, 2:] & mask[1:-1, :-2]
                & mask[2:, 1:-1] & mask[:-2, 1:-1] & (lengths > 1e-10))
    samples = points[1:-1, 1:-1][interior]
    normals = normals[interior] / lengths[interior, None]
    stride = max(1, int(np.ceil(len(samples) / 500)))
    samples, normals = samples[::stride], normals[::stride]
    camera = np.asarray(frame['camera_to_world'])[:3, 3]
    normals *= np.where(np.sum(normals * (camera - samples), axis=1) >= 0, 1, -1)[:, None]
    return cloud, samples, normals


def _basis(normal):
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    ex = np.array([1., 0., 0.]) - normal * normal[0]
    ex /= np.linalg.norm(ex)
    return np.column_stack((ex, np.cross(normal, ex), normal))


def _circle(points, radius=None, initial=None):
    if len(points) < (6 if radius is None else 3):
        raise ValueError('Insufficient curved side depth')
    origin = np.mean(points, axis=0)
    centered = points - origin
    radial_only_information = None
    if radius is None:
        fit = np.linalg.lstsq(np.column_stack((2 * centered, np.ones(len(points)))),
                             np.sum(centered ** 2, axis=1), rcond=None)[0]
        radius_squared = fit[2] + np.sum(fit[:2] ** 2)
        if radius_squared <= 0:
            raise ValueError('Invalid fitted cylinder radius')
        start = np.r_[fit[:2], np.sqrt(radius_squared)]
        result = least_squares(lambda x: np.linalg.norm(centered - x[:2], axis=1) - x[2],
                               start, loss='soft_l1', f_scale=.0003, max_nfev=25)
        center, radius = origin + result.x[:2], float(result.x[2])
    else:
        if initial is None:
            raise ValueError('A partial fixed-radius fit needs a prior center')
        initial = np.asarray(initial)
        radial = points - initial
        radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-9)
        eigenvalues, eigenvectors = np.linalg.eigh(radial.T @ radial / len(points))
        if eigenvalues[0] < .025:
            radial_only_information = float(eigenvalues[0])
            # A narrow arc observes radial displacement but barely constrains
            # tangential displacement. Preserve the propagated tangent instead
            # of allowing an ill-conditioned circle fit to invent that motion.
            axis = eigenvectors[:, -1]
            result = least_squares(
                lambda x: np.linalg.norm(points - initial - axis * x[0], axis=1) - radius,
                np.zeros(1), bounds=(-.009, .009), loss='soft_l1', f_scale=.0003, max_nfev=25)
            center = initial + axis * result.x[0]
        else:
            result = least_squares(lambda x: np.linalg.norm(points - x, axis=1) - radius,
                                   initial, loss='soft_l1', f_scale=.0003, max_nfev=25)
            center = result.x
    vectors = (points - center) / np.maximum(np.linalg.norm(points - center, axis=1)[:, None], 1e-9)
    angular_information = np.linalg.eigvalsh(vectors.T @ vectors / len(points))[0]
    if radial_only_information is not None:
        angular_information = min(angular_information, radial_only_information)
    rms = float(np.sqrt(np.mean((np.linalg.norm(points - center, axis=1) - radius) ** 2)))
    if (not .004 < radius < .06 or rms > .0007
            or (angular_information < .025 and initial is None)):
        raise ValueError('Curved surface does not constrain a cylinder center')
    return center, float(radius), rms, float(angular_information)


def _cylinder_side_axis(normals, prior_up):
    """Observe a cylinder axis only when curved-side normals span two axes."""
    selected = abs(normals @ prior_up) < .5
    if np.count_nonzero(selected) < 12:
        raise ValueError('Too few side normals to observe the cylinder axis')
    for _ in range(3):
        eigenvalues, eigenvectors = np.linalg.eigh(normals[selected].T @ normals[selected] / np.count_nonzero(selected))
        axis = eigenvectors[:, 0]
        axis *= np.sign(axis @ prior_up)
        residual = abs(normals @ axis)
        selected &= residual < max(.015, float(np.percentile(residual[selected], 85)))
        if np.count_nonzero(selected) < 8:
            raise ValueError('Cylinder side-axis evidence is too sparse')
    eigenvalues, eigenvectors = np.linalg.eigh(normals[selected].T @ normals[selected] / np.count_nonzero(selected))
    axis = eigenvectors[:, 0]
    axis *= np.sign(axis @ prior_up)
    if eigenvalues[0] > .0006 or eigenvalues[1] < .035 or axis @ prior_up < np.cos(np.pi/4):
        raise ValueError('Visible normals do not independently constrain a cylinder axis')
    return axis, eigenvalues


def _cylinder_surface_residual(cloud, center, rotation, radius, height):
    local = (cloud-center) @ rotation
    radial = np.linalg.norm(local[:, :2], axis=1)
    side = np.hypot(radial-radius, np.maximum(abs(local[:, 2])-height/2, 0.))
    cap = np.hypot(abs(local[:, 2])-height/2, np.maximum(radial-radius, 0.))
    return np.minimum(side, cap)


def _cylinder(frame, color, model, table_plane, prior):
    cloud, samples, normals = _surface_samples(frame, color)
    if len(samples) < 6:
        raise ValueError('Too few cylinder surface normals')
    up = np.array([0., 0., 1.]) if prior is None else np.asarray(prior['normal'])
    top_mask = normals @ up > .85
    top_samples = samples[top_mask]
    observed_top = len(top_samples) >= 4
    if observed_top:
        _, singular, vh = np.linalg.svd(top_samples - np.mean(top_samples, axis=0), full_matrices=False)
        candidate = vh[-1]
        candidate *= np.sign(candidate[2])
        if singular[1] > .0005 and candidate[2] > .8:
            up = candidate
        else:
            observed_top = False
    side_mask = abs(normals @ up) < .22
    side_samples = samples[side_mask]
    if not observed_top and prior is None:
        raise ValueError('Cylinder top occluded without a prior visual pose')
    rotation = _basis(up)
    projected = side_samples @ rotation[:, :2]
    initial = None if prior is None else np.asarray(prior['center']) @ rotation[:, :2]
    if model is not None and initial is None:
        # The visible side normals point radially outward. Their offset by the
        # learned radius supplies an initialization without assuming a top mask.
        radial = normals[side_mask] @ rotation[:, :2]
        radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-9)
        initial = np.median(projected - radial * model['radius'], axis=0)
    side_axis_observed = False
    side_axis_eigenvalues = None
    top_only = False
    circle_error = None
    try:
        xy, radius, rms, information = _circle(projected, None if model is None else model['radius'], initial)
    except ValueError as error:
        circle_error = error
    if model is not None and prior is not None and circle_error is not None:
        try:
            candidate, eigenvalues = _cylinder_side_axis(normals, np.asarray(prior['normal']))
            candidate_rotation = _basis(candidate)
            candidate_side = abs(normals @ candidate) < .05
            if np.count_nonzero(candidate_side) < 8:
                raise ValueError('Too few side samples to validate the observed cylinder axis')
            candidate_xy, radius, candidate_rms, candidate_information = _circle(
                samples[candidate_side] @ candidate_rotation[:, :2], model['radius'],
                np.asarray(prior['center']) @ candidate_rotation[:, :2])
            if candidate_rms > .0004 or candidate_information < .035:
                raise ValueError('Cylinder side-axis model does not agree with observed curvature')
            up, rotation, xy = candidate, candidate_rotation, candidate_xy
            rms, information = candidate_rms, candidate_information
            side_axis_observed, side_axis_eigenvalues = True, eigenvalues
            top_samples = samples[normals @ up > .95]
            observed_top = len(top_samples) >= 4
            if observed_top:
                offsets = top_samples @ up
                singular = np.linalg.svd(top_samples-top_samples.mean(axis=0), compute_uv=False)
                observed_top = singular[1] > .0005 and np.std(offsets) < .0004
            circle_error = None
        except ValueError:
            pass
    if circle_error is not None:
        if model is None or prior is None or not observed_top:
            raise circle_error
        # A visible planar cap observes the axis and axial position, but it
        # does not establish the hidden circle center. Retain its prior XY only
        # when all but a small edge fraction still match the finite cylinder.
        rotation = _basis(up)
        xy = np.asarray(prior['center']) @ rotation[:, :2]
        radius, information = model['radius'], 0.
        top_level = float(np.median(top_samples @ up))
        candidate_center = rotation[:, :2] @ xy + up*(top_level-model['height']/2)
        residual = _cylinder_surface_residual(cloud, candidate_center, rotation, radius, model['height'])
        if np.percentile(residual, 90) > .0015 or np.mean(residual < .001) < .8:
            raise ValueError('Visible cylinder cap disagrees with propagated side geometry')
        rms = float(np.sqrt(np.mean((top_samples @ up-top_level)**2)))
        if rms > .0004:
            raise ValueError('Visible cylinder cap is not sufficiently planar')
        top_only = True
    if information < .025 and prior is None:
        raise ValueError('Narrow cylinder arc requires a prior visual pose')
    if observed_top:
        top_level = float(np.median(top_samples @ up))
    else:
        top_level = float(np.asarray(prior['center']) @ up + model['height'] / 2)
    if model is None:
        top_center = rotation[:, :2] @ xy + up * top_level
        ground = float(top_center[:2] @ table_plane[:2] + table_plane[2])
        height = (top_center[2] - ground) / up[2]
        if not .009 < height < .15:
            raise ValueError('Unreliable visual cylinder height')
        model = {'height': float(height), 'dims': np.array([2 * radius, 2 * radius]), 'radius': radius}
    height = model['height']
    center = rotation[:, :2] @ xy + up * (top_level - height / 2)
    extent = float(abs(up[2]) * height / 2 + radius * np.sqrt(max(0., 1 - up[2] ** 2)))
    # Sampled rim vertices are visualization/model points, not measured corners.
    angles = np.arange(32) * (2 * np.pi / 32)
    ring = radius * np.column_stack((np.cos(angles), np.sin(angles))) @ rotation[:, :2].T
    top_vertices = center + up * height / 2 + ring
    vertices = np.concatenate((top_vertices, center - up * height / 2 + ring))
    return {'shape': 'cylinder', 'center': center, 'normal': up, 'height': height,
            'dims': np.asarray(model['dims']), 'model': model, 'top': float((center + up * height / 2)[2]),
            'bottom': float(center[2] - extent), 'extent_z': extent, 'vertices': vertices,
            'top_vertices': top_vertices, 'geometry_reliable': bool(observed_top and information >= .025),
            'observed_top': bool(observed_top), 'surface_rms_m': rms,
            'orientation_observed': bool(observed_top or side_axis_observed),
            'orientation_evidence': 'side_normals' if side_axis_observed else ('top_plane' if observed_top else 'prior'),
            'side_axis_eigenvalues': side_axis_eigenvalues, 'top_only_update': top_only,
            'angular_information': information, 'inferred_axial_position': not observed_top,
            'inferred_translation': bool(not observed_top or information < .025),
            'rotation': rotation,
            'constrained_axes': np.array([information >= .025, information >= .025, observed_top])}


def _cuboid_pixel_size_bounds(frame, cloud, table_plane, height, dims):
    """Bound two silhouette endpoint gaps using calibrated top-plane pixels."""
    pose = np.asarray(frame['camera_to_world'])
    camera = pose[:3, 3]
    rotation = pose[:3, :3]
    intrinsics = np.asarray(frame['intrinsics'])
    optical = (np.asarray(cloud) - camera) @ rotation
    projected = optical @ intrinsics.T
    pixels = projected[:, :2] / projected[:, 2, None]
    corners = pixels[:, None] + np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
    rays = np.concatenate((corners, np.ones((*corners.shape[:2], 1))), axis=2)
    rays = rays @ np.linalg.inv(intrinsics).T @ rotation.T
    normal = np.r_[-np.asarray(table_plane)[:2], 1.]
    distance = float(table_plane[2] + height - normal @ camera)
    denominator = rays @ normal
    if np.any(abs(denominator) < 1e-6):
        raise ValueError('Top-plane pixel footprint is ill-conditioned')
    top_points = camera + rays * (distance / denominator)[..., None]
    footprint = np.max(np.ptp(top_points, axis=1), axis=0)[:2]
    if not np.isfinite(footprint).all() or np.any(footprint <= 0):
        raise ValueError('Invalid calibrated pixel footprint')
    return {'initial_dims_m': np.asarray(dims).copy(),
            'dims_upper_m': np.asarray(dims) + 2 * footprint,
            'dimension_pixel_footprint_m': footprint,
            'dimension_refinement_count': 0}


def column_size_uncertainty(frame, color, model, table_plane):
    """Pixel-derived width bounds for an initial tabletop visual box model."""
    cloud, _, _ = _surface_samples(frame, color)
    return _cuboid_pixel_size_bounds(frame, cloud, np.asarray(table_plane),
                                     float(model['height']), np.asarray(model['dims']))


def _cuboid_dimension_indices(sizes, dimensions, rotation):
    """Associate physical box axes with the learned dimension registry."""
    permutations = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2],
                             [1, 2, 0], [2, 0, 1], [2, 1, 0]])
    costs = np.sum((np.asarray(dimensions)[permutations] - sizes) ** 2, axis=1)
    # Equal lengths have no geometric identity; use the initial upright basis
    # only to choose a deterministic registry ordering for those ties.
    costs -= 1e-12 * np.sum(abs(rotation[permutations, np.arange(3)]), axis=1)
    return permutations[int(np.argmin(costs))]


def _finite_cuboid_center(cloud, rotation, sizes, center, constrained, has_prior, samples, normals, model, dimension_indices):
    """Constrain hidden translations by the finite extent of visible surfaces.

    A missing face plane leaves an axis unobserved, but a learned box must still
    contain the measured surface. Robust point bounds limit its possible center;
    they do not manufacture an additional observed face or change learned sizes.
    """
    local = np.asarray(cloud) @ rotation
    # A convex cuboid lies behind every observed outward face plane. Reject
    # depth-edge samples materially in front of those planes, while retaining
    # legitimate narrow faces that have too few normals to fit independently.
    # Interior samples determine the plane levels and signs; learned sizes and
    # predicted centers never decide which points support their own bounds.
    supported = np.ones(len(local), dtype=bool)
    face_levels, face_signs = np.zeros(3), np.zeros(3)
    for axis in np.flatnonzero(constrained):
        facing = normals @ rotation[:, axis]
        aligned = abs(facing) > np.cos(np.deg2rad(15))
        if np.count_nonzero(aligned) < 3:
            raise ValueError('Insufficient independent face samples for finite bounds')
        level = float(np.median(samples[aligned] @ rotation[:, axis]))
        sign = float(np.sign(np.median(facing[aligned])))
        face_levels[axis], face_signs[axis] = level, sign
        supported &= sign * (local[:, axis] - level) <= .00025
    if np.count_nonzero(supported) < 8:
        raise ValueError('Insufficient observed-plane surface points for finite bounds')
    local = local[supported]
    # Round the central 98% interval inward to actual samples. With fewer
    # than 100 pixels, interpolated percentiles retain part of a single mixed
    # silhouette/depth-edge outlier and can contradict otherwise precise
    # interior face planes. Always exclude at least one extreme per side.
    ordered = np.sort(local, axis=0)
    trim = max(1, int(np.ceil(.01 * (len(ordered) - 1))))
    lo, hi = ordered[trim], ordered[-trim - 1]
    tolerance = .00025
    lower, upper = hi - sizes / 2 - tolerance, lo + sizes / 2 + tolerance
    projected = np.asarray(center) @ rotation
    conflict = None
    if np.any(lower > upper):
        conflict = 'Visible cuboid extent contradicts learned dimensions'
    elif np.any(constrained & ((projected < lower) | (projected > upper))):
        conflict = 'Visible cuboid planes contradict finite surface bounds'
    if conflict is not None:
        required = np.maximum(0., hi - lo - 2 * tolerance)
        for axis in np.flatnonzero(constrained):
            opposite = lo[axis] if face_signs[axis] > 0 else hi[axis]
            required[axis] = max(required[axis],
                                 face_signs[axis] * (face_levels[axis] - opposite) - tolerance)
        error = ColumnGeometryConflict(conflict)
        error.minimum_axis_dimensions = required
        error.dimension_indices = np.asarray(dimension_indices)
        error.model = model
        raise error
    width = upper - lower
    finite = ~constrained & (width <= .0015)
    missing = ~constrained
    if not has_prior and np.any(missing & ~finite):
        raise ValueError('Occluded cuboid axes without a prior visual pose or narrow finite bounds')
    projected[missing] = np.clip(projected[missing], lower[missing], upper[missing])
    # A nearly complete visible edge supplies a much narrower center interval
    # than a stale grasp prior. Use its midpoint instead of preserving its bias.
    projected[finite] = (lower[finite] + upper[finite]) / 2
    return rotation @ projected, {
        'finite_surface_point_count': int(np.count_nonzero(supported)),
        'finite_surface_excluded_point_count': int(np.count_nonzero(~supported)),
        'finite_center_bounds_m': np.column_stack((lower, upper)),
        'finite_center_width_m': width,
        'finite_constrained_axes': finite,
    }


def _estimate_column_geometry(frame, color, shape, model=None, table_plane=None, prior=None):
    """Fit an isolated object's surfaces, returning the controller's block schema.

    The caller must isolate its instance mask before calling this function. New
    dimensions are measured only on the tabletop. With a known model, hidden
    axes may use ``prior``; no simulator state or predefined block sizes enter.
    """
    if table_plane is None:
        table_plane = inventory_table(frame)
    table_plane = np.asarray(table_plane)
    if shape == 'cylinder':
        return _cylinder(frame, color, model, table_plane, prior)
    if shape not in ('cube', 'prism', 'thin_prism'):
        raise ValueError('Unsupported column geometry: ' + shape)
    if model is None:
        cloud, _, _ = _surface_samples(frame, color)
        lo, hi = np.percentile(cloud, [0, 100], axis=0)
        midpoint = (lo + hi) / 2
        height = float(hi[2] - midpoint[:2] @ table_plane[:2] - table_plane[2])
        dims = hi[:2] - lo[:2]
        if not .009 < height < .15 or np.min(dims) < .006:
            raise ValueError('Unreliable visual cuboid dimensions')
        if shape == 'cube':
            dims = np.repeat(height, 2)
        elif shape == 'prism':
            dims[:] = max(dims)
        model = {'height': height, 'dims': dims}
        if shape != 'cube':
            model.update(_cuboid_pixel_size_bounds(frame, cloud, table_plane, height, dims))
    if shape == 'cube':
        # A cube's measured height constrains all three dimensions, including
        # when a legacy partial silhouette supplied narrower initial bounds.
        model = dict(model, dims=np.repeat(model['height'], 2))
    dimensions = np.r_[model['dims'], model['height']]
    geometry = estimate_cuboid_geometry(frame, color, dimensions, table_plane=table_plane)
    if np.count_nonzero(geometry['constrained_axes']) == 1 and prior is not None:
        # One plane constrains its normal and one translation component. Its
        # in-plane rotation is unobservable: preserve that part of the last
        # visual attitude rather than constructing arbitrary world-aligned axes.
        previous_rotation = np.asarray(prior.get('rotation', _basis(prior['normal'])))
        sizes = np.asarray(prior.get('axis_dimensions',
                           dimensions[np.argmax(abs(previous_rotation), axis=0)]))
        dimension_indices = np.asarray(prior.get('axis_dimension_indices',
                            _cuboid_dimension_indices(sizes, dimensions, previous_rotation)))
        observed = geometry['rotation'][:, geometry['constrained_axes']][:, 0]
        axis = int(np.argmax(abs(previous_rotation.T @ observed)))
        observed *= np.sign(previous_rotation[:, axis] @ observed)
        before = previous_rotation[:, axis]
        cross = np.cross(before, observed)
        skew = np.array([[0., -cross[2], cross[1]], [cross[2], 0., -cross[0]],
                         [-cross[1], cross[0], 0.]])
        correction = np.eye(3) + skew + skew @ skew / max(1e-8, 1 + before @ observed)
        rotation = correction @ previous_rotation
        cloud, samples, normals = _surface_samples(frame, color)
        agreement = normals @ observed
        aligned = abs(agreement) > np.cos(np.deg2rad(15))
        if np.count_nonzero(aligned) < 3:
            raise ValueError('Insufficient partial cuboid plane samples')
        sign = float(np.sign(np.median(agreement[aligned])))
        level = float(np.median(samples[aligned] @ observed))
        center = np.asarray(prior['center']).copy()
        center += observed * (level - sign * sizes[axis] / 2 - center @ observed)
        constrained = np.arange(3) == axis
        center, finite_bounds = _finite_cuboid_center(
            cloud, rotation, sizes, center, constrained, True, samples, normals, model, dimension_indices)
        signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
        vertices = center + (signs * sizes / 2) @ rotation.T
        up_axis = int(np.argmax(abs(rotation[2])))
        up_sign = float(np.sign(rotation[2, up_axis]))
        normal = rotation[:, up_axis] * up_sign
        constrained = np.arange(3) == axis
        top_vertices = vertices[signs[:, up_axis] == up_sign]
        return {'shape': shape, 'center': center, 'normal': normal,
                'top': float((center + normal * sizes[up_axis] / 2)[2]),
                'bottom': float(np.min(vertices[:, 2])),
                'extent_z': float(np.ptp(vertices[:, 2]) / 2), 'height': model['height'],
                'dims': np.asarray(model['dims']), 'model': model, 'vertices': vertices,
                'top_vertices': top_vertices, 'geometry_reliable': False,
                'observed_face_count': 1, 'constrained_axes': constrained, **finite_bounds,
                'rotation': rotation, 'axis_dimensions': sizes,
                'axis_dimension_indices': dimension_indices,
                'inferred_axial_position': True, 'inferred_orientation': True,
                'surface_rms_m': float(np.sqrt(np.mean((samples[aligned] @ observed - level) ** 2)))}
    center = geometry['centroid'].copy()
    constrained = geometry['constrained_axes']
    rotation = geometry['rotation']
    dimension_indices = _cuboid_dimension_indices(geometry['dimensions_m'], dimensions, rotation)
    cloud, samples, normals = _surface_samples(frame, color)
    if (prior is not None and 'rotation' in prior and 'axis_dimensions' in prior
            and np.ptp(prior['axis_dimensions']) > .000001):
        # Face extraction orders axes by visible normal clusters, not by object
        # identity. Match those axes to the propagated visual basis so a yawed
        # rectangular prism keeps each learned length on its physical axis.
        agreement = abs(rotation.T @ np.asarray(prior['rotation']))
        permutations = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2],
                                 [1, 2, 0], [2, 0, 1], [2, 1, 0]])
        scores = np.sum(agreement[np.arange(3), permutations], axis=1)
        permutation = permutations[int(np.argmax(scores))]
        if np.min(agreement[np.arange(3), permutation]) < np.cos(np.pi / 4):
            raise ValueError('Cuboid visual axes disagree with their learned correspondence')
        sizes = np.asarray(prior['axis_dimensions'])[permutation]
        prior_indices = np.asarray(prior.get('axis_dimension_indices',
                        _cuboid_dimension_indices(np.asarray(prior['axis_dimensions']), dimensions,
                                                  np.asarray(prior['rotation']))))
        dimension_indices = prior_indices[permutation]
        local_center = center @ rotation
        for axis in np.flatnonzero(constrained):
            facing = normals @ rotation[:, axis]
            aligned = abs(facing) > np.cos(np.deg2rad(15))
            if np.count_nonzero(aligned) < 3:
                raise ValueError('Insufficient depth for learned cuboid axis correspondence')
            sign = float(np.sign(np.median(facing[aligned])))
            level = float(np.median(samples[aligned] @ rotation[:, axis]))
            local_center[axis] = level - sign * sizes[axis] / 2
        center = rotation @ local_center
        signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
        vertices = center + (signs * sizes / 2) @ rotation.T
        up_axis = int(np.argmax(abs(rotation[2])))
        up_sign = float(np.sign(rotation[2, up_axis]))
        normal = rotation[:, up_axis] * up_sign
        relative = (cloud - center) @ rotation
        residual = np.min(abs(abs(relative) - sizes / 2), axis=1)
        inside = np.all(abs(relative) <= sizes / 2 + .002, axis=1)
        geometry = dict(geometry, centroid=center, dimensions_m=sizes,
                        vertices=vertices, top_normal=normal,
                        top_center=center + normal * sizes[up_axis] / 2,
                        top_vertices=vertices[signs[:, up_axis] == up_sign],
                        bottom_z=float(np.min(vertices[:, 2])), top_z=float(np.max(vertices[:, 2])),
                        reliable=bool(np.all(constrained) and geometry['plane_rms_m'] < .0008
                                      and np.mean((residual < .0015) & inside) > .8))
    if not np.all(constrained) and prior is not None:
        observed_axes = rotation[:, constrained]
        predicted = np.asarray(prior['center'])
        center = predicted + observed_axes @ (observed_axes.T @ (center - predicted))
    center, finite_bounds = _finite_cuboid_center(
        cloud, rotation, geometry['dimensions_m'], center, constrained, prior is not None, samples, normals, model, dimension_indices)
    shift = center - geometry['centroid']
    return {'shape': shape, 'center': center, 'normal': geometry['top_normal'],
            'top': float(geometry['top_center'][2] + shift[2]),
            'bottom': float(geometry['bottom_z'] + shift[2]),
            'extent_z': float((geometry['top_z'] - geometry['bottom_z']) / 2),
            'height': model['height'], 'dims': np.asarray(model['dims']), 'model': model,
            'vertices': geometry['vertices'] + shift,
            'top_vertices': geometry['top_vertices'] + shift,
            'geometry_reliable': geometry['reliable'], 'observed_face_count': geometry['observed_face_count'],
            'constrained_axes': constrained, 'rotation': rotation, **finite_bounds,
            'axis_dimensions': geometry['dimensions_m'], 'axis_dimension_indices': dimension_indices,
            'inferred_orientation': False,
            'inferred_axial_position': not np.all(constrained), 'surface_rms_m': geometry['plane_rms_m']}


def estimate_column_geometry(frame, color, shape, model=None, table_plane=None, prior=None):
    """Fit visual geometry, refining uncertain tabletop widths from new surfaces.

    Pixel-derived width intervals are created only during initial visual model
    learning. Later observations may increase horizontal sizes inside those
    intervals; fixed supplied sizes, height and cube dimensions remain strict.
    Neither the caller's model nor its prior is mutated.
    """
    try:
        return _estimate_column_geometry(frame, color, shape, model, table_plane, prior)
    except ColumnGeometryConflict as error:
        learned = error.model
        if shape not in ('prism', 'thin_prism') or learned is None or 'dims_upper_m' not in learned:
            raise
        dimensions = np.r_[learned['dims'], learned['height']]
        updated = dimensions.copy()
        for axis, index in enumerate(error.dimension_indices):
            updated[index] = max(updated[index], error.minimum_axis_dimensions[axis])
        if (updated[2] > dimensions[2] + 1e-9
                or np.any(updated[:2] > np.asarray(learned['dims_upper_m']) + 1e-9)
                or np.max(updated - dimensions) <= 1e-9):
            raise
        # A micron inside the independently measured containment bound avoids
        # repeatedly refining the same width due to floating-point roundoff.
        updated[:2] = np.minimum(updated[:2] + np.where(updated[:2] > dimensions[:2], 1e-6, 0.),
                                np.asarray(learned['dims_upper_m']))
        refined = dict(learned, dims=updated[:2],
                       dimension_refinement_count=learned.get('dimension_refinement_count', 0) + 1)
        refined_prior = prior
        if prior is not None and 'axis_dimensions' in prior and 'rotation' in prior:
            indices = np.asarray(prior.get('axis_dimension_indices',
                                 _cuboid_dimension_indices(np.asarray(prior['axis_dimensions']),
                                                           dimensions, np.asarray(prior['rotation']))))
            refined_prior = dict(prior, axis_dimensions=updated[indices], axis_dimension_indices=indices)
        return _estimate_column_geometry(frame, color, shape, refined, table_plane, refined_prior)


def update_partial_column_geometry(frame, color, shape, model, previous, robot_before, robot, table_plane):
    """Correct robot-propagated visual pose on axes still constrained by depth."""
    rotation = np.asarray(robot['tool_rotation']) @ np.asarray(robot_before['tool_rotation']).T
    position = np.asarray(robot['tool_position'])
    old_position = np.asarray(robot_before['tool_position'])
    prior = dict(previous)
    prior['center'] = position + rotation @ (np.asarray(previous['center']) - old_position)
    prior['normal'] = rotation @ np.asarray(previous['normal'])
    if 'rotation' in previous:
        prior['rotation'] = rotation @ np.asarray(previous['rotation'])
    measured = estimate_column_geometry(frame, color, shape, model, table_plane, prior)
    if np.linalg.norm(measured['center'] - prior['center']) > .008:
        raise ValueError('Visible payload disagrees with its last confirmed grasp')
    slip = bool(np.dot(measured['normal'], prior['normal']) < np.cos(.20))
    if slip and measured.get('orientation_evidence') != 'side_normals':
        raise ValueError('Visible payload orientation disagrees with grasp prediction')
    measured['grasp_slip_detected'] = slip
    return measured


def select_column_levels(tracker, levels, frame):
    """Resolve public goal identities without substituting colors for shapes.

    A low-confidence cylinder can be recovered when its measured curved surface
    fits a circle with distributed radial normals. The public goal does not turn
    absent or conflicting visual evidence into a matching object.
    """
    selected = []
    for level in levels:
        color, shape = level['color'], level['shape']
        matches = []
        for key, track in tracker.tracks.items():
            if not track['visible'] or track['color'] != color or key in selected:
                continue
            if track['shape'] == shape:
                matches.append(key)
            elif track['shape'] == 'unknown' and shape == 'cylinder':
                features = track.get('features', {})
                if features.get('side_planarity', 1.) >= .62 or features.get('side_samples', 0) < 10:
                    continue
                try:
                    fit = estimate_column_geometry(tracker.isolated_frame(frame, key), color, shape,
                                                   table_plane=tracker.plane)
                except ValueError:
                    continue
                if fit['angular_information'] > .05 and fit['surface_rms_m'] < .0004:
                    matches.append(key)
        if len(matches) != 1:
            raise ValueError('No unique visually supported ' + color + ' ' + shape)
        key = matches[0]
        tracker.tracks[key]['shape'] = shape
        selected.append(key)
    return selected
