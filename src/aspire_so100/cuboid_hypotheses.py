"""Current-view cuboid dimension correspondence with explicit ambiguity.

A rotated rectangular object need not preserve the nearest-world-axis assignment
of its learned lengths. Test all six assignments against visible face planes and
finite surface bounds. Opposite face signs are kept separate across cameras.
No prior object pose, commanded displacement or simulator state is consumed.
"""
import numpy as np
from aspire_so100.column_geometry import _surface_samples
from aspire_so100.cuboid_geometry import estimate_cuboid_geometry
from aspire_so100.instance_geometry import inventory_points, inventory_color_mask


def cuboid_dimension_hypotheses(cloud, samples, normals, rotation, model, tolerance_m=.00025, point_groups=None):
    """Return every compatible measured-size assignment, including wide intervals.

    Finite bounds are conditional on association, orientation and learned sizes;
    they are not calibrated confidence intervals. This function does not choose
    a hidden center when its coordinate remains ambiguous.
    """
    cloud, samples, normals, rotation = (np.asarray(x, float) for x in (cloud, samples, normals, rotation))
    if (cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 8 or samples.shape != normals.shape
            or samples.ndim != 2 or samples.shape[1] != 3 or rotation.shape != (3, 3)
            or not all(np.isfinite(x).all() for x in (cloud, samples, normals, rotation))
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5)
            or np.linalg.det(rotation) < .999 or not np.isfinite(tolerance_m) or tolerance_m <= 0):
        raise ValueError('Invalid cuboid surface evidence')
    dimensions = np.r_[model['dims'], model['height']].astype(float)
    upper_dimensions = np.r_[model.get('dims_upper_m', model['dims']), model['height']].astype(float)
    if (dimensions.shape != (3,) or np.any(dimensions <= 0) or not np.isfinite(dimensions).all()
            or upper_dimensions.shape != (3,) or not np.isfinite(upper_dimensions).all()
            or np.any(upper_dimensions < dimensions)):
        raise ValueError('Invalid visually learned size intervals')
    levels = {}; constrained = np.zeros(3, bool); residuals = []
    local = cloud@rotation; supported = np.ones(len(cloud), bool)
    for axis in range(3):
        facing = normals@rotation[:, axis]
        for sign in (-1., 1.):
            selected = sign*facing > np.cos(np.deg2rad(12))
            if np.count_nonzero(selected) < 4:
                continue
            projected = samples[selected]@rotation[:, axis]
            level = float(np.median(projected))
            core = abs(projected-level) <= .0008
            if np.count_nonzero(core) < 4 or np.mean(core) < .8:
                raise ValueError('Current views disagree on a cuboid face plane')
            levels[(axis, sign)] = level; constrained[axis] = True
            residuals.extend((projected[core]-level).tolist())
            supported &= sign*(local[:, axis]-level) <= tolerance_m
    if np.count_nonzero(constrained) < 2:
        raise ValueError('Two independent current face directions are required')
    groups = np.zeros(len(cloud), int) if point_groups is None else np.asarray(point_groups)
    if groups.shape != (len(cloud),) or not np.issubdtype(groups.dtype, np.integer):
        raise ValueError('Point groups must identify each input camera cloud')
    ranges = []
    for group in np.unique(groups):
        points = local[supported & (groups == group)]
        if len(points) < 8:
            continue
        ordered = np.sort(points, axis=0); trim = max(1, int(np.ceil(.01*(len(points)-1))))
        ranges.append((ordered[trim], ordered[-trim-1]))
    if not ranges:
        raise ValueError('Insufficient points behind the observed face planes')
    # Trim within each camera before combining bounds. A dense wrist view must
    # not erase the only visible far edge in a much sparser side view.
    lo = np.min([pair[0] for pair in ranges], axis=0)
    hi = np.max([pair[1] for pair in ranges], axis=0)
    local = local[supported]
    permutations = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    hypotheses = []
    for perm in permutations:
        sizes = dimensions[list(perm)].copy(); refined = dimensions.copy(); valid = True
        for axis, index in enumerate(perm):
            if (axis, -1.) in levels and (axis, 1.) in levels:
                span = levels[(axis, 1.)]-levels[(axis, -1.)]
                if span < dimensions[index]-2*tolerance_m or span > upper_dimensions[index]+2*tolerance_m:
                    valid = False; break
                sizes[axis] = float(np.clip(span, dimensions[index], upper_dimensions[index]))
                refined[index] = sizes[axis]
        if not valid:
            continue
        # Single-sided planes plus opposite silhouette points can establish a
        # lower size bound. Refine only inside the initial pixel-derived range;
        # the initially measured height remains fixed.
        for axis, index in enumerate(perm):
            required = hi[axis]-lo[axis]-2*tolerance_m
            for (i, sign), level in levels.items():
                if i == axis:
                    opposite = lo[axis] if sign > 0 else hi[axis]
                    required = max(required, sign*(level-opposite)-tolerance_m)
            if required > upper_dimensions[index]+1e-9:
                valid = False; break
            if required > sizes[axis]:
                sizes[axis] = min(required+1e-6, upper_dimensions[index])
                refined[index] = sizes[axis]
        if not valid:
            continue
        lower, upper = hi-sizes/2-tolerance_m, lo+sizes/2+tolerance_m
        if np.any(lower > upper):
            continue
        center_local = (lower+upper)/2
        for axis in np.flatnonzero(constrained):
            estimates = [level-sign*sizes[axis]/2 for (i, sign), level in levels.items() if i == axis]
            if np.ptp(estimates) > 2*tolerance_m:
                valid = False; break
            center_local[axis] = float(np.mean(estimates))
            if not lower[axis] <= center_local[axis] <= upper[axis]:
                valid = False; break
        if not valid:
            continue
        # Equal lengths carry no physical axis identity. Do not count their
        # permutation as a separate shape/pose hypothesis.
        if any(np.allclose(h['axis_dimensions'], sizes, atol=1e-10, rtol=0) for h in hypotheses):
            continue
        updated_model = dict(model, dims=refined[:2])
        if np.any(refined > dimensions+1e-9):
            updated_model['dimension_refinement_count'] = model.get('dimension_refinement_count', 0)+1
        hypotheses.append({'center': rotation@center_local, 'rotation': rotation.copy(),
                           'axis_dimensions': sizes, 'axis_dimension_indices': np.array(perm),
                           'model': updated_model, 'constrained_axes': constrained.copy(),
                           'finite_center_bounds_m': np.column_stack((lower, upper)),
                           'finite_center_width_m': upper-lower,
                           'finite_surface_point_count': len(local),
                           'finite_surface_excluded_point_count': int(len(cloud)-len(local)),
                           'surface_rms_m': float(np.sqrt(np.mean(np.square(residuals))))})
    return hypotheses


def observe_cuboid_correspondence(frames, color, shape, model, table_plane, max_hidden_width_m=.0015):
    """Resolve a unique current-view correspondence with bounded hidden position.

    Only previously measured cuboid dimensions are accepted. Views must share a
    timestamp. Orientation comes from a view with at least two fitted faces; a
    single plane cannot invent the in-plane rotation. Unknown correspondences or
    wide hidden intervals remain unavailable rather than using an old pose.
    """
    if model is None or shape not in ('prism', 'thin_prism'):
        raise ValueError('Previously measured noncube box dimensions are required')
    if not frames or not np.isfinite(max_hidden_width_m) or max_hidden_width_m <= 0:
        raise ValueError('Current views and a positive interval threshold are required')
    stamps = np.asarray([float(frame['time']) for _, frame in frames])
    if not np.isfinite(stamps).all() or np.ptp(stamps) > 1e-6:
        raise ValueError('All views must have the same finite timestamp')
    clouds = []; samples = []; normals = []; orientations = []; used = []
    for view, frame in frames:
        points, valid = inventory_points(frame)
        height = points[..., 2]-points[..., :2]@np.asarray(table_plane)[:2]-table_plane[2]
        mask = valid & inventory_color_mask(frame['rgb'], color) & (height > .004) & (height < .35)
        frame = dict(frame, valid_depth=mask)
        try:
            cloud, point, normal = _surface_samples(frame, color)
        except ValueError:
            continue
        clouds.append(cloud); samples.append(point); normals.append(normal); used.append(view)
        try:
            raw = estimate_cuboid_geometry(frame, color, np.r_[model['dims'], model['height']], table_plane=table_plane)
            if np.count_nonzero(raw['constrained_axes']) >= 2:
                orientations.append(raw)
        except ValueError:
            continue
    if not orientations:
        raise ValueError('No current view establishes two independent cuboid faces')
    basis = max(orientations, key=lambda fit: (np.count_nonzero(fit['constrained_axes']), -fit['plane_rms_m']))
    hypotheses = cuboid_dimension_hypotheses(np.concatenate(clouds), np.concatenate(samples),
                                             np.concatenate(normals), basis['rotation'], model,
                                             point_groups=np.concatenate([np.full(len(c), i, int) for i, c in enumerate(clouds)]))
    if len(hypotheses) != 1:
        raise ValueError('Cuboid dimension correspondence is ambiguous or inconsistent: '+str(len(hypotheses)))
    fit = hypotheses[0]; axes = fit['constrained_axes']; widths = fit['finite_center_width_m']
    if np.any(widths[~axes] > max_hidden_width_m):
        raise ValueError('Unique cuboid correspondence still has wide hidden position intervals')
    rotation, center, sizes = fit['rotation'], fit['center'], fit['axis_dimensions']
    signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    vertices = center+(signs*sizes/2)@rotation.T
    up = int(np.argmax(abs(rotation[2]))); sign = float(np.sign(rotation[2, up])); normal = sign*rotation[:, up]
    fit.update(shape=shape, normal=normal, height=float(model['height']), dims=np.asarray(fit['model']['dims']),
               vertices=vertices, top_vertices=vertices[signs[:, up] == sign],
               top=float((center+normal*sizes[up]/2)[2]), bottom=float(vertices[:, 2].min()),
               extent_z=float(np.ptp(vertices[:, 2])/2), geometry_reliable=bool(np.all(axes)),
               observed_face_count=int(np.count_nonzero(axes)), position_fully_observed=bool(np.all(axes)),
               inferred_orientation=False, inferred_axial_position=not bool(np.all(axes)),
               finite_constrained_axes=~axes & (widths <= max_hidden_width_m),
               vertices_kind='inferred_rigid_box_corners', correspondence_hypotheses=1,
               view='+'.join(used), coordinate_frame='world', units='meters', observation_time=float(stamps[0]),
               source='current_surface_cuboid_correspondence', temporal_pose_prior_used=False)
    return fit
