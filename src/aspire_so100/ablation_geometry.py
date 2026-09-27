"""Action-free cube observations for the controlled three-cube experiment.

Dimensions are measured from an initial upright tabletop view. Pose observability
is separate from model availability; missing faces never become observed truth.
"""
import numpy as np

from aspire_so100.cube_geometry import estimate_cube_geometry
from aspire_so100.instance_geometry import inventory_points, inventory_table, inventory_color_mask


def observe_cube(frame, color, model=None, table_plane=None, prior=None):
    """Return a measured cube model and a possibly partial visual pose.

    A new model requires an unobstructed upright cube on the table. A stored
    model must be reused after lifting or stacking. With a prior, only observed
    translation components update; without one, hidden components use visible
    extents and can be biased by occlusion. This function never issues actions.
    """
    plane = inventory_table(frame) if table_plane is None else np.asarray(table_plane, float)
    if plane.shape != (3,) or not np.isfinite(plane).all():
        raise ValueError('A finite tabletop plane is required')
    if model is None:
        points, valid = inventory_points(frame)
        elevation = points[..., 2] - points[..., :2] @ plane[:2] - plane[2]
        # Colored image background below the tabletop is not cube evidence.
        mask = valid & inventory_color_mask(frame['rgb'], color)
        mask &= (elevation > .004) & (elevation < .15)
        cloud = points[mask]
        if len(cloud) < 12:
            raise ValueError('Too few colored depth samples to measure cube dimensions')
        heights = cloud[:, 2] - cloud[:, :2] @ plane[:2] - plane[2]
        side = float(np.percentile(heights, 99.5))
        if not .005 < side < .15:
            raise ValueError('Unreliable on-table cube height')
        model = {'height': side, 'dims': np.repeat(side, 2),
                 'source': 'initial_tabletop_rgbd'}
    else:
        side = float(model['height'])
        if not np.isfinite(side) or side <= 0:
            raise ValueError('Stored cube height must be finite and positive')
        model = dict(model, dims=np.repeat(side, 2))
    fit = estimate_cube_geometry(frame, color, side, table_plane=plane)
    center = fit['centroid'].copy()
    constrained = np.asarray(fit['constrained_axes'], bool)
    uses_prior = prior is not None and not np.all(constrained)
    if uses_prior:
        predicted = np.asarray(prior['center'], float)
        if predicted.shape != (3,) or not np.isfinite(predicted).all():
            raise ValueError('Prior center must be a finite three-vector')
        observed_axes = fit['rotation'][:, constrained]
        center = predicted + observed_axes @ (observed_axes.T @ (center - predicted))
    shift = center - fit['centroid']
    return {
        'shape': 'cube', 'center': center, 'normal': fit['top_normal'],
        'rotation': fit['rotation'], 'top': float(fit['top_center'][2] + shift[2]),
        'bottom': float(fit['bottom_z'] + shift[2]),
        'extent_z': float((fit['top_z'] - fit['bottom_z'])/2),
        'height': side, 'dims': np.repeat(side, 2), 'model': model, 'model_valid': True,
        'vertices': fit['vertices'] + shift, 'top_vertices': fit['top_vertices'] + shift,
        'axis_dimensions': np.repeat(side, 3),
        'geometry_reliable': fit['reliable'],
        'observed_face_count': fit['observed_face_count'], 'constrained_axes': constrained,
        'position_fully_observed': bool(np.all(constrained)),
        'inferred_orientation': fit['observed_face_count'] < 2,
        'inferred_axial_position': not bool(np.all(constrained)),
        'position_source': 'prior_and_visible_planes' if uses_prior else 'visible_planes_and_extents',
        'surface_rms_m': fit['plane_rms_m'], 'inlier_fraction': fit['inlier_fraction'],
        'pixel_count': fit['pixel_count'], 'vertices_kind': 'inferred_cube_model',
    }
