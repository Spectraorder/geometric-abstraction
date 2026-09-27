"""Compact current-view constraints; never manufacture an executable point pose.

Rows of basis_world project a world point into the fitted orthonormal frame:
    lower_m <= basis_world @ center_world <= upper_m.
Bounds are conditional on current association, orientation and visual dimensions;
they are not calibrated confidence intervals. Only the active target is retained.
"""
import numpy as np


def partial_geometry_packet(color, frames, descriptor, model, plane):
    fit = fit_current_cuboid_constraints(frames, color, descriptor['shape'], model, plane)
    rotation = np.asarray(fit['rotation'], dtype=float)
    bounds = np.asarray(fit['finite_center_bounds_m'], dtype=float)
    axes = np.asarray(fit['constrained_axes'], dtype=bool)
    if (rotation.shape != (3, 3) or bounds.shape != (3, 2) or axes.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] > bounds[:, 1])):
        raise ValueError('Invalid current cuboid constraints')
    # No center, vertices, pose acceptance, or prediction is exported. The
    # unchanged executor remains unable to use these as a point observation.
    return {'object': color, 'time': float(frames[0][1]['time']),
            'kind': 'current_visual_center_constraints',
            'basis_world': np.round(rotation.T, 6).tolist(),
            'bounds_m': np.round(bounds, 6).tolist(),
            'face_constrained_axes': axes.tolist(),
            'visual_dimensions_m': np.round(fit['axis_dimensions'], 6).tolist(),
            'dimension_refinements': int(fit['model'].get('dimension_refinement_count', 0)),
            'condition': 'fitted_orientation_and_visual_dimensions',
            'pose_available': False}


def current_partial_geometry(packet, color, now):
    """Return a same-target, same-time constraint packet, never stale memory."""
    if packet is None or packet['object'] != color:
        return None
    if not np.isfinite(now) or abs(packet['time']-now) > 1e-6:
        return None
    return metric_copy(packet, plain=True)
