"""Development-only interval geometry from visible points inside a rigid box.

This prototype is excluded from the frozen policy experiments. It assumes the
provided points belong to one object and the measured orientation/dimensions
are sufficiently accurate for the caller's explicit tolerance. It does not
certify contact, infer an action, or read simulator state.
"""
import numpy as np


def containment_bounds(points,rotation,dimensions,tolerance_m=.0015):
    """Bound local center coordinates using point containment, without a pose prior.

    For each axis, p_max - half_width <= center <= p_min + half_width.
    Missing surfaces widen the interval instead of becoming zero-noise evidence.
    Tolerance covers assumed point, dimension and orientation errors; these are
    conditional geometric bounds, not statistically calibrated intervals.
    """
    points=np.asarray(points,dtype=float);rotation=np.asarray(rotation,dtype=float)
    dimensions=np.asarray(dimensions,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<1 or not np.isfinite(points).all():
        raise ValueError('Require finite visible 3D points')
    if rotation.shape!=(3,3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or np.linalg.det(rotation)<0:
        raise ValueError('Require a proper rotation')
    if dimensions.shape!=(3,) or not np.isfinite(dimensions).all() or np.any(dimensions<=0):
        raise ValueError('Require positive measured box dimensions')
    if not np.isfinite(tolerance_m) or tolerance_m<0:raise ValueError('Require nonnegative tolerance')
    local=points@rotation
    lower=local.max(axis=0)-dimensions/2-tolerance_m
    upper=local.min(axis=0)+dimensions/2+tolerance_m
    consistent=bool(np.all(lower<=upper))
    return {'consistent':consistent,'center_local_lower_m':lower,'center_local_upper_m':upper,
            'center_world_midpoint_m':rotation@((lower+upper)/2) if consistent else None,
            'interval_width_m':upper-lower,'tolerance_m':float(tolerance_m),
            'semantics':'conditional_on_object_membership_orientation_and_dimensions'}


def world_center_interval(bounds,rotation):
    """Enclose the local center interval in a world-axis box without independence claims."""
    if not bounds['consistent']:raise ValueError('No interval for inconsistent evidence')
    rotation=np.asarray(rotation,dtype=float)
    lower=np.asarray(bounds['center_local_lower_m']);upper=np.asarray(bounds['center_local_upper_m'])
    midpoint=rotation@((lower+upper)/2);radius=np.abs(rotation)@((upper-lower)/2)
    return midpoint-radius,midpoint+radius


def above_interval(upper,upper_rotation,lower,lower_rotation):
    """Three-valued center-height order from conditional geometric intervals."""
    if not upper.get('consistent',False) or not lower.get('consistent',False):
        return {'predicate':'above','truth':'unknown','reason':'bounds_unavailable'}
    up_min,up_max=world_center_interval(upper,upper_rotation)
    low_min,low_max=world_center_interval(lower,lower_rotation)
    interval=np.array([up_min[2]-low_max[2],up_max[2]-low_min[2]])
    truth='true' if interval[0]>0 else ('false' if interval[1]<=0 else 'unknown')
    return {'predicate':'above','truth':truth,'center_height_difference_interval_mm':(interval*1000).tolist(),
            'semantics':'conditional_center_order_not_contact_or_stability'}
