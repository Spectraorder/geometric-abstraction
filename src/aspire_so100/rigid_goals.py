"""Task-independent desired rigid transforms from point and axis constraints.

Action-free numerical goal operators. Inputs are
caller-supplied measured geometric references and desired references. Outputs
are desired geometry, never observations, contact facts, or robot commands.
"""
import numpy as np


def vector(value,name,unit=False):
    result=np.asarray(value,dtype=float)
    if result.shape!=(3,) or not np.isfinite(result).all():raise ValueError(name+' must be a finite 3-vector')
    if unit:
        length=float(np.linalg.norm(result))
        if length<1e-12:raise ValueError(name+' must be nonzero')
        result=result/length
    return result


def shortest_rotation(source,target):
    """Choose the minimum-angle rotation; antiparallel tie choice is explicit."""
    cross=np.cross(source,target);sine=float(np.linalg.norm(cross));cosine=float(np.clip(source@target,-1,1))
    if sine<1e-10:
        if cosine>0:return np.eye(3),'minimum_angle_zero'
        seed=np.eye(3)[int(np.argmin(np.abs(source)))]
        axis=np.cross(source,seed);axis/=np.linalg.norm(axis)
        return 2*np.outer(axis,axis)-np.eye(3),'antiparallel_minimum_angle_tie'
    axis=cross/sine
    skew=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    return np.eye(3)+sine*skew+(1-cosine)*(skew@skew),'minimum_angle'


def point_axis_goal(source_point,source_axis,target_point,target_axis,
                    source_secondary_axis=None,target_secondary_axis=None):
    """Solve point coincidence and directed axis alignment, optionally fixing twist.

    All points use the same world frame and meters. Axis magnitudes are ignored.
    One point plus one axis leaves one rotational degree of freedom. In that
    case the returned representative uses the minimum-angle rotation; this is
    a selection convention, not evidence that the task constrained its twist.
    A second perpendicular axis fixes twist, useful when edge alignment matters.
    """
    point=vector(source_point,'source point');target=vector(target_point,'target point')
    axis=vector(source_axis,'source axis',True);desired=vector(target_axis,'target axis',True)
    if (source_secondary_axis is None)!=(target_secondary_axis is None):
        raise ValueError('Provide both secondary axes or neither')
    if source_secondary_axis is None:
        rotation,choice=shortest_rotation(axis,desired);free_twist=True
    else:
        second=vector(source_secondary_axis,'source secondary axis',True)
        other=vector(target_secondary_axis,'target secondary axis',True)
        if abs(axis@second)>1e-6 or abs(desired@other)>1e-6:
            raise ValueError('Secondary axes must be perpendicular to primary axes')
        source_basis=np.column_stack([second,np.cross(axis,second),axis])
        target_basis=np.column_stack([other,np.cross(desired,other),desired])
        rotation=target_basis@source_basis.T;choice='two_directed_axes';free_twist=False
    translation=target-rotation@point
    transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=translation
    return {'schema':'desired_rigid_transform_v1','kind':'desired_state','frame':'world','point_units':'m',
            'transform':transform,'rotation':rotation,'translation':translation,
            'unconstrained_twist':free_twist,'representative_selection':choice,
            'point_residual_m':float(np.linalg.norm(rotation@point+translation-target)),
            'axis_residual':float(np.linalg.norm(rotation@axis-desired)),
            'execution_or_contact_certified':False,'uncertainty':'inherited_from_input_references_not_calibrated'}


def transform_points(goal,points):
    """Apply a desired transform to current rigid landmarks, without state mutation."""
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('Require finite Nx3 world points')
    return points@np.asarray(goal['rotation']).T+goal['translation']
