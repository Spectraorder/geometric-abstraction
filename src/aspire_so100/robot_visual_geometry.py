"""Exclude known robot visual spheres using model geometry and current encoders.

This action-free depth mask accepts supplied robot geometry only. It consumes no object identity,
pose, contact or simulator segmentation. Coverage is limited to supplied robot
spheres, not the complete robot or a collision model.
"""
import numpy as np


def exclude_robot_spheres(frame,spheres,robot_time,tolerance_m=.0005):
    """Remove depth samples inside known same-time robot visual geometry."""
    if not np.isfinite(robot_time) or abs(float(frame['time'])-robot_time)>1e-6:
        raise ValueError('Robot geometry must have the camera timestamp')
    if not np.isfinite(tolerance_m) or tolerance_m<0:raise ValueError('Invalid geometric tolerance')
    depth=np.asarray(frame['depth_m'],dtype=float)
    valid=np.asarray(frame['valid_depth'],dtype=bool)&np.isfinite(depth)&(depth>0)
    v,u=np.indices(depth.shape)
    rays=np.stack([u,v,np.ones_like(u)],axis=-1)@np.linalg.inv(frame['intrinsics']).T
    transform=np.asarray(frame['camera_to_world'])
    points=(rays*np.where(valid,depth,0)[...,None])@transform[:3,:3].T+transform[:3,3]
    excluded=np.zeros(depth.shape,dtype=bool)
    for sphere in spheres:
        center=np.asarray(sphere['center_m'],dtype=float);radius=float(sphere['radius_m'])
        if center.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius<=0:
            raise ValueError('Require finite known robot sphere geometry')
        excluded|=valid&(np.sum((points-center)**2,axis=-1)<=(radius+tolerance_m)**2)
    result=dict(frame);result['valid_depth']=valid&~excluded
    return result,{'removed_depth_samples':int(excluded.sum()),'robot_time':float(robot_time),
        'source':'robot_model_and_current_proprioception','coverage':'supplied_visual_spheres_only'}


def filter_robot_visual_geometry(frame,robot_geometry=None):
    """Apply an explicitly supplied robot-model packet; never capture sensors."""
    if robot_geometry is None:
        return frame,{'applied':False,'reason':'robot_geometry_not_supplied'}
    if robot_geometry.get('coordinate_frame')!='world' or robot_geometry.get('units')!='meters':
        raise ValueError('Robot geometry requires world coordinates in meters')
    if robot_geometry.get('source')!='robot_model_and_joint_encoders':
        raise ValueError('Require a robot-model and encoder geometry packet')
    spheres=robot_geometry.get('visual_spheres',[])
    result,evidence=exclude_robot_spheres(frame,spheres,float(robot_geometry['time']))
    return result,dict(evidence,applied=bool(spheres),sphere_count=len(spheres))
