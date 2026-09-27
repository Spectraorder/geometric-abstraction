"""Robot-only jaw pad geometry from the model and measured joint positions."""
import mujoco
import numpy as np


def robot_gripper_geometry(model, data):
    """Export pad boxes and the tool marker, never object poses or contacts.

    Forward kinematics runs on fresh data populated only with robot joint angles.
    Pad boxes describe part of the gripper; the housing meshes are not included.
    """
    scratch = mujoco.MjData(model)
    angles = {}
    for name in ('Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw'):
        address = int(model.joint('robot/' + name).qposadr[0])
        scratch.qpos[address] = data.qpos[address]
        angles[name] = float(data.qpos[address])
    mujoco.mj_kinematics(model, scratch)
    signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    pads = []
    for jaw in ('fixed', 'moving'):
        for index in range(1, 5):
            name = f'robot/{jaw}_jaw_pad_{index}'
            geom = model.geom(name)
            if geom.type[0] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError('Expected robot pad box: ' + name)
            center = scratch.geom_xpos[geom.id].copy()
            rotation = scratch.geom_xmat[geom.id].reshape(3, 3).copy()
            half_size = geom.size.copy()
            pads.append({'name': name, 'center': center, 'rotation': rotation,
                         'half_size_m': half_size,
                         'vertices': center + (signs * half_size) @ rotation.T})
    marker = model.site('robot/ee_site')
    if marker.type[0] != mujoco.mjtGeom.mjGEOM_SPHERE:
        raise ValueError('Expected a spherical robot tool marker')
    visual_spheres = [{'name': 'robot/ee_site',
                       'center_m': scratch.site_xpos[marker.id].copy(),
                       'radius_m': float(marker.size[0])}]
    return {'source': 'robot_model_and_joint_encoders', 'coordinate_frame': 'world',
            'units': 'meters', 'time': float(data.time), 'joint_angles': angles,
            'pads': pads, 'coverage': 'finger_pad_boxes_only',
            'visual_spheres': visual_spheres,
            'visual_geometry_coverage': 'ee_site_marker_only',
            'contact_observed': False}
