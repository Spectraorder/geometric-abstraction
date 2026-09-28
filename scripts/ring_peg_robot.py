"""Object-independent pad-target motion and robot-only kinematic previews."""
import math
import mujoco
import numpy as np


def vector(value):
    value = np.asarray(value, float)
    if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) > 2:
        raise ValueError('Expected finite world-meter vector')
    return value


def jaw_geometry(model, data):
    points, pads = [], {}
    for jaw in ('Fixed_Jaw', 'Moving_Jaw'):
        ids = [g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name == 'robot/'+jaw and model.geom_contype[g]]
        pads[jaw] = np.mean([data.geom_xpos[g] for g in ids if 'pad' in model.geom(g).name], axis=0)
        for g in ids:
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[g]
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                local = model.mesh_vert[start:start+count]
            else:
                local = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*model.geom_size[g]
            points.append(local@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])
    axis = pads['Moving_Jaw']-pads['Fixed_Jaw']
    return {'fixed_pad_m': pads['Fixed_Jaw'], 'moving_pad_m': pads['Moving_Jaw'],
            'closing_axis': axis/np.linalg.norm(axis), 'jaw_vertices_m': np.vstack(points),
            'tool_position': data.site_xpos[model.site('robot/ee_site').id].copy()}


class PadActuator:
    def __init__(self, env):
        self.env = env

    def preview(self, point, roll, jaw):
        point = vector(point)
        if not np.isfinite([roll,jaw]).all() or not -2.79 <= roll <= 2.79 or not -.174 <= jaw <= 1.75:
            raise ValueError('Joint target outside public limits')
        joints = self.env.solve_ik(point, wrist_roll=float(roll), tolerance=.0003, max_iterations=500)
        scratch = mujoco.MjData(self.env.model)
        scratch.qpos[self.env.arm_qpos_addresses] = joints
        scratch.qpos[int(self.env.model.joint('robot/Jaw').qposadr[0])] = jaw
        mujoco.mj_kinematics(self.env.model, scratch)
        result = jaw_geometry(self.env.model, scratch)
        result.update(joint_angles=joints.copy(), roll=float(roll), site_target_m=point, position_error_m=float(np.linalg.norm(result['tool_position']-point)))
        return result

    def plan_pad(self, fixed_pad_m, roll, jaw):
        desired = vector(fixed_pad_m)
        target = desired.copy()
        for _ in range(8):
            result = self.preview(target, roll, jaw)
            error = desired-result['fixed_pad_m']
            if np.linalg.norm(error) < .0008:
                return result
            target += error
        raise ValueError('Fixed pad target did not converge')

    def plan_grasp(self, fixed_pad_m, closing_direction, jaw):
        point, axis = vector(fixed_pad_m), vector(closing_direction)
        axis /= np.linalg.norm(axis)
        current = float(self.env.data.qpos[self.env.arm_qpos_addresses[-1]])
        candidates = []
        for roll in np.linspace(-2.79,2.79,113):
            try:
                result = self.preview(point, float(roll), jaw)
                alignment = float(result['closing_axis']@axis)
                if alignment > math.cos(math.radians(6)):
                    candidates.append((abs(roll-current),float(roll)))
            except RuntimeError:
                pass
        for _, roll in sorted(candidates):
            try:
                return self.plan_pad(point,roll,jaw)
            except (ValueError,RuntimeError):
                pass
        raise ValueError('No reachable roll aligns the requested closing direction')

    def current(self):
        scratch = mujoco.MjData(self.env.model)
        scratch.qpos[self.env.arm_qpos_addresses] = self.env.data.qpos[self.env.arm_qpos_addresses]
        address=int(self.env.model.joint('robot/Jaw').qposadr[0])
        scratch.qpos[address] = self.env.data.qpos[address]
        mujoco.mj_kinematics(self.env.model,scratch)
        result=jaw_geometry(self.env.model,scratch)
        result.update(time=float(self.env.data.time),roll=float(self.env.data.qpos[self.env.arm_qpos_addresses[-1]]),jaw=float(self.env.data.qpos[address]),source='robot_model_and_encoders',contact_observed=False)
        return result

    def move_pad(self, point, roll, seconds):
        if not .1 <= seconds <= 10:
            raise ValueError('Motion duration must be 0.1 through 10 seconds')
        plan=self.plan_pad(point,roll,self.current()['jaw'])
        self.env.move_to_joint_positions(plan['joint_angles'],seconds)
        return self.current()
