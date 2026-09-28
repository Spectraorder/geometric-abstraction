"""Privileged scripted feasibility check for the ring-over-peg scene.

This controller reads simulator truth (ring pose, peg axis, jaw meshes and table
contact). It answers only whether the SO-100 and the shared actuator style can
physically grasp the ring wall and drop it over the peg. It is not a policy, a
perception result, or a success-rate estimate, and no model is called.
"""
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import numpy as np
from PIL import Image

from ring_peg_scene import PEG, RING, TABLE_HEIGHT, RingPegEnv

ROOT = Path(__file__).resolve().parents[1]
OPEN_PRE = .15      # pre-grasp opening: pads about 32 mm apart, clear of the far wall
CLOSED = -.174
ROLL_LIMIT = 2.79


def jaw_geoms(model, jaw):
    return [g for g in range(model.ngeom)
            if model.body(model.geom_bodyid[g]).name == 'robot/'+jaw and model.geom_contype[g]]


def pad_center(model, data, jaw):
    return np.mean([data.geom_xpos[g] for g in jaw_geoms(model, jaw) if 'pad' in model.geom(g).name], axis=0)


def jaw_points(model, data):
    """World vertices of both jaws' collision geometry."""
    points = []
    for g in jaw_geoms(model, 'Fixed_Jaw')+jaw_geoms(model, 'Moving_Jaw'):
        rotation, origin = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            local = model.mesh_vert[start:start+count]
        else:
            half = model.geom_size[g]
            local = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*half
        points.append(local@rotation.T+origin)
    return np.vstack(points)


def kinematics(env, target, roll, jaw):
    """Forward kinematics of an IK solution without touching the live simulation."""
    joints = env.solve_ik(target, wrist_roll=roll)
    probe = mujoco.MjData(env.model)
    probe.qpos[:] = env.data.qpos
    probe.qpos[env.arm_qpos_addresses] = joints
    probe.qpos[int(env.model.joint('robot/Jaw').qposadr[0])] = jaw
    mujoco.mj_forward(env.model, probe)
    fixed, moving = pad_center(env.model, probe, 'Fixed_Jaw'), pad_center(env.model, probe, 'Moving_Jaw')
    axis = (moving-fixed)/np.linalg.norm(moving-fixed)
    return {'site': probe.site_xpos[env.site_id].copy(), 'fixed_pad': fixed, 'moving_pad': moving,
            'axis': axis, 'points': jaw_points(env.model, probe)}


def plan_grasp(env):
    """Choose a wall and wrist roll so the jaws close along that wall's normal.

    The fixed pad sits outside the ring, 2 mm from the outer face; the moving pad
    enters the hole. Candidate rolls are scanned; the smallest roll change wins.
    """
    g = env.geometry
    state = env.private_ring_state()
    center, rotation = state['position'], state['rotation']
    wall_mid = (g.outer_inradius+g.hole_inradius)/2
    grasp_z = center[2]
    best = None
    default_roll = float(env.data.qpos[env.arm_qpos_addresses[-1]])
    for k in range(6):
        normal = rotation@np.array([math.cos(k*math.pi/3), math.sin(k*math.pi/3), 0.])
        normal = np.r_[normal[:2]/np.linalg.norm(normal[:2]), 0.]
        mid = np.r_[center[:2], grasp_z]+wall_mid*normal
        outer_face = np.r_[center[:2], grasp_z]+g.outer_inradius*normal
        for roll in np.linspace(-ROLL_LIMIT, ROLL_LIMIT, 113):
            try:
                fk = kinematics(env, mid+np.r_[0, 0, .02], roll, OPEN_PRE)
            except RuntimeError:
                continue
            # Fixed pad outside the ring means the closing axis points inward.
            alignment = float(-fk['axis'][:2]@normal[:2])
            if alignment < math.cos(math.radians(6)):
                continue
            score = abs(roll-default_roll)
            if best is None or score < best['score']:
                best = dict(score=score, wall=k, roll=float(roll), normal=normal, mid=mid,
                            outer_face=outer_face, alignment=alignment)
    if best is None:
        raise RuntimeError('No reachable wall/roll aligns the jaw axis with a wall normal')
    return best


def site_for_fixed_pad(env, desired_fixed_pad, roll, jaw):
    """Iterate IK targets until the fixed pad reaches the desired world point."""
    target = desired_fixed_pad.copy()
    for _ in range(6):
        fk = kinematics(env, target, roll, jaw)
        error = desired_fixed_pad-fk['fixed_pad']
        if np.linalg.norm(error) < .0008:
            break
        target = target+error
    return target, fk


def render(env, path, lookat):
    renderer = mujoco.Renderer(env.model, 480, 640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance, camera.azimuth, camera.elevation = .42, 60, -30
    renderer.update_scene(env.data, camera=camera)
    Image.fromarray(renderer.render()).save(path)
    renderer.close()


def run(output, seed):
    output.mkdir(parents=True, exist_ok=True)
    env = RingPegEnv(output/'env', scene_seed=seed)
    env.reset()
    g = env.geometry
    log = {'seed': seed, 'layout': env.layout, 'physics': env.physics, 'events': []}
    def note(event, **values):
        log['events'].append(dict(event=event, time=float(env.data.time), **values))
    try:
        plan = plan_grasp(env)
        note('grasp_plan', wall=plan['wall'], roll=plan['roll'], alignment=plan['alignment'])
        roll = plan['roll']
        # The fixed pad's contact face sits 2 mm outside the outer wall face.
        standoff = plan['outer_face']+plan['normal']*(.002+.001)
        above_target, _ = site_for_fixed_pad(env, standoff+np.r_[0, 0, .06], roll, OPEN_PRE)
        env.set_gripper(OPEN_PRE, .6)
        env.move_end_effector(above_target, 1.8, wrist_roll=roll)
        # Guarded descent: stop 2 mm after the lowest jaw point would reach the table.
        for drop in np.arange(.06, -.03, -.004):
            target, fk = site_for_fixed_pad(env, standoff+np.r_[0, 0, drop], roll, OPEN_PRE)
            if fk['points'][:, 2].min() < TABLE_HEIGHT+.002:
                break
            env.move_end_effector(target, .35, wrist_roll=roll)
            if env._objects_in_contact(RING, 'robot'):
                note('ring_contact_during_descent', drop=float(drop))
                break
        pads = (pad_center(env.model, env.data, 'Fixed_Jaw')+pad_center(env.model, env.data, 'Moving_Jaw'))/2
        note('grasp_height', pad_mid_z_above_table=float(pads[2]-TABLE_HEIGHT),
             jaw_min_z_above_table=float(jaw_points(env.model, env.data)[:, 2].min()-TABLE_HEIGHT),
             ring_mid_z_above_table=float(env.private_ring_state()['position'][2]-TABLE_HEIGHT))
        env.set_gripper(CLOSED, 1.)
        env.wait(.35)
        lift_to = TABLE_HEIGHT+g.peg_height+.06
        site = env.data.site_xpos[env.site_id].copy()
        env.move_end_effector(np.r_[site[:2], lift_to], 1.7, wrist_roll=roll)
        ring = env.private_ring_state()
        lifted = bool(ring['position'][2]-TABLE_HEIGHT-g.height/2 > .03 and not env._objects_in_contact(RING, 'table'))
        note('lift', lifted=lifted, ring_z_above_table=float(ring['position'][2]-TABLE_HEIGHT),
             tilt_deg=ring['tilt_deg'])
        render(env, output/'lift.png', ring['position'])
        if not lifted:
            raise RuntimeError('Ring was not lifted')
        # Place: ring center over the peg axis; lowest jaw point 6 mm above the peg top.
        peg_top = TABLE_HEIGHT+g.peg_height
        for attempt in range(4):
            site = env.data.site_xpos[env.site_id].copy()
            ring = env.private_ring_state()
            lowest = jaw_points(env.model, env.data)[:, 2].min()
            ring_bottom = ring['position'][2]-g.height/2
            clearance_z = min(lowest, ring_bottom)
            error_xy = env.peg_axis_xy()-ring['position'][:2]
            target = np.r_[site[:2]+error_xy, site[2]+(peg_top+.006-clearance_z)]
            note('place_step', attempt=attempt, xy_error_m=float(np.linalg.norm(error_xy)),
                 ring_bottom_above_peg_top=float(ring_bottom-peg_top), jaw_min_above_peg_top=float(lowest-peg_top),
                 tilt_deg=ring['tilt_deg'])
            if np.linalg.norm(error_xy) < .0015 and abs(clearance_z-(peg_top+.006)) < .002:
                break
            env.move_end_effector(target, 1.4 if attempt == 0 else .7, wrist_roll=roll)
            env.wait(.2)
        render(env, output/'over_peg.png', np.r_[env.peg_axis_xy(), peg_top])
        env.set_gripper(OPEN_PRE, .6)
        site = env.data.site_xpos[env.site_id].copy()
        env.move_end_effector(site+np.r_[0, 0, .05], 1.2, wrist_roll=roll)
        result = env.evaluate_on_peg()
        note('evaluate', **{k: v for k, v in result.items()})
        render(env, output/'final.png', np.r_[env.peg_axis_xy(), TABLE_HEIGHT+.02])
        log['success'] = bool(result['success'])
    except Exception as error:
        log['success'] = False
        log['error'] = f'{type(error).__name__}: {error}'
        try:
            render(env, output/'failure.png', env.private_ring_state()['position'])
        except Exception:
            pass
    (output/'result.json').write_text(json.dumps(log, indent=2, default=lambda v: np.asarray(v).tolist())+'\n')
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/ring_peg_dev/oracle_01')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    args = parser.parse_args()
    summary = {}
    for seed in args.seeds:
        log = run(args.output/f'seed_{seed}', seed)
        summary[seed] = {'success': log['success'], 'error': log.get('error')}
        print(seed, summary[seed], flush=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
