"""Replay no-motion training IK requests using only robot CAD and proprioception.

This diagnostic never reads object poses, trial success, contacts, or rollout.npz.
It does not alter actor policies or infer collision-free task feasibility.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from run_vision_trial import plan_tool
from aspire_so100.stacking_env import _SCENE_MODULE, StackingEnv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(paths):
    spec = _SCENE_MODULE.get_so100_spec()
    spec.body('Base').pos = np.asarray(spec.body('Base').pos)+np.asarray(_SCENE_MODULE.robot_cfg.init_state.pos)
    model = spec.compile()
    if any(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError('Robot-only model unexpectedly contains a free body')
    joints = [model.joint(name) for name in StackingEnv.ARM_JOINTS]
    env = SimpleNamespace(model=model, data=mujoco.MjData(model), arm_joints=joints,
                          arm_qpos_addresses=np.array([int(j.qposadr[0]) for j in joints]),
                          site_id=model.site('ee_site').id)
    rows, inputs, skipped = [], [], []
    for path in paths:
        calls = json.loads(path.read_text())
        if any(c['method'] in ('move_tool', 'move_joints', 'gripper', 'wait') for c in calls):
            raise ValueError('Only no-motion, no-wait logs qualify: '+str(path))
        inputs.append({'path': str(path), 'sha256': digest(path)})
        state = None
        requests = 0
        for index, call in enumerate(calls):
            if call['method'] == 'proprioception':
                state = call['observation']
            if call['method'] != 'plan_tool':
                continue
            if state is None or abs(state['time']-call['time']) > 1e-9:
                raise ValueError('Missing same-time public joint observation')
            env.data.qpos[env.arm_qpos_addresses] = state['joint_angles']
            mujoco.mj_kinematics(model, env.data)
            position_error = float(np.linalg.norm(env.data.site_xpos[env.site_id]-state['tool_position']))
            rotation_error = float(np.linalg.norm(env.data.site_xmat[env.site_id].reshape(3, 3)-state['tool_rotation']))
            if max(position_error, rotation_error) > 1e-9:
                raise ValueError('Robot CAD does not reproduce public tool pose')
            params = call['params']
            if params.get('orientation') is not None:
                raise ValueError('Diagnostic requires default-orientation requests')
            first = plan_tool(env, params['position'])
            second = plan_tool(env, params['position'], first['tool_rotation'])
            default_error = float(first['position_error'])
            relaxed_error = float(second['position_error'])
            # Match shared move(): default move threshold, then relaxed fallback.
            executable_by_position_test = default_error <= .005 or relaxed_error <= .003
            rows.append({'source': str(path), 'call_index': index,
                         'target_m': params['position'], 'time': call['time'],
                         'robot_pose_replay_error_m': position_error,
                         'robot_rotation_replay_error': rotation_error,
                         'default_error_m': default_error, 'relaxed_error_m': relaxed_error,
                         'actor_2p5mm_guard_passes': default_error <= .0025,
                         'shared_move_position_test_passes': executable_by_position_test,
                         'relaxed_orientation_needed': default_error > .005 and relaxed_error <= .003})
            requests += 1
        if not requests:
            skipped.append({'source': str(path), 'reason': 'No IK requests recorded'})
    disagreement = [r for r in rows if not r['actor_2p5mm_guard_passes'] and r['shared_move_position_test_passes']]
    return {'scope': __doc__, 'requests': len(rows), 'robot_only_model_nq': model.nq,
            'actor_guard_rejects': sum(not r['actor_2p5mm_guard_passes'] for r in rows),
            'shared_position_test_passes': sum(r['shared_move_position_test_passes'] for r in rows),
            'guard_rejects_but_shared_position_test_passes': len(disagreement),
            'relaxed_orientation_needed': sum(r['relaxed_orientation_needed'] for r in rows),
            'max_robot_pose_replay_error_m': max((r['robot_pose_replay_error_m'] for r in rows), default=None),
            'inputs': inputs, 'skipped': skipped, 'records': rows,
            'source_sha256': {str(p): digest(p) for p in (Path(__file__), Path(__file__).with_name('run_vision_trial.py'), _SCENE_MODULE.SO100_XML, Path(_SCENE_MODULE.__file__))},
            'limitations': ['Pointwise kinematic checks from recorded robot joints; not collision, grasp or task feasibility.',
                            'Alternative orientation changes are not validated against held-object orientation constraints.',
                            'A guard/executor mismatch does not prove successful execution or an abstraction benefit.',
                            'Post-training diagnostic only; never supplied to frozen actors.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite diagnostic evidence')
    paths = sorted(args.trials.glob('seed_*/api_calls.json'))
    if not paths:
        raise ValueError('No recorded requests found')
    result = replay(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('records', 'inputs', 'source_sha256', 'scope')})
