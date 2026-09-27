"""Evaluator-side controlled scene interventions, isolated from runtime feedback."""
import hashlib
import time
import mujoco
import numpy as np
from online_bridge_skills import plain
from online_event_skills import drive

KINDS = ('foundation', 'beam', 'loaded_stack')


def operation(skills, index, action='execute'):
    key = skills.order[index]
    return {'action': action, 'skill': ('none' if action == 'clear' else
            'stack' if len(skills.nodes[key]['supports']) == 1 else 'bridge'), 'node': key}


def state_digest(env):
    return hashlib.sha256(env.data.qpos.tobytes()+env.data.qvel.tobytes()).hexdigest()


class RecoveryFixture:
    """One deterministic intervention after a common, sensor-controlled prefix.

    Ground-truth access is restricted to perturbation and private audit records.
    The actor receives a generic event notification and historical observations.
    """
    def __init__(self, kind, displacement=.025):
        if kind not in KINDS or not .015 <= abs(displacement) <= .04:
            raise ValueError('Unsupported recovery fixture')
        self.kind = kind
        self.displacement = displacement

    def __call__(self, env, api, skills):
        from run_online_event_development import dump
        start = time.monotonic()
        prefix_count = {'foundation': 2, 'beam': 3, 'loaded_stack': 5}[self.kind]
        prefix = []
        for i in range(prefix_count):
            op = operation(skills, i)
            outcome = drive(skills, op)
            prefix.append({'operation': op, 'outcome': outcome})
            if outcome['status'] != 'visually_consistent':
                dump(api.output/'fixture.json', {'kind': self.kind, 'status': 'prefix_failed', 'prefix': prefix})
                raise RuntimeError('Shared fixture prefix failed; retain slot as setup failure')
        api.wait(1.)
        images = []; cameras = {}
        for view in ('side', 'wrist'):
            path, calibration = api.image('pre_event_'+view, view)
            images.append(path); cameras[view] = calibration
        before_hash = state_digest(env)
        # Preserve orientation and translate only the selected rigid bodies. This
        # is an external scene intervention, never an action available to agents.
        indices = {'foundation': [1], 'beam': [2], 'loaded_stack': [3, 4]}[self.kind]
        moved = []
        for index in indices:
            name = env.order[index]
            joint = env.model.joint(f'{name}/{name}_joint')
            address, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
            before = env.data.qpos[address:address+7].copy()
            axis = 0 if self.kind == 'loaded_stack' else 1
            env.data.qpos[address+axis] += self.displacement
            env.data.qvel[dof:dof+6] = 0.
            moved.append({'object': name, 'before': before.tolist(),
                          'after': env.data.qpos[address:address+7].tolist()})
        mujoco.mj_forward(env.model, env.data)
        api.wait(1.)
        audit = {'kind': self.kind, 'status': 'ready', 'prefix': prefix,
                 'displacement_m': self.displacement, 'axis': 'x' if self.kind == 'loaded_stack' else 'y',
                 'moved_private': moved,
                 'before_sha256': before_hash, 'after_sha256': state_digest(env),
                 'setup_wall_seconds': time.monotonic()-start,
                 'setup_sim_seconds': float(env.data.time), 'reset_count': env.reset_count,
                 'historical_images': [str(p) for p in images]}
        dump(api.output/'fixture.json', audit)
        public = {'event': 'An external disturbance occurred after the recorded construction prefix. '
                            'Reassess current supports and continue to the original goal without reset.',
                  'completed_prefix': [row['operation'] for row in prefix],
                  'historical_cameras': cameras,
                  'image_semantics': 'pre_event images are historical, before the disturbance; '
                                     'feedback images and geometric text describe current observations.'}
        # The prescribed plan is used only by development feasibility checks.
        repair = ([operation(skills, 4, 'clear'), operation(skills, 3), operation(skills, 4)]
                  if self.kind == 'loaded_stack' else
                  [operation(skills, i) for i in range(indices[0], 5)])
        return {'public': plain(public), 'images': images, 'scripted_plan': repair}
