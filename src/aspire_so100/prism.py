"""Private evaluator for an upright slender prism on a cube in a six-object scene."""
from dataclasses import asdict
import json
import math

import mujoco
import numpy as np

from aspire_so100.stacking_env import StackingEnv, TABLE_HEIGHT, _jsonable
from aspire_so100.tower import TowerEnv, TowerMetrics
from aspire_so100.prism_layouts import PRISM_HALF_SIZES, PRISM_OBJECTS, PRISM_ORDER, prism_positions


class PrismEnv(TowerEnv):
    order = PRISM_ORDER
    objects = PRISM_OBJECTS
    task_id = 'prism'
    task_description = 'place purple slender prism upright on blue cube'

    def __init__(self, output_dir, render=False):
        StackingEnv.__init__(self, output_dir, render, prism=True)
        self.checks = []
        self.failure = None

    def reset(self, seed, layout='nominal'):
        self.seed = seed
        self.traces = []
        self._frame_index = 0
        self.checks = []
        self.failure = None
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        for name, xy in prism_positions(seed, layout).items():
            self._set_freejoint_pose(name, (*xy, TABLE_HEIGHT+PRISM_HALF_SIZES[name][2]), 0.)
        mujoco.mj_forward(self.model, self.data)
        self._step_for(.65)
        return self.observation()

    def observation(self):
        # Private tracing only. This method is never exposed by the sensor RPC.
        return {'seed': self.seed, 'task': self.task_description,
                'end_effector_position': self.data.site_xpos[self.site_id].copy(),
                'robot_joint_positions': self.data.qpos[self.arm_qpos_addresses].copy(),
                'objects': {name: self._object_state(name) for name in self.objects}}

    def structure_status(self, names):
        status = super().structure_status(names)
        target = self.order[1]
        if target in names:
            body = self.model.body(f'{target}/{target}')
            # Unlike a cube, lying on another orthogonal face is not upright.
            up = self.data.xmat[body.id].reshape(3, 3)[:, 2]
            if abs(up[2]) < math.cos(math.radians(8)):
                status['reasons'].append(target+': long axis is not vertical')
            for name in self.objects:
                if name not in self.order and self._objects_in_contact(target, name):
                    status['reasons'].append(target+': touching a distractor')
        status['stable'] = not status['reasons']
        return status

    def evaluate(self):
        placed = self.structure_status(list(self.order))['stable']
        final = self.checks[-1] if self.checks else {}
        stable = bool(placed and final.get('stable') and final.get('blocks') == list(self.order)
                      and final.get('duration_s', 0) >= 5.)
        return TowerMetrics(stable and self.failure is None, float(placed), int(placed), 1, stable, self.failure)

    def write_trace(self, metrics, error=None):
        path = self.output_dir/'trace.json'
        path.write_text(json.dumps(_jsonable({'task':self.task_id, 'seed':self.seed, 'error':error,
            'metrics':asdict(metrics), 'stability_checks':self.checks, 'primitives':self.traces}), indent=2)+'\n')
        return path
