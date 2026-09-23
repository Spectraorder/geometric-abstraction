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
        obs = StackingEnv.observation(self)
        obs['task'] = 'place purple slender prism upright on blue cube'
        obs['objects'] = {name: self._object_state(name) for name in PRISM_OBJECTS}
        return obs

    def structure_status(self, names):
        status = super().structure_status(names)
        if PRISM_ORDER[1] in names:
            body = self.model.body(f'{PRISM_ORDER[1]}/{PRISM_ORDER[1]}')
            # Unlike a cube, lying on another orthogonal face is not upright.
            up = self.data.xmat[body.id].reshape(3, 3)[:, 2]
            if abs(up[2]) < math.cos(math.radians(8)):
                status['reasons'].append('purple_thin_prism: long axis is not vertical')
            for name in PRISM_OBJECTS:
                if name not in PRISM_ORDER and self._objects_in_contact(PRISM_ORDER[1], name):
                    status['reasons'].append('purple_thin_prism: touching a distractor')
        status['stable'] = not status['reasons']
        return status

    def evaluate(self):
        placed = self.structure_status(list(PRISM_ORDER))['stable']
        final = self.checks[-1] if self.checks else {}
        stable = bool(placed and final.get('stable') and final.get('blocks') == list(PRISM_ORDER)
                      and final.get('duration_s', 0) >= 5.)
        return TowerMetrics(stable and self.failure is None, float(placed), int(placed), 1, stable, self.failure)

    def write_trace(self, metrics, error=None):
        path = self.output_dir/'trace.json'
        path.write_text(json.dumps(_jsonable({'task':'prism', 'seed':self.seed, 'error':error,
            'metrics':asdict(metrics), 'stability_checks':self.checks, 'primitives':self.traces}), indent=2)+'\n')
        return path
