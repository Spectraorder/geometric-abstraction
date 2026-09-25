"""Private evaluator for repairing a simulated human's off-center placement.

Only reset initializes the prebuilt stack. All corrections use ordinary robot
physics. A marginal, upright placement is deliberate: catching a dynamically
falling object is a later curriculum stage.
"""
from dataclasses import replace

import mujoco
import numpy as np

from aspire_so100.block_catalog import generate_block_layout
from aspire_so100.prism import PrismEnv
from aspire_so100.stacking_env import StackingEnv, TABLE_HEIGHT


class InventoryRepairEnv(PrismEnv):
    task_id = 'inventory_repair'
    task_description = 'recenter red slender prism on blue cube; preserve foundation'

    def __init__(self, output_dir, render=False, *, scene_seed=1501,
                 object_count=6, repair_case='offset'):
        if repair_case not in ('centered', 'offset'):
            raise ValueError('repair_case must be centered or offset')
        if not 2 <= object_count <= 10:
            raise ValueError('Repair curriculum requires 2..10 objects')
        layout = generate_block_layout(scene_seed, object_count, 'repair_thin_prism')
        self.scene_seed = scene_seed
        self.repair_case = repair_case
        self.objects = tuple(layout.positions)
        self.order = (layout.roles['base'], layout.roles['target'])
        self.scene_metadata = {'generator': 'inventory_repair_v1',
            'scene_task': layout.task, 'seed': scene_seed, 'object_count': object_count,
            'objects': list(self.objects), 'roles': dict(layout.roles),
            'repair_case': repair_case}
        self.monitoring = False
        self.base_anchor = None
        self.target_anchor = None
        self.max_base_drift = 0.
        self.max_target_drift = 0.
        self.collapse_seen = False
        self.reset_progress()
        StackingEnv.__init__(self, output_dir, render, scene_task=layout.task,
                             scene_seed=scene_seed, object_count=object_count)
        self.checks = []
        self.failure = None

    def reset(self, seed, layout='sampled'):
        if seed != self.scene_seed or layout != 'sampled':
            raise ValueError('Repair reset requires the constructor seed and sampled layout')
        self.monitoring = False
        self.seed = seed
        self.traces = []
        self._frame_index = 0
        self.checks = []
        self.failure = None
        self.max_base_drift = self.max_target_drift = 0.
        self.collapse_seen = False
        self.reset_progress()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        base = self._object_state(self.order[0])['position'].copy()
        rng = np.random.default_rng(seed + 982451653)
        offset = np.zeros(2)
        if self.repair_case == 'offset':
            axis = seed % 2
            sign = -1 if (seed // 2) % 2 else 1
            offset[axis] = sign * rng.uniform(.006, .008)
        self._set_freejoint_pose(self.order[1],
            (*list(base[:2]+offset), TABLE_HEIGHT+.06), 0.)
        mujoco.mj_forward(self.model, self.data)
        self._step_for(.65)
        self.base_anchor = self._object_state(self.order[0])['position'].copy()
        self.target_anchor = self._object_state(self.order[1])['position'].copy()
        self.initial_offset = self.target_anchor[:2]-self.base_anchor[:2]
        self.monitoring = True
        return self.observation()

    def _step(self):
        super()._step()
        if self.monitoring:
            base = self._object_state(self.order[0])['position']
            target = self._object_state(self.order[1])['position']
            self.max_base_drift = max(self.max_base_drift,
                                      float(np.linalg.norm(base-self.base_anchor)))
            self.max_target_drift = max(self.max_target_drift,
                                        float(np.linalg.norm(target-self.target_anchor)))
            # A payload lowered onto the table or dropped below the platform
            # has dismantled the prebuilt stack rather than repaired it.
            self.collapse_seen |= bool(target[2]-self.extent(self.order[1])[2]
                                       < TABLE_HEIGHT+.012)
            if not self.verified['grasp'] and self.verified['plan']:
                fingers = self.contact_fingers()
                lifted = target[2]-self.target_anchor[2] >= .005
                self.grasp_hold = self.grasp_hold+self.model.opt.timestep if lifted and len(fingers)==2 else 0.
                if self.grasp_hold >= .1 and self.max_base_drift <= .003 and not self.collapse_seen:
                    self.verified['grasp'] = True

    def reset_progress(self):
        self.verified = {'identify': False, 'plan': False, 'grasp': False}
        self.progress_reports = []
        self.waypoints = {}
        self.grasp_hold = 0.

    def contact_fingers(self):
        target_id = self.model.body(f'{self.order[1]}/{self.order[1]}').id
        jaw_ids = {self.model.body('robot/Fixed_Jaw').id, self.model.body('robot/Moving_Jaw').id}
        touching = set()
        for contact in self.data.contact:
            a, b = (int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if a == target_id and b in jaw_ids:
                touching.add(b)
            if b == target_id and a in jaw_ids:
                touching.add(a)
        return touching

    def record_report(self, report, preview=None):
        """Private milestone verification. RPC returns None, never this verdict."""
        stage, values = report['stage'], report['values']
        result = {'stage': stage, 'accepted': False, 'reasons': []}
        def vector(name):
            value = np.asarray(values[name], float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(name+' must be finite XYZ')
            return value
        try:
            if stage == 'identify':
                base_error = float(np.linalg.norm(vector('base_center')-self._object_state(self.order[0])['position']))
                target_error = float(np.linalg.norm(vector('target_center')-self._object_state(self.order[1])['position']))
                size_error = np.abs(vector('target_dimensions')-2*self.extent(self.order[1]))
                result.update(base_center_error_m=base_error, target_center_error_m=target_error,
                              target_dimension_error_m=size_error.tolist())
                valid = base_error<=.01 and target_error<=.01 and bool(np.all(size_error<=.008))
                result['accepted'] = valid
                self.verified['identify'] |= valid
                if not valid:
                    result['reasons'].append('Reported supported-object geometry is inaccurate')
            elif stage == 'plan':
                role, position = values['role'], vector('position')
                if role not in ('approach','grasp','lift'):
                    raise ValueError('Plan role must be approach, grasp, or lift')
                if not self.verified['identify']:
                    raise ValueError('Supported object has not been identified accurately')
                if not preview or preview.get('error') or 'observation' not in preview:
                    raise ValueError('No matching recorded IK preview')
                if not np.allclose(position, preview['params']['position'], atol=1e-6, rtol=0):
                    raise ValueError('Reported waypoint differs from the last IK preview')
                plan = preview['observation']
                tilt = float(np.arccos(np.clip(np.asarray(plan['tool_rotation'])[2,1],-1,1)))
                result.update(position_error_m=plan['position_error'], tool_tilt_deg=float(np.degrees(tilt)))
                target = self._object_state(self.order[1])['position']
                if plan['position_error'] > .003 or tilt > np.deg2rad(15):
                    raise ValueError('IK preview does not meet position/orientation tolerance')
                if np.linalg.norm(position[:2]-target[:2]) > .025:
                    raise ValueError('Waypoint is not near the supported target')
                if not .005 <= position[2]-target[2] <= .12:
                    raise ValueError('Waypoint height is outside the target grasp/approach region')
                self.waypoints[role] = position.copy()
                result['accepted'] = True
                if set(self.waypoints) == {'approach','grasp','lift'}:
                    grasp = self.waypoints['grasp']
                    self.verified['plan'] = bool(grasp[2]-target[2] <= .07 and all(
                        .005 <= self.waypoints[r][2]-grasp[2] <= .06 for r in ('approach','lift')))
            elif stage not in ('reject','observe'):
                raise ValueError('Report stage must be identify, plan, observe, or reject')
        except (ValueError, KeyError, TypeError) as error:
            result['reasons'].append(str(error))
        self.progress_reports.append(result)

    def progress_metrics(self):
        return {'version': 2, 'verified': dict(self.verified),
                'checks': list(self.progress_reports),
                'plan_scope': 'local robot IK previews; collisions and execution not certified'}

    def structure_status(self, names):
        result = super().structure_status(names)
        if self.monitoring:
            base = self._object_state(self.order[0])['position']
            target = self._object_state(self.order[1])['position']
            if np.linalg.norm(target[:2]-base[:2]) > .004:
                result['reasons'].append('repair: target center error exceeds 4 mm')
            if self.max_base_drift > .003:
                result['reasons'].append('repair: foundation moved more than 3 mm')
            if self.collapse_seen:
                result['reasons'].append('repair: prebuilt stack was dismantled or collapsed')
            if self.repair_case == 'centered' and self.max_target_drift > .004:
                result['reasons'].append('repair: unnecessary intervention on centered control')
        result['stable'] = not result['reasons']
        return result

    def repair_metrics(self):
        base = self._object_state(self.order[0])['position']
        target = self._object_state(self.order[1])['position']
        return {'initial_offset_m': self.initial_offset.tolist(),
                'final_center_error_m': float(np.linalg.norm(target[:2]-base[:2])),
                'max_base_drift_m': self.max_base_drift,
                'max_target_drift_m': self.max_target_drift,
                'collapse_seen': self.collapse_seen, 'case': self.repair_case}

    def evaluate(self):
        metrics = super().evaluate()
        details = self.repair_metrics()
        safe = not self.collapse_seen and self.max_base_drift <= .003
        score = max(0., 1.-details['final_center_error_m']/.02) if safe else 0.
        # Legacy diagnostic only. Loop v2 ranks verified milestones and genuine
        # improvement instead of this absolute score, which can reward no-ops.
        return replace(metrics, score=1. if metrics.success else min(.99, score))
