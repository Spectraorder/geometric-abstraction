"""Shared online skill adapters over training-selected bridge and stack policies.

Only trusted fixed code sees this sensor API. The model chooses validated JSON
actions; it cannot execute Python or inspect the environment through this object.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import time

import numpy as np
from PIL import Image
from aspire_so100.robot_geometry import robot_gripper_geometry
from build_online_bridge_geometry import online_library
from run_vision_trial import plan_tool, move_tool

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'bridge': ('outputs/geometry_parallel_01/replica_01/symbolic/frozen.py',
               'b580b2adb315d171c7bd0148fbc859450fb782bec7aeff0aaeb5b499619c0570'),
    'stack': ('outputs/geometry_cube_stack_01/replica_01/symbolic/frozen.py',
              'a51a64925d6acf1765f13d329df1afad853cbccd90dcd1f719c5c21495e68a87'),
}


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(plain(value), allow_nan=False)+'\n')


class SensorAPI:
    """Expose calibrated sensors and public robot control to the fixed skills."""
    def __init__(self, env, output):
        self.env = env
        self.output = Path(output)
        self.camera_count = 0
        self.camera_seconds = 0.
        self.action_seconds = 0.

    def camera(self, view='side'):
        started = time.monotonic()
        if view not in ('side', 'wrist'):
            raise ValueError('Unknown camera')
        frame = (self.env.get_overhead_rgbd() if view == 'side'
                 else self.env.get_wrist_rgb(depth=True))
        self.camera_count += 1
        self.camera_seconds += time.monotonic()-started
        return frame

    def image(self, name, view='side'):
        frame = self.camera(view)
        path = self.output/(name+'.png')
        Image.fromarray(frame['rgb']).save(path)
        # Archive the permitted sensor packet for input and calibration auditing.
        np.savez_compressed(self.output/(name+'.npz'), **frame)
        return path, {k: plain(frame[k]) for k in
                      ('time', 'intrinsics', 'camera_to_world')}

    def depth_image(self, rgb_path):
        """Render the archived depth sensor on an explicit metric display scale."""
        with np.load(Path(rgb_path).with_suffix('.npz')) as frame:
            depth = frame['depth_m']
            valid = frame['valid_depth'] & np.isfinite(depth)
            near, far = .05, 1.5
            values = np.nan_to_num(depth, nan=far, posinf=far, neginf=far)
            pixels = 1+np.uint8(np.clip((values-near)/(far-near), 0., 1.)*254)
            pixels[~valid] = 0
        path = Path(rgb_path).with_name(Path(rgb_path).stem+'_depth.png')
        Image.fromarray(pixels).save(path)
        return path, {'encoding': 'optical-axis depth in meters; gray 0 means invalid',
                      'gray_1_m': near, 'gray_255_m': far, 'clipped_outside_display_range': True}

    def task(self):
        return self.env.recipe.public_goal()

    def proprioception(self):
        env = self.env
        return {'joint_angles': env.data.qpos[env.arm_qpos_addresses].copy(),
                'tool_position': env.data.site_xpos[env.site_id].copy(),
                'tool_rotation': env.data.site_xmat[env.site_id].reshape(3, 3).copy(),
                'time': float(env.data.time)}

    def gripper_geometry(self):
        return robot_gripper_geometry(self.env.model, self.env.data)

    def record_geometry(self, record):
        append(self.output/'skill_geometry.jsonl', record)

    def note(self, text):
        append(self.output/'skill_notes.jsonl', {'time': float(self.env.data.time),
                                               'text': str(text)})

    def _action(self, method, function, *args):
        started = time.monotonic()
        entry = {'method': method, 'args': plain(args),
                 'sim_time': float(self.env.data.time)}
        try:
            return function(*args)
        except Exception as error:
            entry['error'] = type(error).__name__+': '+str(error)
            raise
        finally:
            entry['wall_seconds'] = time.monotonic()-started
            self.action_seconds += entry['wall_seconds']
            append(self.output/'actions.jsonl', entry)

    def plan_tool(self, position, orientation=None):
        return plan_tool(self.env, position, orientation)

    def move_tool(self, position, seconds=1.5, orientation=None):
        def command(point, duration, rotation):
            return move_tool(self.env, point, duration, rotation)
        return self._action('move_tool', command, position, seconds, orientation)

    def move_joints(self, angles, seconds=1.5):
        angles = np.asarray(angles, float)
        limits = self.env.model.jnt_range[[j.id for j in self.env.arm_joints]]
        if (angles.shape != (5,) or not np.isfinite(angles).all()
                or np.any(angles < limits[:, 0]) or np.any(angles > limits[:, 1])):
            raise ValueError('Invalid robot joint command')
        return self._action('move_joints', self.env.move_to_joint_positions,
                            angles, seconds)

    def gripper(self, angle, seconds=.6):
        if not np.isfinite(angle) or not -.174 <= angle <= 1.75:
            raise ValueError('Invalid jaw command')
        return self._action('gripper', self.env.set_gripper, angle, seconds)

    def wait(self, seconds):
        if not np.isfinite(seconds) or not 0 <= seconds <= 10:
            raise ValueError('Wait must be between zero and ten seconds')
        return self._action('wait', self.env.wait, seconds)


class SharedSkills:
    """Retain initialized shape, workspace and holding state across decisions."""
    def __init__(self, api):
        self.api = api
        bundle = online_library()
        namespace = {}
        exec(compile(bundle, '<online geometry>', 'exec'), namespace)
        self.geometry = namespace['GEOMETRY']
        self.provenance = {'geometry_sha256': hashlib.sha256(bundle.encode()).hexdigest(),
                           'source_policies': {}}
        for kind, (filename, digest) in SOURCES.items():
            source = (ROOT/filename).read_bytes()
            if hashlib.sha256(source).hexdigest() != digest:
                raise ValueError('Training-selected skill source changed: '+filename)
            scope = {'geometry': self.geometry}
            exec(compile(source, filename, 'exec'), scope)
            setattr(self, kind, SimpleNamespace(**scope))
            self.provenance['source_policies'][kind] = {'path': filename, 'sha256': digest}
        self.task = api.task()
        self.nodes = {node['key']: node for node in self.task['nodes']}
        self.session = self.geometry.session(api, self.task['nodes'])
        self.order = self.bridge.construction_order(self.task['nodes'])
        self.counter = [0]
        self.last_result = None
        self.origin = None
        self.scale = None
        self.initialize_workspace()

    def initialize_workspace(self):
        b, api, session = self.bridge, self.api, self.session
        fits = {key: b.observe(session, api, self.nodes[key]['color'], attempts=2)
                for key in self.order}
        if any(value is None for value in fits.values()):
            raise ValueError('Initial five-object geometry is unavailable')
        reference = self.task['length_unit']['observed_cube']
        self.scale = b.reference_side(self.nodes[reference], fits[reference])
        region = b.table_region(api)
        if self.scale is None or region is None:
            raise ValueError('Observed reference scale or table region is unavailable')
        self.origin, _, count = b.select_workspace(api, session, self.nodes,
                                                  self.order, fits, region, self.scale)
        self.api.record_geometry({'kind': 'online_workspace', 'origin': self.origin,
                                  'scale': self.scale, 'candidate_count': count})
        if self.origin is None:
            raise ValueError('No observed reachable five-object workspace')

    def packet(self):
        """Compact current evidence and explicitly typed placement residuals."""
        session = self.session
        session.snapshot()
        fits = {}
        for key, node in self.nodes.items():
            try:
                fits[key] = session.get(node['color'])
            except (RuntimeError, ValueError, KeyError):
                fits[key] = None
        rows = []
        for key in self.order:
            node = self.nodes[key]
            fit = fits[key]
            row = {'node': key, 'status': 'unavailable' if fit is None else 'current_estimate'}
            target = None
            if not node['supports']:
                xy = self.origin+self.scale*np.asarray(node['grid'], float)
                target = np.r_[xy, session.table_z(xy)]
            elif all(fits[parent] is not None and fits[parent].get('position_fully_observed', False)
                     for parent in node['supports']):
                offsets = [fits[parent]['center'][:2]-self.scale*np.asarray(self.nodes[parent]['grid'])
                           for parent in node['supports']]
                xy = np.mean(offsets, axis=0)+self.scale*np.asarray(node['grid'])
                target = np.r_[xy, max(fits[parent]['top'] for parent in node['supports'])]
            if fit is not None:
                row.update(center_m=fit['center'], bottom_top_m=[fit['bottom'], fit['top']],
                           position_fully_observed=fit.get('position_fully_observed', False),
                           orientation_compatible=self.bridge.orientation_evidence(
                               node, self.bridge.fit_geometry(fit)))
                if target is not None and fit.get('position_fully_observed', False):
                    residual = target-np.r_[fit['center'][:2], fit['bottom']]
                    is_stack = len(node['supports']) == 1
                    xy_tolerance = max(.0035, .14*self.scale) if is_stack else max(.0045, .18*self.scale)
                    z_tolerance = max(.0045, .16*self.scale) if is_stack else max(.005, .2*self.scale)
                    row['placement'] = {
                        'remaining_translation_m': residual,
                        'translation_within_tolerance': bool(np.linalg.norm(residual[:2]) <= xy_tolerance
                                                            and abs(residual[2]) <= z_tolerance)}
            row['supports'] = {}
            for parent in node['supports']:
                relation = session.relation('support_candidate', node['color'], self.nodes[parent]['color'])
                row['supports'][parent] = relation['truth']
            rows.append(row)
        def rounded(value):
            if isinstance(value, float):
                return round(value, 4)
            if isinstance(value, dict):
                return {key: rounded(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rounded(item) for item in value]
            return value
        return rounded(plain({'time': session.time(), 'objects': rows,
            'placement_reference': 'center_xy_min_z',
            'placement_tolerance_m': {
                'bridge': {'xy': max(.0045, .18*self.scale), 'z': max(.005, .2*self.scale)},
                'stack': {'xy': max(.0035, .14*self.scale), 'z': max(.0045, .16*self.scale)}},
            'semantics': 'Translation residual = target minus current center XY and lowest Z; '
                         'world coordinates are not errors. Geometry does not certify physical support.'}))

    def holding(self):
        held = self.session.pnp.held
        return None if held is None else {
            'color': held['color'], 'lift_confirmed': bool(held['confirmed'])}

    def execute(self, skill, key):
        if key not in self.nodes or skill not in ('bridge', 'stack'):
            raise ValueError('Unknown skill or public node')
        node = self.nodes[key]
        bridge_node = len(node['supports']) != 1
        if skill == 'bridge' and not bridge_node:
            raise ValueError('Bridge skill accepts one foundation or the beam')
        if skill == 'stack' and bridge_node:
            raise ValueError('Stack skill accepts one of the added cubes')
        started = time.monotonic()
        try:
            held = self.holding()
            if held is not None:
                result = self._resume_held(node, held)
            elif skill == 'bridge':
                point, _, fresh = self.bridge.current_target(self.api, self.session, node,
                                                              self.nodes, self.origin, self.scale)
                good = False
                if fresh:
                    good, _, _ = self.bridge.settled_check(self.api, self.session, node, point, self.scale)
                if not good:
                    good, _ = self.bridge.place_node(self.api, self.session, node,
                                self.nodes, self.origin, self.scale, self.counter)
                result = {'status': 'visually_consistent' if good else 'unconfirmed'}
            else:
                result = self._stack_attempt(node)
        except (RuntimeError, ValueError) as error:
            # Preserve the physical scene and executor state for the next decision.
            result = {'status': 'recoverable_error', 'error': str(error)}
        result.update(skill=skill, node=key, holding=self.holding(),
                      wall_seconds=time.monotonic()-started)
        self.last_result = result
        append(self.api.output/'skill_results.jsonl', result)
        return result

    def _resume_held(self, node, held):
        if held['color'] != node['color']:
            return {'status': 'another_object_held'}
        if not held['lift_confirmed']:
            return {'status': 'holding_uncertain',
                    'instruction': 'Observe or release before another pick'}
        point, _ = self.stack.destination(self.api, self.session, node,
                                          self.nodes, self.origin, self.scale)
        if point is None:
            return {'status': 'support_unavailable'}
        result = self.session.place(point.tolist())
        return {'status': result['status']}

    def release(self):
        """Open at the current position; do not teleport or assume a new pose."""
        self.api.gripper(1.5)
        self.api.wait(.4)
        self.session.pnp.held = None
        self.session.placement_intent = None
        result = {'status': 'opened_unverified', 'holding': None}
        self.last_result = result
        append(self.api.output/'skill_results.jsonl', result)
        return result

    def _stack_attempt(self, node):
        """Adapt the selected stack policy's inner loop to one bounded attempt."""
        s, api, session = self.stack, self.api, self.session
        color = node['color']
        point, supports = s.destination(api, session, node, self.nodes,
                                        self.origin, self.scale)
        if point is None:
            return {'status': 'support_unavailable'}
        current = s.observe_current(api, session, color)
        _, already = s.placement_residual(node, current, point, self.scale)
        if already and s.verify_settled(api, session, node, self.nodes,
                                       self.origin, self.scale)[0]:
            return {'status': 'visually_consistent'}
        if current is None:
            return {'status': 'object_unavailable'}
        if s.orientation_evidence(node, current) is False:
            return {'status': 'orientation_incompatible'}
        picked = session.pick(color, transit_z=max(float(point[2])+.085,
                                                   float(current['top'])+.065))
        s.checkpoint_action(session, color, 'online_pick', picked)
        if not picked['held'] or picked['status'] != 'lift_visually_confirmed':
            return {'status': picked['status']}
        historical_point, historical_supports = point.copy(), supports
        for attempt in range(2):
            point, _, fresh = s.resolve_after_motion(api, session, node, self.nodes,
                self.origin, self.scale, historical_point, historical_supports, self.counter)
            if point is None:
                return {'status': 'support_unavailable_after_pick'}
            placed = session.place(point.tolist())
            s.checkpoint_action(session, color, 'online_place_'+str(attempt), placed)
            if not placed['held']:
                break
            if placed['status'] != 'placement_unconfirmed':
                return {'status': placed['status']}
            api.wait(.25)
        if placed['held']:
            return {'status': placed['status']}
        api.wait(.5)
        good, _ = s.verify_settled(api, session, node, self.nodes, self.origin, self.scale)
        return {'status': 'visually_consistent' if good else 'unconfirmed'}
