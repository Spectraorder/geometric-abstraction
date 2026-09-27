"""Bounded MuJoCo pick/place diagnostics for the opt-in point geometry extension.

The policy runs in the existing sensor-only subprocess sandbox. Generated mesh
parameters, contacts and object poses are private fixture/evaluator data. This is
an actuator/perception diagnostic, not a learned-policy success-rate experiment.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import tempfile
import time

os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

from build_polygon_prism_session import library, validate_grasp_fraction
from run_vision_trial import pack, plan_tool, move_tool, policy_worker_command, validate_source
from aspire_so100.stacking_env import StackingEnv, TABLE_HEIGHT, _SCENE_MODULE as scene_module
from aspire_so100.rgbd import OverheadRGBD
from aspire_so100.robot_geometry import robot_gripper_geometry

ROOT = Path(__file__).resolve().parents[1]
COLLISION_PIPELINES = ('scene', 'native', 'libccd')
NAME = 'red_object'
DESTINATION = [0., -.29, TABLE_HEIGHT]
POLICY = '''"""The same one-action diagnostic controller is used in both arms."""
def run(api):
    task = api.task()
    session = geometry.session(api, task['nodes'])
    api.note('session_ready')
    session.checkpoint('red', 'initial')
    picked = session.pick('red')
    api.note(picked['status'])
    session.checkpoint('red', 'pick', action_status=picked['status'])
    if picked['held']:
        placed = session.place(task['destination_surface_m'])
        api.note(placed['status'])
        session.checkpoint('red', 'place', action_status=placed['status'])
'''


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n')


def fixture(seed):
    """Generate an irregular convex footprint; no named-shape templates."""
    random = np.random.default_rng(seed)
    count = int(random.integers(4, 8))
    angles = (np.arange(count)+random.uniform(-.1, .1, count))*2*np.pi/count
    radius = random.uniform(.020, .024, count)
    polygon = radius[:, None]*np.column_stack((np.cos(angles), np.sin(angles)))
    polygon = polygon[ConvexHull(polygon).vertices]
    following = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0]*following[:, 1]-following[:, 0]*polygon[:, 1]
    centroid = np.sum((polygon+following)*cross[:, None], axis=0)/(3*np.sum(cross))
    polygon -= centroid
    height = float(random.uniform(.028, .042))
    vertices = np.concatenate((np.column_stack((polygon, np.full(len(polygon), -height/2))),
                               np.column_stack((polygon, np.full(len(polygon), height/2)))))
    hull = ConvexHull(vertices)
    faces = []
    for indices, equation in zip(hull.simplices, hull.equations):
        indices = indices.copy()
        a, b, c = vertices[indices]
        if np.cross(b-a, c-a)@equation[:3] < 0:
            indices[[1, 2]] = indices[[2, 1]]
        faces.append(indices)
    return {'seed': seed, 'vertices': vertices, 'faces': np.array(faces), 'height': height,
            'initial_position': [float(random.uniform(-.10, -.07)), float(random.uniform(-.29, -.25)), TABLE_HEIGHT+height/2],
            'yaw': float(random.uniform(-.6, .6)), 'boundary_vertices': len(polygon)}


def mesh_spec(case):
    spec = mujoco.MjSpec()
    spec.add_mesh(name='mesh', uservert=case['vertices'].ravel().tolist(), userface=case['faces'].ravel().tolist())
    body = spec.worldbody.add_body(name=NAME)
    body.add_freejoint(name=NAME+'_joint')
    body.add_geom(name=NAME+'_geom', type=mujoco.mjtGeom.mjGEOM_MESH,
                  meshname='mesh', mass=.03, rgba=(.9, .1, .1, 1.), friction=(1., .005, .0001))
    return spec


def configure_collision_pipeline(model, pipeline):
    """Select a diagnostic collision backend without changing contact parameters."""
    if pipeline not in COLLISION_PIPELINES:
        raise ValueError('Unknown collision pipeline: '+str(pipeline))
    flag = int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    if pipeline == 'libccd':
        model.opt.disableflags |= flag
    elif pipeline == 'native':
        model.opt.disableflags &= ~flag


def physics_settings(model):
    """Record effective settings, including collision flags, for paired audits."""
    fields = ('timestep', 'noslip_iterations', 'solver', 'cone', 'impratio',
              'integrator', 'iterations', 'tolerance', 'ccd_iterations',
              'ccd_tolerance', 'enableflags', 'disableflags')
    settings = {key: getattr(model.opt, key) for key in fields}
    settings['collision_pipeline'] = ('libccd' if model.opt.disableflags &
        int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD) else 'native')
    return settings


class PhysicsEnv(StackingEnv):
    """Use the shared robot, table and primitives with a generated collision mesh."""
    def __init__(self, output, case, width, height, collision_pipeline='scene'):
        self.output_dir, self.case = output, case
        self.render_enabled = False
        cfg = deepcopy(scene_module.scene_cfg)
        cfg.entities = {'robot': deepcopy(scene_module.robot_cfg), 'table': deepcopy(scene_module.table_cfg),
                        NAME: scene_module.make_shape_cfg(lambda: mesh_spec(case), tuple(case['initial_position']))}
        self.scene = scene_module.Scene(cfg, device='cpu')
        self.model = self.scene.compile()
        configure_collision_pipeline(self.model, collision_pipeline)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.site_id = self.model.site('robot/ee_site').id
        self.arm_joints = [self.model.joint('robot/'+name) for name in self.ARM_JOINTS]
        self.arm_qpos_addresses = np.array([int(joint.qposadr[0]) for joint in self.arm_joints])
        self.arm_dof_addresses = np.array([int(joint.dofadr[0]) for joint in self.arm_joints])
        self._renderer = None
        self._overhead_rgbd = OverheadRGBD(self.model, width, height, pixel_aligned=True)
        self.traces, self.frames, self.private_samples = [], [], []
        self.state_type = mujoco.mjtState.mjSTATE_INTEGRATION
        self.seed, self._frame_index, self.last_sample = case['seed'], 0, -1.
        self._set_freejoint_pose(NAME, tuple(case['initial_position']), case['yaw'])
        mujoco.mj_forward(self.model, self.data)
        self._step_for(.65)

    def observation(self):
        # Primitive tracing stays private. Detailed truth is sampled separately.
        return {'time': float(self.data.time)}

    def private_state(self):
        body = self.model.body(NAME+'/'+NAME).id
        rotation = self.data.xmat[body].reshape(3, 3)
        state = self._object_state(NAME)
        points = self.case['vertices']@rotation.T+state['position']
        return dict(state, time=float(self.data.time), bottom=float(points[:, 2].min()),
                    upright=float(rotation[2, 2]), table_contact=self._objects_in_contact(NAME, 'table'),
                    robot_contact=self._objects_in_contact(NAME, 'robot'))

    def _step(self):
        mujoco.mj_step(self.model, self.data)
        if self.data.time-self.last_sample >= .02-1e-9:
            mujoco.mj_forward(self.model, self.data)
            self.last_sample = float(self.data.time)
            self.private_samples.append(self.private_state())
            state = np.empty(mujoco.mj_stateSize(self.model, self.state_type))
            mujoco.mj_getState(self.model, self.data, state, self.state_type)
            self.frames.append(state)

    def evaluate_placement(self):
        self._step_for(.8)
        start = len(self.private_samples)
        self._step_for(2.)
        samples = self.private_samples[start:]
        lifted = [state for state in self.private_samples if state['bottom'] > TABLE_HEIGHT+.025 and state['robot_contact']]
        final = self.private_state()
        initial_position = samples[0]['position']
        checks = {
            'lifted_with_robot_contact': bool(lifted),
            'within_8mm_xy': all(np.linalg.norm(state['position'][:2]-np.asarray(DESTINATION[:2])) <= .008 for state in samples),
            'table_supported': all(state['table_contact'] for state in samples),
            'released_from_robot': all(not state['robot_contact'] for state in samples),
            'upright_within_8deg': all(state['upright'] >= np.cos(np.deg2rad(8)) for state in samples),
            'stable_for_2s': all(np.linalg.norm(state['position']-initial_position) <= .003 and
                                 np.linalg.norm(state['linear_velocity']) <= .015 and
                                 np.linalg.norm(state['angular_velocity']) <= .1 for state in samples),
        }
        stability = {
            'maximum_drift_m': max(float(np.linalg.norm(state['position']-initial_position)) for state in samples),
            'maximum_linear_speed_m_s': max(float(np.linalg.norm(state['linear_velocity'])) for state in samples),
            'maximum_angular_speed_rad_s': max(float(np.linalg.norm(state['angular_velocity'])) for state in samples),
            'sample_count': len(samples),
        }
        return {'physical_success': all(checks.values()), 'checks': checks, 'stability': stability,
                'maximum_bottom_lift_m': max(state['bottom']-TABLE_HEIGHT for state in self.private_samples),
                'lift_with_contact_samples': len(lifted), 'final': final,
                'final_xy_error_m': float(np.linalg.norm(final['position'][:2]-DESTINATION[:2]))}


def run_worker(env, output, geometry_path, timeout):
    calls, images, records = [], [], []
    error, profile = None, None
    with tempfile.TemporaryDirectory(prefix='point-physics-policy-') as tmp:
        scratch = Path(tmp)
        for source, name in ((ROOT/'scripts/vision_policy_worker.py', 'worker.py'),
                             (output.parent/'candidate.py', 'candidate.py'), (geometry_path, 'geometry.py')):
            shutil.copy(source, scratch/name)
        command = policy_worker_command(scratch, scratch/'candidate.py')+[str(scratch/'geometry.py')]
        write_json(output/'sandbox.json', {'isolation': 'bwrap', 'network': 'disabled', 'writable_paths': [tmp]})
        clean = {'PATH': '/usr/bin:/bin', 'HOME': tmp, 'TMPDIR': tmp, 'OPENBLAS_NUM_THREADS': '1',
                 'OMP_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
        with (output/'worker.stderr').open('w') as stderr:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                       text=True, cwd=tmp, env=clean, bufsize=1)
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic()+timeout
            try:
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Policy wall time limit exceeded')
                    if not selector.select(timeout=1):
                        continue
                    line = process.stdout.readline()
                    if not line:
                        raise RuntimeError('Worker exited without completion; see worker.stderr')
                    request = json.loads(line)
                    method, params = request['method'], request['params']
                    calls.append({'time': float(env.data.time), 'method': method, 'params': params})
                    if method in ('done', 'failed'):
                        profile = params.get('cpu_profile')
                        if method == 'failed':
                            error = params['error']
                        break
                    if len(calls) > 300:
                        raise RuntimeError('Diagnostic API call limit exceeded')
                    try:
                        response = None
                        seconds = float(params.get('seconds', 1.5))
                        if not np.isfinite(seconds) or not 0 <= seconds <= 10:
                            raise ValueError('seconds must be within 0..10')
                        if method == 'task':
                            response = {'nodes': [{'color': 'red', 'shape': 'polygon_prism'}],
                                        'destination_surface_m': DESTINATION}
                        elif method == 'camera':
                            view = params.get('view', 'side')
                            if view not in ('side', 'wrist'):
                                raise ValueError('Unknown camera view')
                            response = env.get_overhead_rgbd() if view == 'side' else env.get_wrist_rgb(depth=True)
                            name = 'camera_'+str(len(images)).zfill(3)
                            np.savez_compressed(output/(name+'.npz'), **response)
                            Image.fromarray(response['rgb']).save(output/(name+'.png'))
                            images.append({'file': name+'.npz', 'view': view, 'time': response['time']})
                        elif method == 'proprioception':
                            response = {'time': float(env.data.time), 'tool_position': env.data.site_xpos[env.site_id].copy(),
                                        'tool_rotation': env.data.site_xmat[env.site_id].reshape(3, 3).copy(),
                                        'joint_angles': env.data.qpos[env.arm_qpos_addresses].copy()}
                        elif method == 'gripper_geometry':
                            response = robot_gripper_geometry(env.model, env.data)
                        elif method == 'record_geometry':
                            record = params['record']
                            if not isinstance(record, dict) or len(json.dumps(record, allow_nan=False)) > 131072:
                                raise ValueError('Invalid geometry record')
                            records.append(record)
                        elif method == 'move_tool':
                            move_tool(env, params['position'], max(.1, seconds), params.get('orientation'))
                        elif method == 'plan_tool':
                            response = plan_tool(env, params['position'], params.get('orientation'))
                        elif method == 'gripper':
                            angle = float(params['angle'])
                            if not np.isfinite(angle) or not -.174 <= angle <= 1.75:
                                raise ValueError('Jaw angle outside limits')
                            env.set_gripper(angle, max(.1, seconds))
                        elif method == 'wait':
                            env.wait(seconds)
                        elif method != 'note':
                            raise ValueError('API method not permitted')
                        reply = {'result': pack(response)}
                    except Exception as exc:
                        reply = {'error': type(exc).__name__+': '+str(exc)}
                    process.stdin.write(json.dumps(reply)+'\n')
                    process.stdin.flush()
            except Exception as exc:
                error = type(exc).__name__+': '+str(exc)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                selector.close()
    write_json(output/'api_calls.json', calls)
    write_json(output/'images.json', images)
    write_json(output/'geometry_records.json', records)
    return {'policy_error': error, 'cpu_profile': profile, 'camera_calls': len(images),
            'notes': [call['params']['text'] for call in calls if call['method'] == 'note'],
            'motion_commands': [call for call in calls if call['method'] in ('move_tool', 'gripper', 'wait')]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026092701, 2026092702, 2026092703])
    parser.add_argument('--arms', nargs='+', choices=('basic', 'symbolic'), default=['basic', 'symbolic'])
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--timeout', type=float, default=180.)
    parser.add_argument('--grasp-fraction', type=float, default=None,
                        help='Override the existing generic actuator parameter equally in both arms')
    parser.add_argument('--collision-pipeline', choices=COLLISION_PIPELINES, default='scene',
                        help='Explicit development physics variant; scene preserves compiled defaults')
    args = parser.parse_args()
    if not 160 <= args.width <= 1280 or not 120 <= args.height <= 960 or not 1 <= args.timeout <= 600:
        parser.error('Invalid camera dimensions or worker timeout')
    try:
        validate_grasp_fraction(args.grasp_fraction)
    except ValueError as error:
        parser.error(str(error))
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.arms)) != len(args.arms):
        parser.error('Duplicate seeds or arms')
    args.output.mkdir(parents=True, exist_ok=False)
    validate_source(POLICY)
    (args.output/'candidate.py').write_text(POLICY)
    source_files = [Path(__file__), ROOT/'prototypes/point_geometry.py', ROOT/'prototypes/polygon_prism/geometry.py',
                    ROOT/'prototypes/polygon_prism/partial_boundary.py',
                    ROOT/'scripts/build_polygon_prism_session.py', ROOT/'scripts/vision_policy_worker.py',
                    ROOT/'policies/pick_place_rgbd_shared.py', ROOT/'view_scene.py']
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    libraries = {}
    for arm in args.arms:
        path = args.output/(arm+'_library.py')
        path.write_text(library(arm, grasp_fraction=args.grasp_fraction))
        libraries[arm] = path
    write_json(args.output/'protocol.json', {'schema': 'point_geometry_physics_diagnostic_v1', 'seeds': args.seeds,
        'arms': args.arms, 'collision_pipeline_request': args.collision_pipeline,
        'grasp_fraction_override': args.grasp_fraction, 'width': args.width, 'height': args.height, 'pixel_aligned': True, 'wrist_depth': True,
        'source_sha256': hashes, 'library_sha256': {arm: hashlib.sha256(path.read_bytes()).hexdigest() for arm, path in libraries.items()},
        'candidate_sha256': hashlib.sha256(POLICY.encode()).hexdigest(), 'mujoco_version': mujoco.__version__,
        'model_calls': 0, 'rollout_budget': len(args.seeds)*len(args.arms),
        'purpose': 'Current observer and shared actuator on physical convex meshes; explicit development variants, no policy learning',
        'evaluator': 'Private lift with robot contact; released upright table placement within 8 mm XY, stable for 2 s',
        'public_goal': {'nodes': [{'color': 'red', 'shape': 'polygon_prism'}], 'destination_surface_m': DESTINATION}})
    shutil.copy(__file__, args.output/'runner_snapshot.py')
    results = []
    for seed in args.seeds:
        case = fixture(seed)
        for arm in args.arms:
            output = args.output/(str(seed)+'_'+arm)
            output.mkdir()
            write_json(output/'private_fixture.json', case)
            started = time.monotonic()
            env = PhysicsEnv(output, case, args.width, args.height, args.collision_pipeline)
            try:
                Image.fromarray(env.get_overhead_rgbd()['rgb']).save(output/'initial.png')
                mujoco.mj_saveModel(env.model, str(output/'model.mjb'))
                result = run_worker(env, output, libraries[arm], args.timeout)
                result.update(seed=seed, arm=arm, boundary_vertices=case['boundary_vertices'],
                              evaluation=env.evaluate_placement(), wall_seconds=time.monotonic()-started,
                              physics=physics_settings(env.model))
                result['success'] = result['policy_error'] is None and result['evaluation']['physical_success']
                Image.fromarray(env.get_overhead_rgbd()['rgb']).save(output/'outcome.png')
                np.savez_compressed(output/'rollout.npz', states=env.frames, state_type=int(env.state_type), sample_seconds=.02)
                write_json(output/'private_states.json', env.private_samples)
                write_json(output/'result.json', result)
                results.append(result)
                write_json(args.output/'summary.json', {'status': 'running', 'completed': len(results), 'results': results})
                print(json.dumps({'seed': seed, 'arm': arm, 'success': result['success'],
                                  'error': result['policy_error'], 'lift_m': result['evaluation']['maximum_bottom_lift_m']}), flush=True)
            finally:
                env.close()
    pairs = []
    for seed in args.seeds:
        pair = [row for row in results if row['seed'] == seed]
        if len(pair) == 2:
            pairs.append({'seed': seed, 'identical_motion_commands': pair[0]['motion_commands'] == pair[1]['motion_commands']})
    write_json(args.output/'summary.json', {'status': 'complete', 'completed': len(results),
        'successes': sum(row['success'] for row in results), 'pairs': pairs, 'results': results})


if __name__ == '__main__':
    main()
