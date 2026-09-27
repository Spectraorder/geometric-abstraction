"""Development parity of surface and rigid interfaces with one common starter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from build_mixed_scene import build
from run_vision_trial import validate_source
from run_mixed_scene_experiment import sources

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    runtime = sources()
    common = build(False)+'\n'+(ROOT/'scripts/observation_context_prototype.py').read_text()
    common += '\n'+(ROOT/'scripts/surface_projection_prototype.py').read_text()
    factory = '''
class SessionFactory:
    @staticmethod
    def session(api, objects):
        inventory = api.task()['nodes']
        # Explicit public initial-scene assumptions for this development fixture.
        # Desired goal orientations are deliberately never read here.
        assumptions = {'cube': 'upright', 'prism': 'upright',
                       'thin_prism': 'flat_x', 'cylinder': 'upright'}
        context = {'schema': 'initial_observation_context_v1',
                   'source': 'public_initial_scene_description',
                   'objects': [{'color': obj['color'], 'shape': obj['shape'],
                                'initial_orientation_assumption': assumptions[obj['shape']]}
                               for obj in inventory]}
        session = bind_session(MixedSession, context)(api, objects)
        return SurfaceProjectionSession(session) if SURFACE_ONLY else session
GEOMETRY = SessionFactory()
'''
    for arm in ('surface', 'numeric'):
        library = common+'\nSURFACE_ONLY = '+str(arm == 'surface')+'\n'+factory
        validate_source(library)
        (output/(arm+'_library.py')).write_text(library)
    candidate = output/'candidate.py'
    candidate.write_text('Session = geometry.session\n'+(ROOT/'policies/support_graph_starter.py').read_text())
    seeds = [75103, 75105]
    manifest = {'scope': 'Development starter parity only; not a learning or generalization experiment.',
                'seeds': seeds, 'runtime_sha256': runtime, 'candidate_sha256': digest(candidate),
                'library_sha256': {arm: digest(output/(arm+'_library.py')) for arm in ('surface','numeric')},
                'driver_sha256': digest(Path(__file__)),
                'initial_assumptions': 'Public fixture: sources on table, thin beam flat along world X; other objects upright.'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='egl')
    pairs = []
    for seed in seeds:
        paths = []
        for arm in ('surface', 'numeric'):
            if sources() != runtime:
                raise RuntimeError('Development runtime changed')
            path = output/arm/f'seed_{seed}'
            paths.append(path)
            command = [sys.executable, str(ROOT/'scripts/run_vision_trial.py'),
                       '--candidate', str(candidate), '--output', str(path), '--seed', str(seed),
                       '--task', 'inventory_structure', '--structure-recipe', 'thin_beam_bridge',
                       '--object-count', '4', '--layout', 'sampled', '--wrist-depth', '--record-geometry',
                       '--geometry-library', str(output/(arm+'_library.py')), '--timeout', '120']
            with (output/f'{arm}_{seed}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f'Completed {arm} seed {seed}', flush=True)
        a, b = paths
        results = [json.loads((p/'result.json').read_text()) for p in paths]
        for field in ('success', 'policy_error', 'scene', 'camera_calls', 'physics', 'sensor'):
            if results[0][field] != results[1][field]:
                raise ValueError('Paired result differs: '+field)
        with np.load(a/'rollout.npz') as first, np.load(b/'rollout.npz') as second:
            if first.files != second.files or not all(first[k].dtype == second[k].dtype and first[k].shape == second[k].shape and first[k].tobytes() == second[k].tobytes() for k in first.files):
                raise ValueError('Physical trajectories differ')
        actions = []
        for p in paths:
            calls = json.loads((p/'api_calls.json').read_text())
            actions.append([c for c in calls if c['method'] in ('move_tool','move_joints','gripper','wait')])
        if actions[0] != actions[1] or not actions[0]:
            raise ValueError('Nonempty action traces must match')
        pairs.append({'seed': seed, 'success': results[0]['success'],
                      'identical_physical_arrays': True, 'identical_actions': True,
                      'camera_calls_per_arm': results[0]['camera_calls'],
                      'source_sha256': {str(p.relative_to(output)/name): digest(p/name)
                                        for p in paths for name in ('result.json','rollout.npz','api_calls.json')}})
    report = {'status': 'passed', 'scope': manifest['scope'], 'pairs': pairs,
              'limits': 'Two development layouts, one unchanged starter; no actor evolution or held-out success inference. Physical arrays are offline audit evidence only.'}
    (output/'parity.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())
