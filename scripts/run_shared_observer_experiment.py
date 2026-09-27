"""Matched surface-versus-rigid-symbolic pilot using the existing frozen harness.

Variant hooks are scoped to this process. The original runner and any running
campaign remain unchanged. Source snapshots include this driver and both adapters.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch
import run_mixed_scene_experiment as base
from build_mixed_scene import build

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('basic', 'symbolic')
BASE_SOURCES = base.sources
EXTRA_SOURCES = ('scripts/run_shared_observer_experiment.py',
                 'scripts/observation_context_prototype.py',
                 'scripts/surface_projection_prototype.py')
SURFACE_CONTRACT = '''Your interface is a projection of the SAME mixed rigid RGB-D
observer and PnP executor used by the richer condition. It is not the historical
visible-top estimator. observe/get expose center, normal, height, dims[2], top,
bottom, extent_z, coordinate_frame, units, observation_time, source, view,
position_fully_observed and inferred_orientation. No internal dimension model,
rigid rotation/vertices/axes, history, target, goal_error, reference, relation,
align or text methods are supplied. You may compute additional geometry yourself
from the same sensors within the same planner budget. State/snapshot packets expose
status,last_observed,reason when present,center,normal,top,bottom,extent_z,source,view,
position_fully_observed,inferred_orientation; state packets do not repeat height/dims.
Unavailable or stale rows contain no current pose. Action-return observations use
exactly the same projection as observe. Action statuses and residuals are unchanged.
'''
COMMON = base.COMMON.replace(
    'The observation backend varies\nas specified for your condition; numerical and symbolic conditions share a backend.',
    'Both conditions use exactly the same mixed rigid observation backend and motion loop; only public information differs.')
COMMON += '''\nInitial perception context is bound by the host, independently of desired task
orientations. Public initial assumptions for this scene: sources are on the table,
cubes/cylinders upright, thin prism flat with long axis along world X. Shape/color
descriptors or full task nodes are equivalent; provide the complete public inventory.
Changing goal orientation or descriptor order does not change the initial observer.
These initial assumptions are not sensor measurements or simulator object truth.
'''


def sources():
    hashes = BASE_SOURCES()
    hashes.update({name: base.digest(ROOT/name) for name in EXTRA_SOURCES})
    return hashes


def library(arm):
    if arm not in ARMS:
        raise ValueError('Unsupported shared-observer arm')
    source = build(True)
    for name in EXTRA_SOURCES[1:]:
        source += '\n'+(ROOT/name).read_text()
    source += '\nSURFACE_ONLY = '+str(arm == 'basic')+'\n'
    source += '''
class SessionFactory:
    @staticmethod
    def session(api, objects):
        inventory = api.task()['nodes']
        # Public initial conditions, independent of desired goal orientation.
        initial = {'cube': 'upright', 'prism': 'upright',
                   'thin_prism': 'flat_x', 'cylinder': 'upright'}
        context = {'schema': 'initial_observation_context_v1',
                   'source': 'public_initial_scene_description',
                   'objects': [{'color': obj['color'], 'shape': obj['shape'],
                                'initial_orientation_assumption': initial[obj['shape']]}
                               for obj in inventory]}
        session = bind_session(SymbolicSession, context)(api, objects)
        return SurfaceProjectionSession(session) if SURFACE_ONLY else session
GEOMETRY = SessionFactory()
'''
    base.validate_source(source)
    return source


@contextmanager
def configured():
    """Reuse orchestration, not its old basic-observer treatment definition."""
    with patch.multiple(base, library=library, sources=sources, COMMON=COMMON, BASIC=SURFACE_CONTRACT):
        yield


def prepare(root, train, reserved, rounds=3):
    with configured():
        manifest = base.prepare(root, rounds, train, reserved, ARMS)
    manifest.update({
        'schema': 'shared_observer_abstraction_pilot_v1',
        'arm_labels': {'basic': 'shared surface', 'symbolic': 'rigid metric and symbolic state'},
        'contrasts': ['Identical mixed observer and execution; projected surface state versus supplied rigid metric/symbolic state package.'],
        'shared_initial_context': 'Public initial-scene assumptions bound before actor construction; desired orientations never initialize perception.',
        'primary_metrics': ['reserved SR at each fixed revision', 'first reserved revision reaching 80 percent',
                            'mean reserved SR over all registered revisions', 'training-selected reserved SR'],
        'retained_budget_secondary': 'Highest training SR among revisions available at each stopping budget; earliest tie; evaluate on reserved without selecting by reserved.',
        'limitations': ['One independent actor chain per arm; replication required.',
                       'The baseline has basic geometry; this is richer state versus surface information, not geometry versus none.',
                       'The treatment is a package of rigid fields, history, goals and symbolic operators; no single-component attribution.',
                       'Equal revision/rollout budgets; token counts recorded, not equalized.',
                       'No earlier candidate, development solution, failure audit or reserved outcome enters actor context.']})
    base.write(root/'manifest.json', manifest)
    base.write(root/'progress.json', {'status': 'registered', 'curves': {arm: [] for arm in ARMS}})
    return manifest


def check(root, manifest):
    if manifest.get('schema') != 'shared_observer_abstraction_pilot_v1' or manifest['arms'] != list(ARMS):
        raise ValueError('Wrong experiment variant')
    with configured():
        base.assert_frozen(root, manifest)
    for arm in ARMS:
        if library(arm) != (root/(arm+'_library.py')).read_text():
            raise ValueError('Library does not match the registered variant')


def bootstrap(root, manifest):
    """Run paired starters before any model generation; later campaign reuses them."""
    check(root, manifest)
    if list(root.glob('*/round_*/invocation.json')):
        raise ValueError('Bootstrap must precede all actor generations')
    curves = {}
    with configured():
        for arm in ARMS:
            directory = root/arm/'revision_000'
            bundled = base.bundle(root, arm, (root/'starter.py').read_text(), directory/'bundled.py')
            rows = base.execute(root, arm, bundled, directory/'trials', manifest['train_seeds'], manifest)
            curves[arm] = [{'revision': 0, 'successes': sum(r['success'] for r in rows),
                            'trials': len(rows), 'results': rows}]
            base.write(root/'progress.json', {'status': 'bootstrap', 'curves': curves})
    audit_starters(root, manifest)
    base.write(root/'progress.json', {'status': 'bootstrapped', 'curves': curves})


def audit_starters(root, manifest):
    import numpy as np
    namespace = {}
    exec(compile((root/'symbolic_library.py').read_text(), '<frozen-library>', 'exec'), namespace)
    pairs = []
    for seed in manifest['train_seeds']:
        paths = [root/arm/'revision_000/trials'/f'seed_{seed}' for arm in ARMS]
        results = [json.loads((p/'result.json').read_text()) for p in paths]
        for field in ('success','policy_error','scene','camera_calls','physics','sensor','candidate_sha256'):
            if results[0][field] != results[1][field]:
                raise ValueError('Starter pairing mismatch: '+field)
        with np.load(paths[0]/'rollout.npz') as a, np.load(paths[1]/'rollout.npz') as b:
            if a.files != b.files or not all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes() for k in a.files):
                raise ValueError('Starter physical arrays differ')
        records = [json.loads((p/'geometry_records.json').read_text()) if (p/'geometry_records.json').exists() else [] for p in paths]
        records = [[r for r in entries if r.get('kind') == 'planner_state'] for entries in records]
        if len(records[0]) != len(records[1]):
            raise ValueError('Starter checkpoints missing or mismatched')
        for first, second in zip(*records):
            if first['label'] != second['label'] or first['state'] != namespace['surface_packet'](second['state']):
                raise ValueError('Shared surface state differs')
        pairs.append({'seed': seed, 'identical_physics': True, 'identical_surface_state': True, 'checkpoints': len(records[0]),
                      'source_sha256': {str(p.relative_to(root)/f): base.digest(p/f)
                                        for p in paths for f in ('result.json','rollout.npz','geometry_records.json') if (p/f).exists()}})
    report = {'status': 'passed', 'scope': 'Offline R0 parity; no learning benefit or object truth in actor context.', 'pairs': pairs}
    base.write(root/'starter_parity.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--train-seeds', nargs='+', type=int)
    parser.add_argument('--reserved-seeds', nargs='+', type=int)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--phase', choices=('prepare','bootstrap','train','evaluate','all'), default='prepare')
    args = parser.parse_args()
    root = args.output.resolve()
    if not args.resume and (not args.train_seeds or not args.reserved_seeds):
        parser.error('New registration requires explicit training and reserved seeds')
    manifest = json.loads((root/'manifest.json').read_text()) if args.resume else prepare(root,args.train_seeds,args.reserved_seeds,args.rounds)
    check(root, manifest)
    if args.phase == 'bootstrap':
        bootstrap(root, manifest)
    elif args.phase != 'prepare':
        if not (root/'starter_parity.json').exists():
            bootstrap(root, manifest)
        else:
            audit_starters(root, manifest)
        with configured():
            base.campaign(root, manifest, args.phase)
