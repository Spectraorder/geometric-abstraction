"""Two-arm evolution with target-only feedback and unchanged shared motion control."""
import argparse
import ast
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import run_shared_observer_experiment as shared
from build_geometry_step_session import library

base = shared.base
ROOT = shared.ROOT
ARMS = shared.ARMS
BASE_VALIDATE = base.validate_planner
EXTRA = ('policies/geometry_step_state.py', 'policies/geometry_step_session.py',
         'policies/support_graph_step_starter.py', 'scripts/build_geometry_step_session.py',
         'scripts/run_geometry_step_experiment.py')
COMMON = shared.COMMON
start = COMMON.index('All conditions also have snapshot()')
end = COMMON.index('Runtime RGB-D is always available', start)
COMMON = COMMON[:start]+'''Both conditions provide get(color) for a current estimate, observe(color) for an
explicit refresh, and time(). After each pick/place/check, call
checkpoint(color, step, refresh=False, action_status=None). This logs and returns
ONLY the active object's observation. It performs no camera call unless refresh=True.
Stale/missing geometry remains unknown. Query support objects explicitly as needed.
Do not use global snapshot/state/text/history methods. No scene-wide packet is needed.
'''+COMMON[end:]
COMMON = COMMON.replace("Record {'kind':'planner_state','label':stage,'state':session.snapshot()} at meaningful\nstages so supplied state use is auditable. It changes no task or motion commands.",
'''Use checkpoint at action boundaries; no global snapshots or all-pairs relations.
Each selected training example supplies at most the last eight checkpoint records,
within 16000 characters, alongside the same initial/final image selection in both arms.''')
BASIC = '''The base exposes center, normal, height, dims[2], top, bottom, extent_z,
coordinate_frame, units, observation_time, source/view and observability flags from
the SAME rigid observer and PnP executor as the full condition. Checkpoint records
contain just these current measurements. No supplied references, rigid vertices,
goal/error, step-change or alignment operators. You may compute geometry from your
sensors within the same planner budget. No privileged object state is available.
'''
RICH = '''The full interface additionally exposes shape, rotation[3,3],
axis_dimensions[3], vertices, constrained_axes, inferred_orientation and
position_fully_observed through observe/get. These are RGB-D model estimates;
invisible points are not measured truth. Cylinder rims are samples, not corners.
checkpoint records compact current geometry plus center change since the preceding
checkpoint of that object, with observability flags. Switching object resets this
one-step memory. Missing evidence clears it. Optional goal_center=[x,y,z] stores a
separate desired center and reports remaining_translation_m; it never moves the
robot or changes an observation. reference_names=['color.bottom_face', ...] adds
only requested active-object points/face centers to the record.
After place(), placement_goal automatically records the attempted surface target.
Its reference center_xy_min_z means center X/Y and the estimated lowest world Z,
not a material face center. remaining_translation_m is target minus that current
estimate; it is unknown without a fresh fit. It describes the desired settled
extent, not the next safe robot command (the executor uses release clearance).
It certifies neither contact nor task success. A new pick clears this intent.
'''+base.SYMBOLIC.replace(
    'state() includes these derived references/facts in addition to identical numeric state.\ntext() gives a compact textual summary. No stronger controller or sensing is supplied.',
    'Relations and references are computed only on request. No stronger controller or sensing is supplied.')


def sources():
    result = shared.sources()
    result.update({name: base.digest(ROOT/name) for name in EXTRA})
    return result


def validate_planner(source):
    tree = BASE_VALIDATE(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ('snapshot', 'state', 'text', 'history', 'step_state'):
            raise ValueError('Use target-only checkpoint and explicit geometry queries')
    return tree


def compact_checkpoints(path):
    if not path.exists():
        return []
    rows = [row for row in json.loads(path.read_text()) if row.get('kind') == 'geometry_step'][-8:]
    while rows and len(json.dumps(rows, separators=(',', ':'))) > 16000:
        rows.pop(0)
    return rows


def context(root, arm, revision, manifest):
    prompt = COMMON+({'basic': BASIC, 'symbolic': RICH}[arm])
    prompt += '\nFrozen common motion loop (the supplied Session delegates to this; do not reproduce it):\n'+(root/'executor.py').read_text()
    prompt += '\nIdentical starter planner:\n'+(root/'starter.py').read_text()
    for previous in range(revision):
        directory = root/arm/('revision_000' if previous == 0 else f'round_{previous:03d}')
        prompt += '\nOwn revision '+str(previous)+' feedback:\n'+json.dumps([base.feedback(directory/'trials'/f'seed_{s}') for s in manifest['train_seeds']])
        if previous:
            response = json.loads((directory/'response.json').read_text()); prompt += '\nOwn hypothesis: '+response['hypothesis'][:2000]
    if revision > 1:
        prompt += '\nLatest exact planner:\n'+(root/arm/f'round_{revision-1:03d}'/'candidate.py').read_text()
    examples = []
    for seed in manifest['train_seeds']:
        for previous in range(revision-1, -1, -1):
            directory = root/arm/('revision_000' if previous == 0 else f'round_{previous:03d}')/'trials'/f'seed_{seed}'
            if (directory/'api_calls.json').exists():
                examples.append((seed, previous, directory, base.feedback(directory)['success'])); break
    chosen = []
    for success in (False, True):
        eligible = [row for row in examples if row[3] == success]
        if eligible: chosen.append(eligible[0])
    for row in examples:
        if len(chosen) >= 2: break
        if row not in chosen: chosen.append(row)
    images = []
    for _, previous, directory, success in chosen:
        attached, metadata = base.sensor_context(directory); images.extend(attached)
        prompt += '\nTraining example from revision '+str(previous)+', success='+str(success)+'. First/last side RGB then depth, calibration:\n'+json.dumps(metadata)
        path = directory/'geometry_records.json'
        if path.exists():
            selected = compact_checkpoints(path)
            if selected:
                prompt += '\nOwn recent action checkpoints (estimates, not contact/success certificates):\n'+json.dumps(selected, separators=(',', ':'))
    prompt += '\nReturn the next planner. No manual policy fixes will be inserted.'
    if len(prompt) > manifest['max_prompt_characters']:
        raise RuntimeError('Prompt budget exceeded')
    return prompt, images


@contextmanager
def configured():
    with patch.multiple(base, library=library, sources=sources, context=context,
                        COMMON=COMMON, BASIC=BASIC, NUMERIC=RICH, SYMBOLIC='',
                        validate_planner=validate_planner):
        yield


def prepare(root, train, reserved, rounds=3):
    # Preparation is complete before any rollout or model call is allowed.
    with configured():
        # The inherited preparation validates its historical starter. Our new
        # starter is substituted and independently validated before registration.
        with patch.object(base, 'validate_planner', BASE_VALIDATE):
            manifest = base.prepare(root, rounds, train, reserved, ARMS)
    starter = ROOT/'policies/support_graph_step_starter.py'
    validate_planner(starter.read_text())
    shutil.copyfile(starter, root/'starter.py')
    manifest.update(schema='geometry_step_abstraction_pilot_v1',
        starter_sha256=base.digest(root/'starter.py'),
        arm_labels={'basic':'Base', 'symbolic':'Full geometry'},
        feedback='Own outcomes/notes; same example/image rule; last up to eight active-object checkpoints per example, at most 16000 characters; no global scene packets.',
        contrasts=['Shared observer and executor: current surface measurements versus rigid references, alignment and compact per-step abstraction.'],
        primary_metrics=['reserved SR at every fixed revision', 'first reserved revision reaching 80 percent', 'mean reserved SR across revisions', 'training-selected reserved SR'],
        retained_budget_secondary='Highest training SR among available revisions, earliest tie; no reserved selection.',
        limitations=['One actor chain per arm; replication required.',
                     'Base has basic geometry; this compares a supplied abstraction package, not geometry versus none.',
                     'Same revisions and rollout slots; input/output tokens recorded, not equalized.',
                     'No prior evolved policy, inspected failure diagnosis or reserved outcome enters actor context.',
                     'Character reduction and development parity are not success-rate evidence.'])
    base.write(root/'manifest.json', manifest)
    base.write(root/'progress.json', {'status':'registered'})
    return manifest


def check(root, manifest):
    if manifest.get('schema') != 'geometry_step_abstraction_pilot_v1' or manifest['arms'] != list(ARMS):
        raise ValueError('Wrong experiment variant')
    with configured():
        base.assert_frozen(root, manifest)
    for arm in ARMS:
        if library(arm) != (root/(arm+'_library.py')).read_text():
            raise ValueError('Library differs from registered variant')


def audit_starters(root, manifest):
    pairs = []
    for seed in manifest['train_seeds']:
        paths = [root/arm/'revision_000/trials'/f'seed_{seed}' for arm in ARMS]
        results = [json.loads((p/'result.json').read_text()) for p in paths]
        for key in ('success','policy_error','scene','camera_calls','physics','sensor','candidate_sha256'):
            if results[0][key] != results[1][key]:
                raise ValueError('Starter mismatch: '+key)
        with np.load(paths[0]/'rollout.npz') as a, np.load(paths[1]/'rollout.npz') as b:
            if a.files != b.files or not all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes() for k in a.files):
                raise ValueError('Starter physics differ')
        rows = [json.loads((p/'geometry_records.json').read_text()) if (p/'geometry_records.json').exists() else [] for p in paths]
        if len(rows[0]) != len(rows[1]):
            raise ValueError('Starter checkpoint counts differ')
        for a, b in zip(*rows):
            if a['kind'] != 'geometry_step' or b['kind'] != 'geometry_step':
                raise ValueError('Global feedback returned')
            a, b = a['record'], b['record']
            for key in ('object','step','status','action_status'):
                if a.get(key) != b.get(key):
                    raise ValueError('Starter checkpoint mismatch: '+key)
            if abs(a['time']-b['time']) > 1e-6:
                raise ValueError('Checkpoint time mismatch')
            if a['status'] == 'current_estimate':
                for key, other in (('center','center_m'),('normal','normal')):
                    np.testing.assert_allclose(a['geometry'][key], b['geometry'][other], atol=5.01e-7, rtol=0)
        pairs.append({'seed':seed, 'identical_physics':True, 'matching_focus_estimates':True,
                      'checkpoints':len(rows[0]), 'source_sha256':{str((p/name).relative_to(root)):base.digest(p/name)
                        for p in paths for name in ('result.json','rollout.npz','geometry_records.json') if (p/name).exists()}})
    receipt = {'status':'passed', 'scope':'Starter development/interface parity; no learning benefit inferred.', 'pairs':pairs}
    base.write(root/'starter_parity.json', receipt)
    return receipt


def bootstrap(root, manifest):
    check(root, manifest)
    if list(root.glob('*/round_*/invocation.json')):
        raise ValueError('Bootstrap must precede model calls')
    with configured():
        for arm in ARMS:
            directory = root/arm/'revision_000'
            candidate = base.bundle(root, arm, (root/'starter.py').read_text(), directory/'bundled.py')
            base.execute(root, arm, candidate, directory/'trials', manifest['train_seeds'], manifest)
    audit_starters(root, manifest)
    base.write(root/'progress.json', {'status':'bootstrapped'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--train-seeds', nargs='+', type=int)
    parser.add_argument('--reserved-seeds', nargs='+', type=int)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--phase', choices=('prepare','bootstrap','train','evaluate'), default='prepare')
    args = parser.parse_args(); root = args.output.resolve()
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
