"""Visual-input Base versus supplied compact geometry, with shared CPU actuation."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import run_geometry_step_experiment as step
from build_geometry_session import library

base = step.base
ROOT = step.ROOT
ARMS = step.ARMS
EXTRA = ('policies/visual_input_session.py', 'policies/support_graph_visual_starter.py',
         'scripts/build_geometry_session.py', 'scripts/run_visual_input_experiment.py')
validate_planner = step.validate_planner
COMMON = step.COMMON[:step.COMMON.index('Instantiate exactly Session')]+'''Instantiate exactly Session(api, objects), with the complete public task inventory.
Both conditions share a frozen hidden CPU RGB-D pick/place actuator. Its internal
geometry is estimated from images, not simulator object truth. Do not access, tune,
subclass, redefine or mutate the session or its internals. Common public methods:
time(), tool(), pick(color,transit_z=None), place(surface_xyz), and
pick_and_place(color,surface_xyz). You choose order, targets, workspace and recovery.
place accepts center XY and support-surface Z, in world meters. Every arm may compute
its own perception, geometry and memory from visual input within the same code budget.
Only the Full condition supplies object measurements and geometry query operators.
A released status is not a task-success certificate. Status/held fields report only
the executor's own state; an unconfirmed placement may leave its gripper closed.

After each action call checkpoint(color,step,action_status=None). Base checkpoints
only log identity, time, step and action status. Full checkpoints additionally expose
its current visual geometry; missing/stale evidence remains unknown. Full alone may
request an explicit single-object refresh. Do not use global snapshot/state/text/history
methods. Keep evidence focused on the current object; query supports when needed.

'''+step.COMMON[step.COMMON.index('Runtime RGB-D is always available'):]
BASIC = '''Base receives raw RGB-D through api.camera(), camera calibration, public task
labels and robot-only proprioception/CAD. Session has NO observe, get, table_z,
reference, relation, align, target or goal_error methods. No precomputed object
center, dimensions, vertices, surface heights or geometric residuals are supplied.
Pick/place returns only status and held. Checkpoint returns identity, step, time and
action status; refresh=True is unavailable. Request your visual evidence directly
through api.camera(). Implement any useful geometry yourself, including replacing
the common starter's visible-top estimator. Coordinates you calculate from pixels
are allowed. Neither arm may read simulator object state or evaluator diagnostics.
'''
RICH = '''Full additionally provides observe(color) to fit current RGB-D and get(color)
for a same-time cached fit. table_z(xy) estimates the table from visual depth.
Fits expose center, normal, height, dims[2], top, bottom, extent_z, coordinate_frame,
units, observation_time, source/view, position_fully_observed and inferred_orientation.
Returned geometry is immutable input: copy before modifying. Action-return observations
and geometric residuals are visual estimates. They are never hidden object truth.
'''+step.RICH


def sources():
    result = step.sources()
    result.update({name: base.digest(ROOT/name) for name in EXTRA})
    return result


def context(root, arm, revision, manifest):
    with patch.multiple(step, COMMON=COMMON, BASIC=BASIC, RICH=RICH):
        return step.context(root, arm, revision, manifest)


@contextmanager
def configured():
    with patch.multiple(base, library=library, sources=sources, context=context,
                        COMMON=COMMON, BASIC=BASIC, NUMERIC=RICH, SYMBOLIC='',
                        validate_planner=validate_planner):
        yield


def prepare(root, train, reserved, rounds=3):
    with configured(), patch.object(base, 'validate_planner', step.BASE_VALIDATE):
        manifest = base.prepare(root, rounds, train, reserved, ARMS)
    starter = ROOT/'policies/support_graph_visual_starter.py'
    validate_planner(starter.read_text()); shutil.copyfile(starter, root/'starter.py')
    manifest.update(schema='visual_input_geometry_pilot_v1', starter_sha256=base.digest(root/'starter.py'),
        arm_labels={'basic':'Visual Base', 'symbolic':'Full geometry'},
        feedback='Own outcomes/notes; same example/image rule; up to eight active-object checkpoints per example within 16000 characters. Base has action metadata only; Full adds visual geometry.',
        contrasts=['Raw visual planning versus supplied perception/geometry abstraction; identical hidden visual actuator. Not a pure symbolic-versus-numeric contrast.'],
        primary_metrics=['reserved SR at every fixed revision', 'first reserved revision reaching 80 percent', 'mean reserved SR across revisions', 'training-selected reserved SR'],
        retained_budget_secondary='Highest training SR among available revisions, earliest tie; no reserved selection.',
        limitations=['One isolated actor chain per arm within each replica.',
                     'Treatment includes supplied perception and geometry, not solely text representation.',
                     'The shared hidden actuator uses RGB-D estimates in both arms; this is high-level planning, not pixels-to-torques.',
                     'Same revisions and rollout slots; tokens recorded, not equalized. Rollout duration is not an objective.',
                     'No prior evolved policy, inspected diagnosis or reserved outcome enters actor feedback.'])
    base.write(root/'manifest.json', manifest); base.write(root/'progress.json', {'status':'registered'})
    return manifest


def check(root, manifest):
    if manifest.get('schema') != 'visual_input_geometry_pilot_v1' or manifest['arms'] != list(ARMS):
        raise ValueError('Wrong experiment variant')
    with configured():base.assert_frozen(root, manifest)
    for arm in ARMS:
        if library(arm) != (root/(arm+'_library.py')).read_text():
            raise ValueError('Library differs from registered variant')


def audit_starters(root, manifest):
    pairs = []
    for seed in manifest['train_seeds']:
        paths = [root/arm/'revision_000/trials'/f'seed_{seed}' for arm in ARMS]
        results = [json.loads((p/'result.json').read_text()) for p in paths]
        for key in ('success','policy_error','scene','camera_calls','physics','sensor','candidate_sha256'):
            if results[0][key] != results[1][key]:raise ValueError('Starter mismatch: '+key)
        with np.load(paths[0]/'rollout.npz') as a, np.load(paths[1]/'rollout.npz') as b:
            if a.files != b.files or not all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes() for k in a.files):
                raise ValueError('Starter physics differ')
        rows = [json.loads((p/'geometry_records.json').read_text()) if (p/'geometry_records.json').exists() else [] for p in paths]
        if len(rows[0]) != len(rows[1]):raise ValueError('Starter checkpoint counts differ')
        for a,b in zip(*rows):
            if a['kind'] != 'geometry_step' or b['kind'] != 'geometry_step':raise ValueError('Unexpected starter record')
            a,b=a['record'],b['record']
            if set(a)-{'object','step','time','status','action_status'}:raise ValueError('Base supplied geometry leak')
            if a['status'] != 'visual_input_only':raise ValueError('Wrong Base checkpoint')
            if any(a.get(k)!=b.get(k) for k in ('object','step','action_status')) or abs(a['time']-b['time'])>1e-6:
                raise ValueError('Starter action boundaries differ')
        pairs.append({'seed':seed,'identical_physics':True,'same_action_boundaries':True,
            'base_supplied_geometry_absent':True,'checkpoints':len(rows[0]),
            'source_sha256':{str((p/name).relative_to(root)):base.digest(p/name) for p in paths for name in ('result.json','rollout.npz','geometry_records.json') if (p/name).exists()}})
    receipt={'status':'passed','scope':'Identical starter execution with different public observation interfaces; no learning benefit inferred.','pairs':pairs}
    base.write(root/'starter_parity.json',receipt);return receipt


def bootstrap(root, manifest):
    check(root,manifest)
    if list(root.glob('*/round_*/invocation.json')):raise ValueError('Bootstrap must precede actors')
    with configured():
        for arm in ARMS:
            directory=root/arm/'revision_000'
            candidate=base.bundle(root,arm,(root/'starter.py').read_text(),directory/'bundled.py')
            base.execute(root,arm,candidate,directory/'trials',manifest['train_seeds'],manifest)
    audit_starters(root,manifest);base.write(root/'progress.json',{'status':'bootstrapped'})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--train-seeds',nargs='+',type=int)
    parser.add_argument('--reserved-seeds',nargs='+',type=int)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--phase',choices=('prepare','bootstrap','train','evaluate'),default='prepare')
    args=parser.parse_args();root=args.output.resolve()
    if not args.resume and (not args.train_seeds or not args.reserved_seeds):parser.error('Explicit seeds required')
    manifest=json.loads((root/'manifest.json').read_text()) if args.resume else prepare(root,args.train_seeds,args.reserved_seeds,args.rounds)
    check(root,manifest)
    if args.phase=='bootstrap':bootstrap(root,manifest)
    elif args.phase!='prepare':
        if not (root/'starter_parity.json').exists():bootstrap(root,manifest)
        else:audit_starters(root,manifest)
        with configured():base.campaign(root,manifest,args.phase)
