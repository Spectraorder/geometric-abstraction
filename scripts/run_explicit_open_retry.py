"""Development-only generic open/reacquire/retry after a placement exception.

A human-authored bounded recovery intervention on all five prior guard-development
layouts. Never actor feedback, no model call, no reserved layouts or object truth.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from prototype_open_gripper_session import library
from run_partial_geometry_development import ordinary
from run_visual_guard_counterfactual import ROOT, digest, read, write
from run_vision_trial import validate_source


def run(parent, output):
    parent_manifest=read(parent/'manifest.json')
    study=Path(parent_manifest['parent']); manifest=read(study/'manifest.json')
    seeds=parent_manifest['seeds']
    if seeds != list(range(990201,990206)):
        raise ValueError('Require all five existing guard-development layouts')
    source=(parent/'guard_5mm.py').read_text()
    old="        session.checkpoint(color, kind, action_status='exception: '+str(error)[:180])"
    insertion='''
        if kind == 'place':
            try:
                opened = session.open_gripper()
                session.checkpoint(color, 'recovery_open', action_status=opened['status'])
                picked = session.pick(color, transit_z=float(point[2]+.085))
                session.checkpoint(color, 'recovery_pick', action_status=picked.get('status', 'unknown'))
                if picked.get('held', False):
                    retried = session.place(np.asarray(point, float).tolist())
                    session.checkpoint(color, 'recovery_place', action_status=retried.get('status', 'unknown'))
                    return retried
            except (RuntimeError, ValueError) as recovery_error:
                session.checkpoint(color, 'recovery_exception', action_status='exception: '+str(recovery_error)[:180])
'''
    if source.count(old)!=1:
        raise ValueError('Unexpected action exception boundary')
    candidate_source=source.replace(old,old+insertion,1)
    validate_source(candidate_source)
    output.mkdir(parents=True,exist_ok=False)
    candidate=output/'candidate.py';candidate.write_text(candidate_source)
    lib=output/'library.py';lib.write_text(library((parent/'symbolic_library.py').read_text()))
    for path in (Path(__file__),ROOT/'scripts/prototype_open_gripper_session.py'):
        shutil.copy2(path,output/path.name)
    protocol={'scope':__doc__,'parent':str(parent),'seeds':seeds,'model_calls':0,
        'heldout_seeds_used':[],'not_actor_feedback':True,
        'rule':'After a placement exception, open explicitly, pick the same target again and retry its unchanged goal once. Preserve the original downstream planner and all fit gates.',
        'scope_of_bound':'At most one open/reacquire/retry per invocation of the original place wrapper; no recursive recovery.',
        'source_sha256':{str(p):digest(p) for p in (candidate,lib,Path(__file__),parent/'guard_5mm.py')},
        'exact_insertion':{'after':old,'text':insertion},
        'interpretation':'A control-interface development counterfactual, not a learned-policy or abstraction-effect estimate.'}
    write(output/'manifest.json',protocol)
    rows=[]
    for seed in seeds:
        trial=output/'trials'/f'seed_{seed}'
        command=[sys.executable,str(ROOT/'scripts/run_vision_trial.py'),'--candidate',str(candidate),'--output',str(trial),
            '--seed',str(seed),'--task','inventory_structure','--layout','sampled','--structure-recipe',manifest['recipe'],
            '--structure-color-map',json.dumps(manifest['color_maps'][str(seed)]),'--object-count','4','--wrist-depth',
            '--record-geometry','--timeout','120','--geometry-library',str(lib)]
        with (output/f'seed_{seed}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl'),
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
        prior=parent/'trials'/f'seed_{seed}'
        result,original=read(trial/'result.json'),read(prior/'result.json')
        for key in ('seed','task','layout','scene','physics','sensor','wrist_depth','sensor_blackout'):
            if result[key]!=original[key]:raise ValueError('Settings differ: '+key)
        if digest(trial/'camera_000.npz')!=digest(prior/'camera_000.npz'):
            raise ValueError('Initial visual input differs')
        calls,old_calls=read(trial/'api_calls.json'),read(prior/'api_calls.json')
        records=read(trial/'geometry_records.json') if (trial/'geometry_records.json').exists() else []
        recovery=[r['record'] for r in records if r.get('kind')=='geometry_step' and r['record']['step'].startswith('recovery_')]
        opened=[r for r in recovery if r['step']=='recovery_open']
        if opened:
            index=next((i for i,(a,b) in enumerate(zip(calls,old_calls)) if a!=b),None)
            if index is None or calls[index]['method']!='gripper' or calls[index]['params']!={'angle':.8,'seconds':.6}:
                raise ValueError('First changed call must be explicit opening')
            prefix_frames=sum(c['method']=='camera' for c in calls[:index])
            for i in range(prefix_frames):
                name=f'camera_{i:03d}.npz'
                if digest(trial/name)!=digest(prior/name):raise ValueError('Visual failure prefix differs')
        else:
            if ordinary(calls)!=ordinary(old_calls) or result['success']!=original['success']:
                raise ValueError('Untreated trajectory changed')
            for p in prior.glob('camera_*.npz'):
                if digest(trial/p.name)!=digest(p):raise ValueError('Untreated camera differs')
            index=None;prefix_frames=len(list(prior.glob('camera_*.npz')))
        rows.append({'seed':seed,'original_success':original['success'],'recovery_success':result['success'],
            'policy_error':result['policy_error'],'opening_count':len(opened),'first_changed_api_index':index,
            'unchanged_camera_prefix_count':prefix_frames,'recovery_checkpoints':recovery,
            'artifact_sha256':{str(p):digest(p) for p in (trial/'result.json',trial/'api_calls.json',prior/'result.json',prior/'api_calls.json')}})
        write(output/'progress.json',{'scope':__doc__,'trials':rows})
        print(seed,'original',original['success'],'recovery',result['success'],'opens',len(opened),flush=True)
    report={'scope':__doc__,'status':'completed','protocol_sha256':digest(output/'manifest.json'),
            'trials':rows,'original_successes':sum(r['original_success'] for r in rows),
            'recovery_successes':sum(r['recovery_success'] for r in rows),
            'limitations':['Five previously inspected development layouts, not independent generalization evidence.',
                'Human-authored recovery; no model update or geometry-vs-Base contrast.',
                'Runtime recovery uses visual PnP and public robot commands; final binary success is read only after rollout.',
                'Opening acknowledgment certifies neither physical release nor stability.']}
    write(output/'summary.json',report)
    write(ROOT/'docs/evaluations/explicit_open_retry_development.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.parent.resolve(),args.output.resolve())
