"""Open explicitly after a known development placement exception; no SR claim.

This manually inserted probe is not an evolved policy and never enters actor
feedback. It verifies the shared command after an unchanged visual failure prefix.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from prototype_open_gripper_session import library
from run_visual_guard_counterfactual import ROOT, digest, read, write


def run(parent, output):
    seed=990202
    parent_manifest=read(parent/'manifest.json');study=Path(parent_manifest['parent'])
    manifest=read(study/'manifest.json')
    if seed not in parent_manifest['seeds']:raise ValueError('Require previously inspected development layout')
    source=(parent/'guard_5mm.py').read_text()
    old="        session.checkpoint(color, kind, action_status='exception: '+str(error)[:180])"
    new=old+"\n        if kind == 'place':\n            opened = session.open_gripper()\n            session.checkpoint(color, 'explicit_open', action_status=opened['status'])"
    if source.count(old)!=1:raise ValueError('Unexpected exception boundary')
    output.mkdir(parents=True,exist_ok=False)
    candidate=output/'candidate.py';candidate.write_text(source.replace(old,new,1))
    lib=output/'library.py';lib.write_text(library((parent/'symbolic_library.py').read_text()))
    for path in (Path(__file__),ROOT/'scripts/prototype_open_gripper_session.py'):shutil.copy2(path,output/path.name)
    protocol={'scope':__doc__,'parent':str(parent),'seed':seed,'model_calls':0,'heldout_seeds_used':[],
              'not_actor_feedback':True,'exact_insertion':{'before':old,'after':new},
              'candidate_sha256':digest(candidate),'library_sha256':digest(lib),
              'driver_sha256':digest(Path(__file__))}
    write(output/'manifest.json',protocol)
    trial=output/f'seed_{seed}'
    command=[sys.executable,str(ROOT/'scripts/run_vision_trial.py'),'--candidate',str(candidate),'--output',str(trial),
        '--seed',str(seed),'--task','inventory_structure','--layout','sampled','--structure-recipe',manifest['recipe'],
        '--structure-color-map',json.dumps(manifest['color_maps'][str(seed)]),'--object-count','4','--wrist-depth',
        '--record-geometry','--timeout','120','--geometry-library',str(lib)]
    with (output/'trial.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl'),
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
    prior=parent/'trials'/f'seed_{seed}'
    a,b=read(trial/'api_calls.json'),read(prior/'api_calls.json')
    index=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
    if index is None or a[index]['method']!='gripper':raise ValueError('First change must be the explicit opening command')
    if a[index]['params']!={'angle':.8,'seconds':.6}:raise ValueError('Unexpected gripper command')
    frames=sorted(p.name for p in prior.glob('camera_*.npz'))
    if frames!=sorted(p.name for p in trial.glob('camera_*.npz')):raise ValueError('Unexpected sensing change')
    for name in frames:
        if digest(trial/name)!=digest(prior/name):raise ValueError('Visual failure prefix changed')
    records=read(trial/'geometry_records.json')
    opened=[r['record'] for r in records if r.get('kind')=='geometry_step' and r['record']['step']=='explicit_open']
    if len(opened)!=1 or opened[0]['action_status']!='opened_unverified':raise ValueError('Missing opening checkpoint')
    if opened[0]['status']!='unavailable' or 'geometry' in opened[0]:raise ValueError('Old geometry resurfaced after action')
    report={'status':'completed','scope':__doc__,'protocol_sha256':digest(output/'manifest.json'),
        'unchanged_api_prefix_length':index,'identical_camera_archives':len(frames),
        'opening_command':a[index], 'opening_checkpoint':opened[0],
        'task_success_claim':False,'physical_release_certified':False,
        'limitations':['One manually triggered development case, not agent learning or an SR-effect test.',
            'Explicit opening clears controller bookkeeping only after command acknowledgment.',
            'No task-specific trigger, automatic recovery, retreat or stable-support check is added to the interface.'],
        'artifact_sha256':{str(p):digest(p) for p in (trial/'api_calls.json',trial/'geometry_records.json',prior/'api_calls.json')}}
    write(output/'summary.json',report);write(ROOT/'docs/evaluations/explicit_open_development.json',report)
    print({k:report[k] for k in ('status','unchanged_api_prefix_length','identical_camera_archives','task_success_claim')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.parent.resolve(),args.output.resolve())
