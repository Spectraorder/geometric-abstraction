"""Record one target's visual fits in a behavior-equivalent development replay."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import json
from run_visual_guard_counterfactual import ROOT, digest, read, write
from run_partial_geometry_development import ordinary
from run_vision_trial import validate_source


def run(parent, output, revision, seed):
    manifest=read(parent/'manifest.json')
    if seed not in manifest['train_seeds']:raise ValueError('Only existing training layouts are eligible')
    original_candidate=parent/'symbolic'/f'round_{revision:03d}'/'bundled.py'
    prior=original_candidate.parent/'trials'/f'seed_{seed}'
    if not (prior/'result.json').exists():raise ValueError('Require a completed parent trial')
    source=(parent/'symbolic_library.py').read_text()
    old='        return self.ledger.observe(color, measured)'
    new="        self.api.record_geometry({'kind': 'observer_fit', 'object': color, 'fit': metric_copy(measured, plain=True)})\n"+old
    if source.count(old)!=1:raise ValueError('Expected one current-fit boundary')
    source=source.replace(old,new,1);validate_source(source)
    output.mkdir(parents=True,exist_ok=False)
    candidate=output/'candidate.py';shutil.copy2(original_candidate,candidate)
    lib=output/'library.py';lib.write_text(source);shutil.copy2(Path(__file__),output/'driver_snapshot.py')
    write(output/'manifest.json',{'scope':__doc__,'parent':str(parent),'revision':revision,'seed':seed,
        'candidate_sha256':digest(candidate),'library_sha256':digest(lib),'driver_sha256':digest(Path(__file__)),
        'equivalence_helper_sha256':digest(ROOT/'scripts/run_partial_geometry_development.py'),
        'exact_insertion':{'before':old,'after':new},'not_actor_feedback':True,'model_calls':0,'heldout_seeds_used':[]})
    trial=output/f'seed_{seed}'
    command=[sys.executable,str(ROOT/'scripts/run_vision_trial.py'),'--candidate',str(candidate),'--output',str(trial),
        '--seed',str(seed),'--task','inventory_structure','--layout','sampled','--structure-recipe',manifest['recipe'],
        '--structure-color-map',json.dumps(manifest['color_maps'][str(seed)]),'--object-count','4','--wrist-depth',
        '--record-geometry','--timeout','120','--geometry-library',str(lib)]
    with (output/'trial.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl'),
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
    a,b=read(trial/'result.json'),read(prior/'result.json')
    for key in ('candidate_sha256','seed','task','layout','scene','physics','sensor','wrist_depth','sensor_blackout','success','policy_error'):
        if a[key]!=b[key]:raise ValueError('Logging changed '+key)
    if ordinary(read(trial/'api_calls.json'))!=ordinary(read(prior/'api_calls.json')):raise ValueError('Nonlogging calls changed')
    names=sorted(p.name for p in prior.glob('camera_*.npz'))
    if names!=sorted(p.name for p in trial.glob('camera_*.npz')):raise ValueError('Camera sequence changed')
    for name in [*names,'rollout.npz']:
        if digest(trial/name)!=digest(prior/name):raise ValueError('Camera/opaque physics differs')
    records=read(trial/'geometry_records.json')
    if [r for r in records if r.get('kind')!='observer_fit']!=read(prior/'geometry_records.json'):
        raise ValueError('Original geometry records changed')
    result={'status':'passed','scope':__doc__,'seed':seed,'revision':revision,
        'observer_fits':sum(r.get('kind')=='observer_fit' for r in records),'camera_archives':len(names),
        'identical_nonlogging_calls_excluding_terminal_cpu_profile':True,'identical_opaque_physics_archive':True,
        'identical_original_geometry_records':True,
        'source_sha256':{str(p):digest(p) for p in (candidate,lib,Path(__file__))},
        'artifact_sha256':{str(folder/name):digest(folder/name) for folder in (prior,trial) for name in ('api_calls.json','geometry_records.json','rollout.npz')}}
    write(output/'equivalence.json',result);print({k:v for k,v in result.items() if k not in ('source_sha256','artifact_sha256','scope')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--revision',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    run(args.parent.resolve(),args.output.resolve(),args.revision,args.seed)
