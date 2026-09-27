"""Record visual observer inputs in a separate, behavior-equivalent development run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from run_visual_guard_counterfactual import digest, read, write, ROOT
from run_vision_trial import validate_source


def run(parent, output, seed):
    parent_protocol=read(parent/'manifest.json')
    study=Path(parent_protocol['parent']);manifest=read(study/'manifest.json')
    if seed not in parent_protocol['seeds']:
        raise ValueError('Only existing development training seeds are allowed')
    prior=parent/'trials'/f'seed_{seed}'
    original=parent/'symbolic_library.py';before=original.read_text()
    old='        robot = self.api.gripper_geometry()\n        side, side_exclusion = filter_robot_visual_geometry(frame, robot)'
    new='''        robot = self.api.gripper_geometry()
        self.api.record_geometry(metric_copy({'kind': 'observer_input',
            'object': color, 'time': float(frame['time']),
            'descriptor': self.backend.objects[color], 'visual_model': model,
            'table_plane': plane, 'robot_geometry': robot}, plain=True))
        side, side_exclusion = filter_robot_visual_geometry(frame, robot)'''
    if before.count(old)!=1:
        raise ValueError('Expected exactly one observer callback insertion site')
    after=before.replace(old,new,1);validate_source(after)
    output.mkdir(parents=True,exist_ok=False)
    library=output/'logging_library.py';library.write_text(after)
    candidate=output/'candidate.py';shutil.copy2(parent/'guard_5mm.py',candidate)
    shutil.copy2(Path(__file__),output/'driver_snapshot.py')
    protocol={'scope':__doc__,'seed':seed,'parent':str(parent),'parent_manifest_sha256':digest(parent/'manifest.json'),
              'original_library_sha256':digest(original),'logging_library_sha256':digest(library),
              'candidate_sha256':digest(candidate),'driver_sha256':digest(Path(__file__)),
              'exact_insertion':{'before':old,'after':new},'model_calls':0,'not_actor_feedback':True}
    write(output/'manifest.json',protocol)
    trial=output/f'seed_{seed}'
    command=[sys.executable,str(ROOT/'scripts/run_vision_trial.py'),'--candidate',str(candidate),
             '--output',str(trial),'--seed',str(seed),'--task','inventory_structure','--layout','sampled',
             '--structure-recipe',manifest['recipe'],'--structure-color-map',json.dumps(manifest['color_maps'][str(seed)]),
             '--object-count','4','--wrist-depth','--record-geometry','--timeout','120','--geometry-library',str(library)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl')
    with (output/'trial.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
    verify(parent, output, seed)


def verify(parent, output, seed):
    prior = parent/'trials'/f'seed_{seed}'
    trial = output/f'seed_{seed}'
    original = parent/'symbolic_library.py'
    library = output/'logging_library.py'
    candidate = output/'candidate.py'
    current,previous=read(trial/'result.json'),read(prior/'result.json')
    for key in ('candidate_sha256','seed','task','layout','scene','physics','sensor','wrist_depth','sensor_blackout','success','policy_error'):
        if current[key]!=previous[key]:raise ValueError('Logging changed '+key)
    calls,previous_calls=read(trial/'api_calls.json'),read(prior/'api_calls.json')
    # Geometry logging performs no sensing, robot motion or physics stepping.
    def ordinary(records):
        result = []
        for call in records:
            if call['method'] == 'record_geometry':
                continue
            call = json.loads(json.dumps(call))
            # CPU profiling is nondeterministic timing metadata. Retain the
            # terminal event and every other parameter, sensor and action.
            if call['method'] == 'done':
                call['params'].pop('cpu_profile', None)
            result.append(call)
        return result
    if ordinary(calls)!=ordinary(previous_calls):raise ValueError('Nonlogging API calls changed')
    old_frames=sorted(prior.glob('camera_*.npz'));frames=sorted(trial.glob('camera_*.npz'))
    if [p.name for p in frames]!=[p.name for p in old_frames]:raise ValueError('Camera sequence changed')
    for a,b in zip(frames,old_frames):
        if digest(a)!=digest(b):raise ValueError('Camera data changed')
    # Compare opaque archive bytes, never expose simulator object state to replay.
    if digest(trial/'rollout.npz')!=digest(prior/'rollout.npz'):
        raise ValueError('Opaque physics archive differs')
    records=read(trial/'geometry_records.json');inputs=[r for r in records if r.get('kind')=='observer_input']
    if [r for r in records if r.get('kind')!='observer_input']!=read(prior/'geometry_records.json'):
        raise ValueError('Original visual records changed')
    result={'status':'passed','scope':__doc__,'seed':seed,'observer_inputs':len(inputs),
            'identical_nonlogging_api_calls_excluding_terminal_cpu_profile':True,
            'excluded_metadata':'done.params.cpu_profile (CPU timing and logging-dependent interval count)','identical_camera_archives':True,
            'identical_opaque_physics_archive':True,'identical_original_geometry_records':True,
            'source_sha256':{str(p):digest(p) for p in (original,library,candidate,Path(__file__))},
            'artifact_sha256':{str(p):digest(p) for folder in (prior,trial) for p in (folder/'api_calls.json',folder/'geometry_records.json',folder/'rollout.npz')}}
    write(output/'equivalence.json',result)
    print({k:v for k,v in result.items() if k not in ('source_sha256','artifact_sha256','scope')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--verify-existing',action='store_true')
    args=parser.parse_args()
    operation = verify if args.verify_existing else run
    operation(args.parent.resolve(),args.output.resolve(),args.seed)
