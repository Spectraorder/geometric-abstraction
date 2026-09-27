"""Paired development trials with explicitly different contact-solver settings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[2801,2802,2803])
    parser.add_argument('--noslip-iterations',type=int,default=10)
    parser.add_argument('--jobs',type=int,default=3)
    parser.add_argument('--policies',nargs='+',choices=('baseline','precision','refined'),default=['baseline','precision'])
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or args.jobs<1 or not 0<=args.noslip_iterations<=100:
        parser.error('Distinct seeds, positive jobs and 0..100 NoSlip iterations required')
    args.output.mkdir(parents=True,exist_ok=False)
    candidates={}
    from build_inventory_policy import build_policy
    for label in args.policies:
        controller = 'column_policy' if label == 'baseline' else 'cube_alignment_policy'
        data = build_policy(controller, refine=label == 'refined').encode()
        path=args.output/(label+'.py');path.write_bytes(data)
        candidates[label]=dict(path=str(path.resolve()),sha256=hashlib.sha256(data).hexdigest())
    cases=[dict(policy=label,seed=seed,physics={'noslip_iterations':args.noslip_iterations}) for seed in args.seeds for label in candidates]
    (args.output/'manifest.json').write_text(json.dumps(dict(split='development_contact_profile',candidates=candidates,cases=cases,scope='Changed physics; do not pool with default-physics policy results.'),indent=2)+'\n')
    env=dict(os.environ,MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def run(case):
        path=args.output/case['policy']/f"seed_{case['seed']}"
        path.parent.mkdir(exist_ok=True)
        command=[sys.executable,str(ROOT/'scripts/run_vision_trial.py'),'--candidate',candidates[case['policy']]['path'],'--output',str(path.resolve()),'--seed',str(case['seed']),'--task','inventory_tower','--layout','sampled','--tower-recipe','cube_triple','--object-count','3','--timeout','180','--noslip-iterations',str(args.noslip_iterations)]
        with (path.parent/(path.name+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Trial failed: '+str(path))
        row=json.loads((path/'result.json').read_text())
        if row['physics']!=case['physics'] or row['candidate_sha256']!=candidates[case['policy']]['sha256']:
            raise RuntimeError('Trial profile or source mismatch')
        print(case['policy'],case['seed'],row['success'],flush=True)
        return dict(**case,trial=str(path),result=row)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:rows=list(pool.map(run,cases))
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
