"""Launch a logged learning step inside an existing Slurm GPU allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobid',required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--model',default='gpt-6-astra')
    p.add_argument('--stage',choices=('pickup','fasten'),default='fasten')
    p.add_argument('--grasp-quality',choices=('legacy','grasp_quality_v1'),default='legacy')
    p.add_argument('--max-reported-tokens',type=int,default=60000)
    p.add_argument('--max-prompt-chars',type=int,default=32000)
    p.add_argument('--reference-policy',type=Path,action='append',default=[])
    p.add_argument('--pairs',type=int,choices=(1,3),default=3)
    a=p.parse_args()
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    launch_path=root/'launch.json'
    if launch_path.exists():
        previous=json.loads(launch_path.read_text())
        try: os.kill(previous['pid'],0)
        except ProcessLookupError: pass
        else: raise RuntimeError('Previous launch process still exists; inspect it before resuming')
    command=['srun','--jobid='+a.jobid,'--overlap','--ntasks=1','--cpus-per-task=2',
             '/usr/bin/env','MUJOCO_GL=egl','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
             'MPLCONFIGDIR=/tmp/aspire_mpl_cache',sys.executable,'-u',
             str(ROOT/'scripts/learn_fasten_screws.py'),'--root',str(root),
             '--pairs',str(a.pairs),'--rounds',str(a.rounds),'--model',a.model,'--stage',a.stage,'--grasp-quality',a.grasp_quality,
             '--max-reported-tokens',str(a.max_reported_tokens),
             '--max-prompt-chars',str(a.max_prompt_chars)]
    for reference in a.reference_policy:
        command += ['--reference-policy',str(reference.resolve())]
    with (root/'progress.log').open('a') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch_path.write_text(json.dumps(dict(pid=proc.pid,jobid=a.jobid,command=command),indent=2)+'\n')
    print(f'Learning launch PID {proc.pid}; log {root / "progress.log"}',flush=True)


if __name__=='__main__':main()
