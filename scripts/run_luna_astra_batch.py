"""Run the frozen 3×4 effort grid and single-model controls, without Jev.

Every attempt gets a fresh output folder. Resume skips terminal records only;
failed attempts remain failures and are never silently rerun.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def run_one(job, output):
    trial = output / job['id']
    terminal = output / (job['id'] + '.terminal.json')
    if terminal.exists():
        return json.loads(terminal.read_text())
    if trial.exists():
        raise RuntimeError(f'Unfinished attempt requires manual audit: {trial.name}')
    started = time.time()
    command = [sys.executable, str(ROOT/'scripts/run_luna_astra_trial.py'),
        '--mode',job['mode'],'--fast-effort',job['fast_effort'],
        '--slow-effort',job['slow_effort'],'--fault',job['fault'],
        '--seed',str(job['seed']),'--output',str(trial)]
    with (output/(job['id']+'.log')).open('w') as log:
        try:
            proc = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT/'src')),
                stdout=log, stderr=subprocess.STDOUT, timeout=1080)
            status = 'completed' if proc.returncode == 0 and (trial/'result.json').exists() else 'infrastructure_error'
            error = None if status == 'completed' else 'TrialProcessFailed'
        except subprocess.TimeoutExpired:
            status,error = 'timeout','TrialTimeout'
    record = dict(job,status=status,error_type=error,started_at=started,wall_seconds=time.time()-started)
    terminal.write_text(json.dumps(record,indent=2)+'\n')
    result = json.loads((trial/'result.json').read_text()) if (trial/'result.json').exists() else {}
    print(json.dumps({'id':job['id'],'status':status,'tower_complete':result.get('success'),'wall_seconds':round(record['wall_seconds'],2)}), flush=True)
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ROOT/'benchmarks/luna_astra_campaign.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.workers not in (1,2):p.error('Use one or two workers')
    manifest=json.loads(a.manifest.read_text());a.output.mkdir(parents=True,exist_ok=True)
    saved=a.output/'campaign.json'
    if saved.exists() and json.loads(saved.read_text())!=manifest:raise ValueError('Manifest changed')
    saved.write_text(json.dumps(manifest,indent=2)+'\n')
    sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [ROOT/'scripts/run_luna_astra_trial.py',ROOT/'scripts/run_vision_trial.py',ROOT/'scripts/build_dual_system_policy.py',ROOT/'src/aspire_so100/astra_decision.py',ROOT/'src/aspire_so100/dual_system.py',ROOT/'src/aspire_so100/decision_session.py']}
    frozen=a.output/'frozen_sources.json'
    if frozen.exists() and json.loads(frozen.read_text())!=sources:raise ValueError('Harness changed during campaign')
    frozen.write_text(json.dumps(sources,indent=2)+'\n')
    snapshots=a.output/'sources';snapshots.mkdir(exist_ok=True)
    for name in sources:
        target=snapshots/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        results=list(pool.map(lambda job:run_one(job,a.output),manifest['jobs']))
    (a.output/'batch_terminal.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
