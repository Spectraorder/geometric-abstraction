"""Render all actual recordings without rerunning policies or model calls."""
import argparse,concurrent.futures,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--wait-for-batch',action='store_true');a=p.parse_args()
if a.wait_for_batch:
    while not (a.root/'batch_terminal.json').exists():time.sleep(3)
def render(job):
    folder=a.root/job['id']
    if not (folder/'rollout.npz').exists():return {'id':job['id'],'render':'no_rollout'}
    if (folder/'video.json').exists():return {'id':job['id'],'render':'already_complete'}
    with (folder/'render.log').open('w') as log:
        r=subprocess.run([sys.executable,str(ROOT/'scripts/render_luna_astra_trial.py'),'--trial',str(folder),'--endpoint',job['endpoint'],'--speed','3','--fps','12'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    result={'id':job['id'],'render':'complete' if r.returncode==0 else 'error'};print(json.dumps(result),flush=True);return result
jobs=json.loads((a.root/'campaign.json').read_text())['jobs']
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(render,jobs))
(a.root/'render_status.json').write_text(json.dumps(results,indent=2)+'\n')
