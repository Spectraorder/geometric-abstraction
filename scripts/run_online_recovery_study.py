"""Freeze and run five paired controlled-recovery scenes with balanced parallelism."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

from run_online_bridge_pilot import (ROOT, digest, write, source_paths, package_versions,
                                     client_version, verify_sources)
from build_online_bridge_geometry import online_library
from online_recovery_fixture import RecoveryFixture
from run_online_event_development import run

SEEDS = tuple(range(2028102001, 2028102006))


def prepare(root):
    dev = ROOT/'outputs/online_recovery_dev/pair_01'
    results = {arm: json.loads((dev/arm/'result.json').read_text()) for arm in ('basic','symbolic')}
    if not all(r.get('success') and r['timing']['max_simulation_lag_seconds'] < 2. for r in results.values()):
        raise ValueError('Complete a successful, paced development recovery pair first')
    fixtures = [json.loads((dev/arm/'fixture.json').read_text()) for arm in results]
    if fixtures[0]['after_sha256'] != fixtures[1]['after_sha256']:
        raise ValueError('Development pair did not share the same event state')
    root.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in source_paths():
        relative = path.relative_to(ROOT); target = root/'sources'/relative
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
        hashes[str(relative)] = digest(path)
    geometry=online_library(); (root/'geometry.py').write_text(geometry)
    slots=[]
    for i,seed in enumerate(SEEDS):
        for arm in (('basic','symbolic') if i%2==0 else ('symbolic','basic')):
            slots.append({'seed':seed,'arm':arm,'path':f'episodes/{seed}/{arm}'})
    protocol={'name':'controlled_online_foundation_recovery','registered_at':time.time(),
        'seeds':SEEDS,'slots':slots,'budget_s':240.,'concurrency':4,
        'development_seeds':list(range(2028100101,2028100104)),
        'intervention':{'kind':'foundation','translation_axis':'world_y','translation_m':.025,
                        'phase':'after two shared sensor-controlled foundation placements'},
        'timing':'Recovery clock starts after common accelerated prefix and external intervention; '
                 'no resets or state edits after clock starts; one-to-one pacing',
        'feedback':'Both receive historical pre-event RGB and public prefix; Base current RGB; '
                   'Full current geometric text; both may request current RGB-D',
        'routing':{'initial_and_replan':'gpt-6-astra medium','routine':'gpt-6-luna low'},
        'success':'Unchanged private support graph, five-second stability and 3 mm drift',
        'setup_failures':'Count all slots; report setup failure separately; never replace a seed',
        'parallelism':'Balanced batches of up to two complete pairs; one BLAS/OpenMP thread per worker',
        'max_timing_lag_for_realtime_claim_s':2.,
        'failure_policy':'No automatic retry. Preserve attempts. Infrastructure errors stop new batches.',
        'selection':'Fixed new seeds; no reserved outcome inspected before freezing',
        'source_sha256':hashes,'geometry_sha256':hashlib.sha256(geometry.encode()).hexdigest(),
        'package_versions':package_versions(),'model_client_version':client_version(),
        'development_results':{str(dev/arm/'result.json'):digest(dev/arm/'result.json') for arm in results}}
    write(root/'protocol.json',protocol);(root/'protocol.sha256').write_text(digest(root/'protocol.json')+'\n')
    write(root/'progress.json',{'status':'registered','completed':0,'total':10})


def worker(root,index):
    protocol=verify_sources(root);slot=protocol['slots'][index]
    result=run(root/slot['path'],slot['seed'],slot['arm'],protocol['budget_s'],
               preparation=RecoveryFixture('foundation',.025))
    verify_sources(root)
    print(json.dumps({'slot':index,'seed':slot['seed'],'arm':slot['arm'],
                      'status':result['status'],'success':result['success']}),flush=True)
    if result['status']=='infrastructure_error':
        fixture=root/slot['path']/'fixture.json'
        if fixture.exists() and json.loads(fixture.read_text()).get('status')=='prefix_failed':
            return
        raise RuntimeError('Episode infrastructure error; inspect retained result')


def execute(root):
    import fcntl
    protocol=verify_sources(root)
    if (root/'STOP.json').exists() or (root/'episodes').exists():
        raise FileExistsError('Preserve existing attempts; this runner does not resume')
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.time();completed=0
        def launch(index):
            with (root/f'worker_{index:02d}.log').open('w') as log:
                return subprocess.call([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),
                    '--root',str(root),'--phase','episode','--slot',str(index)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            for offset in range(0,len(protocol['slots']),protocol['concurrency']):
                indices=list(range(offset,min(offset+protocol['concurrency'],len(protocol['slots']))))
                verify_sources(root)
                write(root/'progress.json',{'status':'running','completed':completed,'total':10,
                    'active_slots':indices,'started_at':start})
                with ThreadPoolExecutor(max_workers=protocol['concurrency']) as pool:
                    codes=list(pool.map(launch,indices))
                completed+=len(indices)
                print(json.dumps({'completed':completed,'returncodes':codes}),flush=True)
                if any(codes):raise RuntimeError('A worker failed; no further batch dispatched')
            verify_sources(root)
            from summarize_online_recovery import summarize
            summary=summarize(root);write(root/'summary.json',summary)
            write(root/'progress.json',{'status':'completed','completed':completed,'total':10,
                                        'wall_seconds':time.time()-start})
        except Exception as error:
            write(root/'STOP.json',{'error':str(error),'completed':completed,'time':time.time()})
            write(root/'progress.json',{'status':'stopped','completed':completed,'total':10})
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=('prepare','run','check','episode'),required=True)
    p.add_argument('--slot',type=int)
    a=p.parse_args();root=a.root.resolve()
    if a.phase=='prepare':prepare(root)
    elif a.phase=='run':execute(root)
    elif a.phase=='episode':worker(root,a.slot)
    else:verify_sources(root);print('Registered sources match')
