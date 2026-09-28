"""Register and run one five-pair online Ring study with a 200-second budget."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from statistics import mean
from online_ring_evidence import snapshot, verify_snapshot, audit, read, require, package_versions, client_version
from online_ring_skills import ROOT, choose_training_prior, digest
from run_online_ring_episode import run, write
from evaluate_geometry_learning_curves import hardware

SEEDS=tuple(range(2033101001,2033101006))
ARMS=('basic','symbolic')


def slots():
    return [dict(seed=seed,arm=arm,path=f'episodes/{seed}/{arm}') for i,seed in enumerate(SEEDS)
            for arm in (ARMS if i%2==0 else ARMS[::-1])]


def prepare(root,development):
    require(not root.exists(),'Use a new study root')
    checked=audit(development,[f'seed_{seed}/{arm}' for seed in (1,2) for arm in ARMS])
    require(all(r['completed_requests']>=2 for r in checked['episodes']),'Both arms must exercise the live decision loop')
    require(any(r['depth_queries'] for r in checked['episodes'] if r['arm']=='basic'),'Live Base must exercise depth access')
    physical=read(development.parent/'physical_final/summary.json')
    require(physical['status']=='passed' and physical['seeds']==[1,2],'Missing two-seed physical/interface validation')
    for record in checked['episodes']:
        verify_snapshot(development/record['episode'])
    prior,provenance=choose_training_prior()
    require(shutil.disk_usage(root.parent).free>=15*1024**3,'Insufficient disk reserve')
    for directory in (ROOT/'outputs').iterdir():
        if not directory.is_dir():continue
        for pattern in ('**/protocol.json','**/registration.json','**/manifest.json'):
            for path in directory.resolve().glob(pattern):
                value=read(path)
                for key in ('seeds','train_seeds','heldout_seeds','reserved_seeds','test_seeds'):
                    require(not set(SEEDS).intersection(value.get(key,[])),'Reserved seeds overlap prior registration')
    root.mkdir(parents=True);snapshot(root)
    shutil.copyfile(prior,root/'skill_prior.py')
    protocol=dict(name='online_ring_200_v1',registered_at=time.time(),seeds=list(SEEDS),slots=slots(),
        arms=list(ARMS),budget_s=200.,concurrency=4,model='gpt-6-astra',reasoning='medium',transport='HTTP-only',
        resets_per_episode=1,physics_during_model_wait=True,scenario='Ordinary unassembled ring and fixed peg',
        intervention=None,model_calls_during_actions=False,
        skill=provenance,skill_sha256=digest(root/'skill_prior.py'),
        shared_access='Same learned grasp/lift prefix, public robot kinematics, explicit pad motion, release, images and 16-pixel depth queries',
        contrast='Full additionally receives compact current Ring/peg geometry; Base receives images and qualitative skill outcomes',
        success='Peg axis inside opening, ring center below peg top, no robot contact, low velocity for two seconds and at most 3 mm drift throughout the hold',
        clock='Starts after initial reset, before sensor/skill initialization; includes local perception, requests, actions and settling',
        primary_metrics=['successes / 5','mean capped completion seconds'],
        secondary_metrics=['paired outcomes','complete/incomplete model-request time','tokens and unknown interrupted usage',
                           'image attachments','camera/depth queries','action/feedback time','simulation lag'],
        timing_lag_limit_s=2.,selection='Five fixed fresh layouts; one episode per arm; no replacements or outcome filtering',
        failure_policy='Stop new batches on infrastructure failure; retain all attempts; no automatic retries or budget extension',
        source_sha256=read(root/'sources.json'),package_versions=package_versions(),model_client_version=client_version(),
        development_evidence={str(development/'interface_audit.json'):digest(development/'interface_audit.json'),
                              str(development.parent/'physical_final/summary.json'):digest(development.parent/'physical_final/summary.json')},
        hardware_at_registration=hardware())
    write(root/'protocol.json',protocol);(root/'protocol.sha256').write_text(digest(root/'protocol.json')+'\n')
    write(root/'progress.json',dict(status='registered',completed=0,total=10))
    return protocol


def check(root):
    require(digest(root/'protocol.json')==(root/'protocol.sha256').read_text().strip(),'Protocol changed')
    protocol=read(root/'protocol.json');verify_snapshot(root)
    require(protocol['seeds']==list(SEEDS) and protocol['slots']==slots() and protocol['budget_s']==200.,'Study settings changed')
    require(digest(root/'skill_prior.py')==protocol['skill_sha256'],'Shared prior changed')
    require(package_versions()==protocol['package_versions'] and client_version()==protocol['model_client_version'],'Runtime version changed')
    return protocol


def summarize(root):
    protocol=check(root);checked=audit(root,[s['path'] for s in protocol['slots']])
    records=checked['episodes'];arms={}
    for arm in ARMS:
        subset=[r for r in records if r['arm']==arm]
        arms[arm]=dict(successes=sum(r['success'] for r in subset),episodes=5,
            mean_capped_seconds=mean(r['capped_seconds'] for r in subset),
            usage={k:sum(r['usage'][k] for r in subset) for k in subset[0]['usage']},
            **{k:sum(r[k] for r in subset) for k in ('images','completed_requests','incomplete_requests','model_seconds',
                'incomplete_model_seconds','depth_queries','camera_acquisitions','operations','action_seconds','feedback_seconds')})
    report=dict(status='completed',arms=arms,records=records,
                timing_violations=[r['episode'] for r in records if r['max_simulation_lag_seconds']>2.],
                interpretation='Five paired exploratory layouts; all attempts included. Unknown incomplete token usage is not zero.')
    write(root/'summary.json',report)
    return report


def worker(root,index):
    protocol=check(root);slot=protocol['slots'][index]
    result=run(root/slot['path'],slot['seed'],slot['arm'],root/'skill_prior.py',kind='reserved')
    check(root)
    if result['status']=='infrastructure_error' or result.get('archive_error'):
        raise RuntimeError('Episode infrastructure failure; preserve attempt')
    print(slot,result['status'],flush=True)


def execute(root):
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=check(root)
        require(not (root/'episodes').exists() and not (root/'STOP.json').exists(),'Existing attempts cannot be rerun')
        started=time.time();completed=0
        write(root/'lifecycle.json',dict(status='running',pid=os.getpid(),started_at=started))
        def launch(index):
            with (root/f'worker_{index:02d}.log').open('w') as log:
                return subprocess.call([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),
                    '--root',str(root),'--phase','episode','--slot',str(index)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl'))
        try:
            for offset in range(0,10,4):
                check(root);sample=hardware();write(root/f'hardware_batch_{offset:02d}.json',sample)
                indices=list(range(offset,min(offset+4,10)))
                write(root/'progress.json',dict(status='running',pid=os.getpid(),completed=completed,total=10,active_slots=indices))
                with ThreadPoolExecutor(max_workers=len(indices)) as pool:codes=list(pool.map(launch,indices))
                completed=sum((root/s['path']/'result.json').exists() for s in protocol['slots'])
                require(not any(codes),'Worker failed; no new batch dispatched')
            summarize(root)
            write(root/'progress.json',dict(status='completed',completed=10,total=10,wall_seconds=time.time()-started))
            write(root/'lifecycle.json',dict(status='completed',pid=os.getpid(),wall_seconds=time.time()-started))
        except Exception as error:
            write(root/'STOP.json',dict(error=str(error),completed=completed,time=time.time()))
            write(root/'progress.json',dict(status='stopped',completed=completed,total=10))
            write(root/'lifecycle.json',dict(status='stopped',pid=os.getpid(),error=str(error)))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','check','run','episode','summarize'),required=True)
    parser.add_argument('--development',type=Path,default=ROOT/'outputs/online_ring_dev/live_01')
    parser.add_argument('--slot',type=int,choices=range(10))
    args=parser.parse_args();root=args.root.resolve()
    if args.phase=='prepare':prepare(root,args.development.resolve())
    elif args.phase=='check':check(root);print('Registered sources match')
    elif args.phase=='run':execute(root)
    elif args.phase=='summarize':print(summarize(root)['arms'])
    else:
        if args.slot is None:parser.error('--slot required')
        worker(root,args.slot)
