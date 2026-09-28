"""Register and run the authorized five-pair tower-relocation comparison."""
import argparse
import json
from pathlib import Path
import shutil
import time
from statistics import mean
from online_ring_evidence import snapshot, verify_snapshot, read, require, package_versions, client_version
from online_ring_skills import ROOT, digest
from run_online_ring_episode import write
from run_online_relocation_episode import run
from online_relocation_runtime import clone, relocation_library
from audit_online_relocation import audit
from evaluate_geometry_learning_curves import hardware
import run_online_ring_study as scheduler

SEEDS=tuple(range(2034101001,2034101006))
ARMS=('basic','symbolic')


def slots(seeds=SEEDS,prefix='episodes'):
    return [dict(seed=seed,arm=arm,path=f'{prefix}/{seed}/{arm}') for i,seed in enumerate(seeds)
            for arm in (ARMS if i%2==0 else ARMS[::-1])]


def prepare(root,development):
    require(not root.exists(),'Use a new root; preserve earlier attempts')
    checked=audit(development,slots((1,2)))
    require(all(r['completed_requests']>=2 for r in checked['episodes']),'Both arms must exercise online decisions')
    require(any(r['depth_queries'] for r in checked['episodes'] if r['arm']=='basic'),'Base depth access not exercised')
    physical=read(development.parent/'physical_final/summary.json')
    require(physical['status']=='passed' and physical['seeds']==[1,2],'Missing physical validation')
    for item in checked['episodes']:verify_snapshot(development/item['episode'])
    for seed in (1,2):verify_snapshot(development.parent/f'physical_final/seed_{seed}')
    require(shutil.disk_usage(root.parent).free>=15*1024**3,'Insufficient disk reserve')
    scanned=[]
    for directory in (ROOT/'outputs').iterdir():
        if not directory.is_dir():continue
        for pattern in ('**/protocol.json','**/registration.json','**/manifest.json'):
            for path in directory.resolve().glob(pattern):
                value=read(path)
                for key in ('seeds','train_seeds','heldout_seeds','reserved_seeds','test_seeds'):
                    require(not set(SEEDS).intersection(value.get(key,[])),'Seed overlap: '+str(path))
                scanned.append(str(path))
    root.mkdir(parents=True);snapshot(root)
    (root/'geometry.py').write_text(relocation_library())
    protocol=dict(name='online_tower_relocation_200_v1',registered_at=time.time(),seeds=list(SEEDS),slots=slots(),
        arms=list(ARMS),budget_s=200.,concurrency=4,model='gpt-6-astra',reasoning='medium',transport='HTTP-only',
        resets_per_episode=1,physics_during_model_wait=True,scenario='Three-cube initial column and separate destination cube',
        intervention=None,model_calls_during_actions=False,
        shared_skills='Existing bridge/stack explicit-target pick/place; grasp fraction 0.70; no relocation action sequence supplied',
        size_assumption='Four equal cubes; dimensions measured from RGB-D of table-supported destination; current poses observed separately',
        shared_access='Initial side/wrist images, images on request, 16-pixel calibrated depth queries, qualitative feedback, same skills and action schema',
        contrast='Base side RGB after actions; Full current center/top/bottom references encoded with shared metadata and initial/requested images',
        success='Destination-middle-upper support column, source/destination within 5 mm of initial positions during five-second stability hold, no robot contact, at most 3 mm drift',
        clock='After sole reset, before sensor/skill initialization; all observation, model waiting, motion and settling included',
        timing_lag_limit_s=2.,primary_metrics=['successes / 5','mean capped completion seconds'],
        secondary_metrics=['every paired outcome','tokens with unknown incomplete usage separate','model request time','image attachments','camera/depth queries','physical actions','timing lag'],
        selection='Five new fixed layouts; no replacements, repetitions or outcome screening',
        failure_policy='Stop new batches on infrastructure error; preserve all attempts; no automatic retries or budget extension',
        source_sha256=read(root/'sources.json'),geometry_sha256=digest(root/'geometry.py'),
        package_versions=package_versions(),model_client_version=client_version(),hardware_at_registration=hardware(),
        development_evidence={str(development/'relocation_audit.json'):digest(development/'relocation_audit.json'),
                              str(development.parent/'physical_final/summary.json'):digest(development.parent/'physical_final/summary.json')})
    write(root/'seed_overlap_audit.json',dict(status='passed',registrations=len(scanned),seeds=list(SEEDS)))
    write(root/'protocol.json',protocol);(root/'protocol.sha256').write_text(digest(root/'protocol.json')+'\n')
    write(root/'progress.json',dict(status='registered',completed=0,total=10))
    return protocol


def check(root):
    require(digest(root/'protocol.json')==(root/'protocol.sha256').read_text().strip(),'Protocol changed')
    p=read(root/'protocol.json');verify_snapshot(root)
    require(p['seeds']==list(SEEDS) and p['slots']==slots() and p['budget_s']==200.,'Settings changed')
    require(digest(root/'geometry.py')==p['geometry_sha256'],'Geometry archive changed')
    import hashlib
    require(hashlib.sha256(relocation_library().encode()).hexdigest()==p['geometry_sha256'],'Generated geometry changed')
    require(package_versions()==p['package_versions'] and client_version()==p['model_client_version'],'Runtime changed')
    return p


def summarize(root):
    protocol=check(root);report=audit(root,protocol['slots']);records=report['episodes'];arms={}
    require(len(records)==10,'Missing registered episodes')
    for arm in ARMS:
        subset=[r for r in records if r['arm']==arm]
        arms[arm]=dict(successes=sum(r['success'] for r in subset),episodes=len(subset),
            mean_capped_seconds=mean(r['capped_seconds'] for r in subset),
            usage={k:sum(r['usage'][k] for r in subset) for k in subset[0]['usage']},
            **{k:sum(r[k] for r in subset) for k in ('images','completed_requests','incomplete_requests','model_seconds',
                'incomplete_model_seconds','depth_queries','camera_acquisitions','target_bindings','action_seconds','feedback_seconds')})
    value=dict(status='completed',arms=arms,records=records,
        timing_violations=[r['episode'] for r in records if r['max_simulation_lag_seconds']>2.],
        interpretation='Five exploratory paired layouts; all failures retained; unknown incomplete usage is not zero.')
    write(root/'summary.json',value);return value


def worker(root,index):
    protocol=check(root);slot=protocol['slots'][index]
    result=run(root/slot['path'],slot['seed'],slot['arm'],kind='relocation_reserved')
    check(root)
    require(result['status']!='infrastructure_error' and not result.get('archive_error'),'Worker infrastructure error')
    print(slot,result['status'],flush=True)


execute=clone(scheduler.execute,__file__=__file__,check=check,summarize=summarize)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','check','run','episode','summarize','audit-development'),required=True)
    parser.add_argument('--development',type=Path,default=ROOT/'outputs/online_relocation_dev/live_01')
    parser.add_argument('--slot',type=int,choices=range(10))
    args=parser.parse_args();root=args.root.resolve()
    if args.phase=='prepare':prepare(root,args.development.resolve())
    elif args.phase=='check':check(root);print('Registered sources match')
    elif args.phase=='run':execute(root)
    elif args.phase=='summarize':print(summarize(root)['arms'])
    elif args.phase=='audit-development':print(audit(root,slots((1,2)))['status'])
    else:
        if args.slot is None:parser.error('--slot required')
        worker(root,args.slot)
