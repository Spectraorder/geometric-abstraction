"""Freeze and run five fresh paired explicit-target episodes with bounded parallelism."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import time
from run_online_bridge_pilot import (
    ROOT,client_version,digest,package_versions,source_paths,verify_sources,write,
)
from build_online_bridge_geometry import online_library
from run_online_bound_development import DEFAULT_BUDGET_S, _run_episode
from audit_online_bound_development import audit

SEEDS=tuple(range(2028104001,2028104006))
ARMS=('basic','symbolic')


def slots(seeds=SEEDS):
    return [dict(seed=seed,arm=arm,path=f'episodes/{seed}/{arm}')
            for i,seed in enumerate(seeds) for arm in (ARMS if i%2==0 else ARMS[::-1])]


def prepare(root, *, seeds=SEEDS, compact_text=False):
    if len(seeds)!=5 or len(set(seeds))!=5:
        raise ValueError('Exactly five unique seeds are required')
    development=ROOT/'outputs/online_bound_dev'
    physical=development/'physical_01/result.json'
    paired=development/'pairs_01/interface_audit.json'
    if not json.loads(physical.read_text())['success'] or json.loads(paired.read_text())['status']!='passed':
        raise ValueError('Physical and live interface checks must pass before registration')
    for arm in ARMS:
        result=json.loads((development/'pairs_01'/arm/'result.json').read_text())
        if result['model_requests_completed']<1 or result['depth_queries']<1 or result['incomplete_requests']:
            raise ValueError('Live schema, sensor access and complete usage must be exercised in both arms')
    compact_evidence=None
    if compact_text:
        compact_evidence=ROOT/'outputs/online_compact_text_dev/live_01/compact_audit.json'
        checked=json.loads(compact_evidence.read_text())
        if checked['status']!='passed' or any(not r['success'] or r['incomplete_requests']
                                             for r in checked['variants'].values()):
            raise ValueError('Complete live compact-text validation is required')
        for variant in ('raw','compact'):
            episode=compact_evidence.parent/variant
            for name,expected in json.loads((episode/'sources.json').read_text()).items():
                location=json.loads((episode/'source_locations.json').read_text())[name]
                if digest(ROOT/location)!=expected:
                    raise ValueError('Development-validated runtime changed: '+name)
    root.mkdir(parents=True,exist_ok=False)
    hashes={}
    for source in source_paths():
        relative=source.relative_to(ROOT);target=root/'sources'/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        hashes[str(relative)]=digest(source)
    geometry=online_library();(root/'geometry.py').write_text(geometry)
    protocol=dict(name='online_compact_explicit_target_200' if compact_text else 'online_explicit_target_bridge_tower',registered_at=time.time(),
        seeds=seeds,arms=ARMS,slots=slots(seeds),repetitions=1,budget_s=DEFAULT_BUDGET_S,concurrency=4,
        compact_text=compact_text,
        serialization='Lossless shared reference metadata and one-time explanation for Full; Base unchanged' if compact_text else 'Raw reference records',
        scenario='Ordinary unassembled five-object bridge with two cubes stacked at the beam center',
        intervention=None,one_reset_per_episode=True,physics_during_model_wait=True,
        routing='One persistent Astra medium context per arm/episode; one decision per placement or query',
        actor_responsibility='Choose construction origin/order and bind numeric world surface targets; '
                             'shared actuator does not infer or replace task destinations',
        shared_access='Initial side/wrist RGB, calibration, metric pixel queries (up to 16 per query), '
                      'additional observations on request, identical adapted skills and actuator',
        feedback='Base: current side RGB after physical actions and qualitative feedback. '
                 'Full: typed current object references, initial and requested images. '
                 'Both receive query results without recapturing the queried image.',
        success='Complete private support graph and five-second stability with at most 3 mm drift; '
                'robot contact exclusion includes unnamed collision geometry by body ancestry',
        clock='Starts immediately after one reset, before sensor/skill initialization. '
              'Includes model requests, sensor queries, actions and settling, with 1:1 physics pacing.',
        max_timing_lag_for_realtime_claim_s=2.,
        parallelism='Balanced batches of two complete pairs, two pairs, then one pair; one BLAS/OpenMP thread per worker',
        failure_policy='No automatic retries, replacement seeds or budget extension. '
                       'Keep every attempt; infrastructure errors stop new batches after current workers finish.',
        selection='Five fixed fresh seeds with no outcome screening; prior development used 2028100101–2028100103',
        primary_metrics=['successes / 5','mean capped completion seconds including failures'],
        secondary_metrics=['all paired outcomes/time differences','input/output/cached tokens',
                           'unknown usage of incomplete requests','image attachments','metric queries',
                           'request wall time','camera acquisitions','feedback/action time','campaign wall time'],
        interpretation='Exploratory five-pair study of this explicit-target interface. No assumed favorable outcome; '
                       'do not combine with node-only studies or attribute all effects to model latency.',
        source_sha256=hashes,geometry_sha256=hashlib.sha256(geometry.encode()).hexdigest(),
        package_versions=package_versions(),model_client_version=client_version(),
        development_evidence={str(p.relative_to(ROOT)):digest(p) for p in (physical,paired)})
    if compact_evidence is not None:
        protocol['development_evidence'][str(compact_evidence.relative_to(ROOT))]=digest(compact_evidence)
    write(root/'protocol.json',protocol);(root/'protocol.sha256').write_text(digest(root/'protocol.json')+'\n')
    write(root/'progress.json',dict(status='registered',completed=0,total=10))


def summarize(root):
    protocol=verify_sources(root)
    report=audit(root,[slot['path'] for slot in protocol['slots']])
    if protocol.get('compact_text'):
        from audit_online_compact_prompts import audit_prompts
        audit_prompts(root,protocol['slots'])
    records=report['episodes'];arms={};pairs=[]
    if len(records)!=10 or {(r['seed'],r['arm']) for r in records}!={(s['seed'],s['arm']) for s in protocol['slots']}:
        raise ValueError('Registered slots missing or duplicated')
    violations=[]
    for row in records:
        result=json.loads((root/row['episode']/'result.json').read_text())
        if (result['kind']!='explicit_target_reserved' or result['scripted'] or result['budget_s']!=protocol['budget_s']
                or result['intervention'] is not None):raise ValueError('Episode settings mismatch')
        expected='shared_reference_metadata_v1' if protocol.get('compact_text') else 'reference_records'
        if result.get('geometry_serialization','reference_records')!=expected:
            raise ValueError('Episode serialization mismatch')
        if row['max_simulation_lag_seconds']>protocol['max_timing_lag_for_realtime_claim_s']:
            violations.append(row['episode']+': excessive timing lag')
        row.update({key:result[key] for key in ('camera_acquisitions','action_seconds','feedback_seconds','depth_query_seconds')})
    for arm in ARMS:
        selected=[row for row in records if row['arm']==arm]
        arms[arm]=dict(successes=sum(r['success'] for r in selected),episodes=5,
            mean_capped_seconds=mean(r['capped_seconds'] for r in selected),
            usage={key:sum(r['usage'][key] for r in selected) for key in selected[0]['usage']},
            usage_complete=all(r['incomplete_requests']==0 for r in selected),
            **{key:sum(r[key] for r in selected) for key in ('completed_requests','incomplete_requests',
              'images','model_seconds','target_bindings','depth_queries','camera_acquisitions',
              'feedback_seconds','action_seconds','depth_query_seconds','invalid_operations')})
    for seed in protocol['seeds']:
        base,full=[next(r for r in records if r['seed']==seed and r['arm']==arm) for arm in ARMS]
        pairs.append(dict(seed=seed,base_success=base['success'],full_success=full['success'],
                          base_seconds=base['capped_seconds'],full_seconds=full['capped_seconds'],
                          full_minus_base_seconds=full['capped_seconds']-base['capped_seconds']))
    result=dict(arms=arms,pairs=pairs,records=records,timing_violations=violations,
                interpretation='Five exploratory paired layouts. All failures retained. '
                               'Incomplete request usage remains unknown, not zero.')
    write(root/'summary.json',result)
    return result


def worker(root,index):
    protocol=verify_sources(root);slot=protocol['slots'][index]
    result=_run_episode(root/slot['path'],slot['seed'],slot['arm'],protocol['budget_s'],
                        kind='explicit_target_reserved',compact_text=protocol.get('compact_text',False))
    verify_sources(root)
    if result['status']=='infrastructure_error' or result.get('archive_error'):
        raise RuntimeError('Worker failed; preserve artifacts and stop new batches')
    print(json.dumps(dict(slot,status=result['status'],success=result['success'])),flush=True)


def execute(root):
    import fcntl
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=verify_sources(root)
        if (root/'episodes').exists() or (root/'STOP.json').exists():
            raise FileExistsError('Existing attempts cannot be resumed or overwritten automatically')
        started=time.time();completed=0
        def launch(index):
            environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl')
            with (root/f'worker_{index:02d}.log').open('w') as log:
                return subprocess.call([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),
                    '--root',str(root),'--phase','episode','--slot',str(index)],
                    cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        try:
            for offset in range(0,len(protocol['slots']),protocol['concurrency']):
                verify_sources(root)
                indices=list(range(offset,min(offset+protocol['concurrency'],len(protocol['slots']))))
                write(root/'progress.json',dict(status='running',completed=completed,total=10,
                                               active_slots=indices,started_at=started))
                with ThreadPoolExecutor(max_workers=len(indices)) as pool:codes=list(pool.map(launch,indices))
                completed=sum((root/s['path']/'result.json').exists() for s in protocol['slots'])
                if any(codes):raise RuntimeError('A worker failed; no further batch dispatched')
            summarize(root)
            write(root/'progress.json',dict(status='completed',completed=completed,total=10,
                                           wall_seconds=time.time()-started))
        except Exception as error:
            write(root/'STOP.json',dict(error=str(error),completed=completed,time=time.time()))
            write(root/'progress.json',dict(status='stopped',completed=completed,total=10))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','check','run','episode','summarize'),required=True)
    parser.add_argument('--slot',type=int,choices=range(10));args=parser.parse_args();root=args.root.resolve()
    if args.phase=='prepare':prepare(root)
    elif args.phase=='check':verify_sources(root);print('Registered sources match')
    elif args.phase=='run':execute(root)
    elif args.phase=='summarize':summarize(root)
    else:
        if args.slot is None:parser.error('--slot is required for episode')
        worker(root,args.slot)
