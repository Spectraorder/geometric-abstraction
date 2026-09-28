"""Post-hoc 100-layout evaluation of the frozen contract-only stacking policies."""
import argparse
import copy
from pathlib import Path
import os
import shutil
import statistics
import time

import evaluate_three_cube_contract as previous
from experiment_parallel import bounded_map, lease, check_stop, stop, write_json, audit_leases
from run_mixed_scene_experiment import color_map
from summarize_mixed_scene_learning import require, paired

runner=previous.runner
base=previous.base
read=previous.read
curves=previous.curves
DIRECTORY='evaluation_100'
SCOPE='Post-hoc extension after inspecting the original 20-layout results; frozen policies, no retraining or test-based selection.'


def expanded_manifest(manifest,seeds):
    result=copy.deepcopy(manifest)
    result['heldout_seeds']=list(seeds)
    result['color_maps'].update({str(seed):color_map(seed) for seed in seeds})
    return result


def matrix(old_records,new_seeds):
    records=[dict(row,reused=True) for row in old_records]
    policies={}
    for row in old_records:policies.setdefault((row['replica'],row['arm'],row['revision']),row)
    require(len(policies)==40 and len(old_records)==800, 'Expected all 40 frozen policies and 800 previous trials')
    for (name,arm,revision),row in sorted(policies.items()):
        for seed in new_seeds:
            records.append(dict(row,seed=seed,reused=False,
                trial=f'{DIRECTORY}/rollouts/{name}/{arm}/revision_{revision:03d}/seed_{seed}'))
    return records


def prepare(root):
    output=root/DIRECTORY
    require(not output.exists(),'Use a new extension directory')
    old=previous.check(root)
    require(read(root/'learning_curves/summary.json')['status']=='completed','Previous evaluation must be complete')
    runner.study.check_gate(root,runner.check(root))
    old_seeds=old['heldout_seeds']
    new_seeds=list(range(max(old_seeds)+1,max(old_seeds)+81))
    inspected=set()
    for pattern in ('**/manifest.json','**/registration.json','**/protocol.json'):
        for path in (runner.ROOT/'outputs').glob(pattern):
            record=read(path)
            for key in ('train_seeds','heldout_seeds','reserved_seeds','test_seeds'):
                inspected.update(record.get(key,[]))
    require(not set(new_seeds)&inspected,'Additional layouts overlap a prior registration')
    seeds=old_seeds+new_seeds
    require(len(set(seeds))==100,'Expected 100 unique shared layouts')
    records=matrix(old['records'],new_seeds)
    require(len(records)==4000,'Incorrect extension budget')
    output.mkdir()
    preserved={name:base.digest(root/name) for name in ('protocol.json','all_candidates_frozen.json',
        'evaluation_registration.json','learning_curves/registration.json','learning_curves/summary.json')}
    preserved.update(old['preserved_files'])
    preserved.update(read(root/'learning_curves/summary.json')['result_hashes'])
    for name in sorted({row['replica'] for row in records}):
        path=output/'manifests'/f'{name}.json';path.parent.mkdir(exist_ok=True)
        write_json(path,expanded_manifest(read(root/name/'manifest.json'),seeds))
        preserved[str(path.relative_to(root))]=base.digest(path)
    shutil.copyfile(Path(__file__),output/'runner_snapshot.py')
    registration={'schema':'contract_cube_100_extension_v1','registered_at':time.time(),
        'scope':SCOPE,'heldout_seeds':seeds,'new_seeds':new_seeds,'slots':4000,
        'new_slots':3200,'reused_slots':800,'model_calls':0,'max_workers':24,
        'minimum_free_disk_bytes':15*1024**3,'pilot_rollouts':48,
        'selection':'Reuse training-retained choices; evaluate every saved R0-R3 policy.',
        'records':records,'preserved_files':preserved,'runner_sha256':base.digest(Path(__file__)),
        'hardware_at_registration':curves.hardware()}
    write_json(output/'registration.json',registration)
    write_json(root/'evaluation_100_receipt.json',{'registration_sha256':base.digest(output/'registration.json')})
    write_json(output/'progress.json',{'status':'registered','new_completed':0,'new_total':3200,'reused':800,'total':4000})
    check(root)
    return registration


def check(root):
    previous.check(root)
    output=root/DIRECTORY;r=read(output/'registration.json')
    require(base.digest(output/'registration.json')==read(root/'evaluation_100_receipt.json')['registration_sha256'],'Extension registration changed')
    require(r['schema']=='contract_cube_100_extension_v1' and r['slots']==4000 and r['new_slots']==3200,'Wrong extension budget')
    require(base.digest(Path(__file__))==r['runner_sha256']==base.digest(output/'runner_snapshot.py'),'Extension source changed')
    for name,sha in r['preserved_files'].items():require(base.digest(root/name)==sha,'Frozen evidence changed: '+name)
    require(r['records']==matrix(read(root/'learning_curves/registration.json')['records'],r['new_seeds']), 'Evaluation matrix changed')
    return r


def summarize_rows(root,rows,seeds):
    replicas={}
    for name in sorted({key[0] for key in rows}):
        training=read(root/name/'training_summary.json')
        accounting=read(root/'learning_curves/training'/name/'resource_accounting.json')
        replicas[name]={}
        for arm in ('basic','symbolic'):
            raw=[statistics.mean(r['success'] for r in rows[name,arm,rev] if r['seed'] in seeds) for rev in range(4)]
            tokens=[0]
            for turn in accounting['arms'][arm]['turns']:tokens.append(tokens[-1]+turn['usage']['total_tokens'])
            replicas[name][arm]=curves.curves(training['curves'][arm],raw,tokens,[.8,.6,.7])
        replicas[name]['paired_by_raw_revision']=[paired(
            [r for r in rows[name,'basic',rev] if r['seed'] in seeds],
            [r for r in rows[name,'symbolic',rev] if r['seed'] in seeds]) for rev in range(4)]
    mean={arm:{key:[statistics.mean(row[arm][key][i] for row in replicas.values()) for i in range(4)]
               for key in ('raw','retained','cumulative_generation_tokens')} for arm in ('basic','symbolic')}
    return {'replicas':replicas,'mean_curves':mean,'layouts':len(seeds)}


def audit(root):
    r=check(root);output=root/DIRECTORY;rows={};initial={};hashes={}
    manifests={name:read(output/'manifests'/f'{name}.json') for name in {row['replica'] for row in r['records']}}
    for row in r['records']:
        trial=root/row['trial']
        fingerprint=base.digest(trial/'initial.png') if not row['validation_error'] else None
        if fingerprint is not None:initial.setdefault(row['seed'],fingerprint)
        result=curves.audit_trial(root,row,manifests[row['replica']],initial.get(row['seed']))
        if not row['reused']:require((trial/'result.json').stat().st_mtime>=r['registered_at'],'New trial predates registration')
        rows.setdefault((row['replica'],row['arm'],row['revision']),[]).append(result)
        hashes[row['trial']+'/result.json']=base.digest(trial/'result.json')
    report={'status':'completed','scope':SCOPE,'slots':4000,'new_slots':3200,'reused_slots':800,
        **summarize_rows(root,rows,set(r['heldout_seeds'])),
        'additional_80':summarize_rows(root,rows,set(r['new_seeds'])),
        'result_hashes':hashes,'registration_sha256':base.digest(output/'registration.json'),
        'scheduler':audit_leases(output,{'rollout':24})}
    write_json(output/'summary.json',report);plot(output,report)
    return report


def plot(output,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for axis,kind,cost in zip(axes,('raw','retained','retained'),(False,False,True)):
        for arm,label in (('basic','Base'),('symbolic','Full')):
            mean=report['mean_curves'][arm]
            x=[v/1000 for v in mean['cumulative_generation_tokens']] if cost else range(4)
            axis.plot(x,mean[kind],'-o',label=label)
        axis.set(ylim=(0,1),ylabel='Test success rate',xlabel='Cumulative tokens (thousands)' if cost else 'Revision',title=kind.capitalize())
        axis.grid(alpha=.2);axis.legend()
    fig.suptitle('Contract-only stacking: 5 paired chains, 100 shared layouts; post-hoc extension')
    fig.savefig(output/'learning_curves.pdf');fig.savefig(output/'learning_curves.png',dpi=180);plt.close(fig)


def run(root):
    output=root/DIRECTORY
    with runner.study.exclusive(root/'pipeline.lock'),runner.study.exclusive(root/'coordinator.lock'):
        r=check(root);check_stop(output)
        require(not (output/'summary.json').exists(),'Extension already complete')
        pending=[row for row in r['records'] if not row['reused']]
        require(not any((root/row['trial']).exists() for row in pending),'Existing attempts retained; inspect before resuming')
        pending.sort(key=lambda row:(row['revision']==0,row['seed'],row['revision'],row['replica'],row['arm']))
        completed=0;started=time.time();samples=[]
        manifests={name:read(output/'manifests'/f'{name}.json') for name in {row['replica'] for row in pending}}
        def execute(row):
            with lease(output,'rollout',24,row['trial']):
                require(shutil.disk_usage(root).free>=r['minimum_free_disk_bytes'],'Disk reserve reached')
                return runner.variant.execute(root/row['replica'],row['arm'],root/row['candidate'],
                    (root/row['trial']).parent,[row['seed']],manifests[row['replica']],row['validation_error'])
        def batch(jobs,workers):
            nonlocal completed
            for _,_ in bounded_map(execute,jobs,workers,on_error=lambda e:stop(output,e)):
                completed+=1
                write_json(output/'progress.json',{'status':'evaluating','pid':os.getpid(),'new_completed':completed,
                    'new_total':3200,'reused':800,'total':4000,'workers':workers,'elapsed_seconds':time.time()-started,'time':time.time()})
        write_json(root/'progress.json',{'status':'evaluation_100','details':DIRECTORY+'/progress.json','training_slots':200,'test_slots':4000})
        try:
            with curves.telemetry(output,samples),runner.variant.configured():
                batch(pending[:48],24)
                workers=curves.pilot_workers(samples,24)
                batch(pending[48:],workers)
            report=audit(root)
            write_json(output/'progress.json',{'status':'completed','new_completed':3200,'new_total':3200,'reused':800,'total':4000,'elapsed_seconds':time.time()-started})
            write_json(root/'progress.json',{'status':'completed','training_slots':200,'test_slots':4000,'details':DIRECTORY+'/summary.json'})
            write_json(root/'lifecycle.json',{'status':'completed','phase':'evaluation_100','pid':os.getpid(),'time':time.time()})
            return report
        except BaseException as error:
            stop(output,error);write_json(output/'progress.json',{'status':'failed','error':str(error),'new_completed':completed,'new_total':3200})
            write_json(root/'lifecycle.json',{'status':'failed','phase':'evaluation_100','error':str(error),'time':time.time()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=('prepare','run','audit','check'),required=True);a=p.parse_args()
    with runner.installed():
        root=a.root.resolve()
        if a.phase=='prepare':
            with runner.study.exclusive(root/'pipeline.lock'),runner.study.exclusive(root/'coordinator.lock'):r=prepare(root)
        else:r={'run':run,'audit':audit,'check':check}[a.phase](root)
        print({k:r[k] for k in ('status','slots','new_slots','reused_slots','mean_curves') if k in r})
