"""Preserve completed extension trials and repair only the legacy log path adapter."""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil
import time
from unittest.mock import patch
import extend_three_cube_contract_evaluation as extension

read=extension.read
write=extension.write_json
base=extension.base
require=extension.require
REPAIR='path_repair'


def corrected_run():
    source=Path(extension.__file__).read_text()
    old="pending=[row for row in r['records'] if not row['reused']]"
    new="pending=[row for row in r['records'] if not row['reused'] and not (root/row['trial']/'result.json').exists()]"
    require(source.count(old)==1,'Expected one pending trial selector')
    source=source.replace(old,new)
    old='completed=0;started=time.time();samples=[]'
    require(source.count(old)==1,'Expected one completion counter')
    source=source.replace(old,"completed=r['new_slots']-len(pending);started=time.time();samples=[]")
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run']
    require(len(functions)==1,'Expected one run function')
    namespace=dict(vars(extension))
    exec(compile(ast.Module(body=functions,type_ignores=[]),extension.__file__,'exec'),namespace)
    return namespace['run'],source


def register(root):
    out=root/extension.DIRECTORY;dest=out/REPAIR
    require(not dest.exists(),'Repair already registered')
    r=extension.check(root)
    stop=read(out/'STOP.json')
    require('is not in the subpath' in stop['error'],'Unexpected stop reason')
    completed={};initial={}
    for row in r['records']:
        if row['reused']:continue
        trial=root/row['trial']
        if not trial.exists():continue
        require((trial/'result.json').exists(),'Incomplete rollout retained; cannot resume')
        fingerprint=base.digest(trial/'initial.png') if not row['validation_error'] else None
        if fingerprint:initial.setdefault(row['seed'],fingerprint)
        manifest=read(out/'manifests'/f"{row['replica']}.json")
        extension.curves.audit_trial(root,row,manifest,initial.get(row['seed']))
        completed[row['trial']+'/result.json']=base.digest(trial/'result.json')
    require(len(completed)==24,'Expected the drained 24-rollout first batch')
    dest.mkdir()
    for name in ('STOP.json','progress.json','campaign.log','launch.json'):
        if (out/name).exists():shutil.copyfile(out/name,dest/(name+'.before_resume'))
    shutil.copyfile(Path(__file__),dest/Path(__file__).name)
    _,source=corrected_run();(dest/'corrected_runner.py').write_text(source)
    aliases={}
    for row in read(root/'protocol.json')['replicas']:
        name=row['directory'];alias=root/name/'evaluation_100_rollouts';target=out/'rollouts'/name
        require(not alias.exists() and not alias.is_symlink(),'Unexpected rollout alias')
        target.mkdir(parents=True,exist_ok=True);alias.symlink_to(target,target_is_directory=True)
        aliases[str(alias.relative_to(root))]=str(target.relative_to(root))
    write(dest/'registration.json',{'scope':'Repair legacy relative log path; reuse verified completed trials. No simulator, policy, seed or budget changes.',
        'time':time.time(),'adapter_sha256':base.digest(Path(__file__)),
        'corrected_runner_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'preserved_results':completed,'aliases':aliases,'stop_sha256':base.digest(out/'STOP.json')})
    # The exact original STOP receipt remains archived, with all completed outcomes.
    (out/'STOP.json').unlink()
    return len(completed)


def run(root):
    out=root/extension.DIRECTORY;dest=out/REPAIR;r=read(dest/'registration.json')
    require(base.digest(Path(__file__))==r['adapter_sha256']==base.digest(dest/Path(__file__).name),'Repair adapter changed')
    require(base.digest(dest/'STOP.json.before_resume')==r['stop_sha256'],'Original stop changed')
    function,source=corrected_run()
    require(hashlib.sha256(source.encode()).hexdigest()==r['corrected_runner_sha256'],'Correction changed')
    for relative,sha in r['preserved_results'].items():require(base.digest(root/relative)==sha,'Completed result changed')
    for alias,target in r['aliases'].items():require((root/alias).is_symlink() and (root/alias).resolve()==root/target,'Alias changed')
    original=extension.runner.variant.execute
    def execute(replica,arm,candidate,directory,seeds,manifest,error=None):
        relative=directory.relative_to(out/'rollouts'/replica.name)
        return original(replica,arm,candidate,replica/'evaluation_100_rollouts'/relative,seeds,manifest,error)
    with patch.object(extension.runner.variant,'execute',execute):return function(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=('register','run'),required=True);a=p.parse_args()
    with extension.runner.installed():
        root=a.root.resolve()
        if a.phase=='register':
            with extension.runner.study.exclusive(root/'pipeline.lock'),extension.runner.study.exclusive(root/'coordinator.lock'):print('Verified completed trials:',register(root))
        else:print(run(root)['mean_curves'])
