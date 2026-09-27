"""Budgeted, resumable LLM policy search for one- or three-pair fastening."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_vision_trial import validate_source
from aspire_so100.fasteners import SCENE_VERSION
from aspire_so100.fasten_screws import TRACKING_CONTROLLER, CONTROLLERS

TRAIN = [5101,5102,5103]
HELDOUT = [5201,5202,5203]


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def reference_context(root,state,sources):
    """Snapshot prior code as generation references, never as scored candidates."""
    references=state.get('references',[])
    if sources and not references:
        if state.get('history'):
            raise ValueError('Reference policies must be supplied when starting a new campaign')
        prepared=[]
        for source in sources:
            source=source.resolve()
            code=source.read_text()
            validate_source(code)
            prepared.append((source,code,hashlib.sha256(code.encode()).hexdigest()))
        folder=root/'reference_policies';folder.mkdir(exist_ok=True)
        for index,(source,code,digest) in enumerate(prepared,1):
            snapshot=folder/f'{index:02d}_{source.name}'
            if snapshot.exists() and snapshot.read_text()!=code:
                raise ValueError('Conflicting reference snapshot; use a fresh campaign root')
            snapshot.write_text(code)
            references.append(dict(source=str(source),snapshot=str(snapshot),sha256=digest))
        state['references']=references
    elif sources:
        supplied=[(str(p.resolve()),hashlib.sha256(p.read_text().encode()).hexdigest()) for p in sources]
        if supplied!=[(r['source'],r['sha256']) for r in references]:
            raise ValueError('Reference policies changed; preserve campaign and use a new root')
    sections=[]
    for index,reference in enumerate(references,1):
        code=Path(reference['snapshot']).read_text()
        if hashlib.sha256(code.encode()).hexdigest()!=reference['sha256']:
            raise ValueError('Reference snapshot checksum mismatch')
        sections += [f'Reference policy {index} from {reference["source"]}:',code]
    if sections:
        sections.insert(0,'Prior LLM-generated code for adaptation ONLY. No old scores or successes '
                        'carry over. Generate a complete full-task candidate and test it in this scene.')
    return '\n'.join(sections)


def trial(candidate,folder,seed,timeout,stage='fasten',controller='legacy',grasp_quality='legacy',pair_count=3):
    result_file=folder/'result.json'
    if result_file.exists():
        result=json.loads(result_file.read_text())
        if (result['candidate_sha256']!=hashlib.sha256(candidate.read_bytes()).hexdigest()
                or result['seed']!=seed or result['task']!='fasten_screws'
                or result.get('scene',{}).get('generator')!=SCENE_VERSION
                or result['scene'].get('controller','legacy')!=controller
                or result['scene'].get('learning_stage','fasten')!=stage
                or result['scene'].get('grasp_quality','legacy')!=grasp_quality
                or result['scene'].get('pair_count',3)!=pair_count):
            raise RuntimeError('Existing trial does not match candidate/seed/task/scene version')
        return result
    if folder.exists():
        folder.rename(folder.with_name(folder.name+'.interrupted-'+datetime.now().strftime('%Y%m%d%H%M%S')))
    folder.parent.mkdir(parents=True,exist_ok=True)
    with folder.with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/run_vision_trial.py'),
            '--candidate',str(candidate),'--output',str(folder),'--task','fasten_screws',
            '--layout','sampled','--seed',str(seed),'--timeout',str(timeout),
            '--fasten-pairs',str(pair_count),'--fasten-controller',controller,'--fasten-stage',stage,'--grasp-quality',grasp_quality],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout+180)
    return json.loads(result_file.read_text())


def rank(entry):
    results=entry.get('results',[])
    if entry.get('stage') == 'pickup':
        pickups=[r['metrics']['pickup'] for r in results]
        return (sum(r['success'] for r in results),
                sum(p['lift'] for p in pickups),
                sum(p['sustained_grasp'] for p in pickups),
                sum(p['grasp'] for p in pickups),
                sum(min(p['max_hold_seconds'],2.) for p in pickups),
                sum(min(p['max_lift_m'],.03) for p in pickups),
                -sum(bool(r['policy_error']) for r in results))
    if entry.get('grasp_quality','legacy')!='legacy':
        # Completed fastening remains primary. Stable transport/grasp outrank
        # the old transient touch/lift counts, not actual threading progress.
        qualities=[q for r in results for q in r['metrics']['grasp_quality'].values()]
        return (sum(r['success'] for r in results),sum(r['metrics']['score'] for r in results),
                sum(sum(r['metrics']['milestones'][k] for k in ('engage','rotate','depth','fasten')) for r in results),
                sum(q['stable_transport'] for q in qualities),sum(q['stable_grasp'] for q in qualities),
                sum(sum(r['metrics']['milestones'].values()) for r in results),
                -sum(bool(r['policy_error']) for r in results))
    return (sum(r['success'] for r in results),
            sum(r['metrics']['score'] for r in results),
            sum(sum(r['metrics']['milestones'].values()) for r in results),
            -sum(bool(r['policy_error']) for r in results))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model',default='gpt-6-astra')
    p.add_argument('--stage',choices=('pickup','fasten'),default='fasten')
    p.add_argument('--controller',choices=CONTROLLERS,default=TRACKING_CONTROLLER)
    p.add_argument('--grasp-quality',choices=('legacy','grasp_quality_v1'),default='legacy')
    p.add_argument('--rounds',type=int,default=3,help='Total round cap; increase when resuming')
    p.add_argument('--max-reported-tokens',type=int,default=60000)
    p.add_argument('--llm-timeout',type=int,default=300)
    p.add_argument('--trial-timeout',type=int,default=300)
    p.add_argument('--max-prompt-chars',type=int,default=32000)
    p.add_argument('--reference-policy',type=Path,action='append',default=[],
                   help='Prior policy code for first-round adaptation, not imported scores; repeatable')
    p.add_argument('--pairs',type=int,choices=(1,3),default=3)
    a=p.parse_args()
    if min(a.rounds,a.max_reported_tokens,a.llm_timeout,a.trial_timeout,a.max_prompt_chars)<=0:
        p.error('All limits must be positive')
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    state_path=root/'campaign.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(
        created_at=datetime.now(timezone.utc).isoformat(),task='fasten_screws',model=a.model,
        history=[],reported_tokens=0,model_calls=0,train_seeds=TRAIN,heldout_seeds=HELDOUT,
        scene_version=SCENE_VERSION,stage=a.stage,controller=a.controller,grasp_quality=a.grasp_quality,pair_count=a.pairs)
    if state.get('scene_version') != SCENE_VERSION:
        raise RuntimeError('Campaign uses older fastening geometry/physics. Preserve it and use a new --root; '
                           'old scores cannot be reused with the current scene version.')
    if state.get('stage','fasten')!=a.stage or state.get('controller','legacy')!=a.controller:
        raise RuntimeError('Stage/controller differ from campaign. Use a new --root; do not mix historical scores.')
    if state.get('grasp_quality','legacy')!=a.grasp_quality:
        raise RuntimeError('Grasp-quality evaluator changed. Use a new --root; do not mix ranks.')
    if state.get('pair_count',3)!=a.pairs:
        raise RuntimeError('Pair count differs from campaign. Use a new --root.')
    references=reference_context(root,state,a.reference_policy)
    state.update(status='initializing',limits=vars(a)|{'root':str(root),
                 'reference_policy':[str(p.resolve()) for p in a.reference_policy]})
    write(state_path,state)
    try:
        if not (root/'initial.png').exists():
            from aspire_so100.fasten_screws import FastenScrewsEnv
            from PIL import Image
            env=FastenScrewsEnv(root/'scene_check',scene_seed=TRAIN[0],
                               controller=a.controller,learning_stage=a.stage,grasp_quality=a.grasp_quality,pair_count=a.pairs)
            try:
                env.reset(TRAIN[0])
                Image.fromarray(env.get_overhead_rgbd()['rgb']).save(root/'initial.png')
                write(root/'scene_check'/'metrics.json',asdict(env.evaluate()))
            finally: env.close()
        for index in range(len(state['history'])+1,a.rounds+1):
            if state['reported_tokens']>=a.max_reported_tokens:
                state['status']='reported_token_limit';break
            directory=root/f'round_{index:03d}'
            sections=[f'Active objective: {a.stage}; robot controller: {a.controller}; number of pairs: {a.pairs}.'];images=[root/'initial.png']
            if not state['history'] and references:
                sections += [references]
            if state.get('best'):
                sections+=['Current best policy:',Path(state['best']['candidate']).read_text()]
            if state['history']:
                latest=state['history'][-1]
                sections+=['Latest attempt feedback:',json.dumps(latest.get('feedback',latest.get('error','')))]
                response=Path(latest['candidate']).parent/'response.json'
                if response.exists(): sections+=['Latest hypothesis:',json.loads(response.read_text())['hypothesis']]
                for result in latest.get('results',[])[:1]:
                    images.append(Path(result['directory'])/'outcome.png')
            sections+=['Recent hypothesis outcomes:',json.dumps([
                dict(round=h['round'],rank=h.get('rank'),error=h.get('error')) for h in state['history'][-4:]])]
            context='\n'.join(sections)
            contract=ROOT/'docs'/('FASTEN_PICKUP_PROMPT.txt' if a.stage=='pickup' else 'FASTEN_SINGLE_NUT_PROMPT.txt' if a.pairs==1 else 'FASTEN_SCREWS_PROMPT.txt')
            if a.grasp_quality!='legacy':
                sections.insert(0,(ROOT/'docs/FASTEN_GRASP_QUALITY_PROMPT.txt').read_text())
                context='\n'.join(sections)
            if len(context)+len(contract.read_text())>a.max_prompt_chars:
                raise RuntimeError('Prompt exceeds cap; no model call made')
            context_path=root/f'context_{index:03d}.txt';context_path.write_text(context)
            candidate=directory/'candidate.py'
            state.update(status='proposing',active_round=index);write(state_path,state)
            if not candidate.exists():
                if directory.exists():
                    directory.rename(directory.with_name(directory.name+'.interrupted-'+datetime.now().strftime('%Y%m%d%H%M%S')))
                state['model_calls']+=1;write(state_path,state)
                command=[sys.executable,'-u',str(ROOT/'scripts/propose_vision_policy.py'),
                         '--round-dir',str(directory),'--contract-file',str(contract),
                         '--context-file',str(context_path),'--model',a.model,'--timeout',str(a.llm_timeout)]
                for image in images: command+=['--image',str(image)]
                print(f'Round {index}: asking {a.model}; {len(context)+len(contract.read_text())} prompt chars',flush=True)
                subprocess.run(command,check=True,timeout=a.llm_timeout+60)
            events=[json.loads(line) for line in (directory/'model.jsonl').read_text().splitlines() if line.startswith('{')]
            if any(e.get('item',{}).get('type') in {'command_execution','mcp_tool_call','web_search','file_change'} for e in events):
                raise RuntimeError('Actor tool-isolation audit failed')
            usage=[e['usage'] for e in events if e.get('type')=='turn.completed' and 'usage' in e]
            # Count each completed model call once, including resumptions mid-evaluation.
            if str(index) not in state.setdefault('usage_by_round',{}):
                state['usage_by_round'][str(index)]=usage
                state['reported_tokens']+=sum(u.get('input_tokens',0)+u.get('output_tokens',0) for u in usage)
            state['status']='evaluating';write(state_path,state)
            entry=dict(round=index,candidate=str(candidate),stage=a.stage,grasp_quality=a.grasp_quality,results=[],feedback=[])
            try: validate_source(candidate.read_text())
            except (ValueError,SyntaxError) as exc:
                entry['error']=str(exc)
                state['history'].append(entry);write(state_path,state)
                print(f'Round {index}: validation rejected; 0 trials: {exc}',flush=True)
                continue
            for seed in TRAIN:
                folder=directory/'trials'/f'seed_{seed}'
                result=trial(candidate,folder,seed,a.trial_timeout,a.stage,a.controller,a.grasp_quality,a.pairs)|{'directory':str(folder)}
                entry['results'].append(result)
                feedback=json.loads((folder/'actor_feedback.json').read_text())
                notes=feedback['notes']
                feedback['notes']=notes if len(notes)<=18 else notes[:6]+notes[-12:]
                entry['feedback'].append(feedback)
                write(directory/'evaluation.json',entry)
                metrics=result['metrics']
                if a.grasp_quality!='legacy':
                    quality=metrics['grasp_quality']
                    print(f'Round {index}, seed {seed}: stable grasp '
                          f'{sum(q["stable_grasp"] for q in quality.values())}/{a.pairs}; stable transport '
                          f'{sum(q["stable_transport"] for q in quality.values())}/{a.pairs}; '
                          f'slip episodes {sum(q["slip_events"] for q in quality.values())}',flush=True)
                if a.stage=='pickup':
                    pickup=metrics['pickup']
                    print(f'Round {index}, seed {seed}: pickup={result["success"]}; '
                          f'grasp={pickup["grasp"]}; sustained={pickup["sustained_grasp"]}; '
                          f'lift={pickup["max_lift_m"]*1000:.1f} mm; '
                          f'hold={pickup["hold_seconds"]:.2f} s; error={result["policy_error"]}',flush=True)
                    continue
                print(f'Round {index}, seed {seed}: score {metrics["score"]}/{20 if a.pairs==1 else 100}; '
                      f'fastened {metrics["fastened_pairs"]}/{a.pairs}; stages {metrics["milestones"]}; '
                      f'error={result["policy_error"]}',flush=True)
            entry['rank']=rank(entry)
            state['history'].append(entry)
            if not state.get('best') or rank(entry)>=rank(state['best']):
                state['best']=entry
                shutil.copyfile(candidate,root/'.best_policy.py.tmp')
                (root/'.best_policy.py.tmp').replace(root/'best_policy.py')
                best_trial=max(entry['results'],key=lambda r:rank(dict(stage=a.stage,grasp_quality=a.grasp_quality,results=[r])))
                (root/'best_trial.txt').write_text(best_trial['directory']+'\n')
            write(state_path,state)
            print(f'Round {index} complete: {rank(entry)[0]}/3 {a.stage} trials passed; '
                  f'reported tokens so far {state["reported_tokens"]}',flush=True)
            if rank(entry)[0]==len(TRAIN):
                state['status']='frozen_evaluation';write(state_path,state)
                state['heldout']=[trial(root/'best_policy.py',root/'heldout'/f'seed_{seed}',seed,a.trial_timeout,a.stage,a.controller,a.grasp_quality,a.pairs) for seed in HELDOUT]
                state['status']='completed';break
        else: state['status']='round_limit'
        write(state_path,state)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'error',
                     error=f'{type(exc).__name__}: {exc}')
        write(state_path,state)
        raise


if __name__=='__main__': main()
