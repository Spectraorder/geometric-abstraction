"""Budgeted policy revision for a single matching hoop and fixed pole."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from learn_fasten_screws import write,reference_context
from run_vision_trial import validate_source
from aspire_so100.hoop_insertion import HoopPoleEnv,CONTROLLER
from aspire_so100.hoop_pole import LEARNING_SCENE_VERSIONS,HOOP_VARIANTS

ROOT=Path(__file__).resolve().parents[1]
TRAIN=[6101,6102,6103]
HELDOUT=[6201,6202,6203]


def trial(candidate,folder,seed,timeout,variant="standard"):
    path=folder/'result.json'
    if path.exists():
        result=json.loads(path.read_text())
        if (result['candidate_sha256']!=hashlib.sha256(candidate.read_bytes()).hexdigest()
            or result['seed']!=seed or result['task']!='hoop_pole'
            or result['scene']['generator']!=LEARNING_SCENE_VERSIONS[variant]
            or result['scene'].get('hoop_variant','standard')!=variant
            or (variant!='standard' and result['scene'].get('hoop_inner_radius_m')!=HOOP_VARIANTS[variant])
            or result['scene']['controller']!=CONTROLLER
            or result['scene']['evaluator']!='hoop_seated_v1'):
            raise RuntimeError('Cached trial does not match this candidate/task/scene')
        return result
    if folder.exists():
        folder.rename(folder.with_name(folder.name+'.interrupted-'+datetime.now().strftime('%Y%m%d%H%M%S')))
    folder.parent.mkdir(parents=True,exist_ok=True)
    with folder.with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/run_vision_trial.py'),
            '--candidate',str(candidate),'--output',str(folder),'--task','hoop_pole',
            '--layout','sampled','--seed',str(seed),'--timeout',str(timeout),'--hoop-variant',variant],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout+180)
    return json.loads(path.read_text())


def rank(entry):
    results=entry.get('results',[])
    return (sum(r['success'] for r in results),
            sum(sum(r['metrics']['milestones'].values()) for r in results),
            sum(min(r['metrics']['stable_seconds'],2.) for r in results),
            -sum(bool(r['policy_error']) for r in results)-int('error' in entry))


def evaluate_candidate(candidate,folder,seeds,a):
    entry=dict(candidate=str(candidate),results=[],feedback=[])
    for seed in seeds:
        directory=folder/f'seed_{seed}'
        result=trial(candidate,directory,seed,a.trial_timeout,a.hoop_variant)|dict(directory=str(directory))
        entry['results'].append(result)
        feedback=json.loads((directory/'actor_feedback.json').read_text())
        notes=feedback['notes'];feedback['notes']=notes if len(notes)<=18 else notes[:6]+notes[-12:]
        entry['feedback'].append(feedback)
        print(f'{folder.name}, seed {seed}: success={result["success"]}; center_error_mm={1000*result["metrics"]["center_error_m"]:.3f}',flush=True)
    return entry


def publish_best(root,state,entry):
    state['best']=entry
    shutil.copyfile(entry['candidate'],root/'.best_policy.py.tmp')
    (root/'.best_policy.py.tmp').replace(root/'best_policy.py')
    best=max(entry['results'],key=lambda r:rank(dict(results=[r])))
    (root/'best_trial.txt').write_text(best['directory']+'\n')


def frozen_evaluation(a,root,state):
    candidate=root/'best_policy.py'
    digest=hashlib.sha256(candidate.read_bytes()).hexdigest()
    if state.get('frozen_sha256',digest)!=digest:
        raise RuntimeError('Frozen policy checksum changed')
    state.update(status='frozen_evaluation',frozen_sha256=digest);write(root/'campaign.json',state)
    state['heldout']=[trial(candidate,root/'heldout'/f'seed_{seed}',seed,a.trial_timeout,a.hoop_variant)
                      for seed in a.heldout_seeds]
    # Comparison happens only after policy selection is final; neither result feeds revision.
    if state.get('baseline'):
        baseline=Path(state['baseline']['candidate'])
        state['baseline_heldout']=[trial(baseline,root/'baseline_heldout'/f'seed_{seed}',seed,a.trial_timeout,a.hoop_variant)
                                  for seed in a.heldout_seeds]
        state['comparison']=dict(seeds=a.heldout_seeds,
            baseline_passed=sum(r['success'] for r in state['baseline_heldout']),
            learned_passed=sum(r['success'] for r in state['heldout']),trials=len(a.heldout_seeds))
        print('Fresh-layout comparison: '+json.dumps(state['comparison']),flush=True)
    state['status']='completed';write(root/'campaign.json',state)


def run(a,root):
    path=root/'campaign.json'
    state=json.loads(path.read_text()) if path.exists() else dict(
        created_at=datetime.now(timezone.utc).isoformat(),task='hoop_pole',model=a.model,
        scene_version=LEARNING_SCENE_VERSIONS[a.hoop_variant],hoop_variant=a.hoop_variant,controller=CONTROLLER,evaluator='hoop_seated_v1',
        history=[],reported_tokens=0,model_calls=0,train_seeds=a.train_seeds,heldout_seeds=a.heldout_seeds)
    if (state['task']!='hoop_pole' or state['scene_version']!=LEARNING_SCENE_VERSIONS[a.hoop_variant]
        or state.get('hoop_variant','standard')!=a.hoop_variant
        or state['train_seeds']!=a.train_seeds or state['heldout_seeds']!=a.heldout_seeds
        or state['controller']!=CONTROLLER or state['evaluator']!='hoop_seated_v1'
        or state['model']!=a.model):
        raise RuntimeError('Campaign configuration changed; use a fresh root')
    if state.get('status')=='completed':
        print('Campaign already completed frozen evaluation.',flush=True);return
    references=reference_context(root,state,([a.initial_policy] if a.initial_policy else [])+a.reference_policy)
    if bool(state.get('initial_policy',False)) != bool(a.initial_policy) and state.get('history'):
        raise RuntimeError('Initial policy setting changed; use a fresh root')
    state['initial_policy']=bool(a.initial_policy)
    if state.get('frozen_sha256'):
        frozen_evaluation(a,root,state);return
    state.update(status='initializing',limits=vars(a)|dict(root=str(root),
        reference_policy=[str(p.resolve()) for p in a.reference_policy],
        initial_policy=str(a.initial_policy.resolve()) if a.initial_policy else None))
    write(path,state)
    try:
        if not (root/'initial.png').exists():
            from PIL import Image
            e=HoopPoleEnv(root/'scene_check',scene_seed=a.train_seeds[0],hoop_variant=a.hoop_variant)
            try:
                e.reset(a.train_seeds[0]);Image.fromarray(e.get_overhead_rgbd()['rgb']).save(root/'initial.png')
                write(root/'scene_check/metrics.json',asdict(e.evaluate()))
            finally:e.close()
        if a.initial_policy and not state.get('baseline'):
            state['status']='baseline_evaluation';write(path,state)
            baseline=evaluate_candidate(Path(state['references'][0]['snapshot']),root/'baseline',a.train_seeds,a)
            baseline['round']=0;baseline['rank']=rank(baseline);state['baseline']=baseline
            publish_best(root,state,baseline);write(path,state)
        for index in range(len(state['history'])+1,a.rounds+1):
            if state['reported_tokens']>=a.max_reported_tokens:
                state['status']='reported_token_limit';break
            folder=root/f'round_{index:03d}';candidate=folder/'candidate.py'
            sections=['Active task: one matching red hoop and fixed red pole; ordinary contact; no screwing.']
            if a.hoop_variant=='tight':
                sections += ['This continuation uses a narrower hoop opening. Improve centering robustness using live RGB-D, especially hoop-to-tool offset and release drift. Keep successful grasp/transport behavior. No simulator dimensions or seed-specific coordinates may enter the policy.']
            images=[root/'initial.png']
            if not state['history'] and state.get('baseline'):
                sections += ['Measured baseline development feedback:',json.dumps(state['baseline']['feedback'])]
                failed=[r for r in state['baseline']['results'] if not r['success']]
                if failed:images += [Path(failed[0]['directory'])/'outcome.png']
            if not state['history'] and references and not state.get('best'):sections += [references]
            if state.get('best'):sections += ['Current best policy:',Path(state['best']['candidate']).read_text()]
            if state['history']:
                last=state['history'][-1]
                sections += ['Latest development feedback:',json.dumps(last.get('feedback',last.get('error','')))]
                if last.get('results'):images += [Path(last['results'][0]['directory'])/'outcome.png']
            sections += ['Recent outcomes:',json.dumps([dict(round=h['round'],rank=h.get('rank'),error=h.get('error')) for h in state['history'][-4:]])]
            contract=ROOT/('docs/HOOP_POLE_TIGHT_PROMPT.txt' if a.hoop_variant=='tight' else 'docs/HOOP_POLE_PROMPT.txt');context='\n'.join(sections)
            if len(context)+len(contract.read_text())>a.max_prompt_chars:
                raise RuntimeError('Prompt cap exceeded; no model call made')
            context_path=root/f'context_{index:03d}.txt';context_path.write_text(context)
            state.update(status='proposing',active_round=index);write(path,state)
            if not candidate.exists():
                if folder.exists():folder.rename(folder.with_name(folder.name+'.interrupted-'+datetime.now().strftime('%Y%m%d%H%M%S')))
                state['model_calls']+=1;write(path,state)
                command=[sys.executable,'-u',str(ROOT/'scripts/propose_vision_policy.py'),
                    '--round-dir',str(folder),'--contract-file',str(contract),'--context-file',str(context_path),
                    '--model',a.model,'--timeout',str(a.llm_timeout)]
                for image in images:command += ['--image',str(image)]
                print(f'Round {index}: asking {a.model}; {len(context)+len(contract.read_text())} prompt chars',flush=True)
                subprocess.run(command,check=True,timeout=a.llm_timeout+60)
            events=[json.loads(line) for line in (folder/'model.jsonl').read_text().splitlines() if line.startswith('{')]
            if any(e.get('item',{}).get('type') in {'command_execution','mcp_tool_call','web_search','file_change'} for e in events):
                raise RuntimeError('Actor tool-isolation audit failed')
            usage=[e['usage'] for e in events if e.get('type')=='turn.completed' and 'usage' in e]
            if str(index) not in state.setdefault('usage_by_round',{}):
                state['usage_by_round'][str(index)]=usage
                state['reported_tokens']+=sum(u.get('input_tokens',0)+u.get('output_tokens',0) for u in usage)
            state['status']='evaluating';write(path,state)
            entry=dict(round=index,candidate=str(candidate),results=[],feedback=[])
            try:validate_source(candidate.read_text())
            except (ValueError,SyntaxError) as error:
                entry['error']=str(error);state['history'].append(entry);write(path,state)
                print(f'Round {index}: policy rejected: {error}',flush=True);continue
            for seed in a.train_seeds:
                directory=folder/'trials'/f'seed_{seed}'
                result=trial(candidate,directory,seed,a.trial_timeout,a.hoop_variant)|dict(directory=str(directory))
                entry['results'].append(result)
                feedback=json.loads((directory/'actor_feedback.json').read_text())
                notes=feedback['notes'];feedback['notes']=notes if len(notes)<=18 else notes[:6]+notes[-12:]
                entry['feedback'].append(feedback);write(folder/'evaluation.json',entry)
                print(f'Round {index}, seed {seed}: success={result["success"]}; stages={result["metrics"]["milestones"]}; error={result["policy_error"]}',flush=True)
            entry['rank']=rank(entry);state['history'].append(entry)
            if not state.get('best') or rank(entry)>=rank(state['best']):
                publish_best(root,state,entry)
            write(path,state)
            print(f'Round {index} complete: {rank(entry)[0]}/{len(a.train_seeds)} passed; reported tokens {state["reported_tokens"]}',flush=True)
            if rank(entry)[0]==len(a.train_seeds):
                state['status']='development_passed';break
        else:state['status']='round_limit'
        state['development_stop_reason']=state['status']
        if state.get('best'):
            frozen_evaluation(a,root,state)
        write(path,state)
        print(f'Campaign stopped: {state["status"]}',flush=True)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',error=f'{type(error).__name__}: {error}')
        write(path,state);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--model',default='gpt-6-astra')
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--max-reported-tokens',type=int,default=60000)
    p.add_argument('--max-prompt-chars',type=int,default=40000)
    p.add_argument('--llm-timeout',type=int,default=300);p.add_argument('--trial-timeout',type=int,default=300)
    p.add_argument('--reference-policy',type=Path,action='append',default=[])
    p.add_argument('--initial-policy',type=Path,help='Evaluate and retain this policy as the starting best')
    p.add_argument('--hoop-variant',choices=tuple(HOOP_VARIANTS),default='standard')
    p.add_argument('--train-seeds',type=int,nargs='+',default=TRAIN)
    p.add_argument('--heldout-seeds',type=int,nargs='+',default=HELDOUT)
    a=p.parse_args()
    if set(a.train_seeds)&set(a.heldout_seeds):p.error('Development and held-out seeds must be disjoint')
    if len(set(a.train_seeds))!=len(a.train_seeds) or len(set(a.heldout_seeds))!=len(a.heldout_seeds):
        p.error('Seed lists must not contain duplicates')
    if min(a.rounds,a.max_reported_tokens,a.max_prompt_chars,a.llm_timeout,a.trial_timeout)<=0:p.error('Limits must be positive')
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a,root)


if __name__=='__main__':main()
