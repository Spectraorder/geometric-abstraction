"""Extend a completed actor chain without rewriting its registered evidence.

The original manifest and completed summaries remain immutable. A separately
hashed amendment authorizes additional own-training-only session turns. This
runner never evaluates reserved layouts or exposes their outcomes to an actor.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import time

import run_constraint_geometry_experiment as experiment
from geometry_actor_session import audit_session
from geometry_iteration import behavior_feedback, retention
from run_geometry_ablation import audit_events
from summarize_mixed_scene_learning import require

base = experiment.base


def read(path):
    return json.loads(path.read_text())


def effective_manifest(original, rounds):
    require(type(rounds) is int and rounds > original['rounds'], 'Extension must add proposal rounds')
    require(original.get('actor_context_mode') == 'persistent', 'Require persistent actor sessions')
    result = deepcopy(original)
    result['rounds'] = rounds
    return result


def extension_dir(root, rounds):
    return root/f'extension_{rounds:02d}'


def stage(root, arm, revision):
    return root/arm/('revision_000' if revision == 0 else f'round_{revision:03d}')


def prepare(root, rounds):
    original = read(root/'manifest.json')
    effective_manifest(original, rounds)
    experiment.check(root, original)
    old_audit = experiment.audit(root, allow_partial=False)
    require(old_audit['status'] == 'completed', 'Require a completed audited original study')
    directory = extension_dir(root, rounds)
    require(not directory.exists(), 'Extension is already registered')
    require(not list(root.glob('extension_*/amendment.json')), 'An extension already exists; do not fork a live chain')
    require(not any(stage(root,a,r).exists() for a in original['arms']
                    for r in range(original['rounds']+1, rounds+1)), 'Future revision artifacts already exist')
    paths = [root/name for name in ('manifest.json','training_summary.json','all_candidates_frozen.json',
                                    'summary.json','audit_final.json','starter.py','executor.py','starter_parity.json')]
    paths += [root/name for name in old_audit['result_hashes']]
    for arm in original['arms']:
        paths += [root/arm/'frozen.py', root/(arm+'_library.py')]
        paths += list((root/arm/'frozen_revisions').glob('*.py'))
        for revision in range(1, original['rounds']+1):
            paths += [stage(root,arm,revision)/name for name in
                      ('candidate.py','bundled.py','prompt.txt','response.json','generation.json',
                       'invocation.json','model.jsonl','session.jsonl','hypothesis_review.json')]
        state = read(root/arm/'.actor_session/state.json')
        require(state['revision'] == original['rounds'], 'Live session is not at the original final turn')
    preserved = {str(path.relative_to(root)):base.digest(path) for path in paths}
    amendment = {'schema':'geometry_round_extension_v1','registered_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                 'original_rounds':original['rounds'],'total_rounds':rounds,
                 'additional_model_calls':len(original['arms'])*(rounds-original['rounds']),
                 'additional_training_slots':len(original['arms'])*(rounds-original['rounds'])*len(original['train_seeds']),
                 'runner_sha256':base.digest(Path(__file__)),'preserved_files':preserved,
                 'scope':'Extend existing same-arm sessions using only registered training evidence. No reserved evaluation.',
                 'limitations':['Extension was requested after inspecting original reserved results; this is exploratory continuation.',
                                'No reserved results or human diagnosis are inserted into actor prompts.',
                                'Future confirmation requires separately registered fresh reserved layouts.']}
    directory.mkdir()
    base.write(directory/'amendment.json',amendment)
    base.write(directory/'progress.json',{'status':'registered'})
    return amendment


def check(root, rounds):
    directory = extension_dir(root,rounds)
    amendment = read(directory/'amendment.json')
    require(amendment['total_rounds'] == rounds, 'Round budget differs from amendment')
    require(amendment['runner_sha256'] == base.digest(Path(__file__)), 'Extension runner changed after registration')
    for name,sha in amendment['preserved_files'].items():
        require(base.digest(root/name) == sha, 'Original evidence changed: '+name)
    original = read(root/'manifest.json')
    require(original['rounds'] == amendment['original_rounds'], 'Original budget changed')
    manifest = effective_manifest(original,rounds)
    experiment.check(root,original)
    return directory, amendment, manifest


def run(root, rounds):
    directory, amendment, manifest = check(root,rounds)
    require(not (directory/'training_summary.json').exists(), 'Extended training already completed')
    curves = deepcopy(read(root/'training_summary.json')['curves'])
    with experiment.configured():
        for revision in range(amendment['original_rounds']+1,rounds+1):
            check(root,rounds)
            base.write(directory/'progress.json',{'status':'proposing','revision':revision,'curves':curves})
            with ThreadPoolExecutor(max_workers=len(manifest['arms'])) as pool:
                pending = []
                for arm in manifest['arms']:
                    output = stage(root,arm,revision)
                    if not (output/'generation.json').exists():
                        prompt,images = experiment.context(root,arm,revision,manifest)
                        pending.append(pool.submit(base.propose,output,prompt,images,manifest))
                for future in pending:
                    future.result()
            for arm in (manifest['arms'] if revision%2 else list(reversed(manifest['arms']))):
                output = stage(root,arm,revision)
                generation = read(output/'generation.json')
                require(base.digest(output/'candidate.py') == generation['candidate_sha256'], 'Submitted source changed')
                source = (output/'candidate.py').read_text()
                error = None
                try:
                    base.validate_planner(source)
                except (ValueError,SyntaxError) as exc:
                    error = type(exc).__name__+': '+str(exc)
                candidate = base.bundle(root,arm,source,output/'bundled.py')
                base.write(directory/'progress.json',{'status':'executing','arm':arm,'revision':revision,'curves':curves})
                rows = base.execute(root,arm,candidate,output/'trials',manifest['train_seeds'],manifest,error)
                curves[arm].append({'revision':revision,'successes':sum(r['success'] for r in rows),
                                    'trials':len(rows),'results':rows,'generation':generation,'validation_error':error})
                base.write(output/'hypothesis_review.json',{'revision':revision,
                           'command_comparisons':behavior_feedback(root,arm,revision,manifest['train_seeds'])})
                base.write(directory/'progress.json',{'status':'training','curves':curves})
    frozen = {}; revisions = {}
    for arm,curve in curves.items():
        selected = retention(curve)[-1]['selected_revision']
        target = directory/arm/'frozen.py'
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(stage(root,arm,selected)/'bundled.py',target)
        frozen[arm] = {'revision':selected,'sha256':base.digest(target)}
        revisions[arm] = []
        for row in curve:
            target = directory/arm/f"revision_{row['revision']:03d}.py"
            shutil.copyfile(stage(root,arm,row['revision'])/'bundled.py',target)
            revisions[arm].append({'revision':row['revision'],'sha256':base.digest(target)})
    summary = {'status':'trained_extension','manifest_sha256':base.digest(root/'manifest.json'),
               'amendment_sha256':base.digest(directory/'amendment.json'),'curves':curves,'frozen':frozen,
               'revision_frozen':revisions,'reserved_evaluation_started':False}
    base.write(directory/'training_summary.json',summary)
    base.write(directory/'all_candidates_frozen.json',{'selected':frozen,'revisions':revisions})
    report = audit(root,rounds)
    base.write(directory/'audit.json',report)
    base.write(directory/'progress.json',{'status':'completed_training_extension',
                                         'selected':frozen,'audit':'audit.json'})
    return report


def audit(root, rounds):
    directory, amendment, manifest = check(root,rounds)
    original = experiment.audit(root,allow_partial=False)
    summary = read(directory/'training_summary.json')
    require(summary['manifest_sha256'] == base.digest(root/'manifest.json') and
            summary['amendment_sha256'] == base.digest(directory/'amendment.json'), 'Summary provenance mismatch')
    require(read(directory/'all_candidates_frozen.json') ==
            {'selected':summary['frozen'],'revisions':summary['revision_frozen']}, 'Extension freeze gate differs')
    old_curves = read(root/'training_summary.json')['curves']
    actors=[]; valid=invalid=0
    for arm in manifest['arms']:
        curve=summary['curves'][arm]
        require(curve[:amendment['original_rounds']+1] == old_curves[arm], 'Inherited training curve changed')
        require(len(curve) == rounds+1, 'Incomplete extension curve')
        for revision in range(amendment['original_rounds']+1,rounds+1):
            output=stage(root,arm,revision)
            invocation=read(output/'invocation.json');generation=read(output/'generation.json')
            audit_session(output,manifest,invocation,generation)
            events=[json.loads(line) for line in (output/'model.jsonl').read_text().splitlines() if line.startswith('{')]
            require(audit_events(events)==generation['usage'],'Model usage mismatch')
            require(invocation['tools_disabled'] and not invocation['repository_context'] and
                    not invocation['user_config_loaded'] and not invocation['skills_loaded'], 'Actor isolation mismatch')
            command=invocation['command']
            require(command[command.index('--model')+1]==manifest['model'] and
                    'model_reasoning_effort="high"' in command and '--ignore-user-config' in command,'Actor settings changed')
            prompt,images=experiment.context(root,arm,revision,manifest)
            require(prompt==(output/'prompt.txt').read_text() and base.digest(output/'prompt.txt')==invocation['prompt_sha256'], 'Prompt reconstruction differs')
            require([base.digest(p) for p in images]==[r['sha256'] for r in invocation['images']], 'Image reconstruction differs')
            for item in invocation['images']:
                image=(root/item['name']).resolve()
                require(image.is_relative_to((root/arm).resolve()) and 'heldout' not in image.parts and
                        base.digest(image)==item['sha256'], 'Invalid actor image provenance')
            source=(output/'candidate.py').read_text()
            require(source==read(output/'response.json')['code'] and
                    base.digest(output/'candidate.py')==generation['candidate_sha256'],'Candidate differs from actor response')
            require((output/'bundled.py').read_text()=='Session = geometry.session\n'+source,'Candidate was repaired')
            rows=[]
            for seed in manifest['train_seeds']:
                trial=output/'trials'/f'seed_{seed}';result=read(trial/'result.json')
                require(result['seed']==seed and type(result['success']) is bool and
                        result['candidate_sha256']==base.digest(output/'bundled.py'),'Training result identity mismatch')
                if result.get('invalid_candidate'):
                    require(not result['success'] and not result['rollout_executed'],'Invalid slot counted as success');invalid+=1
                else:
                    require(result['geometry_library_sha256']==manifest['library_sha256'][arm] and
                            result['task']==manifest['task'] and result['scene']['recipe']==manifest['recipe'] and
                            result['scene']['color_map']==manifest['color_maps'][str(seed)] and
                            result['scene']['object_count']==manifest['object_count'] and
                            result['physics']==manifest['physics'] and result['sensor']==manifest['sensor'] and
                            result['wrist_depth'] and not result['sensor_blackout'],'Training runtime settings changed');valid+=1
                feedback=read(trial/'actor_feedback.json')
                require(feedback['success']==result['success'] and feedback['policy_error']==result['policy_error'],'Actor feedback differs from outcome')
                rows.append({key:result.get(key) for key in ('seed','success','policy_error')})
            require(rows==[{key:r[key] for key in ('seed','success','policy_error')} for r in curve[revision]['results']] and
                    curve[revision]['successes']==sum(r['success'] for r in rows),'Training curve differs from results')
            expected={'revision':revision,'command_comparisons':behavior_feedback(root,arm,revision,manifest['train_seeds'])}
            require(read(output/'hypothesis_review.json')==json.loads(json.dumps(expected)),'Hypothesis review differs')
            actors.append({'arm':arm,'revision':revision,'usage':generation['usage'],
                           'wall_seconds':generation['wall_seconds'],'image_count':len(images),'session_id':generation['session_id']})
        selected=retention(curve)[-1]['selected_revision']
        require(selected==summary['frozen'][arm]['revision'],'Final selection differs from training guard')
        require(base.digest(directory/arm/'frozen.py')==summary['frozen'][arm]['sha256']==
                base.digest(stage(root,arm,selected)/'bundled.py'),'Selected policy source mismatch')
        for entry in summary['revision_frozen'][arm]:
            require(base.digest(directory/arm/f"revision_{entry['revision']:03d}.py")==entry['sha256']==
                    base.digest(stage(root,arm,entry['revision'])/'bundled.py'),'Frozen revision source mismatch')
    require(valid+invalid==amendment['additional_training_slots'] and
            len(actors)==amendment['additional_model_calls'],'Extension budget mismatch')
    return {'status':'completed_training_extension','original_study_audit':original['status'],
            'additional_valid_rollouts':valid,'additional_invalid_slots':invalid,'actors':actors,
            'selected':summary['frozen'],'amendment_sha256':summary['amendment_sha256'],
            'reserved_evaluation_started':False,'limitations':amendment['limitations']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--phase',choices=('prepare','train','audit'),required=True)
    args=parser.parse_args();root=args.root.resolve()
    with (root/'training_coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='prepare': report=prepare(root,args.rounds)
        elif args.phase=='train': report=run(root,args.rounds)
        else: report=audit(root,args.rounds)
        print(json.dumps(report,indent=2),flush=True)
