"""Audit frozen mixed-scene evidence without calling models or running simulation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import math
from scipy.stats import binomtest
from run_geometry_ablation import audit_events

ARMS = ('basic', 'numeric', 'symbolic')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interval(k, n):
    if not n:
        return None
    z = 1.959963984540054; p = k/n; denominator = 1+z*z/n
    midpoint = (p+z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0., midpoint-half), min(1., midpoint+half)]


def paired(first, second):
    a = {r['seed']: bool(r['success']) for r in first}
    b = {r['seed']: bool(r['success']) for r in second}
    require(a and len(a) == len(first) and len(b) == len(second), 'Empty or duplicate paired seeds')
    require(a.keys() == b.keys(), 'Paired seed mismatch')
    left = sum(a[s] and not b[s] for s in a)
    right = sum(b[s] and not a[s] for s in a)
    discordant = left+right
    return {'both_success': sum(a[s] and b[s] for s in a), 'first_only': left,
            'second_only': right, 'neither': sum(not a[s] and not b[s] for s in a),
            'second_minus_first_sr': (right-left)/len(a),
            'exact_mcnemar_p_unadjusted': float(binomtest(left, discordant, .5).pvalue) if discordant else 1.,
            'scope': 'Conditional on these policies, exploratory seed-level comparison; not independent actor replication.'}


def audit(root, allow_partial=False):
    manifest = read(root/'manifest.json')
    arms = tuple(manifest['arms'])
    require(arms and len(set(arms)) == len(arms) and all(a in arms for a in arms), 'Invalid experiment arms')
    for relative, sha in manifest['source_sha256'].items():
        require(digest(root/'source_snapshot'/relative) == sha, 'Archived runtime source changed: '+relative)
    for arm, sha in manifest['library_sha256'].items():
        require(digest(root/(arm+'_library.py')) == sha, 'Frozen observer changed')
    require(digest(root/'starter.py') == manifest['starter_sha256'], 'Starter changed')
    require(digest(root/'executor.py') == manifest['executor_sha256'], 'Executor changed')
    require(manifest['rounds'] >= 1 and manifest['train_seeds'] and manifest['heldout_seeds'], 'Empty protocol')
    all_seeds = manifest['train_seeds']+manifest['heldout_seeds']
    require(len(all_seeds) == len(set(all_seeds)), 'Overlapping or duplicate protocol seeds')
    complete = (root/'summary.json').exists()
    require(complete or allow_partial, 'Campaign incomplete; pass --allow-partial for descriptive progress only')
    summary = read(root/'summary.json') if complete else read(root/'progress.json')
    report = {'status': 'completed' if complete else summary['status'], 'manifest_sha256': digest(root/'manifest.json'),
              'schema': manifest['schema'], 'source_audit': 'passed', 'curves': {}, 'actors': [],
              'valid_rollouts': 0, 'invalid_slots': 0, 'result_hashes': {}, 'limitations': manifest['limitations']}
    if manifest.get('protocol_correction'):
        lineage = manifest['protocol_correction']
        parent = Path(lineage['parent_directory'])
        if not parent.exists():
            parent = root.parent/parent.name
        require(digest(parent/'manifest.json') == lineage['parent_manifest_sha256'], 'Correction parent manifest changed')
        original = read(parent/'manifest.json')
        changed = [name for name, sha in original['source_sha256'].items() if manifest['source_sha256'].get(name) != sha]
        require(changed == ['scripts/run_mixed_scene_experiment.py'] == lineage['changed_runtime_snapshot_files'],
                'Correction changed more than the host validator')
        before = (parent/'source_snapshot/scripts/run_mixed_scene_experiment.py').read_text()
        after = (root/'source_snapshot/scripts/run_mixed_scene_experiment.py').read_text()
        require(before.replace("'MixedPnPObservation', 'ndimage', 'geometry'", "'MixedPnPObservation', 'geometry'") == after,
                'Correction includes undeclared implementation edits')
        for key in ('library_sha256','starter_sha256','executor_sha256','train_seeds','heldout_seeds','color_maps'):
            require(manifest[key] == original[key], 'Correction changed frozen experimental settings')
        for arm in arms:
            for name in ('candidate.py','generation.json','model.jsonl','invocation.json','prompt.txt','response.json'):
                relative = Path(arm)/'round_001'/name
                require(digest(root/relative) == digest(parent/relative), 'First-generation evidence changed in correction')
            for revision in (('revision_000',) if arm == 'basic' else ('revision_000','round_001')):
                for seed in manifest['train_seeds']:
                    relative = Path(arm)/revision/'trials'/f'seed_{seed}'/'result.json'
                    require(digest(root/relative) == digest(parent/relative), 'Reused trial changed in correction')
        require(not list(parent.glob('*/round_002/invocation.json')), 'Erroneous feedback reached a later actor')
        report['protocol_correction'] = dict(lineage, audit='passed')
    if manifest.get('observation_contract_repair'):
        lineage = manifest['observation_contract_repair']
        parent = Path(lineage['parent_directory'])
        if not parent.exists():
            parent = root.parent/parent.name
        require(digest(parent/'manifest.json') == lineage['parent_manifest_sha256'], 'Schema repair parent changed')
        previous = read(parent/'manifest.json')
        changed = [name for name, sha in previous['source_sha256'].items() if manifest['source_sha256'].get(name) != sha]
        require(changed == ['policies/mixed_scene_session.py'] == lineage['changed_runtime_snapshot_files'], 'Unexpected schema repair source change')
        before = (parent/'source_snapshot/policies/mixed_scene_session.py').read_text()
        after = (root/'source_snapshot/policies/mixed_scene_session.py').read_text()
        expected = before.replace('        self.ledger.observe(color, measured)\n        return measured\n',
            '        # Expose the same validated schema through observe(), get() and state().\n'
            '        # Shape-specific fitters may omit fields such as axis_dimensions.\n'
            '        return self.ledger.observe(color, measured)\n')
        require(expected == after and before != after, 'Schema repair contains undeclared behavior changes')
        for key in ('starter_sha256','executor_sha256','train_seeds','heldout_seeds','color_maps'):
            require(previous[key] == manifest[key], 'Schema repair changed the experimental problem')
        require(previous['library_sha256']['basic'] == manifest['library_sha256']['basic'], 'Basic observer changed during schema repair')
        for arm in arms:
            for revision in lineage['retained_generation_revisions']:
                for name in ('candidate.py','generation.json','model.jsonl','invocation.json','prompt.txt','response.json'):
                    relative = Path(arm)/f'round_{revision:03d}'/name
                    require(digest(root/relative) == digest(parent/relative), 'Retained model output changed during schema repair')
        gate_path = root/'repair_equivalence.json'
        gate = read(gate_path) if gate_path.exists() else {'status':'pending'}
        if complete or list(root.glob('*/round_003/invocation.json')):
            require(gate['status'] == 'passed', 'Model continuation before repair equivalence')
        if gate['status'] == 'passed':
            expected_trials = {(arm, revision, seed) for arm in arms for revision in (0,1) for seed in manifest['train_seeds']}
            checked_trials = {(row['arm'],row['revision'],row['seed']) for row in gate['trial_checks']}
            require(checked_trials == expected_trials and len(gate['trial_checks']) == len(expected_trials), 'Incomplete repair trial pairing')
            require(all(row['feedback_equal'] and row['physics_equal'] for row in gate['trial_checks']), 'Repair changed prior execution feedback')
            expected_contexts = {(arm,revision) for arm in arms for revision in (1,2)}
            require({(row['arm'],row['revision']) for row in gate['context_checks']} == expected_contexts
                    and len(gate['context_checks']) == 6, 'Incomplete retained-input equivalence')
            require(all(row['prompt_equal'] and row['images_equal'] for row in gate['context_checks']), 'Repair changed a retained model input')
            for row in gate['context_checks']:
                invocation = read(root/row['arm']/f"round_{row['revision']:03d}"/'invocation.json')
                require(row['image_sha256'] == [item['sha256'] for item in invocation['images']], 'Gate image hashes disagree with invocation')
        discarded = [{'arm':arm,'revision':3,'usage':read(parent/arm/'round_003/generation.json')['usage']} for arm in ARMS]
        report['observation_contract_repair'] = dict(lineage, equivalence_status=gate['status'], discarded_generations=discarded,
            accounting='These three host-defect-affected calls count in service usage but not as extra improvement iterations.')
    initial = {}; max_generation_mtime = 0.
    def inspect(directory, candidate, arm, seeds):
        rows = []
        for seed in seeds:
            path = directory/f'seed_{seed}'/'result.json'
            if not path.exists():
                if allow_partial and not complete:
                    continue
                raise ValueError('Missing trial: '+str(path))
            result = read(path)
            require(type(result['success']) is bool, 'Success must be a literal boolean')
            require(result['seed'] == seed and result['candidate_sha256'] == digest(candidate), 'Trial provenance mismatch')
            report['result_hashes'][str(path.relative_to(root))] = digest(path)
            if result.get('invalid_candidate'):
                require(not result['success'] and not result['rollout_executed'], 'Invalid candidate counted as execution')
                report['invalid_slots'] += 1
            else:
                require(result['geometry_library_sha256'] == manifest['library_sha256'][arm], 'Observer provenance mismatch')
                require(result['scene']['color_map'] == manifest['color_maps'][str(seed)] and result['scene']['object_count'] == 4, 'Scene mismatch')
                require(result['task'] == manifest['task'] and result['scene']['recipe'] == manifest['recipe'], 'Task mismatch')
                require(result['physics'] == manifest['physics'] and result['sensor'] == manifest['sensor'], 'Simulation settings mismatch')
                require(result['wrist_depth'] and not result['sensor_blackout'], 'Sensor availability mismatch')
                report['valid_rollouts'] += 1
                # The host captures this before invoking any candidate. Its first
                # queried camera need not be initial if the policy waits first.
                image = path.parent/'initial.png'
                fingerprint = digest(image)
                if seed in initial:
                    require(initial[seed] == fingerprint, 'Different initial RGB scene for seed '+str(seed))
                else:
                    initial[seed] = fingerprint
            rows.append({'seed': seed, 'success': bool(result['success']), 'policy_error': result['policy_error'],
                         'rollout_executed': not result.get('invalid_candidate', False)})
        return rows
    training_rows = {}
    for arm in arms:
        report['curves'][arm] = {'training': []}
        for revision in range(manifest['rounds']+1):
            directory = root/arm/('revision_000' if revision == 0 else f'round_{revision:03d}')
            candidate = directory/'bundled.py'
            if revision and (directory/'generation.json').exists():
                generation = read(directory/'generation.json'); invocation = read(directory/'invocation.json')
                command = invocation['command']
                require(command.count('--model') == 1 and command[command.index('--model')+1] == manifest['model'], 'Actor model mismatch')
                require('model_reasoning_effort="high"' in command and manifest['reasoning_effort'] == 'high', 'Actor reasoning mismatch')
                persistent = manifest.get('actor_context_mode', 'fresh') == 'persistent'
                require('--ignore-user-config' in command, 'Actor inherited configuration')
                if persistent:
                    from geometry_actor_session import audit_session
                    audit_session(directory, manifest, invocation, generation)
                else:
                    require('--ephemeral' in command and invocation['fresh_context'], 'Actor inherited configuration')
                require(len((directory/'prompt.txt').read_text()) <= manifest['max_prompt_characters'], 'Actor prompt exceeds frozen budget')
                response = read(directory/'response.json')
                require(response['code'] == (directory/'candidate.py').read_text(), 'Candidate differs from returned source')
                require(digest(directory/'candidate.py') == generation['candidate_sha256'], 'Actor source changed')
                require(digest(directory/'prompt.txt') == invocation['prompt_sha256'], 'Actor context changed')
                require(invocation['tools_disabled'] and not invocation['repository_context']
                        and not invocation['user_config_loaded'] and not invocation['skills_loaded'], 'Actor isolation metadata mismatch')
                events = [json.loads(line) for line in (directory/'model.jsonl').read_text().splitlines() if line.startswith('{')]
                usage = audit_events(events)
                require(usage == generation['usage'], 'Actor usage mismatch')
                for item in invocation['images']:
                    path = root/item['name']
                    require(path.resolve().is_relative_to((root/arm).resolve()), 'Other-arm image in context')
                    require('heldout' not in path.parts and 'reserved_revisions' not in path.parts, 'Reserved image in actor context')
                    require(digest(path) == item['sha256'], 'Actor image changed')
                # Static references describe code, not execution or use of the
                # precomputed state packet in the model's reasoning.
                source = (directory/'candidate.py').read_text()
                try:
                    parsed = ast.parse(source); syntax_error = None; interface_calls = {}
                    for node in ast.walk(parsed):
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                            name = node.func.attr
                            if name in {'reference','relation','align','text','target','goal_error','state','snapshot'}:
                                interface_calls[name] = interface_calls.get(name, 0)+1
                except SyntaxError as error:
                    interface_calls = {}; syntax_error = str(error)
                report['actors'].append({'arm': arm, 'revision': revision, 'usage': usage,
                                         'static_interface_call_counts': interface_calls, 'source_syntax_error': syntax_error,
                                         'static_usage_scope': 'Syntactic call sites only; not runtime frequency or evidence of how feedback was used.',
                                         'wall_seconds': generation['wall_seconds'], 'audit': 'passed',
                                         'image_count': len(invocation['images']),
                                         'actor_context_mode': manifest.get('actor_context_mode', 'fresh'),
                                         'session_id': generation.get('session_id'),
                                         'compaction_records_total': generation.get('compaction_records_total'),
                                         'planner_characters': len((directory/'candidate.py').read_text())})
                max_generation_mtime = max(max_generation_mtime, (directory/'generation.json').stat().st_mtime)
            if not candidate.exists():
                if not complete:
                    continue
                raise ValueError('Missing candidate')
            expected_source = (root/'starter.py').read_text() if not revision else (directory/'candidate.py').read_text()
            require(candidate.read_text() == 'Session = geometry.session\n'+expected_source, 'Actor candidate was repaired or replaced')
            rows = inspect(directory/'trials', candidate, arm, manifest['train_seeds'])
            training_rows[arm, revision] = rows
            report['curves'][arm]['training'].append({'revision': revision, 'successes': sum(r['success'] for r in rows),
                                                      'slots_completed': len(rows), 'total_slots': len(manifest['train_seeds'])})
    if complete:
        require(summary['status'] == 'completed' and summary['manifest_sha256'] == report['manifest_sha256'], 'Summary mismatch')
        trained = read(root/'training_summary.json'); gate = read(root/'all_candidates_frozen.json')
        require(trained['manifest_sha256'] == report['manifest_sha256'], 'Training protocol mismatch')
        require(gate == {'selected': summary['frozen'], 'revisions': summary['revision_frozen']}, 'Global freeze mismatch')
        require(trained['frozen'] == summary['frozen'] and trained['revision_frozen'] == summary['revision_frozen'], 'Training freeze metadata changed')
        require(trained['curves'] == summary['curves'], 'Training history changed after freeze')
        require((root/'all_candidates_frozen.json').stat().st_mtime >= max_generation_mtime, 'Freeze before model generation finished')
        heldout_rows = {}; final_rows = {}
        for arm in arms:
            training = report['curves'][arm]['training']
            for i, actual in enumerate(training):
                declared = summary['curves'][arm][i]
                require(declared['revision'] == i and declared['successes'] == actual['successes']
                        and declared['trials'] == actual['slots_completed'], 'Training summary differs from trials')
                require([{k: r[k] for k in ('seed','success','policy_error','rollout_executed')} for r in declared['results']]
                        == training_rows[arm,i], 'Training outcomes differ from trials')
            from geometry_iteration import select_revision
            verified_curve = [dict(row, results=training_rows[arm, row['revision']]) for row in training]
            selected = select_revision(verified_curve, manifest)
            require(selected == summary['frozen'][arm]['revision'], 'Selection used non-training information')
            require(digest(root/arm/'frozen.py') == summary['frozen'][arm]['sha256'], 'Selected source changed')
            reserved = []
            for index, row in enumerate(summary['heldout_curves'][arm]):
                require(row['revision'] == index, 'Reserved revision ordering mismatch')
                entry = summary['revision_frozen'][arm][index]
                candidate = root/arm/'frozen_revisions'/f'revision_{index:03d}.py'
                require(digest(candidate) == entry['sha256'], 'Archived revision changed')
                original = root/arm/('revision_000' if index == 0 else f'round_{index:03d}')/'bundled.py'
                require(digest(candidate) == digest(original), 'Archived revision differs from training source')
                if index == selected:
                    require(digest(root/arm/'frozen.py') == digest(candidate), 'Selected policy differs from selected training revision')
                directory = root/row['trial_path']
                expected = root/arm/'heldout' if index == selected else root/arm/'reserved_revisions'/f'revision_{index:03d}'/'heldout'
                require(directory == expected, 'Reserved result path mismatch')
                rows = inspect(directory, candidate, arm, manifest['heldout_seeds'])
                for r in rows:
                    trial_dir = directory/f"seed_{r['seed']}"
                    first_artifact = trial_dir/'initial.png' if r['rollout_executed'] else trial_dir/'result.json'
                    require(first_artifact.stat().st_mtime >= (root/'all_candidates_frozen.json').stat().st_mtime,
                            'Reserved trial started before global freeze')
                    require((directory/f"seed_{r['seed']}"/'result.json').stat().st_mtime >= (root/'all_candidates_frozen.json').stat().st_mtime,
                            'Reserved result predates global freeze')
                require([{k: r[k] for k in ('seed','success','policy_error','rollout_executed')} for r in row['results']] == rows,
                        'Reserved outcomes differ from trials')
                k = sum(r['success'] for r in rows); n = len(rows)
                require(k == row['successes'] and n == row['trials'], 'Learning curve summary differs from trials')
                reserved.append({'revision': index, 'successes': k, 'trials': n, 'sr': k/n, 'wilson95': interval(k,n)})
                heldout_rows[arm,index] = rows
                if index == selected:
                    final_rows[arm] = rows
                    require([{k: r[k] for k in ('seed','success','policy_error','rollout_executed')} for r in summary['heldout'][arm]] == rows,
                            'Selected reserved outcomes differ from trials')
            require(len(reserved) == manifest['rounds']+1, 'Missing reserved revision')
            target = manifest['training_threshold']
            report['curves'][arm].update({'reserved': reserved, 'selected_revision': selected,
                'first_training_revision_at_80_percent': next((r['revision'] for r in training if r['successes']/r['total_slots'] >= target), None),
                'first_reserved_revision_at_80_percent': next((r['revision'] for r in reserved if r['sr'] >= target), None),
                'mean_reserved_sr_over_revisions': sum(r['sr'] for r in reserved)/len(reserved),
                'selected_reserved': reserved[selected]})
        report['comparisons'] = {}
        for first, second in (('basic','numeric'), ('numeric','symbolic')):
            if first not in arms or second not in arms:
                continue
            report['comparisons'][first+'_versus_'+second] = {
                'by_revision': [{'revision': i, **paired(heldout_rows[first,i], heldout_rows[second,i])}
                                for i in range(manifest['rounds']+1)],
                'training_selected': paired(final_rows[first], final_rows[second])}
        require(len(report['actors']) == len(arms)*manifest['rounds'], 'Missing actor generations')
        expected = len(arms)*(manifest['rounds']+1)*(len(manifest['train_seeds'])+len(manifest['heldout_seeds']))
        require(report['valid_rollouts']+report['invalid_slots'] == expected, 'Slot accounting mismatch')
        report['global_freeze_audit'] = 'passed'
    report['model_generation_accounting'] = {'retained_completed': len(report['actors']),
        'discarded_due_to_schema_defect': len(report.get('observation_contract_repair', {}).get('discarded_generations', []))}
    report['model_generation_accounting']['actual_completed'] = sum(report['model_generation_accounting'].values())
    report['initial_sensor_pairing'] = {'status':'passed', 'unique_seeds_checked':len(initial), 'checked':'host pre-policy RGB render plus seed, color map and scene settings'}
    return report


def plot(report, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10,4), sharey=True)
    colors = {'basic':'#666666','numeric':'#267db3','symbolic':'#b15d23'}
    for arm in report['curves']:
        for axis, field in zip(axes, ('training','reserved')):
            rows = report['curves'][arm][field]
            y = [r['successes']/r.get('total_slots',r.get('trials',1)) for r in rows]
            axis.plot([r['revision'] for r in rows], y, marker='o', label=report.get('arm_labels', {}).get(arm, arm), color=colors[arm])
    for axis, title in zip(axes, ('Training (5 layouts)', 'Reserved (20 layouts, after global freeze)')):
        axis.set(title=title, xlabel='Policy revision', ylim=(-.03,1.03), xticks=range(4))
        axis.axhline(.8,color='#aaaaaa',linewidth=.7,linestyle=':'); axis.grid(alpha=.2)
    axes[0].set_ylabel('Success rate'); axes[1].legend()
    fig.suptitle('Mixed-shape pilot: one actor chain per arm', fontsize=12)
    fig.tight_layout(); target.parent.mkdir(parents=True,exist_ok=True); fig.savefig(target,dpi=170); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true'); parser.add_argument('--figure',type=Path)
    args = parser.parse_args(); result = audit(args.root.resolve(), args.allow_partial)
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+'\n')
    if args.figure:
        require(result['status']=='completed','Only completed curves may be plotted'); plot(result,args.figure)
    print(json.dumps({k:result[k] for k in ('status','valid_rollouts','invalid_slots','initial_sensor_pairing')},indent=2))
