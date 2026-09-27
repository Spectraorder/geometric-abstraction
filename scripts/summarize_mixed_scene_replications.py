"""Audit independent learning chains without pooling dependent rollout slots."""
import argparse
import json
from pathlib import Path
from summarize_mixed_scene_learning import audit, digest, read, require



def audit_validator_repair(root, protocol):
    lineage=protocol.get('validator_repair')
    if not lineage:
        return None
    path=root/'repair_equivalence.json'
    require(digest(path)==lineage['equivalence_sha256'], 'Repair equivalence changed')
    proof=read(path); parent=Path(lineage['parent'])
    if not parent.exists(): parent=root.parent/parent.name
    require(digest(parent/'protocol.json')==lineage['parent_protocol_sha256'], 'Repair parent protocol changed')
    require(proof['status']=='passed' and proof['parent_protocol_sha256']==lineage['parent_protocol_sha256'], 'Repair gate did not pass')
    original=read(parent/'protocol.json')
    for key in ('arms','heldout_seeds','training_revisions_per_arm','training_slots','reserved_slots','model_generations','coordinator_sha256','selection','method'):
        require(protocol[key]==original[key], 'Repair changed study settings: '+key)
    old_names="    reserved = {'Session', 'RGBDPickPlace', 'ObservedRGBDPickPlace', 'BasicSession', 'MixedSession',\n                'SymbolicSession', 'MetricScene', 'SymbolicScene', 'MixedPnPObservation', 'geometry'}"
    new_names="    # Only the documented constructor exists in the actor's public namespace.\n    # Backend implementation names live in a separate module; numerical helpers\n    # cannot overwrite them. The injected factory's .session is not a public API.\n    reserved = {'Session'}"
    for row in protocol['replicas']:
        before=read(parent/row['directory']/'manifest.json');after=read(root/row['directory']/'manifest.json')
        changed=[k for k in before['source_sha256'] if before['source_sha256'][k]!=after['source_sha256'].get(k)]
        require(set(before['source_sha256'])==set(after['source_sha256']) and changed==['scripts/run_mixed_scene_experiment.py'], 'Repair changed runtime beyond the host validator')
        a=(parent/row['directory']/'source_snapshot/scripts/run_mixed_scene_experiment.py').read_text()
        b=(root/row['directory']/'source_snapshot/scripts/run_mixed_scene_experiment.py').read_text()
        require(a.replace("FORBIDDEN = {'api', 'pnp'", "FORBIDDEN = {'session', 'api', 'pnp'",1).replace(old_names,new_names)==b, 'Undeclared runner edit during repair')
        for key in ('library_sha256','starter_sha256','executor_sha256','train_seeds','heldout_seeds','model','reasoning_effort','max_prompt_characters','max_candidate_characters','physics','sensor','feedback'):
            require(before[key]==after[key], 'Repair changed replica settings: '+key)
    expected={(f'replica_01',arm,revision) for arm in protocol['arms'] for revision in (1,2,3)}
    expected.update({('replica_02',arm,1) for arm in protocol['arms']})
    require({(r['replica'],r['arm'],r['revision']) for r in proof['retained_generations']}==expected, 'Unexpected retained generations')
    for row in proof['retained_generations']:
        relative=Path(row['replica'])/row['arm']/f"round_{row['revision']:03d}"
        a=parent/relative;b=root/relative
        require(row['context_byte_identical'] and digest(a/'prompt.txt')==digest(b/'prompt.txt')==row['prompt_sha256'], 'Retained prompt differs')
        for name,sha in row['generation_files_sha256'].items():
            require(digest(a/name)==digest(b/name)==sha, 'Retained generation artifact changed')
        invocation=read(b/'invocation.json')
        require([image['sha256'] for image in invocation['images']]==row['image_sha256'], 'Retained image list differs')
        for image in invocation['images']:
            require(digest(root/row['replica']/image['name'])==image['sha256'], 'Retained input image changed')
    expected_trials=set()
    for row in protocol['replicas'][:2]:
        replica=row['directory']
        combinations=[(arm,revision) for arm in protocol['arms'] for revision in range(4)] if replica=='replica_01' else [('numeric',0),('symbolic',0),('symbolic',1)]
        for arm,revision in combinations:
            stage='revision_000' if revision==0 else f'round_{revision:03d}'
            expected_trials.update((replica,str(Path(arm)/stage/'trials'/f'seed_{seed}'/'result.json')) for seed in row['train_seeds'])
    require({(r['replica'],r['path']) for r in proof['copied_valid_trials']}==expected_trials, 'Unexpected reused trial selection')
    for row in proof['copied_valid_trials']:
        a=parent/row['replica']/row['path'];b=root/row['replica']/row['path']
        require(digest(a)==digest(b)==row['result_sha256'], 'Reused trial outcome changed')
        files=[p for p in a.parent.rglob('*') if p.is_file()]
        require(len(files)==row['file_count'], 'Original retained trial file set changed')
        for p in files:
            require(digest(p)==digest(b.parent/p.relative_to(a.parent)), 'Reused trial artifact changed')
    discarded=proof['discarded_generations']
    require({(r['replica'],r['arm'],r['revision']) for r in discarded}=={('replica_02',a,2) for a in protocol['arms']}, 'Unexpected discarded generations')
    require(not list(parent.glob('replica_02/*/round_002/trials/seed_*/result.json')), 'Discarded policies were executed')
    require(not list(parent.glob('replica_*/*/**/heldout/seed_*/result.json')), 'Reserved evaluation occurred before repair')
    usage=[]
    for row in discarded:
        generation=read(parent/row['replica']/row['arm']/f"round_{row['revision']:03d}"/'generation.json')
        usage.append({'replica':row['replica'],'arm':row['arm'],'revision':row['revision'],'usage':generation['usage']})
    return {'status':'passed','retained_generations':len(expected),'reused_valid_trials':len(expected_trials),
            'discarded_generations':usage,'not_an_independent_replication':True}


def summarize(root, allow_partial=False):
    protocol=read(root/'protocol.json')
    repair=audit_validator_repair(root,protocol)
    require(digest(root/'coordinator_snapshot.py')==protocol['coordinator_sha256'], 'Coordinator snapshot changed')
    arms=protocol['arms']
    require(arms==['numeric','symbolic'], 'Unexpected replication contrast')
    reserved=protocol['heldout_seeds']; all_train=[]; reports={}
    gate_path=root/'all_candidates_frozen.json'
    gate=read(gate_path) if gate_path.exists() else None
    if gate:
        require(gate['protocol_sha256']==digest(root/'protocol.json'), 'Study protocol changed after freeze')
        require(set(gate['replicas'])=={r['directory'] for r in protocol['replicas']}, 'Incomplete global gate')
    for row in protocol['replicas']:
        directory=root/row['directory']; manifest=read(directory/'manifest.json')
        require(digest(directory/'manifest.json')==row['manifest_sha256'], 'Replica manifest changed')
        require(manifest['arms']==arms and manifest['heldout_seeds']==reserved, 'Replica distribution mismatch')
        require(manifest['train_seeds']==row['train_seeds'], 'Replica training split changed')
        all_train.extend(manifest['train_seeds'])
        if gate:
            require(gate['replicas'][row['directory']]==read(directory/'all_candidates_frozen.json'), 'Replica/global freeze mismatch')
            require((directory/'training_summary.json').stat().st_mtime_ns<=gate_path.stat().st_mtime_ns,
                    'Training changed after study-wide freeze')
        # Check any reserved artifact even if a campaign is not yet complete.
        for result in directory.glob('*/**/heldout/seed_*/result.json'):
            require(gate is not None, 'Reserved trial exists before study-wide freeze')
            require(result.stat().st_mtime_ns>=gate_path.stat().st_mtime_ns, 'Reserved result predates study freeze')
        for image in directory.glob('*/**/heldout/seed_*/initial.png'):
            require(gate is not None and image.stat().st_mtime_ns>=gate_path.stat().st_mtime_ns,
                    'Reserved trial started before study-wide freeze')
        if (directory/'progress.json').exists() or (directory/'summary.json').exists():
            reports[row['directory']]=audit(directory,allow_partial)
        else:
            require(allow_partial, 'Replica has not started')
            for relative,sha in manifest['source_sha256'].items():
                require(digest(directory/'source_snapshot'/relative)==sha, 'Unstarted replica source changed')
            reports[row['directory']]={'status':'registered','valid_rollouts':0,'invalid_slots':0,'actors':[]}
    require(len(all_train)==len(set(all_train)), 'Training splits overlap across chains')
    require(not set(all_train)&set(reserved), 'Training overlaps reserved scenes')
    complete=all(r['status']=='completed' for r in reports.values())
    result={'schema':'mixed_scene_replications_audit_v1','status':'completed' if complete else 'in_progress',
            'protocol_sha256':digest(root/'protocol.json'),'replicas':reports,
            'valid_rollouts':sum(r['valid_rollouts'] for r in reports.values()),
            'invalid_slots':sum(r['invalid_slots'] for r in reports.values()),
            'completed_actor_generations':sum(len(r['actors']) for r in reports.values()),
            'study_freeze_audit':'passed' if gate else 'not_yet_frozen',
            'interpretation':'Independent model chains, common reserved layouts; no pooled Bernoulli confidence interval or p-value.'}
    if repair:
        result['validator_repair']=repair
    discarded=0 if repair is None else len(repair['discarded_generations'])
    result['model_generation_accounting']={'retained_completed':result['completed_actor_generations'],
        'discarded_due_to_validator_defect':discarded,
        'actual_completed':result['completed_actor_generations']+discarded}
    if complete:
        require(gate is not None, 'Missing study-wide freeze')
        require(result['valid_rollouts']+result['invalid_slots']==protocol['training_slots']+protocol['reserved_slots'], 'Study slot accounting mismatch')
        require(result['completed_actor_generations']==protocol['model_generations'], 'Study generation accounting mismatch')
        comparisons=[]
        for name,report in reports.items():
            a=report['curves']['numeric']; b=report['curves']['symbolic']
            comparisons.append({'replica':name,
                'reserved_sr_by_revision':{arm:[r['sr'] for r in report['curves'][arm]['reserved']] for arm in arms},
                'symbolic_minus_numeric_mean_curve_sr':b['mean_reserved_sr_over_revisions']-a['mean_reserved_sr_over_revisions'],
                'symbolic_minus_numeric_selected_sr':b['selected_reserved']['sr']-a['selected_reserved']['sr'],
                'first_reserved_revision_at_80_percent':{arm:report['curves'][arm]['first_reserved_revision_at_80_percent'] for arm in arms}})
        result['per_chain_comparisons']=comparisons
        result['mean_paired_curve_sr_difference']=sum(r['symbolic_minus_numeric_mean_curve_sr'] for r in comparisons)/len(comparisons)
    administrative=root/'administrative_status.json'
    if administrative.exists():
        result['administrative_status']=read(administrative)
        if result['administrative_status']['status'].startswith('withdrawn'):
            result['status']=result['administrative_status']['status']
            result['excluded_from_learning_effect_claims']=True
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args(); report=summarize(args.root.resolve(),args.allow_partial)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('status','valid_rollouts','invalid_slots','completed_actor_generations','study_freeze_audit')},indent=2))
