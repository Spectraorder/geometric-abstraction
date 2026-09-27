"""Audit all visual-input chains and summarize paired, dependent learning curves."""
import argparse
from pathlib import Path
import numpy as np
from summarize_mixed_scene_learning import read, digest, require
from summarize_visual_input_experiment import audit
from run_mixed_scene_experiment import write


def summarize(root,allow_partial=False):
    protocol=read(root/'protocol.json')
    require(protocol['schema']=='visual_input_replications_v1' and protocol['arms']==['basic','symbolic'],'Wrong study contrast')
    for name,key in (('coordinator_snapshot.py','coordinator_sha256'),('shared_coordinator_snapshot.py','shared_coordinator_sha256')):
        require(digest(root/name)==protocol[key],'Coordinator archive changed')
    gate_path=root/'all_candidates_frozen.json';gate=read(gate_path) if gate_path.exists() else None
    if gate:
        require(gate['protocol_sha256']==digest(root/'protocol.json'),'Protocol changed after global freeze')
        require(set(gate['replicas'])=={r['directory'] for r in protocol['replicas']},'Global freeze incomplete')
    reports={};training=[];attempts=[]
    for row in protocol['replicas']:
        directory=root/row['directory'];manifest=read(directory/'manifest.json')
        require(digest(directory/'manifest.json')==row['manifest_sha256'],'Replica manifest changed')
        require(manifest['arms']==protocol['arms'] and manifest['heldout_seeds']==protocol['heldout_seeds'] and manifest['train_seeds']==row['train_seeds'],'Replica split changed')
        training.extend(manifest['train_seeds'])
        if gate:
            require(gate['replicas'][row['directory']]==read(directory/'all_candidates_frozen.json'),'Child/global freeze mismatch')
            for name in ('training_summary.json','train_audit.json'):
                require((directory/name).stat().st_mtime_ns<=gate_path.stat().st_mtime_ns,'Training/audit changed after global freeze')
            receipt=read(directory/'train_audit.json')
            require(receipt['exact_actor_contexts_verified']==len(protocol['arms'])*manifest['rounds'],'Training contexts incomplete at freeze')
        for pattern in ('*/**/heldout/seed_*/result.json','*/**/heldout/seed_*/initial.png'):
            for path in directory.glob(pattern):
                require(gate is not None and path.stat().st_mtime_ns>=gate_path.stat().st_mtime_ns,'Reserved trial started before global freeze')
        reports[row['directory']]=audit(directory,allow_partial)
        for p in directory.glob('*/round_*/invocation.json'):
            attempts.append({'path':str(p.relative_to(root)),'sha256':digest(p),'completed':(p.parent/'generation.json').exists()})
    require(len(training)==len(set(training)) and not set(training)&set(protocol['heldout_seeds']),'Overlapping training/reserved splits')
    complete=all(r['status']=='completed' for r in reports.values())
    require(allow_partial or complete,'Study incomplete')
    report={'schema':'visual_input_replications_audit_v1','status':'completed' if complete else 'in_progress',
        'protocol_sha256':digest(root/'protocol.json'),'replicas':reports,
        'valid_rollouts':sum(r['valid_rollouts'] for r in reports.values()),'invalid_slots':sum(r['invalid_slots'] for r in reports.values()),
        'completed_actor_generations':sum(len(r['actors']) for r in reports.values()),
        'model_invocation_attempts':attempts,'study_freeze_audit':'passed' if gate else 'not_yet_frozen',
        'interpretation':'Three independent chain pairs with common test layouts. Correlated policy revisions and test episodes are not additional actor replications. No pooled Bernoulli confidence interval or p-value.',
        'auditor_sha256':digest(Path(__file__))}
    if complete:
        require(gate is not None,'Missing global freeze')
        require(report['valid_rollouts']+report['invalid_slots']==protocol['training_slots']+protocol['reserved_slots'],'Study slot accounting mismatch')
        require(report['completed_actor_generations']==protocol['model_generations'],'Study generation accounting mismatch')
        comparisons=[]
        for name,result in reports.items():
            a,b=result['curves']['basic'],result['curves']['symbolic']
            comparisons.append({'replica':name,
                'reserved_sr_by_revision':{arm:[r['sr'] for r in result['curves'][arm]['reserved']] for arm in protocol['arms']},
                'geometry_minus_visual_mean_curve_sr':b['mean_reserved_sr_over_revisions']-a['mean_reserved_sr_over_revisions'],
                'geometry_minus_visual_selected_sr':b['selected_reserved']['sr']-a['selected_reserved']['sr'],
                'first_reserved_revision_at_80_percent':{arm:result['curves'][arm]['first_reserved_revision_at_80_percent'] for arm in protocol['arms']},
                'selected_revision':{arm:result['curves'][arm]['selected_revision'] for arm in protocol['arms']}})
        report['per_chain_comparisons']=comparisons
        for field in ('geometry_minus_visual_mean_curve_sr','geometry_minus_visual_selected_sr'):
            values=[r[field] for r in comparisons]
            report[field]={'mean':float(np.mean(values)),'range':[min(values),max(values)],'paired_chains':len(values)}
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    report=summarize(args.root.resolve(),args.allow_partial)
    args.output.parent.mkdir(parents=True,exist_ok=True);write(args.output,report)
    print({k:report[k] for k in ('status','valid_rollouts','invalid_slots','completed_actor_generations','study_freeze_audit')})
