"""Audit and summarize independent paired campaigns without pseudoreplication."""
import argparse
import json
from pathlib import Path
import numpy as np
from summarize_symbolic_policy_experiment import summarize,sha


def threshold_order(raw,symbolic):
    """Compare censored threshold crossings without inventing an extra revision."""
    if raw is None and symbolic is None:return 'neither_reached'
    if symbolic is None:return 'raw_earlier'
    if raw is None:return 'symbolic_earlier'
    if raw==symbolic:return 'same_revision'
    return 'symbolic_earlier' if symbolic<raw else 'raw_earlier'


def audit_campaign(directory,manifest,summary):
    """Cross-check summaries against physical outcomes and exact actor submissions."""
    counts={'executed_rollouts':0,'invalid_slots':0,'actor_turns':0,'sensor_attachments':0}
    visited=set();first_images={}
    for filename,key in [('executor.py','executor_sha256'),('base_policy.py','base_policy_sha256')]:
        if sha(directory/filename)!=manifest[key]:raise ValueError('Shared execution changed')

    def outcomes(rows,folder,candidate,library_hash):
        if [row['seed'] for row in rows] not in [manifest['train_seeds'],manifest['heldout_seeds']]:
            raise ValueError('Outcome seed sequence mismatch')
        for row in rows:
            path=folder/f'seed_{row["seed"]}'/'result.json'
            result=json.loads(path.read_text())
            for key in ['seed','success','policy_error']:
                if result.get(key)!=row.get(key):raise ValueError('Summary/result mismatch: '+str(path))
            if row['rollout_executed']:
                if result.get('candidate_sha256')!=sha(candidate):raise ValueError('Executed policy hash mismatch')
                if result.get('geometry_library_sha256')!=library_hash:raise ValueError('Executed geometry hash mismatch')
                for key in ['physics','sensor','task','layout','wrist_depth']:
                    if result[key]!=manifest[key]:raise ValueError('Physical trial setting mismatch: '+key)
            if path not in visited:
                counts['executed_rollouts' if row['rollout_executed'] else 'invalid_slots']+=1
                visited.add(path)

    for arm,curve in summary['curves'].items():
        if [r['revision'] for r in curve]!=list(range(manifest['rounds']+1)):
            raise ValueError('Incomplete revision sequence')
        best=max(curve,key=lambda row:row['successes'])
        if summary['frozen'][arm]['revision']!=best['revision']:raise ValueError('Selection rule mismatch')
        for row in curve:
            if row['successes']!=sum(bool(x['success']) for x in row['results']):
                raise ValueError('Training success count mismatch')
            revision=row['revision']
            if revision==0:
                outcomes(row['results'],directory/'bootstrap',directory/'base_policy.py',None)
                continue
            folder=directory/arm/f'round_{revision:03d}'
            generation=json.loads((folder/'generation.json').read_text())
            response=json.loads((folder/'response.json').read_text())
            if generation!=row['generation'] or sha(folder/'candidate.py')!=generation['candidate_sha256']:
                raise ValueError('Generation metadata mismatch')
            if (folder/'candidate.py').read_text()!=response['code']:raise ValueError('Submitted source was edited')
            if (folder/'bundled.py').read_bytes()!=(directory/'executor.py').read_bytes()+b'\n'+(folder/'candidate.py').read_bytes():
                raise ValueError('Execution bundle differs from actor source')
            invocation=json.loads((folder/'invocation.json').read_text())
            for key,expected in [('fresh_context',True),('tools_disabled',True),('user_config_loaded',False),
                                 ('skills_loaded',False),('repository_context',False)]:
                if invocation.get(key)!=expected:raise ValueError('Actor isolation metadata mismatch')
            if sha(folder/'prompt.txt')!=invocation['prompt_sha256']:raise ValueError('Actor prompt changed')
            for image in invocation['images']:
                if sha(directory/image['name'])!=image['sha256']:raise ValueError('Actor sensor image changed')
            counts['sensor_attachments']+=len(invocation['images'])
            if revision==1:first_images[arm]=[image['sha256'] for image in invocation['images']]
            events=[json.loads(line) for line in (folder/'model.jsonl').read_text().splitlines() if line.startswith('{')]
            if any(e.get('item',{}).get('type') not in [None,'reasoning','agent_message'] for e in events):
                raise ValueError('Actor invoked an unexpected tool')
            completed=[e for e in events if e.get('type')=='turn.completed']
            if len(completed)!=1 or completed[0].get('usage',{})!=generation['usage']:
                raise ValueError('Actor completion/usage mismatch')
            counts['actor_turns']+=1
            outcomes(row['results'],folder/'trials',folder/'bundled.py',
                     manifest['geometry_sha256'] if arm=='symbolic' else None)
        outcomes(summary['heldout'][arm],directory/arm/'heldout',directory/arm/'frozen.py',
                 manifest['geometry_sha256'] if arm=='symbolic' else None)
    if first_images['raw']!=first_images['symbolic']:raise ValueError('Initial image feedback differs between arms')
    return dict(counts,initial_sensor_feedback_matched=True,unchanged_actor_submissions=True,
                recorded_actor_events_tool_free=True,physical_outcomes_match_summary=True)


def report(root):
    protocol=json.loads((root/'protocol.json').read_text())
    progress=json.loads((root/'progress.json').read_text())
    if progress['status']!='completed':raise ValueError('Independent campaigns are not all complete')
    if progress['protocol_sha256']!=sha(root/'protocol.json'):raise ValueError('Protocol hash mismatch')
    if sha(root/'coordinator_snapshot.py')!=protocol['coordinator_sha256']:raise ValueError('Coordinator snapshot mismatch')
    gate=json.loads((root/'all_candidates_frozen.json').read_text())
    replicas={};rows=[];all_train=[];library_hashes=set();earliest_reserved=float('inf');latest_frozen=0.
    for spec in protocol['replicas']:
        directory=root/spec['directory'];manifest=json.loads((directory/'manifest.json').read_text())
        if sha(directory/'manifest.json')!=spec['manifest_sha256']:raise ValueError('Replica manifest mismatch')
        if manifest['heldout_seeds']!=protocol['heldout_seeds'] or manifest['train_seeds']!=spec['train_seeds']:
            raise ValueError('Seed protocol mismatch')
        all_train.extend(manifest['train_seeds']);library_hashes.add(manifest['geometry_sha256'])
        for relative,expected in manifest['source_sha256'].items():
            if sha(directory/'source_snapshot'/relative)!=expected:raise ValueError('Source snapshot mismatch: '+relative)
        for name,expected in manifest['experiment_sha256'].items():
            if sha(directory/'source_snapshot/scripts'/name)!=expected:raise ValueError('Protocol source snapshot mismatch')
        if sha(directory/'geometry.py')!=manifest['geometry_sha256']:raise ValueError('Frozen online library changed')
        summary=json.loads((directory/'summary.json').read_text())
        if summary['frozen']!=gate[spec['directory']]:raise ValueError('Policy differs from global freeze gate')
        for arm,frozen in summary['frozen'].items():
            latest_frozen=max(latest_frozen,(directory/arm/'frozen.py').stat().st_mtime)
            selected=frozen['revision']
            expected=(directory/'base_policy.py').read_bytes() if selected==0 else (directory/'executor.py').read_bytes()+b'\n'+(directory/arm/f'round_{selected:03d}'/'candidate.py').read_bytes()
            if (directory/arm/'frozen.py').read_bytes()!=expected:raise ValueError('Selected planner/executor changed')
            for timing in (directory/arm/'heldout').glob('*/trial_timing.json'):
                duration=json.loads(timing.read_text())['host_trial_wall_seconds']
                earliest_reserved=min(earliest_reserved,timing.stat().st_mtime-duration)
        one=summarize(directory);one['integrity_audit']=audit_campaign(directory,manifest,summary)
        replicas[spec['directory']]=one
        raw=one['arms']['raw'];symbolic=one['arms']['symbolic']
        rows.append({'replica':spec['directory'],
            'raw_training_curve':[v['successes'] for v in raw['training_curve']],
            'symbolic_training_curve':[v['successes'] for v in symbolic['training_curve']],
            'raw_threshold_revision':raw['first_revision_at_80_percent'],
            'symbolic_threshold_revision':symbolic['first_revision_at_80_percent'],
            'threshold_order':threshold_order(raw['first_revision_at_80_percent'],symbolic['first_revision_at_80_percent']),
            'raw_reserved_successes':raw['heldout_successes'],'symbolic_reserved_successes':symbolic['heldout_successes'],
            'reserved_trials_per_arm':raw['heldout_trials'],
            'reserved_difference_pp':100*(symbolic['heldout_sr']-raw['heldout_sr']),
            'paired_reserved':one['paired_reserved']})
    if len(set(all_train))!=len(all_train) or set(all_train)&set(protocol['heldout_seeds']):raise ValueError('Train/reserved overlap')
    if len(library_hashes)!=1:raise ValueError('Replicas used different geometry libraries')
    if latest_frozen>earliest_reserved:raise ValueError('Cannot establish that all policies were frozen before reserved execution')
    pooled={arm:{'successes':sum(r['arms'][arm]['heldout_successes'] for r in replicas.values()),
                 'episodes':sum(r['arms'][arm]['heldout_trials'] for r in replicas.values())} for arm in ['raw','symbolic']}
    return {'schema':'online_geometry_replications_report_v2','protocol_sha256':sha(root/'protocol.json'),
            'method':protocol['method'],'replica_rows':rows,'replicas':replicas,
            'pooled_descriptive_counts':pooled,
            'direction_counts':{'symbolic_reaches_threshold_earlier':sum(r['threshold_order']=='symbolic_earlier' for r in rows),
                'raw_reaches_threshold_earlier':sum(r['threshold_order']=='raw_earlier' for r in rows),
                'symbolic_has_higher_reserved_sr':sum(r['reserved_difference_pp']>0 for r in rows),
                'raw_has_higher_reserved_sr':sum(r['reserved_difference_pp']<0 for r in rows)},
            'audit':{'same_geometry_library_across_replicas':True,'distinct_training_layouts':True,
                     'all_selected_sources_preserved':True,'source_snapshots_match_manifests':True,
                     'latest_freeze_before_earliest_reserved_start_s':earliest_reserved-latest_frozen},
            'limits':['Three independent coding-agent chains; simple uniquely colored cubes in one simulator.',
                'The shared PnP baseline already exposes basic visible-surface position/size estimates; this tests an additional rigid/symbolic state abstraction, not all geometry versus none.',
                'Shared reserved layouts and within-policy episodes are dependent; pooled counts have no binomial confidence interval.',
                'This combined treatment supplies online metric/symbolic geometry and derived feedback, not isolated relation names or temporal memory.',
                'Do not pool prior v1 or feedback-only pilots with this version.']}


def export(root,output,figure=None):
    value=report(root);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(value,indent=2)+'\n')
    if figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained');flat=axes.ravel()
        for index,(name,replica) in enumerate(value['replicas'].items()):
            axis=flat[index]
            for arm,color in [('raw','#697785'),('symbolic','#087e8b')]:
                data=replica['arms'][arm];curve=data['training_curve']
                axis.plot([x['revision'] for x in curve],[x['successes']/x['allotted_rollout_slots'] for x in curve],marker='o' if arm=='raw' else 'x',markerfacecolor='white',linestyle='--' if arm=='raw' else '-',label='Basic PnP' if arm=='raw' else '+ Geometry',color=color)
                for row in curve:
                    if row['validation_error']:axis.scatter(row['revision'],0,marker='x',s=90,color=color)
            axis.axhline(.8,color='#aaaaaa',ls=':');axis.set(title=name.replace('_',' '),xlabel='Policy revision',ylabel='Training success / allotted slots',xticks=[0,1,2,3])
            axis.set_ylim(-.04,1.04);axis.legend();axis.grid(axis='y',alpha=.2)
        axis=flat[3]
        for index,row in enumerate(value['replica_rows']):
            n=row['reserved_trials_per_arm']
            for offset,key,color,label in [(-.18,'raw_reserved_successes','#697785','Basic PnP'),
                                            (.18,'symbolic_reserved_successes','#087e8b','+ Geometry')]:
                count=row[key];x=index+offset
                axis.bar(x,count/n,width=.32,color=color,label=label if index==0 else None)
                axis.text(x,count/n+.015,f'{count}/{n}',ha='center',fontsize=8)
        axis.set(xticks=range(len(value['replica_rows'])),xticklabels=['Campaign '+str(i+1) for i in range(len(value['replica_rows']))],
                 ylim=(0,1.12),title='Frozen policies on reserved layouts',ylabel='Success rate')
        axis.axhline(.9,color='#aaaaaa',ls=':');axis.grid(axis='y',alpha=.2);axis.legend(fontsize=8,loc='lower right')
        fig.suptitle('Added rigid/symbolic geometry on top of shared RGB-D PnP')
        figure.parent.mkdir(parents=True,exist_ok=True);fig.savefig(figure,dpi=180);plt.close(fig)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figure',type=Path)
    args=parser.parse_args();value=export(args.campaign,args.output,args.figure)
    print(json.dumps({'rows':value['replica_rows'],'directions':value['direction_counts'],'pooled_descriptive':value['pooled_descriptive_counts']},indent=2))
