"""Audit delivered target evidence; counts do not reveal an actor's reasoning."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root):
    rows=[]
    for invocation in sorted(root.glob('replica_*/*/round_*/invocation.json')):
        stage=invocation.parent;replica=stage.parent.parent
        manifest=json.loads((replica/'manifest.json').read_text())
        if manifest.get('variant')!='partial_constraints_all_training_checkpoints_v1':
            raise ValueError('Require the registered complete-feedback variant')
        receipt=json.loads(invocation.read_text());prompt_path=stage/'prompt.txt'
        if sha(prompt_path)!=receipt['prompt_sha256']:raise ValueError('Submitted prompt changed')
        prompt=prompt_path.read_text();marker='Own latest target checkpoints for every training layout:\n'
        if prompt.count(marker)!=1:raise ValueError('Expected one complete target feedback block')
        body=prompt.split(marker)[1].split('\nReturn the next planner.')[0]
        feedback=json.loads(body)
        if [r['seed'] for r in feedback]!=manifest['train_seeds']:raise ValueError('Training-layout coverage differs')
        checkpoints=[c for r in feedback for c in r['checkpoints']]
        arm=stage.parent.name
        if arm=='basic' and any(set(c)-{'object','step','time','status','action_status'} for c in checkpoints):
            raise ValueError('Visual Base received supplied geometric fields')
        rows.append({'stage':str(stage.relative_to(root)), 'arm':arm,
            'submitted_prompt_sha256':receipt['prompt_sha256'], 'invocation_sha256':sha(invocation),
            'generation_complete':(stage/'generation.json').exists(), 'image_count':len(receipt['images']),
            'covered_layouts':len(feedback),'layouts_without_checkpoints':sum(not r['checkpoints'] for r in feedback),
            'checkpoint_count':len(checkpoints),'feedback_characters':len(body),
            'current_geometry_packets':sum('geometry' in c for c in checkpoints),
            'partial_constraint_packets':sum('constraints' in c for c in checkpoints),
            'placement_goal_packets':sum('placement_goal' in c for c in checkpoints),
            'per_layout':[{'seed':r['seed'],'revision':r['revision'], 'checkpoint_count':len(r['checkpoints']),
                           'partial_constraint_count':sum('constraints' in c for c in r['checkpoints'])} for r in feedback]})
    runtime=[]
    for path in sorted(root.glob('replica_*/*/*/trials/seed_*/geometry_records.json')):
        data=json.loads(path.read_text());packets=[r['record'] for r in data if r.get('kind')=='geometry_step']
        count=sum('constraints' in r for r in packets)
        if count:runtime.append({'trial':str(path.parent.relative_to(root)),'partial_constraint_count':count,'sha256':sha(path)})
    return {'scope':__doc__,'source_sha256':sha(Path(__file__)), 'submitted_contexts':rows,
        'training_trials_with_partial_constraints':runtime,
        'interpretation':['Counts describe actual delivered evidence, not internal model reasoning or runtime call frequency.',
            'Absent partial constraints cannot explain an improvement through that feature in those contexts.',
            'The registered contrast is the full geometry package versus visual Base, not a partial-constraint ablation.',
            'This read-only analysis is never actor feedback.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=analyze(args.root.resolve());args.output.write_text(json.dumps(report,indent=2)+'\n')
    for row in report['submitted_contexts']:
        print({k:row[k] for k in ('stage','covered_layouts','checkpoint_count','partial_constraint_packets','feedback_characters')})
    print('training trials with partial constraints',len(report['training_trials_with_partial_constraints']))
