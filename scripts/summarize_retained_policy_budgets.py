"""Secondary stopping-budget analysis with training-only candidate retention."""
import argparse
import json
from pathlib import Path


def retained_revisions(training):
    if [r['revision'] for r in training]!=list(range(len(training))):
        raise ValueError('Require ordered consecutive training revisions')
    if not training or len({r['total_slots'] for r in training})!=1:
        raise ValueError('Require equal nonempty training budgets')
    selected=[];best=0
    for index,row in enumerate(training):
        if row['successes']>training[best]['successes']:
            best=index
        selected.append(best)
    return selected


def summarize(report):
    if report.get('status')!='completed' or report.get('study_freeze_audit')!='passed':
        raise ValueError('Require a completed audited study')
    result={'analysis':'secondary_training_selected_policy_at_each_stopping_budget',
            'selection':'Highest training SR among revisions available at the budget, earliest revision breaks ties.',
            'interpretation':'A stopping-budget evaluation of the existing chains; it does not simulate evolving from a different parent policy.',
            'replicas':{}}
    for name,replica in report['replicas'].items():
        arms={}
        for arm,curve in replica['curves'].items():
            selected=retained_revisions(curve['training'])
            if selected[-1]!=curve['selected_revision']:
                raise ValueError('Final selection differs from the registered training rule')
            rows=[{'budget':budget,'selected_revision':revision,
                   'reserved_successes':curve['reserved'][revision]['successes'],
                   'reserved_trials':curve['reserved'][revision]['trials'],
                   'reserved_sr':curve['reserved'][revision]['sr']} for budget,revision in enumerate(selected)]
            arms[arm]={'budgets':rows,'mean_reserved_sr_over_budgets':sum(r['reserved_sr'] for r in rows)/len(rows)}
        result['replicas'][name]=arms
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=summarize(json.loads(args.report.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
