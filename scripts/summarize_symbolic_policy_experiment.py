"""Export paired outcomes without exposing reserved feedback to policy actors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes, trials):
    if trials==0:return None
    z=1.959963984540054;p=successes/trials;den=1+z*z/trials
    center=(p+z*z/(2*trials))/den
    half=z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/den
    return [max(0.,center-half),min(1.,center+half)]


def summarize(root):
    summary=json.loads((root/'summary.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    if summary['status']!='completed' or summary['manifest_sha256']!=sha(root/'manifest.json'):
        raise ValueError('Require completed experiment with matching frozen manifest')
    report={'campaign':str(root),'manifest_sha256':summary['manifest_sha256'],
            'treatment':manifest['treatment'],'runtime_library':manifest.get('runtime_library'),
            'temporal_memory':manifest.get('temporal_memory',False),'arms':{},'limitations':summary['limitations']}
    for arm in manifest['arms']:
        held=summary['heldout'][arm];s=sum(bool(r['success']) for r in held);n=len(held)
        if [r['seed'] for r in held]!=manifest['heldout_seeds']:
            raise ValueError('Reserved seed list mismatch')
        frozen=summary['frozen'][arm]
        if sha(root/arm/'frozen.py')!=frozen['sha256']:
            raise ValueError('Selected policy hash mismatch')
        curve=summary['curves'][arm]
        reached=[r['revision'] for r in curve if r['successes']/r['trials']>=.8]
        usage={};generation_seconds=0.
        for row in curve:
            g=row.get('generation',{});generation_seconds+=g.get('wall_seconds',0.)
            for key,value in g.get('usage',{}).items():
                if isinstance(value,(int,float)):usage[key]=usage.get(key,0)+value
        report['arms'][arm]={'training_curve':[{'revision':r['revision'],'successes':r['successes'],
            'allotted_rollout_slots':r['trials'],'executed_rollouts':sum(bool(x['rollout_executed']) for x in r['results']),
            'validation_error':r.get('validation_error')} for r in curve],
            'first_revision_at_80_percent':min(reached) if reached else None,
            'rollout_slots_to_80_percent':(min(reached)+1)*len(manifest['train_seeds']) if reached else None,
            'selected_revision':frozen['revision'],'heldout_successes':s,'heldout_trials':n,
            'heldout_sr':s/n,'heldout_wilson_95':wilson(s,n),
            'generation_usage':usage,'generation_seconds':generation_seconds,
            'reserved_results':held}
    paired=list(zip(summary['heldout']['raw'],summary['heldout']['symbolic']))
    raw_only=sum(bool(a['success']) and not b['success'] for a,b in paired)
    symbolic_only=sum(bool(b['success']) and not a['success'] for a,b in paired)
    discordant=raw_only+symbolic_only
    p=1. if not discordant else min(1.,2*sum(math.comb(discordant,k) for k in range(min(raw_only,symbolic_only)+1))/2**discordant)
    report['paired_reserved']={'both_success':sum(bool(a['success']) and bool(b['success']) for a,b in paired),
        'raw_only':raw_only,'symbolic_only':symbolic_only,
        'both_fail':sum(not a['success'] and not b['success'] for a,b in paired),
        'exact_mcnemar_two_sided_p':p,
        'interpretation':'Descriptive pilot statistic; independent agent campaigns are still required.'}
    return report


def export(root,output,figure=None):
    report=summarize(root);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    if figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
        rejected_labels=set()
        for index,(arm,stats) in enumerate(report['arms'].items()):
            rows=stats['training_curve'];color=['#687787','#147d92'][index]
            axes[0].plot([r['revision'] for r in rows],[r['successes']/r['allotted_rollout_slots'] for r in rows],
                marker='o',label=arm,color=color)
            for row in rows:
                if row['validation_error']:
                    axes[0].scatter(row['revision'],0,marker='x',s=90,color=color,zorder=4)
                    if row['revision'] not in rejected_labels:
                        axes[0].annotate('code rejected',(row['revision'],0),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
                        rejected_labels.add(row['revision'])
            low,high=stats['heldout_wilson_95'];p=stats['heldout_sr']
            axes[1].errorbar(index,p,yerr=np.array([[p-low],[high-p]]),fmt='o',capsize=5,color=color)
            axes[1].annotate(f"{stats['heldout_successes']}/{stats['heldout_trials']}",(index,p),xytext=(8,0),textcoords='offset points')
        axes[0].axhline(.8,ls=':',color='#999999');axes[0].set(xlabel='Policy revision',ylabel='Successes / allotted rollout slots',title='Training: five slots per revision')
        axes[0].legend();axes[0].set_xticks(range(len(rows)))
        axes[1].set(xticks=[0,1],xticklabels=list(report['arms']),xlim=(-.5,1.6),title='Reserved layouts: 95% Wilson intervals')
        for axis in axes:axis.set_ylim(-.04,1.04);axis.grid(axis='y',alpha=.2)
        fig.suptitle('Online symbolic/metric state; shared execution' if report['runtime_library'] else
                     'Frozen execution; additional symbolic/metric sensor feedback')
        figure.parent.mkdir(parents=True,exist_ok=True);fig.savefig(figure,dpi=180);plt.close(fig)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figure',type=Path)
    args=parser.parse_args();r=export(args.campaign,args.output,args.figure)
    print(json.dumps({arm:{k:v for k,v in data.items() if k not in ['reserved_results','training_curve']} for arm,data in r['arms'].items()},indent=2))
