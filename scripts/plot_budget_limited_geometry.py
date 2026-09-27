"""Plot retained learning evidence without inventing missing test scores."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def render(report_path, output):
    report=json.loads(report_path.read_text())
    if report['status']!='user_budget_stopped' or report['registered_study_complete']:
        raise ValueError('Require the explicitly budget-limited report')
    for relative,digest in report['source_sha256'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Source changed: '+relative)
    audit_path=ROOT/'docs/evaluations/constraint_geometry_learning_progress.json'
    audit=json.loads(audit_path.read_text())
    if audit['study_freeze_audit']!='passed':raise ValueError('Global freeze is not verified')
    replicas=audit['replicas'];batches=report['batches']
    if sum(b['completed_slots'] for b in batches)!=report['reserved_rollouts']:
        raise ValueError('Reserved slot accounting differs')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(2,len(replicas),figsize=(13,7.6),sharex=True,sharey=True,
                           squeeze=False,layout='constrained')
    arms=[('basic','Visual Base','#2868b2'),('symbolic','Full geometry','#c66b19')]
    displayed=[]
    for column,(name,replica) in enumerate(replicas.items()):
        relevant=[b for b in batches if b['replica']==name]
        revisions=sorted({r['revision'] for c in replica['curves'].values() for r in c['training']})
        for arm,label,color in arms:
            training=replica['curves'][arm]['training']
            if any(r['slots_completed']!=r['total_slots'] for r in training):
                raise ValueError('Incomplete training curve')
            axes[0,column].plot([r['revision'] for r in training],
                [100*r['successes']/r['total_slots'] for r in training], 'o-',color=color,label=label)
            rows={b['revision']:b for b in relevant if b['arm']==arm}
            values=[]
            for rev in revisions:
                row=rows[rev]
                if row['batch_complete']:
                    if row['completed_slots']!=row['planned_slots'] or abs(row['sr']-row['successes']/row['planned_slots'])>1e-12:
                        raise ValueError('Incorrect completed-batch SR')
                    values.append(100*row['sr']);displayed.append(row)
                else:
                    if row['sr'] is not None:raise ValueError('Partial batch has an SR')
                    values.append(np.nan)
            # NaNs preserve genuine gaps: no line from Full R1 to R3 across R2.
            axes[1,column].plot(revisions,values,'o-',color=color,label=label)
            for row in rows.values():
                if row['selected_from_training'] and row['batch_complete']:
                    axes[1,column].scatter(row['revision'],100*row['sr'],s=180,
                        facecolors='none',edgecolors=color,linewidths=2,zorder=4)
        axes[0,column].set_title('Chain '+name.split('_')[-1],fontweight='bold')
        for row in range(2):
            axes[row,column].set(ylim=(-5,105),xticks=revisions,yticks=range(0,101,20))
            axes[row,column].grid(alpha=.2)
        completed=sum(b['completed_slots'] for b in relevant)
        planned=sum(b['planned_slots'] for b in relevant)
        ax=axes[1,column]
        if not completed:
            ax.set_facecolor('#f3f4f5')
            ax.text(.5,.60,'Not evaluated',transform=ax.transAxes,ha='center',fontsize=15,color='#555555')
            ax.text(.5,.43,f'0/{planned} reserved rollouts\nUser capped evaluation',transform=ax.transAxes,
                ha='center',va='center',fontsize=11,color='#555555')
        else:
            ax.axhline(80,color='gray',ls=':',lw=1)
            ax.text(.97,.03,f'{completed}/{planned} reserved rollouts retained',transform=ax.transAxes,
                ha='right',fontsize=8,color='#555555')
        ax.set_xlabel('Policy revision (R0 = common starter)')
    axes[0,0].set_ylabel('Training success (%)\n8 layouts per chain')
    axes[1,0].set_ylabel('Reserved success (%)\n20 common layouts')
    axes[0,0].legend(loc='lower left')
    fig.suptitle('Preliminary geometry comparison: evaluation stopped at 103/480 rollouts',fontsize=15,fontweight='bold')
    fig.supxlabel('Only complete test batches have scores. Base R2 has 3/20 trials; its SR is not plotted.\n'
        'Gaps are unexecuted/incomplete, not failures. Open circles mark training-selected versions.\n'
        'Reserved evidence covers one learning chain; repeated layouts are not independent replications.',fontsize=10)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,facecolor='white');plt.close(fig)
    receipt={'scope':__doc__,'report_sha256':sha(report_path),'audit_sha256':sha(audit_path),
        'renderer_sha256':sha(Path(__file__)),'figure_sha256':sha(output),
        'displayed_complete_reserved_batches':displayed,
        'partial_or_unexecuted_scores_plotted':False,'missing_revisions_connected':False}
    output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();render(args.report.resolve(),args.output.resolve())
