"""Plot every registered learning chain from a completed, audited study report."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot(report, target):
    if report.get('status')!='completed' or report.get('study_freeze_audit')!='passed':
        raise ValueError('Require a completed study with a passed global freeze audit')
    replicas=report['replicas']
    if len(replicas)!=3 or any(r.get('global_freeze_audit')!='passed' for r in replicas.values()):
        raise ValueError('Require all three independently audited chains')
    if report.get('validator_repair',{}).get('status','passed')!='passed':
        raise ValueError('Correction lineage has not passed')
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
    colors={'numeric':'#267db3','symbolic':'#b15d23'}
    for col,(name,replica) in enumerate(sorted(replicas.items())):
        for arm,color in colors.items():
            curve=replica['curves'][arm]
            for row,field in enumerate(('training','reserved')):
                records=curve[field]
                if [r['revision'] for r in records]!=[0,1,2,3]:
                    raise ValueError('A registered revision is missing')
                totals=[r['total_slots'] if field=='training' else r['trials'] for r in records]
                if totals!=([5]*4 if field=='training' else [20]*4):
                    raise ValueError('Unexpected scene counts in a learning curve')
                values=[r['successes']/n for r,n in zip(records,totals)]
                ax=axes[row,col]
                ax.plot(range(4),values,color=color,marker='o',lw=1.8,label=arm)
                if field=='reserved':
                    selected=curve['selected_revision']
                    ax.scatter([selected],[values[selected]],s=140,facecolors='none',edgecolors=color,lw=1.5)
        axes[0,col].set_title(name.replace('_',' ').title(),fontsize=11)
    for ax in axes.flat:
        ax.axhline(.8,color='#888888',lw=.8,ls=':')
        ax.set(ylim=(-.04,1.04),xticks=range(4))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(alpha=.2)
    axes[0,0].set_ylabel('Training SR (5 layouts per chain)')
    axes[1,0].set_ylabel('Reserved SR (20 common layouts)')
    for ax in axes[1]:ax.set_xlabel('Policy revision (R0 is the common starter)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.5,.955),frameon=False)
    fig.suptitle('Mixed-shape representation study: three independent actor chains',fontsize=13)
    fig.text(.5,.015,'Circles around reserved points mark training-selected policies. All revisions froze before evaluation. Shared layouts and revisions are dependent.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.91))
    target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target,dpi=170);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plot(json.loads(args.report.read_text()),args.output)
    print(args.output)
