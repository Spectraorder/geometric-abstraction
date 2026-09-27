"""Export descriptive figures for the frozen five-layout recovery comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root):
    summary=json.loads((root/'summary.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.1),layout='constrained')
    colors={'basic':'#456b96','symbolic':'#24866f'};labels={'basic':'Base','symbolic':'Full'}
    for arm in colors:
        rows=sorted((r for r in summary['records'] if r['arm']==arm),key=lambda r:r['seed'])
        times=np.array([r['capped_seconds'] if r['success'] else np.inf for r in rows])
        xs=np.r_[0,sorted(t for t in times if np.isfinite(t)),protocol['budget_s']]
        axes[0].step(xs,[(times<=x).mean() for x in xs],where='post',label=labels[arm],color=colors[arm])
        axes[1].plot(range(1,6),[r['capped_seconds'] for r in rows],'o-',label=labels[arm],color=colors[arm])
        axes[2].plot(range(1,6),[r['usage']['total_tokens']/1000 for r in rows],'o-',label=labels[arm],color=colors[arm])
    axes[0].set(xlabel='Recovery wall time (s)',ylabel='Success fraction',ylim=(-.03,1.03),title='(a) Success within budget')
    axes[0].legend(frameon=False)
    axes[1].set(xlabel='Paired layout',ylabel='Capped completion time (s)',title='(b) All episodes, timeout = 240 s',xticks=range(1,6))
    axes[2].set(xlabel='Paired layout',ylabel='Completed-request tokens (thousands)',title='(c) Measured model usage',xticks=range(1,6))
    for ax in axes:
        ax.spines[['right','top']].set_visible(False);ax.grid(alpha=.15)
    for suffix in ('png','pdf'):
        fig.savefig(root/f'recovery_comparison.{suffix}',dpi=220)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    plot(p.parse_args().root)
