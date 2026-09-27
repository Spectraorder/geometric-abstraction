"""Plot completed audited visual-input chains without rerunning any policies."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(report_path, output):
    report = json.loads(report_path.read_text())
    if report['status'] != 'completed' or report['study_freeze_audit'] != 'passed':
        raise ValueError('A completed, globally frozen audit is required')
    replicas = report['replicas']
    train_counts = {row['total_slots'] for replica in replicas.values()
                    for curve in replica['curves'].values() for row in curve['training']}
    train_label = (str(next(iter(train_counts))) + ' layouts per chain'
                   if len(train_counts) == 1 else 'Layout counts vary by chain')
    reserved_counts = {row['trials'] for replica in replicas.values()
                       for curve in replica['curves'].values() for row in curve['reserved']}
    reserved_label = (str(next(iter(reserved_counts))) + ' common layouts'
                      if len(reserved_counts) == 1 else 'Layout counts vary by chain')
    revisions = sorted({row['revision'] for replica in replicas.values()
                        for curve in replica['curves'].values() for row in curve['training']})
    fig, axes = plt.subplots(2, len(replicas), figsize=(13, 7),
                             sharex=True, sharey=True, squeeze=False,
                             layout='constrained')
    arms = [('basic', 'Visual Base', '#2868b2'),
            ('symbolic', 'Full geometry', '#c66b19')]
    for column, (name, replica) in enumerate(replicas.items()):
        for arm, label, color in arms:
            curve = replica['curves'][arm]
            train, test = curve['training'], curve['reserved']
            if any(r['slots_completed'] != r['total_slots'] for r in train):
                raise ValueError('Incomplete training slots')
            axes[0, column].plot([r['revision'] for r in train],
                                [100*r['successes']/r['total_slots'] for r in train],
                                'o-', color=color, label=label)
            axes[1, column].plot([r['revision'] for r in test],
                                [100*r['sr'] for r in test], 'o-', color=color)
            selected = curve['selected_revision']
            axes[1, column].scatter(selected, 100*test[selected]['sr'], s=180,
                                    facecolors='none', edgecolors=color, linewidths=2)
        axes[0, column].set_title(name.replace('_', ' ').title())
        axes[1, column].set_xlabel('Policy revision (R0 = common visual starter)')
        for row in range(2):
            ax = axes[row, column]
            ax.set(ylim=(-4, 104), xticks=revisions, yticks=range(0, 101, 20))
            ax.grid(alpha=.2)
        axes[1, column].axhline(80, color='gray', linestyle=':', linewidth=1)
    axes[0, 0].set_ylabel('Training success (%)\n' + train_label)
    axes[1, 0].set_ylabel('Reserved success (%)\n' + reserved_label)
    axes[0, 0].legend(loc='lower left')
    fig.suptitle('Visual input versus supplied compact geometry: all registered chains', fontsize=15)
    fig.supxlabel('Open circles mark training-selected revisions. All candidates froze before reserved evaluation.\n'
                  f'{len(replicas)} chain pairs; repeated layouts and revisions are dependent. No pooled significance claim.', fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    metadata = {'audit': str(report_path),
                'training_layout_counts': sorted(train_counts),
                'reserved_layout_counts': sorted(reserved_counts),
                'revisions': revisions,
                'audit_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
                'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'figure_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'source': 'Completed independent audit; no simulator object state used'}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.audit, args.output)
