"""Create descriptive figures for the completed compact 200-second comparison."""
import argparse
import csv
import json
from pathlib import Path
from analyze_online_bound_results import analyze


def plot(root, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    report = analyze(root)
    audit = json.loads((root / 'compact_prompt_audit.json').read_text())
    if audit['status'] != 'passed' or len(audit['episodes']) != len(report['episodes']):
        raise ValueError('Compact prompt audit does not cover all episodes')
    report['compact_prompt_audit'] = audit
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    fields = ('seed', 'arm', 'success', 'capped_seconds', 'completed_requests',
              'incomplete_requests', 'images', 'model_seconds', 'first_request_seconds',
              'target_bindings', 'depth_queries', 'camera_acquisitions')
    with prefix.with_suffix('.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(report['episodes'])
    budget = report['budget_s']
    colors = {'basic': '#537fa6', 'symbolic': '#d18a46'}
    names = {'basic': 'Base', 'symbolic': 'Full (compact)'}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    for arm in ('basic', 'symbolic'):
        rows = sorted((r for r in report['episodes'] if r['arm'] == arm), key=lambda r: r['seed'])
        times = sorted(r['capped_seconds'] for r in rows if r['success'])
        axes[0].step([0, *times, budget], [0, *[100*(i+1)/len(rows) for i in range(len(times))],
                     100*len(times)/len(rows)], where='post', color=colors[arm],
                     linestyle='--' if arm == 'basic' else '-', label=names[arm])
        x = np.arange(len(rows)) + (-.12 if arm == 'basic' else .12)
        for i, row in enumerate(rows):
            axes[1].scatter(x[i], row['capped_seconds'], color=colors[arm], s=42,
                            marker='o' if row['success'] else 'x')
        axes[1].plot(x, [r['capped_seconds'] for r in rows], color=colors[arm], alpha=.25)
    axes[0].set(title='(a) Success within time budget', xlabel='Episode time (s)',
                ylabel='Success (%)', xlim=(0, budget), ylim=(-3, 103))
    axes[0].legend(frameon=False)
    axes[1].set(title='(b) Every paired outcome', xlabel='Layout index',
                ylabel='Capped completion time (s)', xticks=range(5), xticklabels=range(1, 6), ylim=(0, budget*1.08))
    axes[1].text(.02, .04, 'x: timeout; circle: success', transform=axes[1].transAxes, fontsize=8)
    base, full = (report['arms'][arm] for arm in ('basic', 'symbolic'))
    keys = [('Completed model time', 'model_seconds'), ('Image attachments', 'images'),
            ('Camera acquisitions', 'camera_acquisitions')]
    ratios = [full[key]/base[key] for _, key in keys] + [full['usage']['total_tokens']/base['usage']['total_tokens']]
    axes[2].barh([name for name, _ in keys]+['Completed total tokens'], ratios, color=colors['symbolic'], alpha=.8)
    axes[2].axvline(1., color=colors['basic'], linestyle='--', linewidth=1)
    for i, ratio in enumerate(ratios):
        axes[2].text(ratio+.025, i, f'{ratio:.2f}', va='center', fontsize=9)
    axes[2].set(title='(c) Aggregate resource ratios', xlabel='Full / Base (Base = 1)',
                xlim=(0, max(1.5, max(ratios)+.25)))
    axes[2].invert_yaxis()
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'Online bridge-tower: 5 paired layouts, {budget:g} s, no within-episode reset', fontsize=12)
    fig.savefig(prefix.with_suffix('.pdf')); fig.savefig(prefix.with_suffix('.png'), dpi=180)
    plt.close(fig)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    report = plot(args.root, args.output_prefix)
    print(json.dumps(report['arms'], indent=2))
