"""Analyze frozen paired outcomes without running episodes or changing their evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root):
    protocol = read(root / 'protocol.json')
    if digest(root / 'protocol.json') != (root / 'protocol.sha256').read_text().strip():
        raise ValueError('Registered protocol changed')
    for name, expected in protocol['source_sha256'].items():
        if digest(root / 'sources' / name) != expected:
            raise ValueError('Archived source changed: ' + name)
    if digest(root / 'geometry.py') != protocol['geometry_sha256']:
        raise ValueError('Archived geometry changed')
    progress = read(root / 'progress.json')
    if progress['status'] != 'completed' or progress['completed'] != len(protocol['slots']):
        raise ValueError('Study is incomplete')
    audits = {name: read(root / name) for name in ('audit.json', 'interface_audit.json')}
    expected_paths = {slot['path'] for slot in protocol['slots']}
    for audit in audits.values():
        if audit['status'] != 'passed' or {r['episode'] for r in audit['episodes']} != expected_paths:
            raise ValueError('Audit does not cover the complete registered study')
    resource_rows = {r['episode']: r for r in audits['audit.json']['episodes']}
    interface_rows = {r['episode']: r for r in audits['interface_audit.json']['episodes']}
    summary = read(root / 'summary.json')
    rows = []
    evidence = {name: digest(root / name) for name in
                ('protocol.json', 'summary.json', 'audit.json', 'interface_audit.json', 'progress.json')}
    for slot in protocol['slots']:
        episode = root / slot['path']
        result = read(episode / 'result.json')
        if any(result[k] != slot[k] for k in ('seed', 'arm')) or result['budget_s'] != protocol['budget_s']:
            raise ValueError('Episode identity or budget mismatch')
        if digest(episode / 'result.json') != resource_rows[slot['path']]['result_sha256']:
            raise ValueError('Result changed since resource audit')
        receipts = [read(p) for p in sorted(episode.glob('context_*/decision_*/receipt.json'))]
        usage = {k: sum(r['usage'].get(k, 0) for r in receipts) for k in result['usage']}
        incomplete = len(list(episode.glob('context_*/decision_*/attempt_failure.json')))
        if (usage != result['usage'] or len(receipts) != result['model_requests_completed']
                or incomplete != result['incomplete_requests']):
            raise ValueError('Receipts disagree with the episode aggregate')
        row = dict(interface_rows[slot['path']])
        if row['success'] != result['success'] or row['status'] != result['status']:
            raise ValueError('Interface audit disagrees with the episode outcome')
        durations = [r['wall_seconds'] for r in receipts]
        invocations = [read(p) for p in episode.glob('context_*/decision_*/invocation.json')]
        images = sum(len(r['images']) for r in invocations)
        if images != row['images'] or abs(sum(durations) - row['model_seconds']) > 1e-6:
            raise ValueError('Receipt timing or images disagree with the interface audit')
        row.update(first_request_seconds=durations[0], request_seconds=durations,
                   completed_request_images=sum(r['image_count'] for r in receipts),
                   completed_operations=result['completed_operations'],
                   dispatched_operations=result['dispatched_operations'],
                   camera_acquisitions=result['camera_acquisitions'],
                   camera_seconds=result['camera_seconds'],
                   feedback_seconds=result['feedback_seconds'], action_seconds=result['action_seconds'])
        rows.append(row)
    arms = {}
    for arm in protocol['arms']:
        selected = [r for r in rows if r['arm'] == arm]
        durations = [v for r in selected for v in r['request_seconds']]
        totals = dict(summary['arms'][arm])
        if totals['successes'] != sum(r['success'] for r in selected):
            raise ValueError('Summary success count mismatch')
        if abs(totals['mean_capped_seconds'] - mean(r['capped_seconds'] for r in selected)) > 1e-6:
            raise ValueError('Summary time mismatch')
        if totals['usage'] != {k: sum(r['usage'][k] for r in selected) for k in totals['usage']}:
            raise ValueError('Summary token mismatch')
        for key in ('completed_requests', 'incomplete_requests', 'images', 'model_seconds',
                    'target_bindings', 'depth_queries', 'camera_acquisitions'):
            if abs(totals[key] - sum(r[key] for r in selected)) > 1e-6:
                raise ValueError('Summary resource mismatch: ' + key)
        totals.update(mean_request_seconds=mean(durations), median_request_seconds=median(durations),
                      mean_first_request_seconds=mean(r['first_request_seconds'] for r in selected),
                      uncached_input_tokens=totals['usage']['input_tokens'] - totals['usage']['cached_input_tokens'])
        arms[arm] = totals
    return dict(budget_s=protocol['budget_s'], campaign_wall_seconds=progress['wall_seconds'],
                arms=arms, episodes=rows, evidence_sha256=evidence,
                scope='Descriptive analysis of all registered pairs. Images include attempted requests; '
                      'model time and token counts cover completed requests only. Incomplete token usage remains unknown.')


def save(report, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    fields = ('seed', 'arm', 'success', 'capped_seconds', 'completed_requests',
              'incomplete_requests', 'images', 'model_seconds', 'first_request_seconds',
              'target_bindings', 'depth_queries', 'camera_acquisitions')
    with prefix.with_suffix('.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['episodes'])
    colors = {'basic': '#537fa6', 'symbolic': '#d18a46'}
    labels = {'basic': 'Base', 'symbolic': 'Full'}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    budget = report['budget_s']
    for arm in ('basic', 'symbolic'):
        rows = sorted((r for r in report['episodes'] if r['arm'] == arm), key=lambda r: r['seed'])
        successes = sorted(r['capped_seconds'] for r in rows if r['success'])
        axes[0].step([0, *successes, budget], [0, *[100 * (i + 1) / len(rows) for i in range(len(successes))],
                     100 * len(successes) / len(rows)], where='post', color=colors[arm], label=labels[arm])
        x = np.arange(len(rows)) + (-0.12 if arm == 'basic' else 0.12)
        for i, row in enumerate(rows):
            axes[1].scatter(x[i], row['capped_seconds'], color=colors[arm],
                            marker='o' if row['success'] else 'x', s=42)
        axes[1].plot(x, [r['capped_seconds'] for r in rows], color=colors[arm], alpha=.3)
    axes[0].set(title='(a) Success within time budget', xlabel='Elapsed episode time (s)',
                ylabel='Success (%)', xlim=(0, budget), ylim=(-3, 103))
    axes[0].legend(frameon=False)
    axes[1].set(title='(b) All five paired layouts', xlabel='Layout index',
                ylabel='Capped completion time (s)', xticks=range(5), xticklabels=range(1, 6), ylim=(145, 185))
    axes[1].text(.02, .04, 'x: timeout; circle: success', transform=axes[1].transAxes, fontsize=8)
    base, full = (report['arms'][arm] for arm in ('basic', 'symbolic'))
    metrics = [('Completed model time', 'model_seconds'), ('Image attachments', 'images'),
               ('Camera acquisitions', 'camera_acquisitions')]
    ratios = [full[k] / base[k] for _, k in metrics]
    ratios.append(full['usage']['total_tokens'] / base['usage']['total_tokens'])
    names = [name for name, _ in metrics] + ['Completed total tokens']
    axes[2].barh(names, ratios, color=colors['symbolic'], alpha=.8)
    axes[2].axvline(1., color=colors['basic'], linestyle='--', linewidth=1)
    for i, ratio in enumerate(ratios):
        axes[2].text(ratio + .025, i, f'{ratio:.2f}', va='center', fontsize=9)
    axes[2].set(title='(c) Aggregate resource ratios', xlabel='Full / Base (Base = 1)', xlim=(0, 1.7))
    axes[2].invert_yaxis()
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'Explicit-target online pilot: 5 paired layouts, {budget:g} s, no resets', fontsize=12)
    fig.savefig(prefix.with_suffix('.pdf'))
    fig.savefig(prefix.with_suffix('.png'), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root)
    save(report, args.output_prefix)
    print(json.dumps({'arms': report['arms'], 'campaign_wall_seconds': report['campaign_wall_seconds']}, indent=2))
