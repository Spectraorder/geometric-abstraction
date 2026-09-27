"""Recompute test curves for policies retained using training outcomes only.

This analysis reads frozen artifacts. It launches no model calls or rollouts and
keeps the original full-revision summary and figures unchanged.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from evaluate_geometry_learning_curves import check
from geometry_iteration import RULE, retention


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recompute(root, curve_directory):
    registration = check(root, curve_directory)
    original = read(curve_directory/'summary.json')
    if original['registration_sha256'] != digest(curve_directory/'registration.json'):
        raise ValueError('Curve summary does not match registration')
    results = {}
    for entry in registration['records']:
        relative = entry['trial']+'/result.json'
        path = root/relative
        if digest(path) != original['result_hashes'][relative]:
            raise ValueError('Audited result changed: '+relative)
        result = read(path)
        if (result['seed'] != entry['seed'] or type(result['success']) is not bool
                or result['candidate_sha256'] != entry['candidate_sha256']):
            raise ValueError('Result identity or candidate mismatch')
        key = (entry['replica'], entry['arm'], entry['revision'])
        group = results.setdefault(key, {})
        if entry['seed'] in group:
            raise ValueError('Duplicate test layout within a revision')
        group[entry['seed']] = result['success']
    protocol = read(root/'protocol.json')
    seeds = set(protocol['heldout_seeds'])
    if any(set(group) != seeds for group in results.values()):
        raise ValueError('Test layout sets differ between revisions')
    rows = []
    inputs = {'curve_summary_sha256': digest(curve_directory/'summary.json'),
              'registration_sha256': digest(curve_directory/'registration.json'),
              'replicas': {}}
    for replica in protocol['replicas']:
        name = replica['directory']
        directory = root/name
        manifest = read(directory/'manifest.json')
        if manifest['retention_rule'] != RULE:
            raise ValueError('Unexpected training retention rule')
        training = read(directory/'training_summary.json')
        resources = read(directory/'resource_accounting.json')
        inputs['replicas'][name] = {file: digest(directory/file) for file in
                                   ('training_summary.json', 'resource_accounting.json')}
        for arm in ('basic', 'symbolic'):
            train_curve = training['curves'][arm]
            decisions = retention(train_curve)
            tokens = [0]
            for turn in resources['arms'][arm]['turns']:
                if turn['revision'] != len(tokens):
                    raise ValueError('Generation cost revisions are not consecutive')
                tokens.append(tokens[-1]+turn['usage']['total_tokens'])
            if len(tokens) != len(decisions):
                raise ValueError('Missing generation costs')
            for decision in decisions:
                iteration = decision['revision']
                selected = decision['selected_revision']
                scores = results[name, arm, selected]
                selected_train = train_curve[selected]
                row = {'replica': name, 'arm': arm, 'iteration': iteration,
                       'selected_revision': selected,
                       'training_successes': sum(r['success'] for r in selected_train['results']),
                       'training_trials': len(selected_train['results']),
                       'test_successes': sum(scores.values()), 'test_trials': len(scores),
                       'test_sr': statistics.mean(scores.values()),
                       'cumulative_generation_tokens': tokens[iteration]}
                previous = original['replicas'][name][arm]
                if (selected != previous['retained_revisions'][iteration]
                        or abs(row['test_sr']-previous['retained'][iteration]) > 1e-12
                        or tokens[iteration] != previous['cumulative_generation_tokens'][iteration]):
                    raise ValueError('Recomputed retention disagrees with the audited curve')
                rows.append(row)
    means = []
    for iteration in sorted({r['iteration'] for r in rows}):
        row = {'iteration': iteration}
        for arm in ('basic', 'symbolic'):
            subset = [r for r in rows if r['arm'] == arm and r['iteration'] == iteration]
            row[arm+'_sr'] = statistics.mean(r['test_sr'] for r in subset)
            row[arm+'_tokens'] = statistics.mean(r['cumulative_generation_tokens'] for r in subset)
        row['full_minus_base_pp'] = 100*(row['symbolic_sr']-row['basic_sr'])
        means.append(row)
    return {'schema': 'training_retained_test_curve_v1', 'status': 'verified',
            'selection_rule': RULE,
            'selection_scope': 'Per chain, using training success sets only; ties keep the incumbent.',
            'test_selection_used': False, 'new_model_calls': 0, 'new_rollouts': 0,
            'replicas_per_arm': len(protocol['replicas']), 'shared_test_layouts': len(seeds),
            'audited_result_files_verified': len(registration['records']),
            'token_scope': 'Cumulative input plus output usage, including rejected proposals.',
            'analysis_scope': registration['scope'], 'inputs': inputs,
            'mean_curve': means, 'rows': rows}


def export(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if (output/'summary.json').exists() and read(output/'summary.json').get('schema') != report['schema']:
        raise ValueError('Refusing to replace an unrelated or frozen summary')
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    for name, rows in (('replica_curves.csv', report['rows']), ('mean_curve.csv', report['mean_curve'])):
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained')
    for axis, token_axis in zip(axes, (False, True)):
        for arm, label, color in (('basic', 'Base', '#2463a5'), ('symbolic', 'Full geometry', '#c75c24')):
            subset = [r for r in report['rows'] if r['arm'] == arm]
            for replica in sorted({r['replica'] for r in subset}):
                rows = [r for r in subset if r['replica'] == replica]
                x = [r['cumulative_generation_tokens']/1000 if token_axis else r['iteration'] for r in rows]
                axis.plot(x, [100*r['test_sr'] for r in rows], color=color, alpha=.16, linewidth=1)
            means = report['mean_curve']
            x = [r[arm+'_tokens']/1000 if token_axis else r['iteration'] for r in means]
            y = [100*r[arm+'_sr'] for r in means]
            axis.plot(x, y, '-o', color=color, linewidth=2.6, label=label, zorder=3)
            if not token_axis:
                for iteration, sr in zip(x[1:], y[1:]):
                    axis.annotate(f'{sr:.1f}%', (iteration, sr), xytext=(0, 9 if arm == 'symbolic' else -18),
                                  textcoords='offset points', ha='center', color=color, fontsize=10)
        axis.axhline(80, color='gray', linestyle=':', linewidth=1)
        axis.set_ylim(0, 100)
        axis.set_ylabel('Test success rate (%)')
        axis.set_xlabel('Cumulative generation tokens (thousands)' if token_axis else 'Completed improvement rounds')
        if not token_axis:
            axis.set_xticks([r['iteration'] for r in report['mean_curve']])
        axis.grid(alpha=.2)
        axis.legend(loc='lower right')
    axes[0].set_title('Training-retained best policy')
    axes[1].set_title('Retained performance vs cumulative token budget')
    figure.suptitle('Five paired training chains; 100 shared test layouts\nSelection uses training outcomes only. Thin lines: chains; thick lines: means.', fontsize=12)
    figure.savefig(output/'retained_curve.png', dpi=180)
    figure.savefig(output/'retained_curve.pdf')
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    curve_directory = root/'learning_curves'
    if args.output.resolve() == curve_directory:
        parser.error('Use a separate directory to preserve the original curve artifacts')
    report = recompute(root, curve_directory)
    export(report, args.output)
    print(json.dumps(report['mean_curve'], indent=2))
