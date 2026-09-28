#!/usr/bin/env python3
"""Export frozen single-policy evidence; never run policies or select on test data."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / 'docs/evaluations/single_policy'
STUDIES = {
    'bridge': ('geometry_bridge_contract_01', 'learning_curves/summary.json'),
    'stacking': ('geometry_cube_stack_contract_01', 'evaluation_100/summary.json'),
    'ring': ('ring_peg_comparison_01', 'summary.json'),
}
ARMS = {'basic': 'Base', 'symbolic': 'Full'}
TOKEN_KEYS = ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
              'output_tokens', 'reasoning_output_tokens', 'total_tokens')
SOURCES = {}


def read(path, expected=None):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f'Frozen result hash mismatch: {path}')
    SOURCES[str(path.relative_to(ROOT))] = digest
    return json.loads(data)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, context):
    check(math.isclose(actual, expected, abs_tol=1e-10),
          f'{context}: {actual} != {expected}')


def write_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def trial_row(task, study, replica, arm, revision, value, source):
    return dict(task=task, study=study, replica=replica, arm=ARMS[arm],
                revision=revision, seed=value['seed'], success=int(value['success']),
                status=value.get('status'), rollout_executed=value.get('rollout_executed'),
                error=value.get('policy_error') or value.get('error'),
                camera_calls=value.get('camera_calls'),
                policy_sim_seconds=value.get('policy_sim_seconds'),
                wall_seconds=value.get('wall_seconds'),
                policy_compute_cpu_seconds=value.get('cpu_profile', {}).get('policy_compute_cpu_seconds'),
                candidate_sha256=value.get('candidate_sha256'),
                source=str(source.relative_to(ROOT)), source_sha256=SOURCES[str(source.relative_to(ROOT))])


def collect():
    tests, training, requests, curves, means, resources = [], [], [], [], [], []
    sources = {}
    for task, (study, summary_name) in STUDIES.items():
        root = ROOT / 'outputs' / study
        summary_path = root / summary_name
        summary = read(summary_path)
        check(summary['status'] == 'completed', f'{task} is incomplete')
        check(len(summary['result_hashes']) == 4000, f'{task} slot count')
        sources[task] = str(summary_path.relative_to(ROOT))
        for path in [root / 'protocol.json', summary_path.parent / 'registration.json']:
            if path.exists():
                read(path)
        grouped = defaultdict(list)
        for relative, digest in sorted(summary['result_hashes'].items()):
            path = root / relative
            value = read(path, digest)
            replica = int(re.search(r'replica_(\d+)', relative)[1])
            arm = next(part for part in path.parts if part in ARMS)
            revision = int(re.search(r'revision_(\d+)', relative)[1])
            check(value['seed'] == int(re.search(r'seed_(\d+)', relative)[1]), 'Seed mismatch')
            row = trial_row(task, study, replica, arm, revision, value, path)
            row['layout_subset'] = ('original_20' if row['seed'] <= 2029091020 else 'additional_80') if task == 'stacking' else 'registered_100'
            tests.append(row)
            grouped[replica, arm, revision].append(row)
        expected_seeds = {r['seed'] for r in grouped[1, 'basic', 0]}
        check(len(expected_seeds) == 100 and len(grouped) == 40, 'Incomplete evaluation matrix')
        for rows in grouped.values():
            check(len(rows) == 100 and {r['seed'] for r in rows} == expected_seeds, 'Unpaired test layouts')
            check(len({r['candidate_sha256'] for r in rows}) == 1, 'Multiple candidates in one version')
        for replica in range(1, 6):
            directory = root / f'replica_{replica:02d}'
            resource_path = directory / 'resource_accounting.json'
            accounting = read(resource_path)
            common_train = read(directory / 'training_summary.json') if task != 'ring' else None
            for arm, label in ARMS.items():
                ref = summary['replicas'][directory.name][arm]
                turns = accounting['arms'][arm]['turns']
                check([turn['revision'] for turn in turns] == [1, 2, 3], 'Missing completed turn')
                cumulative = [0]
                for turn in turns:
                    usage = turn['usage']
                    equal(usage['input_tokens'] + usage['output_tokens'], usage['total_tokens'], 'Token decomposition')
                    requests.append(dict(task=task, study=study, replica=replica, arm=label,
                        revision=turn['revision'], attempt_kind='completed', usage_complete=True,
                        wall_seconds=turn['wall_seconds'], image_count=turn['image_count'], **usage,
                        source=str(resource_path.relative_to(ROOT))))
                    cumulative.append(cumulative[-1] + usage['total_tokens'])
                train_sets = []
                if task == 'ring':
                    train_curve = read(directory / arm / 'training_curve.json')
                else:
                    train_curve = common_train['curves'][arm]
                for revision in range(4):
                    if task == 'ring':
                        version = 'revision_000' if revision == 0 else f'round_{revision:03d}'
                        paths = sorted((directory / arm / version / 'trials').glob('seed_*/result.json'))
                        results = [(read(path), path) for path in paths]
                    else:
                        results = [(result, directory / 'training_summary.json')
                                   for result in train_curve[revision]['results']]
                    check(len(results) == 5, f'{task} {replica} {arm} R{revision}: training slots')
                    train_sets.append({value['seed'] for value, _ in results if value['success']})
                    equal(len(train_sets[-1]), train_curve[revision]['successes'], 'Training count')
                    for value, path in results:
                        training.append(trial_row(task, study, replica, arm, revision, value, path))
                selected = 0
                for revision in range(4):
                    if train_sets[revision] > train_sets[selected]:
                        selected = revision
                    reference_selected = ref.get('retained_revisions', ref.get('selected_revisions'))[revision]
                    check(selected == reference_selected, 'Training-only retention mismatch')
                    equal(cumulative[revision], ref.get('cumulative_generation_tokens', ref.get('cumulative_tokens'))[revision], 'Cumulative tokens')
                    for subset in (['all_100', 'additional_80'] if task == 'stacking' else ['all_100']):
                        raw_rows = grouped[replica, arm, revision]
                        retained_rows = grouped[replica, arm, selected]
                        if subset == 'additional_80':
                            raw_rows = [r for r in raw_rows if r['layout_subset'] == subset]
                            retained_rows = [r for r in retained_rows if r['layout_subset'] == subset]
                        raw = statistics.mean(r['success'] for r in raw_rows)
                        retained = statistics.mean(r['success'] for r in retained_rows)
                        reference = ref if subset == 'all_100' else summary['additional_80']['replicas'][directory.name][arm]
                        equal(raw, reference['raw'][revision], 'Raw success')
                        equal(retained, reference['retained'][revision], 'Retained success')
                        curves.append(dict(task=task, study=study, subset=subset, replica=replica,
                            arm=label, revision=revision, test_layouts=len(raw_rows), raw_sr=raw,
                            retained_sr=retained, retained_revision=selected,
                            training_successes=len(train_sets[revision]),
                            retained_training_successes=len(train_sets[selected]), training_layouts=5,
                            cumulative_completed_tokens=cumulative[revision]))
            for index, attempt in enumerate(accounting.get('incomplete_attempts', []), 1):
                requests.append(dict(task=task, study=study, replica=replica, arm=ARMS[attempt['arm']],
                    revision=attempt['revision'], attempt_kind=f'interrupted_{index}', usage_complete=False,
                    wall_seconds=attempt['wall_seconds'], image_count=attempt['image_count'],
                    **{key: None for key in TOKEN_KEYS}, source=str(resource_path.relative_to(ROOT))))
        for arm, label in ARMS.items():
            for subset in (['all_100', 'additional_80'] if task == 'stacking' else ['all_100']):
                for revision in range(4):
                    rows = [r for r in curves if (r['task'], r['arm'], r['subset'], r['revision']) == (task, label, subset, revision)]
                    row = dict(task=task, subset=subset, arm=label, revision=revision, chains=5,
                        distinct_test_layouts=rows[0]['test_layouts'])
                    for metric in ('raw_sr', 'retained_sr', 'cumulative_completed_tokens', 'training_successes', 'retained_training_successes'):
                        row[metric + '_mean'] = statistics.mean(r[metric] for r in rows)
                        row[metric + '_sd'] = statistics.stdev(r[metric] for r in rows)
                    reference = summary['mean_curves'][arm] if subset == 'all_100' else summary['additional_80']['mean_curves'][arm]
                    equal(row['raw_sr_mean'], reference['raw'][revision], 'Mean raw')
                    equal(row['retained_sr_mean'], reference['retained'][revision], 'Mean retained')
                    means.append(row)
            arm_requests = [r for r in requests if r['task'] == task and r['arm'] == label]
            complete = [r for r in arm_requests if r['usage_complete']]
            incomplete = [r for r in arm_requests if not r['usage_complete']]
            resources.append(dict(task=task, arm=label, chains=5, completed_requests=len(complete),
                interrupted_requests=len(incomplete), all_attempt_usage_complete=not incomplete,
                **{key: sum(r[key] for r in complete) for key in TOKEN_KEYS},
                completed_request_wall_seconds=sum(r['wall_seconds'] for r in complete),
                interrupted_request_wall_seconds=sum(r['wall_seconds'] for r in incomplete),
                completed_image_attachments=sum(r['image_count'] for r in complete),
                interrupted_image_attachments=sum(r['image_count'] for r in incomplete)))
    check(len(tests) == 12000 and len(training) == 600 and len(requests) == 92, 'Export totals')
    for task in STUDIES:
        for replica in range(1, 6):
            sets = [{r['seed'] for r in training if r['task'] == task and r['replica'] == replica and r['arm'] == arm} for arm in ARMS.values()]
            check(sets[0] == sets[1] and len(sets[0]) == 5, 'Training layout pairing')
            check(not sets[0] & {r['seed'] for r in tests if r['task'] == task}, 'Train/test overlap')
    return tests, training, requests, curves, means, resources, sources


def figures(curves, means):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'Base': '#416A9C', 'Full': '#BD6833'}
    for token_axis in (False, True):
        fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.4), sharey=True, layout='constrained')
        for column, task in enumerate(STUDIES):
            for row_index, metric in enumerate(('raw_sr', 'retained_sr')):
                ax = axes[row_index, column]
                for arm in ARMS.values():
                    for replica in range(1, 6):
                        points = [r for r in curves if (r['task'], r['arm'], r['replica'], r['subset']) == (task, arm, replica, 'all_100')]
                        xs = [r['cumulative_completed_tokens'] / 1000 if token_axis else r['revision'] for r in points]
                        ax.plot(xs, [r[metric] * 100 for r in points], color=colors[arm], alpha=.18, linewidth=.9)
                    points = [r for r in means if (r['task'], r['arm'], r['subset']) == (task, arm, 'all_100')]
                    xs = [r['cumulative_completed_tokens_mean'] / 1000 if token_axis else r['revision'] for r in points]
                    ax.plot(xs, [r[metric + '_mean'] * 100 for r in points], color=colors[arm], linewidth=2, marker='o', markersize=4, label=arm)
                ax.set_ylim(-3, 103)
                ax.set_yticks(range(0, 101, 20))
                ax.grid(axis='y', alpha=.17)
                if not token_axis:
                    ax.set_xticks(range(4), ['R0', 'R1', 'R2', 'R3'])
                if row_index == 0:
                    ax.set_title({'bridge': 'Bridge + cylinder', 'stacking': 'Three-cube stacking', 'ring': 'Ring over peg'}[task])
                if column == 0:
                    ax.set_ylabel(('Raw candidates' if row_index == 0 else 'Training-retained policies') + '\nTest success (%)')
                if row_index == 1:
                    ax.set_xlabel('Completed tokens / chain (thousands)' if token_axis else 'Policy revision')
        axes[0, 0].legend(frameon=False, loc='upper left')
        name = 'learning_curves_tokens' if token_axis else 'learning_curves'
        fig.savefig(OUT / f'{name}.pdf', metadata={'CreationDate': None})
        fig.savefig(OUT / f'{name}.png', dpi=180)
        plt.close(fig)


def iteration_cost_figure(means, requests):
    """Pair retained test success with cumulative completed generation tokens."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams['svg.fonttype'] = 'none'
    rows = []
    for row in means:
        if row['subset'] != 'all_100':
            continue
        task, arm, revision = row['task'], row['arm'], row['revision']
        turns = [r for r in requests if (r['task'], r['arm'], r['revision']) == (task, arm, revision)]
        completed = [r for r in turns if r['usage_complete']]
        check(len(completed) == (0 if revision == 0 else 5), 'Iteration token sample count')
        tokens = sum(r['total_tokens'] for r in completed) / 5
        previous = next((r['cumulative_completed_tokens_mean'] for r in means
                         if (r['task'], r['arm'], r['subset'], r['revision'])
                         == (task, arm, 'all_100', revision - 1)), 0)
        equal(tokens, row['cumulative_completed_tokens_mean'] - previous, 'Iteration token delta')
        history = [r for r in requests if r['task'] == task and r['arm'] == arm
                   and r['revision'] <= revision]
        cumulative_tokens = sum(r['total_tokens'] for r in history if r['usage_complete']) / 5
        equal(cumulative_tokens, row['cumulative_completed_tokens_mean'], 'Cumulative bar tokens')
        rows.append(dict(task=task, arm=arm, iteration=revision,
                         retained_sr_percent=100 * row['retained_sr_mean'],
                         mean_iteration_completed_tokens=tokens,
                         mean_cumulative_completed_tokens=cumulative_tokens, chains=5,
                         incomplete_requests=sum(not r['usage_complete'] for r in turns),
                         cumulative_incomplete_requests=sum(not r['usage_complete'] for r in history)))
    write_csv('iteration_sr_tokens.csv', rows)

    colors = {'Base': '#416A9C', 'Full': '#BD6833'}
    fills = {'Base': '#D7E2F0', 'Full': '#F0DBCC'}
    titles = {'bridge': '(a) Bridge + cylinder', 'stacking': '(b) Three-cube stacking', 'ring': '(c) Ring over peg'}
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.95))
    fig.subplots_adjust(left=.053, right=.947, bottom=.23, top=.77, wspace=.57)
    for ax, task in zip(axes, STUDIES):
        token_ax = ax.twinx()
        token_ax.set_ylim(0, 200)
        token_ax.set_yticks([0, 50, 100, 150, 200])
        token_ax.set_ylabel('Cumulative tokens (k)', fontsize=10, labelpad=7)
        token_ax.tick_params(axis='y', labelsize=9, length=3, color='.35')
        for spine in token_ax.spines.values():
            spine.set_visible(False)
        ax.set_zorder(token_ax.get_zorder() + 1)
        ax.patch.set_visible(False)
        for arm, offset, marker in [('Base', -.15, 'o'), ('Full', .15, 's')]:
            values = sorted([r for r in rows if r['task'] == task and r['arm'] == arm],
                            key=lambda r: r['iteration'])
            xs = [r['iteration'] for r in values]
            token_ax.bar([x + offset for x in xs], [r['mean_cumulative_completed_tokens']/1000 for r in values],
                         width=.27, color=fills[arm], edgecolor=colors[arm], linewidth=.6, zorder=1)
            ax.plot(xs, [r['retained_sr_percent'] for r in values], color=colors[arm],
                    marker=marker, markersize=6.5 if arm == 'Base' else 4.5,
                    markerfacecolor='white', markeredgewidth=1.4, linewidth=2,
                    zorder=4, clip_on=False)
        ax.set(xlim=(-.43, 3.43), ylim=(0, 105), xticks=range(4), yticks=range(0, 101, 20))
        ax.set_xlabel('Iteration', fontsize=10, labelpad=7)
        ax.set_ylabel('SR (%)', fontsize=10, labelpad=6)
        ax.set_title(titles[task], fontsize=11, pad=19)
        ax.tick_params(labelsize=9, length=3, color='.35')
        ax.grid(axis='y', color='#E4E7EB', linewidth=.65, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('.35')
        ax.spines['bottom'].set_linewidth(.7)
        for x in (0, 1):
            ax.annotate('', xy=(x, 1.065), xytext=(x, -.018), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='-|>', color='#343A40', lw=.85, mutation_scale=9),
                        annotation_clip=False, zorder=5)
    handles = [Line2D([], [], color=colors[arm], marker='o' if arm == 'Base' else 's',
                      markerfacecolor='white', lw=2, markersize=5, label=f'{arm} SR') for arm in colors]
    handles += [Patch(facecolor=fills[arm], edgecolor=colors[arm], linewidth=.6,
                      label=f'{arm} tokens') for arm in colors]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1.005),
               ncol=4, frameon=False, fontsize=10, handlelength=2.1, columnspacing=2.1)
    fig.text(.5, .045, 'Lines: Test SR (retained policy). Bars: cumulative tokens through each iteration, averaged over five chains.',
             ha='center', fontsize=8.4, color='#4A4F55')
    for suffix in ('pdf', 'svg', 'png'):
        options = {'metadata': {'CreationDate': None}} if suffix == 'pdf' else {}
        fig.savefig(OUT / f'iteration_sr_tokens.{suffix}', dpi=220, **options)
        if suffix == 'svg':
            path = OUT / f'iteration_sr_tokens.{suffix}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    tests, training, requests, curves, means, resources, sources = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, rows in [('test_outcomes.csv', tests), ('training_outcomes.csv', training),
                           ('model_requests.csv', requests), ('chain_curves.csv', curves),
                           ('mean_curves.csv', means), ('resources.csv', resources)]:
        write_csv(filename, rows)
    summary = dict(schema_version=1, source_summaries=sources, test_slots=len(tests),
                   training_slots=len(training), completed_model_requests=90,
                   interrupted_model_requests=2, mean_curves=means, resources=resources,
                   notes=['SR values are fractions; SD describes five chains, not a confidence interval.',
                          'Retained policies are recomputed from strict training-success-set inclusion.',
                          'All 12000 test result hashes and all curve values match the frozen summaries.',
                          'Stacking additional_80 is a subset, not an extra study.',
                          'Interrupted token usage is null; completed tokens exclude unknown consumption.'])
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    figures(curves, means)
    iteration_cost_figure(means, requests)
    provenance = dict(exporter='scripts/export_single_policy_results.py',
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_files=SOURCES,
        exported_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())
                        if p.suffix in {'.csv', '.json', '.pdf', '.svg', '.png'} and p.name != 'provenance.json'})
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: summary[key] for key in ('test_slots', 'training_slots', 'completed_model_requests', 'interrupted_model_requests')}, indent=2))
    for row in means:
        if row['revision'] == 3:
            print(row['task'], row['subset'], row['arm'], 'raw', row['raw_sr_mean'], 'retained', row['retained_sr_mean'])


if __name__ == '__main__':
    main()
