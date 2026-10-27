#!/usr/bin/env python3
"""Export the completed 50-pair study; never execute or modify experiments."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from export_online_results import arrow_axes
from online_interrupted_usage import audit_interrupted_requests, apply_recoveries

ROOT = Path(__file__).absolute().parents[1]
SOURCE = ROOT / 'outputs/online_extension_50_01/terminal_resume_01'
REPEAT = ROOT / 'outputs/online_targeted_repeat_01'
INTEGRATION = REPEAT / 'report_integration_01'
OUT = ROOT / 'docs/evaluations/online_50'
TASKS = ('tower', 'ring', 'relocation')
ARMS = ('Base', 'Full')
LABELS = ('Bridge-tower', 'Ring-on-peg', 'Tower relocation')
SHORT_LABELS = ('Bridge-tower', 'Ring', 'Relocation')
COLORS = {'Base': '#416A9C', 'Full': '#BD6833'}
FILLS = {'Base': '#D7E2F0', 'Full': '#F0DBCC'}
SOURCES = {}
MEASURES = ('success', 'capped_seconds', 'total_tokens', 'all_model_seconds',
            'completed_requests', 'images', 'camera_acquisitions', 'action_seconds')
COHORTS = ('original_5', 'new_45', 'pooled_50', 'sensitivity_valid_pairs_new',
           'sensitivity_valid_pairs_pooled', 'sensitivity_without_archive_recoveries_new',
           'original_attempts_50')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
            f'{label}: {actual} != {expected}')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path):
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    for archive in (SOURCE.parent, REPEAT):
        if path.is_relative_to(archive.resolve()):
            return str(archive.relative_to(ROOT) / path.relative_to(archive.resolve()))
    return str(path.relative_to(ROOT))


def read(path, expected=None):
    path = ROOT / path
    actual = digest(path)
    require(expected is None or actual == expected, f'Changed evidence: {path}')
    SOURCES[portable(path)] = actual
    return json.loads(path.read_text())


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for r in rows for k in r)),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def paired(rows):
    result = []
    for task, seed in sorted({(r['task'], r['seed']) for r in rows}):
        selected = [r for r in rows if (r['task'], r['seed']) == (task, seed)]
        require(len(selected) == 2 and {r['arm'] for r in selected} == set(ARMS), 'Incomplete pair')
        base, full = [next(r for r in selected if r['arm'] == arm) for arm in ARMS]
        require(base['initial_state_sha256'] == full['initial_state_sha256'], 'Initial states differ')
        require(base['cohort'] == full['cohort'], 'Cohorts differ within pair')
        result.append(dict(task=task, seed=seed, cohort=base['cohort'],
            base_success=base['success'], full_success=full['success'],
            base_capped_seconds=base['capped_seconds'], full_capped_seconds=full['capped_seconds'],
            full_minus_base_sr=full['success'] - base['success'],
            full_minus_base_seconds=full['capped_seconds'] - base['capped_seconds'],
            quality_eligible=all(r.get('timing_valid') is not False and
                                 r.get('gpu_dispatch_query_ok') is not False for r in selected),
            historical_quality_eligible=all(r.get('historical_timing_valid', r.get('timing_valid')) is not False and
                r.get('historical_gpu_dispatch_query_ok', r.get('gpu_dispatch_query_ok')) is not False for r in selected),
            replaced_pair=any(r.get('attempt_replaced', False) for r in selected),
            archive_recovered=any(r.get('archive_recovery_verified', False) for r in selected)))
    return result


def aggregate(rows):
    arms, differences = [], []
    for task in TASKS:
        for arm in ARMS:
            selected = [r for r in rows if r['task'] == task and r['arm'] == arm]
            require(bool(selected), 'Missing task/arm')
            record = dict(task=task, arm=arm, episodes=len(selected),
                successes=sum(r['success'] for r in selected),
                usage_complete=all(r['usage_complete'] for r in selected),
                episodes_with_unknown_tokens=sum(not r['usage_complete'] for r in selected),
                incomplete_requests=sum(r['incomplete_requests'] for r in selected),
                recovered_interrupted_requests=sum(r.get('recovered_interrupted_requests', 0) for r in selected),
                recovered_interrupted_tokens=sum(r.get('recovered_interrupted_tokens', 0) for r in selected),
                unknown_token_requests=sum(r.get('unknown_token_requests', r['incomplete_requests']) for r in selected),
                timing_flagged=sum(r.get('timing_valid') is False for r in selected),
                gpu_query_flagged=sum(r.get('gpu_dispatch_query_ok') is False for r in selected),
                archive_recovered=sum(r.get('archive_recovery_verified', False) for r in selected))
            record.update({'mean_' + k: statistics.mean(r[k] for r in selected) for k in MEASURES})
            arms.append(record)
        pairs = paired([r for r in rows if r['task'] == task])
        indices = np.random.default_rng(1050).integers(0, len(pairs), size=(10000, len(pairs)))
        for metric in ('full_minus_base_sr', 'full_minus_base_seconds'):
            values = np.array([p[metric] for p in pairs])
            low, high = np.quantile(values[indices].mean(axis=1), [.025, .975])
            differences.append(dict(task=task, pairs=len(pairs), metric=metric,
                mean=float(values.mean()), ci95_low=float(low), ci95_high=float(high)))
    return dict(arms=arms, paired_differences=differences)


def check_aggregate(actual, frozen):
    for row, ref in zip(actual['arms'], frozen['arms'], strict=True):
        require((row['task'], row['arm']) == (ref['task'], ref['arm']), 'Aggregate order')
        for key, value in ref.items():
            if key not in ('task', 'arm'):
                equal(row[key], value, key)
    for row, ref in zip(actual['paired_differences'], frozen['paired_differences'], strict=True):
        require((row['task'], row['metric']) == (ref['task'], ref['metric']), 'Difference order')
        for key in ('pairs', 'mean'):
            equal(row[key], ref[key], key)
        equal(row['ci95_low'], ref['ci95'][0], 'Lower interval')
        equal(row['ci95_high'], ref['ci95'][1], 'Upper interval')


def collect_records(records):
    episodes, requests = [], []
    for original in records:
        row = dict(original)
        result = read(row['source'], row['source_sha256'])
        equal(result['success'], row['success'], 'Success')
        equal(result['capped_completion_seconds'], row['capped_seconds'], 'Capped time')
        require(row['budget_seconds'] == 200 and result['budget_s'] == 200, 'Changed budget')
        require(row['status'] in ('success', 'timeout'), 'Unexpected status')
        require((row['success_seconds'] is not None) == bool(row['success']), 'Missing success time')
        equal(row['capped_seconds'], row['success_seconds'] if row['success'] else 200, 'Timeout handling')
        require(0 <= row['capped_seconds'] <= 200, 'Time outside budget')
        equal(row['input_tokens'] + row['output_tokens'], row['total_tokens'], 'Token arithmetic')
        directory = (ROOT / row['source']).parent
        current = []
        common = {k: row[k] for k in ('task', 'cohort', 'seed', 'arm')}
        for path in sorted(directory.glob('context_*/decision_*/receipt.json')):
            receipt = read(path)
            equal(receipt['usage']['input_tokens'] + receipt['usage']['output_tokens'],
                  receipt['usage']['total_tokens'], 'Receipt tokens')
            current.append(dict(**common, context=path.parent.parent.name, decision=path.parent.name,
                status='completed', usage_complete=True, wall_seconds=receipt['wall_seconds'],
                image_count=receipt['image_count'], **receipt['usage'], source=portable(path)))
        for path in sorted(directory.glob('context_*/decision_*/attempt_failure.json')):
            failure, invocation = read(path), read(path.parent / 'invocation.json')
            current.append(dict(**common, context=path.parent.parent.name, decision=path.parent.name,
                status=failure['type'], usage_complete=False, wall_seconds=failure['wall_seconds'],
                image_count=len(invocation['images']), input_tokens=None, cached_input_tokens=None,
                output_tokens=None, total_tokens=None, source=portable(path)))
        completed = [r for r in current if r['usage_complete']]
        equal(len(completed), row['completed_requests'], 'Completed requests')
        equal(len(current) - len(completed), row['incomplete_requests'], 'Interrupted requests')
        require(row['usage_complete'] == (len(current) == len(completed)), 'Usage completeness')
        for key in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'total_tokens'):
            equal(sum(r[key] for r in completed), row[key], key)
        equal(sum(r['wall_seconds'] for r in current), row['all_model_seconds'], 'Model wait total')
        equal(sum(r['image_count'] for r in current), row['images'], 'Image attachments')
        row['source'] = portable(row['source'])
        for key in ('audit_source',):
            if row.get(key):
                row[key] = portable(row[key])
        row['completed_images'] = sum(r['image_count'] for r in completed)
        row['incomplete_images'] = row['images'] - row['completed_images']
        episodes.append(row)
        requests.extend(current)
    return episodes, requests


def collect_original():
    audit = read(SOURCE / 'completion_audit.json')
    require(audit['status'] == 'verified_complete_with_preserved_quality_flags', 'Incomplete audit')
    summary = read(SOURCE / 'summary.json', audit['summary_sha256'])
    require(digest(SOURCE / 'episodes.csv') == audit['csv_sha256'], 'Changed source CSV')
    SOURCES[portable(SOURCE / 'episodes.csv')] = audit['csv_sha256']
    episodes, requests = collect_records(summary['records'])
    require(len(episodes) == 300, 'Expected 300 episodes')
    pairs = paired(episodes)
    require(len(pairs) == 150, 'Expected 150 pairs')
    eligible = {(p['task'], p['seed']) for p in pairs if p['quality_eligible']}
    recovered = {(p['task'], p['seed']) for p in pairs if p['archive_recovered']}
    for row in episodes:
        row['pair_in_quality_sensitivity'] = (row['task'], row['seed']) in eligible
    subsets = {
        'original_5': [r for r in episodes if r['cohort'] == 'original_5'],
        'new_45': [r for r in episodes if r['cohort'] == 'new_45'],
        'pooled_50': episodes,
        'sensitivity_valid_pairs_new': [r for r in episodes if r['cohort'] == 'new_45' and r['pair_in_quality_sensitivity']],
        'sensitivity_valid_pairs_pooled': [r for r in episodes if r['pair_in_quality_sensitivity']],
        'sensitivity_without_archive_recoveries_new': [r for r in episodes if r['cohort'] == 'new_45' and
            r['pair_in_quality_sensitivity'] and (r['task'], r['seed']) not in recovered],
    }
    aggregates = {}
    for cohort, rows in subsets.items():
        aggregates[cohort] = aggregate(rows)
        check_aggregate(aggregates[cohort], summary[cohort])
    epochs = {}
    for epoch, ref in summary['by_runtime_epoch'].items():
        epochs[epoch] = aggregate([r for r in episodes if r.get('runtime_epoch') == epoch])
        check_aggregate(epochs[epoch], ref)
    require(sum(r.get('timing_valid') is False for r in episodes) == 4, 'Timing flags')
    require(sum(r.get('gpu_dispatch_query_ok') is False for r in episodes) == 16, 'GPU flags')
    require(sum(not r['pair_in_quality_sensitivity'] for r in episodes) == 18, 'Quality pair union')
    require(sum(r.get('archive_recovery_verified', False) for r in episodes) == 2, 'Archive labels')
    # Validate the frozen receipt-based totals first, then add only independently
    # verified terminal usage. The original results and decision counts stay fixed.
    usage_audit = audit_interrupted_requests(requests)
    SOURCES.update(usage_audit['source_files'])
    apply_recoveries(episodes, requests, usage_audit)
    aggregates = {cohort: aggregate(rows) for cohort, rows in subsets.items()}
    epochs = {epoch: aggregate([r for r in episodes if r.get('runtime_epoch') == epoch])
              for epoch in summary['by_runtime_epoch']}
    return summary, episodes, requests, pairs, aggregates, epochs, usage_audit


def substitute_pairs(originals, repeats, selected_keys):
    """Use every selected repeat, including regressions, and preserve pair identity."""
    key = lambda r: (r['task'], r['seed'], r['arm'])
    previous = {key(r): r for r in originals}
    replacement = {key(r): r for r in repeats}
    require(len(previous) == len(originals), 'Duplicate original slot')
    require(len(replacement) == len(repeats), 'Duplicate repeat slot')
    expected = {(task, seed, arm) for task, seed in selected_keys for arm in ARMS}
    require(set(replacement) == expected and expected <= set(previous), 'Repeat coverage differs from selected pairs')
    merged, changes = [], []
    for old in originals:
        new = dict(replacement.get(key(old), old))
        changed = key(old) in expected
        if changed:
            require(new['initial_state_sha256'] == old['initial_state_sha256'], 'Repeated initial state differs')
            require(new['timing_valid'] and new['gpu_dispatch_query_ok'], 'Repeat quality audit failed')
            require(new['budget_seconds'] == old['budget_seconds'] == 200, 'Repeated budget differs')
            new['cohort'] = old['cohort']
            changes.append(dict(task=old['task'], seed=old['seed'], arm=old['arm'],
                original_source=old['source'], original_source_sha256=old['source_sha256'],
                selected_source=new['source'], selected_source_sha256=new['source_sha256'],
                original_success=old['success'], selected_success=new['success'],
                original_capped_seconds=old['capped_seconds'], selected_capped_seconds=new['capped_seconds'],
                original_tokens=old['total_tokens'], selected_tokens=new['total_tokens'],
                original_usage_complete=old['usage_complete'], selected_usage_complete=new['usage_complete']))
        new.update(attempt_replaced=changed, selected_attempt='targeted_repeat' if changed else 'original',
            historical_timing_valid=old.get('timing_valid'),
            historical_gpu_dispatch_query_ok=old.get('gpu_dispatch_query_ok'),
            pair_in_quality_sensitivity=(old['task'], old['seed']) not in selected_keys)
        merged.append(new)
    paired(merged)
    return merged, changes


def collect():
    frozen, original, original_requests, _, original_aggregates, _, original_usage = collect_original()
    registration = read(INTEGRATION / 'registration.json')
    require(digest(INTEGRATION / 'registration.json') ==
            (INTEGRATION / 'registration.sha256').read_text().strip(), 'Integration registration changed')
    for name, sha in registration['source_sha256'].items():
        path = ROOT / name
        require(digest(path) == sha, 'Integration evidence changed: ' + name)
        SOURCES[portable(path)] = sha
    repeated = read(REPEAT / 'summary.json')
    require(repeated['status'] == 'completed' and repeated['episodes'] == 18, 'Incomplete repeat')
    require(read(REPEAT / 'progress.json')['audited'] == 18, 'Repeat audits incomplete')
    for path in sorted((REPEAT / 'audits/final').rglob('*.json')):
        audit = read(path)
        if isinstance(audit, dict):
            require(audit['status'] == 'passed', 'Failed final repeat audit')
    original_by_key = {(r['task'], r['seed'], r['arm']): r for r in original}
    raw_repeats = []
    for row in repeated['records']:
        old = original_by_key[(row['task'], row['seed'], row['arm'])]
        require(row['original_source_sha256'] == old['source_sha256'], 'Repeat refers to different original')
        raw_repeats.append(dict(row, cohort=old['cohort']))
    repeats, repeat_requests = collect_records(raw_repeats)
    repeat_usage = audit_interrupted_requests(repeat_requests)
    SOURCES.update(repeat_usage['source_files'])
    apply_recoveries(repeats, repeat_requests, repeat_usage)
    selected_keys = {(r['task'], r['seed']) for r in original if not r['pair_in_quality_sensitivity']}
    require(len(selected_keys) == 9, 'Expected nine historical quality pairs')
    registered = read(REPEAT / 'registration.json')['slots']
    require({(s['task'], s['seed']) for s in registered} == selected_keys, 'Repeat selection changed')
    episodes, changes = substitute_pairs(original, repeats, selected_keys)
    requests = [r for r in original_requests if (r['task'], r['seed']) not in selected_keys] + repeat_requests
    usage_records = [r for r in original_usage['records'] if (r['task'], r['seed']) not in selected_keys] + repeat_usage['records']
    recovered = [r for r in usage_records if r['recovered']]
    usage_audit = dict(status='verified', interrupted_requests=len(usage_records),
        recovered_requests=len(recovered), unknown_requests=len(usage_records) - len(recovered),
        recovered_total_tokens=sum(r['usage']['total_tokens'] for r in recovered), records=usage_records,
        source_files={**original_usage['source_files'], **repeat_usage['source_files']},
        reason_counts={kind: sum(r['interruption_type'] == kind for r in usage_records)
                       for kind in sorted({r['interruption_type'] for r in usage_records})},
        interpretation='Selected 300-episode accounting; superseded attempts are reported separately. Unknown usage is not zero.')
    pairs = paired(episodes)
    require(len(episodes) == 300 and len(pairs) == 150, 'Replacement changed sample size')
    subsets = {
        'original_5': [r for r in episodes if r['cohort'] == 'original_5'],
        'new_45': [r for r in episodes if r['cohort'] == 'new_45'],
        'pooled_50': episodes,
        'sensitivity_valid_pairs_new': [r for r in episodes if r['cohort'] == 'new_45' and r['pair_in_quality_sensitivity']],
        'sensitivity_valid_pairs_pooled': [r for r in episodes if r['pair_in_quality_sensitivity']],
        'sensitivity_without_archive_recoveries_new': [r for r in episodes if r['cohort'] == 'new_45' and
            r['pair_in_quality_sensitivity'] and not r.get('archive_recovery_verified', False)],
        'original_attempts_50': original,
    }
    aggregates = {cohort: aggregate(rows) for cohort, rows in subsets.items()}
    for name in ('original_5', 'sensitivity_valid_pairs_new', 'sensitivity_valid_pairs_pooled',
                 'sensitivity_without_archive_recoveries_new'):
        require(aggregates[name] == original_aggregates[name], 'Unaffected cohort changed: ' + name)
    epochs = {epoch: aggregate([r for r in episodes if r.get('runtime_epoch') == epoch])
              for epoch in sorted({r['runtime_epoch'] for r in episodes if r.get('runtime_epoch')})}
    superseded = [r for r in original if (r['task'], r['seed']) in selected_keys]
    integration = dict(registration=portable(INTEGRATION / 'registration.json'),
        replaced_pairs=9, replaced_episodes=18, changes=changes,
        original_attempts=original, superseded_attempts=superseded,
        superseded_recorded_tokens=sum(r['total_tokens'] for r in superseded),
        superseded_unknown_requests=sum(r['unknown_token_requests'] for r in superseded),
        repeat_recorded_tokens=sum(r['total_tokens'] for r in repeats),
        repeat_unknown_requests=repeat_usage['unknown_requests'],
        interpretation=registration['selection'], timing=registration['timing'])
    return frozen, episodes, requests, pairs, aggregates, epochs, usage_audit, integration


def curves_for(episodes):
    curves = []
    for task in TASKS:
        for arm in ARMS:
            rows = [r for r in episodes if r['task'] == task and r['arm'] == arm]
            times = sorted({0., 200., *(r['success_seconds'] for r in rows if r['success'])})
            for time in times:
                n = sum(r['success'] and r['success_seconds'] <= time for r in rows)
                curves.append(dict(task=task, arm=arm, budget_seconds=time, successes=n,
                                   episodes=len(rows), sr_percent=100 * n / len(rows)))
    return curves


def save(fig, name):
    import matplotlib.pyplot as plt
    for suffix in ('pdf', 'png', 'svg'):
        metadata = {'CreationDate': None} if suffix == 'pdf' else {'Date': None} if suffix == 'svg' else {}
        fig.savefig(OUT / f'{name}.{suffix}', dpi=300, metadata=metadata)
        if suffix == 'svg':
            path = OUT / f'{name}.{suffix}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def plot(episodes, summaries, pairs, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'online-50'})
    bars = [Patch(facecolor=FILLS[a], edgecolor=COLORS[a], label=a) for a in ARMS]
    lines = [Line2D([], [], color=COLORS[a], lw=2, label=a) for a in ARMS]
    pooled = summaries['pooled_50']['arms']
    foot = '50 paired layouts per task. Audited repeats replace both arms of 9 timing/GPU-flagged pairs.'

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.95))
    fig.subplots_adjust(left=.065, right=.982, bottom=.25, top=.77, wspace=.36)
    for index, task in enumerate(TASKS):
        for arm, offset in [('Base', -.18), ('Full', .18)]:
            row = next(r for r in pooled if r['task'] == task and r['arm'] == arm)
            ys = [100 * row['mean_success'], row['mean_capped_seconds'], row['mean_total_tokens'] / 1000]
            labels = [f"{row['successes']}/50", f'{ys[1]:.1f}', f'{ys[2]:.1f}' + ('†' if not row['usage_complete'] else '')]
            for ax, y, label, pad in zip(axes, ys, labels, [2.5, 5, 6]):
                ax.bar(index + offset, y, width=.31, color=FILLS[arm], edgecolor=COLORS[arm], lw=1.1, zorder=2)
                ax.text(index + offset, y + pad, label, fontsize=7.8, ha='center', color=COLORS[arm])
    for ax, title, label, limit, ticks in zip(axes,
            ['(a) Success within 200 s', '(b) Capped completion time', '(c) Recorded tokens'],
            ['SR (%)', 'Time per episode (s)', 'Tokens per episode (k)'],
            [115, 235, 270], [range(0, 101, 20), range(0, 201, 50), range(0, 251, 50)]):
        arrow_axes(ax)
        ax.set_title(title, fontsize=11, pad=19)
        ax.set_ylabel(label, fontsize=10)
        ax.set(xticks=range(3), xticklabels=SHORT_LABELS, xlim=(-.55, 2.55), ylim=(0, limit), yticks=list(ticks))
    fig.legend(handles=bars, loc='upper center', ncol=2, frameon=False, fontsize=10)
    fig.text(.5, .085, foot, ha='center', fontsize=8.1, color='#4A4F55')
    fig.text(.5, .032, 'Time and tokens: episode means; timeouts = 200 s.  † Unknown interrupted tokens excluded.',
             ha='center', fontsize=8.1, color='#4A4F55')
    save(fig, 'outcome_bars')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.95))
    fig.subplots_adjust(left=.065, right=.978, bottom=.24, top=.77, wspace=.30)
    for index, (ax, task) in enumerate(zip(axes, TASKS)):
        for arm in ARMS:
            rows = [r for r in curves if r['task'] == task and r['arm'] == arm]
            ax.step([r['budget_seconds'] for r in rows], [r['sr_percent'] for r in rows],
                    where='post', color=COLORS[arm], lw=2)
        arrow_axes(ax)
        ax.set(xlabel='Elapsed wall time (s)', ylabel='SR (%)', xlim=(0, 203), ylim=(0, 105),
               xticks=[0, 50, 100, 150, 200], yticks=range(0, 101, 20))
        ax.set_title(f'({chr(97 + index)}) {LABELS[index]}', fontsize=11, pad=19)
    fig.legend(handles=lines, loc='upper center', ncol=2, frameon=False, fontsize=10)
    fig.text(.5, .075, foot, ha='center', fontsize=8.1, color='#4A4F55')
    fig.text(.5, .027, 'Steps mark first verified success in the same 200-second episodes; timeouts remain failures.',
             ha='center', fontsize=8.1, color='#4A4F55')
    save(fig, 'success_vs_time')

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    fig.subplots_adjust(left=.075, right=.98, bottom=.25, top=.76, wspace=.29)
    for index, task in enumerate(TASKS):
        for arm, offset in [('Base', -.18), ('Full', .18)]:
            rows = sorted([r for r in episodes if r['task'] == task and r['arm'] == arm], key=lambda r: r['seed'])
            # Identical bounded offsets for both arms make paired layouts reproducible.
            jitter = np.random.default_rng(1050 + index).permutation(np.linspace(-.12, .12, len(rows)))
            for ax, key, scale in zip(axes, ['all_model_seconds', 'total_tokens'], [1, 1000]):
                ax.bar(index + offset, statistics.mean(r[key] / scale for r in rows), width=.31,
                       color=FILLS[arm], edgecolor=COLORS[arm], lw=.8, zorder=2)
                for j, row in enumerate(rows):
                    unknown = key == 'total_tokens' and not row['usage_complete']
                    ax.scatter(index + offset + jitter[j], row[key] / scale, s=18 if unknown else 12,
                               marker='^' if unknown else 'o', facecolors='white', edgecolors=COLORS[arm],
                               lw=.8, alpha=.85 if unknown else .65, zorder=4 if unknown else 3)
    for ax, title, ylabel in zip(axes, ['(a) Model-request time', '(b) Recorded token consumption'],
                                ['Model time per episode (s)', 'Tokens per episode (k)']):
        arrow_axes(ax)
        ax.set_title(title, fontsize=11, pad=19)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set(xticks=range(3), xticklabels=SHORT_LABELS, xlim=(-.55, 2.55), ylim=(0, ax.get_ylim()[1] * 1.04))
    handles = bars + [Line2D([], [], marker=m, ls='', markerfacecolor='white', color='.3', label=l)
                     for m, l in [('o', 'Episode'), ('^', 'Unknown extra tokens')]]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=10)
    fig.text(.5, .080, 'Bars: mean of 50 episodes per arm/task. Points: selected episodes, including timeouts and 9 paired replacements.',
             ha='center', fontsize=8.1, color='#4A4F55')
    fig.text(.5, .028, 'Model time includes interrupted waits. Token totals exclude unknown interrupted usage.',
             ha='center', fontsize=8.1, color='#4A4F55')
    save(fig, 'model_resources')

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.65))
    fig.subplots_adjust(left=.15, right=.97, bottom=.25, top=.76, wspace=.36)
    cohorts = [('pooled_50', 'Revised 50 pairs', '#BD6833', 'o', -.19),
               ('new_45', 'New 45 pairs', '#687585', 's', 0),
               ('sensitivity_valid_pairs_pooled', 'Quality sensitivity', '#267C70', 'D', .19)]
    for ax, metric, scale, title, xlabel in zip(axes,
            ['full_minus_base_sr', 'full_minus_base_seconds'], [100, 1],
            ['(a) Success difference', '(b) Capped-time difference'],
            ['Full − Base SR (percentage points)', 'Full − Base time (s)']):
        for cohort, label, color, marker, offset in cohorts:
            for i, task in enumerate(TASKS):
                row = next(r for r in summaries[cohort]['paired_differences'] if r['task'] == task and r['metric'] == metric)
                x, low, high = [row[k] * scale for k in ('mean', 'ci95_low', 'ci95_high')]
                ax.errorbar(x, i + offset, xerr=[[x - low], [high - x]], fmt=marker, color=color,
                            markersize=4.5, capsize=2.5, lw=1.15, label=label if i == 0 else None)
        ax.axvline(0, color='.5', ls=':', lw=.8)
        ax.set(yticks=range(3), yticklabels=LABELS, ylim=(2.55, -.55), xlabel=xlabel)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('.5')
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#E4E7EB', lw=.65)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=11, pad=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=9.5)
    fig.text(.5, .075, '95% paired-layout bootstrap intervals (10,000 resamples). Quality sensitivity excludes whole flagged pairs:',
             ha='center', fontsize=8.1, color='#4A4F55')
    fig.text(.5, .028, '48 Bridge-tower, 46 Ring, 47 Relocation pairs. Positive SR and negative time differences favor Full.',
             ha='center', fontsize=8.1, color='#4A4F55')
    save(fig, 'paired_effects')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.subplots_adjust(left=.065, right=.98, bottom=.25, top=.78, wspace=.25)
    for index, (task, ax) in enumerate(zip(TASKS, axes)):
        selected = [p for p in pairs if p['task'] == task]
        selected.sort(key=lambda p: (p['cohort'] != 'original_5', p['seed']))
        for j, pair in enumerate(selected, 1):
            ax.plot([j - .12, j + .12], [pair['base_capped_seconds'], pair['full_capped_seconds']], color='.8', lw=.65)
            for arm, offset in [('Base', -.12), ('Full', .12)]:
                ax.scatter(j + offset, pair[f'{arm.lower()}_capped_seconds'], color=COLORS[arm],
                    marker='o' if pair[f'{arm.lower()}_success'] else 'x', s=12, linewidths=.7, zorder=3)
            if pair.get('replaced_pair', False):
                ax.scatter(j, 207, marker='v', color='.3', s=13)
        ax.axhline(200, color='.5', ls=':', lw=.8)
        ax.axvline(5.5, color='.7', ls=':', lw=.7)
        arrow_axes(ax)
        ax.set(xlabel='Paired layout (original 5, then new 45)', xticks=[1, 10, 20, 30, 40, 50], xlim=(0, 51), ylim=(0, 216))
        ax.set_title(f'({chr(97 + index)}) {LABELS[index]}', fontsize=10, pad=17)
    axes[0].set_ylabel('Capped completion time (s)')
    fig.legend(handles=lines, loc='upper center', ncol=2, frameon=False)
    fig.text(.5, .055, 'Circle: success. Cross: timeout. Triangle above 200 s: both arms replaced by their audited repeat.',
             ha='center', fontsize=8.1, color='#4A4F55')
    save(fig, 'paired_completion_times')


def main():
    frozen, episodes, requests, pairs, aggregates, epochs, usage_audit, integration = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    curves = curves_for(episodes)
    write_csv('episodes.csv', episodes)
    write_csv('model_requests.csv', requests)
    write_csv('paired_outcomes.csv', pairs)
    write_csv('success_vs_time.csv', curves)
    write_csv('arm_summary.csv', aggregates['pooled_50']['arms'])
    write_csv('cohort_summary.csv', [dict(cohort=c, **r) for c in COHORTS for r in aggregates[c]['arms']])
    write_csv('paired_differences.csv', [dict(cohort=c, **r) for c in COHORTS for r in aggregates[c]['paired_differences']])
    write_csv('runtime_epochs.csv', [dict(runtime_epoch=e, **r) for e in epochs for r in epochs[e]['arms']])
    write_csv('quality_flags.csv', [r for r in episodes if not r['pair_in_quality_sensitivity'] or r.get('archive_recovery_verified')])
    write_csv('replacements.csv', integration['changes'])
    write_csv('original_attempts.csv', integration['original_attempts'])
    write_csv('superseded_attempts.csv', integration['superseded_attempts'])
    (OUT / 'replacement_manifest.json').write_text(json.dumps({k: v for k, v in integration.items()
        if k not in ('original_attempts', 'superseded_attempts')}, indent=2) + '\n')
    (OUT / 'interrupted_usage_audit.json').write_text(json.dumps(usage_audit, indent=2) + '\n')
    write_csv('recovered_interrupted_usage.csv', [dict(
        **{k: v for k, v in r.items() if k != 'usage'}, **r['usage'])
        for r in usage_audit['records'] if r['recovered']])
    for name in ('outcome_bars.csv', 'model_resources.csv'):
        write_csv(name, aggregates['pooled_50']['arms'])
    summary = dict(status='completed_with_audited_paired_replacements', episodes=len(episodes), paired_layouts=len(pairs),
        completed_requests=sum(r['completed_requests'] for r in episodes),
        incomplete_requests=sum(r['incomplete_requests'] for r in episodes),
        unknown_token_requests=usage_audit['unknown_requests'],
        recovered_interrupted_requests=usage_audit['recovered_requests'],
        recovered_interrupted_tokens=usage_audit['recovered_total_tokens'],
        completed_receipt_total_tokens=sum(r['receipt_total_tokens'] for r in episodes),
        total_recorded_tokens=sum(r['total_tokens'] for r in episodes),
        quality=dict(timing_flagged=sum(r.get('timing_valid') is False for r in episodes),
            gpu_dispatch_query_failed=sum(r.get('gpu_dispatch_query_ok') is False for r in episodes),
            historical_timing_flagged=frozen['quality']['timing_flagged'],
            historical_gpu_dispatch_query_failed=frozen['quality']['gpu_dispatch_query_failed'],
            replaced_pairs=9, replaced_episodes=18, archive_recovered_episodes=2),
        cohorts=aggregates, runtime_epochs=epochs,
        runtime_change=frozen['runtime_change'], archive_recovery=frozen['archive_recovery'],
        infrastructure_overhead=frozen['infrastructure_overhead'],
        replacement={k: v for k, v in integration.items() if k not in ('changes', 'original_attempts', 'superseded_attempts')},
        campaign_recorded_tokens_including_superseded_and_startup=sum(r['total_tokens'] for r in episodes) +
            integration['superseded_recorded_tokens'] + frozen['infrastructure_overhead']['total_recorded_tokens'],
        interval_method='Paired-layout percentile bootstrap, 10000 resamples, seed 1050 per task; no multiplicity correction.',
        scope='300 selected episodes: 282 original and 18 audited repeats replacing nine whole pairs. '
            'Reporting amendment made after inspecting repeats; every selected repeat is used, including regressions. '
            'Original outcomes and superseded costs remain separate. No component ablations.')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plot(episodes, aggregates, pairs, curves)
    provenance = dict(exporter=str(Path(__file__).relative_to(ROOT)), exporter_sha256=digest(Path(__file__)),
        usage_auditor_sha256=digest(ROOT / 'scripts/online_interrupted_usage.py'),
        style_source_sha256=digest(ROOT / 'scripts/export_online_results.py'), source_files=SOURCES,
        source_summary_sha256=digest(SOURCE / 'summary.json'),
        exported_files={p.name: digest(p) for p in sorted(OUT.iterdir())
                        if p.suffix in {'.csv', '.json', '.pdf', '.png', '.svg'} and p.name != 'provenance.json'})
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: summary[k] for k in ('status', 'episodes', 'paired_layouts', 'completed_requests',
                                            'incomplete_requests', 'total_recorded_tokens')}, indent=2))
    print(f'Exported to {OUT}')


if __name__ == '__main__':
    main()
