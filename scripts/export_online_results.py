#!/usr/bin/env python3
"""Consolidate audited online studies without modifying or executing experiments."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / 'docs/evaluations/online'
STUDIES = {'tower': 'online_compact_tower_200_01', 'ring': 'online_ring_200_01',
           'relocation': 'online_relocation_200_01'}
LABELS = {'tower': 'Bridge-tower', 'ring': 'Ring over peg', 'relocation': 'Tower relocation'}
ARMS = {'basic': 'Base', 'symbolic': 'Full'}
TOKENS = ('input_tokens', 'cached_input_tokens', 'output_tokens', 'total_tokens')
SOURCES = {}


def read(path, expected=None):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(expected is None or expected == digest, f'Hash mismatch: {path}')
    SOURCES[str(path.relative_to(ROOT))] = digest
    return json.loads(raw)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(a, b, label):
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), f'{label}: {a} != {b}')


def csv_file(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in rows for k in row)), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect():
    episodes, requests, totals, pairs, curves, campaigns = [], [], [], [], [], []
    for task, study in STUDIES.items():
        root = ROOT / 'outputs' / study
        summary = read(root / 'summary.json')
        progress = read(root / 'progress.json')
        require(progress['status'] == 'completed' and progress['completed'] == 10, 'Incomplete study')
        require(not summary['timing_violations'], 'Timing violation')
        read(root / 'protocol.json')
        audits = {}
        for name in ('audit.json', 'interface_audit.json', 'compact_prompt_audit.json', 'relocation_audit.json'):
            if (root / name).exists():
                audits[name] = read(root / name)
                require(audits[name]['status'] == 'passed', f'{study} {name} failed')
        audited = {r['episode']: r for r in audits['audit.json']['episodes']}
        campaigns.append(dict(task=task, study=study, status='completed', episodes=10,
            campaign_wall_seconds=progress['wall_seconds'], audits={k: v['status'] for k,v in audits.items()}))
        for record in summary['records']:
            directory = root / record['episode']
            result_path = directory / 'result.json'
            result = read(result_path, audited[record['episode']]['result_sha256'])
            require(result['success'] == record['success'], 'Success mismatch')
            equal(result['capped_completion_seconds'], record['capped_seconds'], 'Completion time')
            require(result['status'] in ('success', 'timeout'), 'Unexpected episode status')
            common = dict(task=task, study=study, seed=record['seed'], arm=ARMS[record['arm']])
            ep_requests = []
            for path in sorted(directory.glob('context_*/decision_*/receipt.json')):
                receipt = read(path)
                usage = receipt['usage']
                equal(usage['input_tokens'] + usage['output_tokens'], usage['total_tokens'], 'Token decomposition')
                ep_requests.append(dict(**common, context=path.parent.parent.name,
                    decision=path.parent.name, status='completed', usage_complete=True,
                    wall_seconds=receipt['wall_seconds'], image_count=receipt['image_count'],
                    **{k: usage[k] for k in TOKENS}, source=str(path.relative_to(ROOT))))
            failures = []
            for path in sorted(directory.glob('context_*/decision_*/attempt_failure.json')):
                failure = read(path)
                invocation = read(path.parent / 'invocation.json')
                failures.append(failure)
                ep_requests.append(dict(**common, context=path.parent.parent.name,
                    decision=path.parent.name, status=failure['type'], usage_complete=False,
                    wall_seconds=failure['wall_seconds'], image_count=len(invocation['images']),
                    **{k: None for k in TOKENS}, source=str(path.relative_to(ROOT))))
            completed = [r for r in ep_requests if r['usage_complete']]
            require(len(completed) == record['completed_requests'], 'Completed request count')
            require(len(failures) == record['incomplete_requests'], 'Interrupted request count')
            for key in TOKENS:
                total = sum(r[key] for r in completed)
                equal(total, record['usage'][key], key)
                equal(total, audited[record['episode']]['verified_usage'][key], 'Audited tokens')
            equal(sum(r['wall_seconds'] for r in completed), record['model_seconds'], 'Model time')
            equal(sum(r['image_count'] for r in ep_requests), record['images'], 'Image count')
            incomplete_seconds = sum(f['wall_seconds'] for f in failures)
            if 'incomplete_model_seconds' in record:
                equal(incomplete_seconds, record['incomplete_model_seconds'], 'Interrupted time')
            row = dict(**common, success=int(record['success']), status=result['status'],
                budget_seconds=200, capped_seconds=record['capped_seconds'],
                success_seconds=record['capped_seconds'] if record['success'] else None,
                completed_model_seconds=record['model_seconds'], incomplete_model_seconds=incomplete_seconds,
                all_model_seconds=record['model_seconds'] + incomplete_seconds,
                completed_requests=len(completed), incomplete_requests=len(failures),
                usage_complete=not failures, **record['usage'], images=record['images'],
                completed_images=sum(r['image_count'] for r in completed),
                incomplete_images=sum(r['image_count'] for r in ep_requests if not r['usage_complete']),
                camera_acquisitions=record['camera_acquisitions'], depth_queries=record['depth_queries'],
                action_seconds=record['action_seconds'], feedback_seconds=record['feedback_seconds'],
                operations=record.get('operations'), target_bindings=record.get('target_bindings'),
                initial_state_sha256=record['initial_state_sha256'],
                max_simulation_lag_seconds=record['max_simulation_lag_seconds'],
                source=str(result_path.relative_to(ROOT)), source_sha256=SOURCES[str(result_path.relative_to(ROOT))])
            episodes.append(row)
            requests.extend(ep_requests)
        task_rows = [r for r in episodes if r['task'] == task]
        seeds = sorted({r['seed'] for r in task_rows})
        require(len(seeds) == 5 and len(task_rows) == 10, 'Paired study size')
        for seed in seeds:
            pair = {r['arm']: r for r in task_rows if r['seed'] == seed}
            require(set(pair) == {'Base', 'Full'}, 'Missing pair')
            require(pair['Base']['initial_state_sha256'] == pair['Full']['initial_state_sha256'], 'Initial-state mismatch')
            pairs.append(dict(task=task, seed=seed, base_success=pair['Base']['success'],
                full_success=pair['Full']['success'], base_capped_seconds=pair['Base']['capped_seconds'],
                full_capped_seconds=pair['Full']['capped_seconds'],
                full_minus_base_seconds=pair['Full']['capped_seconds'] - pair['Base']['capped_seconds']))
        for original_arm, arm in ARMS.items():
            rows = [r for r in task_rows if r['arm'] == arm]
            successes = [r for r in rows if r['success']]
            aggregate = dict(task=task, arm=arm, episodes=5, successes=len(successes),
                success_rate=len(successes)/5,
                mean_capped_seconds=statistics.mean(r['capped_seconds'] for r in rows),
                success_conditioned_mean_seconds=statistics.mean(r['capped_seconds'] for r in successes),
                usage_complete=all(r['usage_complete'] for r in rows))
            for key in (*TOKENS, 'completed_model_seconds', 'incomplete_model_seconds', 'all_model_seconds',
                        'completed_requests', 'incomplete_requests', 'images', 'camera_acquisitions',
                        'depth_queries', 'action_seconds', 'feedback_seconds'):
                aggregate[key] = sum(r[key] for r in rows)
            for key in ('total_tokens', 'completed_model_seconds', 'all_model_seconds'):
                aggregate['mean_' + key] = aggregate[key] / 5
            aggregate['mean_completed_request_seconds'] = aggregate['completed_model_seconds'] / aggregate['completed_requests']
            ref = summary['arms'][original_arm]
            for key in ('successes', 'mean_capped_seconds', 'images', 'camera_acquisitions', 'completed_requests', 'incomplete_requests', 'action_seconds', 'feedback_seconds'):
                equal(aggregate[key], ref[key], key)
            for key in TOKENS:
                equal(aggregate[key], ref['usage'][key], key)
            totals.append(aggregate)
            for time in sorted({0., 200., *(r['capped_seconds'] for r in successes)}):
                curves.append(dict(task=task, arm=arm, budget_seconds=time,
                    successes=sum(r['success'] and r['capped_seconds'] <= time for r in rows), episodes=5))
    require(len(episodes) == 30 and len(requests) == 271, 'Overall counts')
    return episodes, requests, totals, pairs, curves, campaigns


def plot(episodes, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'Base':'#416A9C', 'Full':'#BD6833'}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True, layout='constrained')
    for ax, task in zip(axes, STUDIES):
        for arm in ARMS.values():
            rows = [r for r in curves if r['task'] == task and r['arm'] == arm]
            ax.step([r['budget_seconds'] for r in rows], [r['successes']*20 for r in rows],
                    where='post', color=colors[arm], linewidth=2, label=arm)
        ax.set(title=LABELS[task], xlabel='Elapsed wall time (s)', xlim=(0, 203), ylim=(-3, 103))
        ax.set_xticks([0,50,100,150,200]);ax.set_yticks(range(0,101,20));ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Success within time budget (%)');axes[0].legend(frameon=False)
    for ext in ('pdf','png'):
        fig.savefig(OUT / f'success_vs_time.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(10,3.2),sharey=True,layout='constrained')
    for ax, task in zip(axes, STUDIES):
        rows=[r for r in episodes if r['task']==task]
        for index,seed in enumerate(sorted({r['seed'] for r in rows}),1):
            pair={r['arm']:r for r in rows if r['seed']==seed}
            ax.plot([index-.12,index+.12],[pair[a]['capped_seconds'] for a in ARMS.values()],color='.75',lw=1)
            for arm,offset in [('Base',-.12),('Full',.12)]:
                r=pair[arm];ax.scatter(index+offset,r['capped_seconds'],color=colors[arm],marker='o' if r['success'] else 'x',s=34)
        ax.axhline(200,color='.5',linestyle=':',lw=1)
        ax.set(title=LABELS[task],xlabel='Paired layout',xticks=range(1,6),ylim=(0,214))
    axes[0].set_ylabel('Capped completion time (s)')
    fig.suptitle('Blue: Base · Orange: Full     Circle: success · Cross: timeout',fontsize=9)
    for ext in ('pdf','png'):
        fig.savefig(OUT / f'paired_completion_times.{ext}', dpi=180)
    plt.close(fig)


def main():
    episodes, requests, totals, pairs, curves, campaigns = collect()
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in [('episodes.csv',episodes),('model_requests.csv',requests),('arm_summary.csv',totals),
                      ('paired_outcomes.csv',pairs),('success_vs_time.csv',curves)]:
        csv_file(name,rows)
    summary=dict(schema_version=1,episodes=30,paired_layouts=15,completed_requests=262,
        incomplete_requests=9,arms=totals,campaigns=campaigns,
        scope='Three separate exploratory studies; do not pool task success rates. Unknown tokens are not zero.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(episodes,curves)
    provenance=dict(exporter='scripts/export_online_results.py',
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_files=SOURCES,
        exported_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())
                        if p.suffix in {'.csv','.json','.pdf','.png'} and p.name!='provenance.json'})
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
