"""Descriptive paired online outcomes; layouts remain the unit of scene variation."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def summarize(root):
    root = Path(root)
    protocol = read(root/'protocol.json')
    records = []
    for slot in protocol['slots']:
        folder = root/slot['path']
        result = read(folder/'result.json')
        for name, expected in read(folder/'source_hashes.json').items():
            if (protocol['source_sha256'].get('scripts/'+name) != expected
                    or hashlib.sha256((folder/'sources'/name).read_bytes()).hexdigest() != expected):
                raise ValueError('Episode source snapshot differs from registration')
        for key in ('seed', 'arm', 'budget_s'):
            if result[key] != slot.get(key, protocol.get(key)):
                raise ValueError('Episode result differs from registration: '+key)
        if result.get('timing', {}).get('reset_count') != 1:
            raise ValueError('Episode did not preserve the one-reset contract')
        if (not result['timing'].get('physics_during_model_wait')
                or not result['timing'].get('paced_actions')):
            raise ValueError('Episode did not use the registered timing implementation')
        success = result['success']
        if success and (not 0 < result['time_to_success_s'] <= protocol['budget_s']
                        or not result['private_final_status']['stable']):
            raise ValueError('Invalid success evidence')
        evidence = result.get('private_success_evidence')
        if success and (not evidence or evidence['duration_sim_seconds'] < 5.
                        or evidence['maximum_final_displacement_m'] > .003):
            raise ValueError('Missing sustained-success evidence')
        provenance = read(folder/'skill_provenance.json')
        if provenance['geometry_sha256'] != protocol['geometry_sha256']:
            raise ValueError('Episode used a different geometry bundle')
        for kind, (source, digest) in protocol['shared_skills'].items():
            if provenance['source_policies'][kind] != {'path': source, 'sha256': digest}:
                raise ValueError('Episode used different shared skill experience')
        invocations = list(folder.glob('decision_*/invocation.json'))
        dispatches = rows(folder/'dispatches.jsonl')
        requests = result.get('model_requests', [])
        records.append({
            'seed': slot['seed'], 'repetition': slot['repetition'], 'arm': slot['arm'],
            'status': result['status'], 'success': success,
            'time_to_success_s': result.get('time_to_success_s'),
            'capped_completion_seconds': result['capped_completion_seconds'],
            'decisions': result['decisions'],
            'new_image_attachments': sum(len(read(path)['images']) for path in invocations),
            'model_requests': len(invocations), 'completed_model_requests': len(requests),
            'model_seconds_completed_requests': sum(row['wall_seconds'] for row in requests),
            'model_wait_seconds_all_attempts': (sum(row['wall_seconds'] for row in requests)
                +sum(read(path)['wall_seconds'] for path in folder.glob('decision_*/attempt_failure.json'))),
            'completed_request_latencies_by_model': {
                model: [r['wall_seconds'] for r in requests if r['model'] == model]
                for model in ('gpt-6-astra', 'gpt-6-luna')},
            'completed_request_usage': result.get('completed_request_usage', {}),
            'usage_complete': result.get('usage_complete', False),
            'feedback_to_dispatch_seconds': [r['feedback_to_dispatch_seconds'] for r in dispatches],
            'max_simulation_lag_seconds': result['timing']['max_simulation_lag_seconds'],
            'timing_lag_within_limit': result['timing']['max_simulation_lag_seconds'] <=
                                      protocol['max_timing_lag_for_realtime_claim_s'],
            'initial_physics_state_sha256': result['initial_physics_state_sha256'],
            'camera_acquisitions': result.get('camera_acquisitions'),
        })
    aggregate = {}
    for arm in ('basic', 'symbolic'):
        selected = [r for r in records if r['arm'] == arm]
        latencies = [value for r in selected for value in r['feedback_to_dispatch_seconds']]
        aggregate[arm] = {
            'episodes': len(selected), 'successes': sum(r['success'] for r in selected),
            'success_rate': statistics.mean(r['success'] for r in selected),
            'mean_capped_completion_seconds': statistics.mean(r['capped_completion_seconds'] for r in selected),
            'median_feedback_to_dispatch_seconds': statistics.median(latencies) if latencies else None,
            'new_image_attachments': sum(r['new_image_attachments'] for r in selected),
            'model_requests': sum(r['model_requests'] for r in selected),
            'model_wait_seconds_all_attempts': sum(r['model_wait_seconds_all_attempts'] for r in selected),
            'usage_complete': all(r['usage_complete'] for r in selected),
            'completed_request_usage': {key: sum(r['completed_request_usage'].get(key, 0) for r in selected)
                                        for key in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'total_tokens')},
            'infrastructure_failures': sum(r['status'] == 'infrastructure_error' for r in selected),
            'timing_limit_violations': sum(not r['timing_lag_within_limit'] for r in selected),
        }
        aggregate[arm]['completed_requests_by_model'] = {}
        for model in ('gpt-6-astra', 'gpt-6-luna'):
            values = [value for r in selected for value in r['completed_request_latencies_by_model'][model]]
            aggregate[arm]['completed_requests_by_model'][model] = {
                'count': len(values), 'median_seconds': statistics.median(values) if values else None,
                'mean_seconds': statistics.mean(values) if values else None}
    paired = []
    for seed in protocol['seeds']:
        states = {r['initial_physics_state_sha256'] for r in records if r['seed'] == seed}
        if len(states) != 1:
            raise ValueError('Initial physics differs across a paired layout')
        row = {'seed': seed}
        for arm in aggregate:
            selected = [r for r in records if r['seed'] == seed and r['arm'] == arm]
            row[arm] = {'successes': sum(r['success'] for r in selected), 'episodes': len(selected),
                        'mean_capped_seconds': statistics.mean(r['capped_completion_seconds'] for r in selected)}
        row['full_minus_base_success_rate'] = (row['symbolic']['successes']/row['symbolic']['episodes']
                                               -row['basic']['successes']/row['basic']['episodes'])
        paired.append(row)
    return {'aggregate': aggregate, 'paired_layouts': paired, 'episodes': records,
            'interpretation': 'Five distinct layouts with repeated episodes, not twenty independent layouts. '
                              'Natural failure subsets are descriptive and are not paired recovery trials. '
                              'Completed-request token totals exclude any unfinished requests; inspect usage_complete.'}


def plot(summary, root, budget):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    for arm, label, color in (('basic', 'Visual Base', '#527e9b'), ('symbolic', 'Geometry Full', '#ac7542')):
        records = [r for r in summary['episodes'] if r['arm'] == arm]
        times = sorted(r['time_to_success_s'] for r in records if r['success'])
        axes[0].step([0]+times+[budget], [0]+[(i+1)/len(records) for i in range(len(times))]
                     +[len(times)/len(records)], where='post', label=label, color=color)
        means = [row[arm]['mean_capped_seconds'] for row in summary['paired_layouts']]
        axes[1].plot(np.arange(1, len(means)+1), means, 'o-', label=label, color=color)
    axes[0].set(xlabel='Wall-clock budget (s)', ylabel='Success rate', ylim=(0, 1.03), xlim=(0, budget))
    axes[1].set(xlabel='Shared layout', ylabel='Mean capped completion time (s)', ylim=(0, budget*1.03))
    axes[1].set_xticks(range(1, len(summary['paired_layouts'])+1))
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
    for extension in ('png', 'pdf'):
        fig.savefig(Path(root)/('online_curves.'+extension), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    target = args.root/'summary.json'
    if target.exists():
        raise FileExistsError('Preserve the previous report; verify it instead of overwriting')
    target.write_text(json.dumps(result, indent=2)+'\n')
    plot(result, args.root, read(args.root/'protocol.json')['budget_s'])
    print(json.dumps(result['aggregate'], indent=2))
