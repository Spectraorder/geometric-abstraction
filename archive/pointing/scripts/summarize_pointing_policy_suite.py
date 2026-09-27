"""Summarize paired real-policy rollouts and export offline-only pointing checks."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trial_record(path):
    result = json.loads((path / 'result.json').read_text())
    calls = json.loads((path / 'api_calls.json').read_text())
    notes = [c['params']['text'] for c in calls if c['method'] == 'note']
    return result, calls, notes


def action_calls(calls):
    return [{'method': c['method'], 'params': c['params']} for c in calls
            if c['method'] in ('move_tool', 'move_joints', 'gripper', 'wait')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / 'manifest.json').read_text())
    rows, frames, predictions = [], [], []
    for case in manifest['cases']:
        name = case['case_id']
        base_path, point_path = args.root / 'baseline' / name, args.root / 'pointing' / name
        base, base_calls, base_notes = trial_record(base_path)
        point, point_calls, point_notes = trial_record(point_path)
        assert base['candidate_sha256'] == manifest['baseline_sha256']
        assert point['candidate_sha256'] == manifest['pointing_sha256']
        request_indices = [i for i, c in enumerate(point_calls) if c['method'] == 'point_target']
        pre = action_calls(point_calls[:request_indices[0]]) if request_indices else action_calls(point_calls)
        row = dict(case, baseline_success=base['success'], pointing_success=point['success'],
                   baseline_final_note=base_notes[-1], pointing_final_note=point_notes[-1],
                   baseline_confirmed_nodes=sum(n.startswith('NODE confirmed') for n in base_notes),
                   pointing_confirmed_nodes=sum(n.startswith('NODE confirmed') for n in point_notes),
                   baseline_policy_error=base['policy_error'], pointing_policy_error=point['policy_error'],
                   baseline_sim_seconds=base['policy_sim_seconds'], pointing_sim_seconds=point['policy_sim_seconds'],
                   baseline_cpu_seconds=base['cpu_profile']['policy_compute_cpu_seconds'],
                   pointing_cpu_seconds=point['cpu_profile']['policy_compute_cpu_seconds'],
                   pointing_requests=point['pointing']['request_count'],
                   pointing_host_wait_seconds=point['pointing']['host_wait_seconds'],
                   pre_pointing_actions_match_baseline_prefix=pre == action_calls(base_calls)[:len(pre)],
                   complete_action_sequence_unchanged=action_calls(point_calls) == action_calls(base_calls),
                   pointing_notes=[n for n in point_notes if n.startswith('POINTING')],
                   baseline_result_sha256=digest(base_path / 'result.json'),
                   pointing_result_sha256=digest(point_path / 'result.json'))
        rows.append(row)
        records = iter(point['pointing']['records'])
        camera_index, last_side_index = -1, None
        for call in point_calls:
            if call['method'] == 'camera':
                camera_index += 1
                if call['params'].get('view', 'side') == 'side':
                    last_side_index = camera_index
            elif call['method'] == 'point_target':
                record = next(records)
                assert record['status'] == 'completed'
                source = point_path / f'camera_{last_side_index:03d}.npz'
                assert digest(source.with_suffix('.png')) == record['request']['image_sha256']
                target = call['params']['color'] + '_' + call['params']['shape']
                predictions.append(record['reply'])
                frames.append({'task_id': record['task_id'], 'trial': str(point_path),
                               'frame_path': str(source), 'color': call['params']['color'],
                               'evaluation_target_object': target, 'stratum': 'runtime_recovery',
                               'distractors': case['distractors'], 'case_id': name})
    failure_cases = [r for r in rows if not r['archived_success']]
    controls = [r for r in rows if r['archived_success']]
    report = {'selection': manifest['selection'], 'candidate_hashes': {k: manifest[k] for k in
              ('baseline_sha256', 'pointing_sha256')}, 'summary': {
                  'paired_cases': len(rows), 'actual_rollouts': 2 * len(rows),
                  'previous_failure_cases': len(failure_cases),
                  'baseline_successes_on_previous_failures': sum(r['baseline_success'] for r in failure_cases),
                  'pointing_successes_on_previous_failures': sum(r['pointing_success'] for r in failure_cases),
                  'success_controls': len(controls),
                  'baseline_control_successes': sum(r['baseline_success'] for r in controls),
                  'pointing_control_successes': sum(r['pointing_success'] for r in controls),
                  'pointing_requests': sum(r['pointing_requests'] for r in rows),
                  'point_proposals': sum(p['status'] == 'point' for p in predictions),
                  'abstentions': sum(p['status'] == 'abstain' for p in predictions),
                  'worker_errors': sum(r['baseline_policy_error'] is not None or r['pointing_policy_error'] is not None for r in rows)},
              'protocol': 'Fresh simulator rollouts, fixed seeds/goals, baseline paired with optional point-guided reassociation. Geometry/manipulation thresholds unchanged. RGB-only hosted subagents; CPU worker remains network/file isolated. Hosted wait excluded from the simulation wall-time budget and worker CPU, recorded separately.',
              'cases': rows}
    (args.root / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    offline = args.root / 'offline_point_checks';offline.mkdir(exist_ok=True)
    (offline / 'frames.json').write_text(json.dumps(frames, indent=2) + '\n')
    (offline / 'predictions.json').write_text(json.dumps(predictions, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
