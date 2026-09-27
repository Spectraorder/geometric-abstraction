"""Verify retained live numeric/symbolic integration sources, states and physics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    if json.loads((root/'progress.json').read_text())['status'] != 'completed':
        raise ValueError('Integration is not complete')
    manifest = json.loads((root/'manifest.json').read_text()); reports = []
    for seed in manifest['seeds']:
        paths = [root/f'{arm}_{seed}' for arm in ('numeric', 'symbolic')]
        results = [json.loads((path/'result.json').read_text()) for path in paths]
        for arm, result in zip(('numeric', 'symbolic'), results):
            if result['candidate_sha256'] != manifest['candidate_sha256'][arm] or digest(root/(arm+'.py')) != manifest['candidate_sha256'][arm]:
                raise ValueError('Source mismatch')
            if result['policy_error'] is not None or result['seed'] != seed:
                raise ValueError('Execution or seed mismatch')
        for field in ('scene', 'sensor', 'physics', 'wrist_depth', 'camera_calls', 'success'):
            if results[0][field] != results[1][field]:
                raise ValueError('Paired '+field+' differs')
        if digest(paths[0]/'initial.png') != digest(paths[1]/'initial.png'):
            raise ValueError('Initial images differ')
        with np.load(paths[0]/'rollout.npz') as first, np.load(paths[1]/'rollout.npz') as second:
            trajectory = bool(np.array_equal(first['states'], second['states']))
        packets = [[row for row in json.loads((p/'geometry_records.json').read_text()) if row.get('kind') == 'scene_checkpoint'] for p in paths]
        numeric_equal = True
        for first, second in zip(*packets, strict=True):
            if first['phase'] != second['phase']:
                raise ValueError('Checkpoint phases differ')
            projection = json.loads(json.dumps(second['state']))
            projection['schema'] = 'mixed_metric_scene_v1'; projection.pop('facts'); projection.pop('contact_state')
            for row in projection['objects'].values():
                row.pop('references', None)
            numeric_equal &= first['state'] == projection
        if not trajectory or not numeric_equal:
            raise ValueError('Physical trajectory or numerical state differs')
        reports.append({'seed': seed, 'numeric_success': results[0]['success'], 'symbolic_success': results[1]['success'],
                        'identical_physical_trajectory': trajectory, 'identical_numeric_states': numeric_equal,
                        'state_checkpoints': len(packets[0]), 'camera_calls_per_arm': results[0]['camera_calls']})
    return {'purpose': 'Interface parity integration on inspected development layouts, not agent-learning evidence',
            'manifest': manifest, 'comparisons': reports,
            'limitations': ['The same manual planner is used; symbolic queries do not select different actions in this check.',
                            'Sources and packets are the archived integration version; later text or guard changes do not rewrite them.',
                            'Conditional prediction is disabled, and no reserved scenes or actor revisions are included.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); report = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['comparisons'], indent=2))
