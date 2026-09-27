"""Describe completed training command traces without inferring physical contact."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    protocol = read(root / 'protocol.json')
    rows = []
    counts = defaultdict(Counter)
    for replica in protocol['replicas']:
        for arm in protocol['arms']:
            for revision in range(protocol['training_revisions_per_arm'] + 1):
                stage = 'revision_000' if revision == 0 else f'round_{revision:03d}'
                for seed in replica['train_seeds']:
                    trial = root / replica['directory'] / arm / stage / 'trials' / f'seed_{seed}'
                    result = read(trial / 'result.json')
                    calls = read(trial / 'api_calls.json')
                    actions = [c for c in calls if c['method'] in ('move_tool', 'move_joints', 'gripper')]
                    jaws = [(i, c) for i, c in enumerate(calls) if c['method'] == 'gripper']
                    notes = [c['params']['text'] for c in calls if c['method'] == 'note']
                    moves = []
                    if result['success']:
                        category = 'successful_task'
                    elif not actions:
                        category = 'no_actuation'
                    elif not jaws:
                        category = 'motion_without_gripper_command'
                    else:
                        index, jaw = jaws[-1]
                        pair = (jaw['params']['angle'], jaw['params']['seconds'])
                        moves = [c['params']['seconds'] for c in calls[index+1:] if c['method'] == 'move_tool']
                        if pair == (-.174, 1.):
                            # The frozen shared executor uses 1.6 s for the first
                            # placement descent, after the 1.7 s lift. A retry can
                            # duplicate move calls, so preserve the raw sequence.
                            category = 'closed_after_placement_descent' if 1.7 in moves and 1.6 in moves else 'closed_stage_unresolved'
                        elif pair == (1.4, 1.):
                            category = 'after_release_opening'
                        elif pair == (.8, .5):
                            category = 'after_unconfirmed_grasp_reopening'
                        else:
                            category = 'other_command_phase'
                    counts[arm][category] += 1
                    rows.append({'replica': replica['directory'], 'arm': arm, 'revision': revision,
                                 'seed': seed, 'category': category, 'success': result['success'],
                                 'policy_error': result.get('policy_error'), 'notes': notes,
                                 'last_gripper_command': jaws[-1][1]['params'] if jaws else None,
                                 'subsequent_move_durations': moves,
                                 'worker_failures': [c['params']['error'] for c in calls if c['method'] == 'failed'],
                                 'source_sha256': {name: digest(trial/name) for name in ('result.json', 'api_calls.json')}})
    if len(rows) != protocol['training_slots']:
        raise ValueError('Incomplete training coverage')
    return {'scope': 'All registered training slots only; no reserved outcomes or simulator object states.',
            'interpretation': 'Command-phase localization, not a diagnosis of physical grasp, contact, stability or causal failure. Revisions and repeated layouts are dependent.',
            'protocol_sha256': digest(root/'protocol.json'), 'auditor_sha256': digest(Path(__file__)),
            'counts_by_arm': dict(counts), 'total_slots': len(rows), 'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['counts_by_arm'], indent=2))
