"""Report per-turn usage from cumulative counters in archived actor sessions.

Raw generation records are preserved. This report cross-checks session totals,
new token events and CLI usage before subtracting the preceding turn's counters.
"""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def turn_usage(previous, current, raw_usage, new_events):
    if any(current.get(key) != value for key, value in raw_usage.items()):
        raise ValueError('CLI usage differs from archived session totals')
    delta = {key: value - previous.get(key, 0) for key, value in current.items()}
    if any(value < 0 for value in delta.values()):
        raise ValueError('Cumulative usage decreased; accounting needs investigation')
    event_sum = {key: sum(event['last_token_usage'].get(key, 0) for event in new_events)
                 for key in current}
    if delta != event_sum:
        raise ValueError('Counter delta differs from new token events')
    if delta['total_tokens'] != delta['input_tokens'] + delta['output_tokens']:
        raise ValueError('Total tokens differ from input plus output')
    return delta


def summarize(root, rounds):
    manifest = read(root/'manifest.json')
    report = {'accounting': 'Per-turn deltas of verified cumulative session counters',
              'notes': ['Cached input is included in input; reasoning is included in output.',
                        'Raw generation usage records contain cumulative session totals.',
                        'No monetary cost or isolated latency advantage is inferred.'], 'arms': {}}
    for arm in manifest['arms']:
        previous = {}; previous_events = []; session_id = None; rows = []
        for revision in range(1, rounds + 1):
            stage = root/arm/f'round_{revision:03d}'
            generation = read(stage/'generation.json')
            invocation = read(stage/'invocation.json')
            events = [json.loads(line) for line in (stage/'session.jsonl').read_text().splitlines()]
            counts = [e['payload']['info'] for e in events
                      if e.get('type') == 'event_msg' and e.get('payload', {}).get('type') == 'token_count'
                      and e['payload'].get('info') is not None]
            if not counts or counts[:len(previous_events)] != previous_events:
                raise ValueError('Missing or changed session token history')
            if session_id is not None and generation['session_id'] != session_id:
                raise ValueError('Actor session changed')
            session_id = generation['session_id']
            current = counts[-1]['total_token_usage']
            usage = turn_usage(previous, current, generation['usage'], counts[len(previous_events):])
            rows.append({'revision': revision, 'usage': usage,
                         'wall_seconds': generation['wall_seconds'], 'image_count': len(invocation['images'])})
            previous = current; previous_events = counts
        def aggregate(subset):
            return {'usage': {key: sum(row['usage'][key] for row in subset) for key in previous},
                    'generation_wall_seconds': sum(row['wall_seconds'] for row in subset),
                    'new_image_attachments': sum(row['image_count'] for row in subset)}
        report['arms'][arm] = {'session_id': session_id, 'turns': rows,
                               'all_rounds': aggregate(rows),
                               'original_rounds': aggregate(rows[:manifest['rounds']]),
                               'extension_rounds': aggregate(rows[manifest['rounds']:])}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--rounds', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.rounds)
    if args.output.exists():
        raise FileExistsError('Keep prior accounting reports; choose a new output path')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
