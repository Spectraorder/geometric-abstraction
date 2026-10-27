"""Recover terminal request usage from agreeing immutable client/session counters.

This only reads archived evidence. A usage recovery never turns an interrupted
decision into an executed one, or changes episode outcomes and timing.
"""
import hashlib
import json
from pathlib import Path

from verified_token_usage import turn_usage

ROOT = Path(__file__).absolute().parents[1]


def audit_interrupted_requests(requests):
    sources, records = {}, []

    def snapshot(path):
        path = ROOT / path
        raw = path.read_bytes()
        key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        sources[key] = hashlib.sha256(raw).hexdigest()
        return raw

    def events(path):
        return [json.loads(line) for line in snapshot(path).splitlines() if line.strip()]

    def counts(values):
        return [v['payload']['info'] for v in values if v.get('type') == 'event_msg'
                and v.get('payload', {}).get('type') == 'token_count'
                and v['payload'].get('info') is not None]

    for request in requests:
        if request['status'] == 'completed':
            continue
        directory = (ROOT / request['source']).parent
        failure = json.loads(snapshot(directory / 'attempt_failure.json'))
        invocation = json.loads(snapshot(directory / 'invocation.json'))
        cli = events(directory / 'model.jsonl')
        record = {k: request[k] for k in ('task', 'cohort', 'seed', 'arm', 'context', 'decision', 'source')}
        record.update(interruption_type=failure['type'], recovered=False, usage=None)
        previous = sorted(p for p in directory.parent.glob('decision_*/receipt.json')
                          if p.parent.name < directory.name)
        sid = invocation.get('session_before')
        sessions = list((directory.parent / 'actor_home/sessions').rglob(f'*{sid}.jsonl')) if sid else []
        if not previous or len(sessions) != 1:
            record['reason'] = 'No unique session/baseline evidence; usage remains unknown'
            records.append(record)
            continue
        previous_receipt = json.loads(snapshot(previous[-1]))
        baseline_path = previous[-1].parent / 'session.jsonl'
        baseline_raw, current_raw = snapshot(baseline_path), snapshot(sessions[0])
        if not current_raw.startswith(baseline_raw):
            raise ValueError('Session history changed: ' + str(directory))
        baseline, current = counts(events(baseline_path)), counts(events(sessions[0]))
        meta = events(sessions[0])[0]
        if meta.get('type') != 'session_meta' or meta.get('payload', {}).get('id') != sid:
            raise ValueError('Session identity differs: ' + str(directory))
        if not baseline or current[:len(baseline)] != baseline:
            raise ValueError('Token history changed: ' + str(directory))
        before = baseline[-1]['total_token_usage']
        if any(before.get(k) != v for k, v in previous_receipt['raw_cumulative_usage'].items()):
            raise ValueError('Baseline receipt disagrees with session: ' + str(directory))
        finished = [e for e in cli if e.get('type') == 'turn.completed']
        if not finished:
            record['reason'] = 'No terminal CLI usage; absence of a new counter does not prove zero consumption'
        else:
            if len(finished) != 1 or {e['thread_id'] for e in cli if e.get('type') == 'thread.started'} != {sid}:
                raise ValueError('Ambiguous completed turn: ' + str(directory))
            usage = turn_usage(before, current[-1]['total_token_usage'],
                               finished[0]['usage'], current[len(baseline):])
            if usage['total_tokens'] <= 0:
                raise ValueError('No positive verified terminal usage: ' + str(directory))
            record.update(recovered=True, usage=usage,
                reason='Final CLI totals agree with append-only session counter deltas and last-turn usage',
                baseline_receipt=str(previous[-1].relative_to(ROOT)),
                session_source=str(sessions[0].relative_to(ROOT)))
        records.append(record)
    recovered = [r for r in records if r['recovered']]
    return dict(status='verified', interrupted_requests=len(records),
        recovered_requests=len(recovered), unknown_requests=len(records) - len(recovered),
        recovered_total_tokens=sum(r['usage']['total_tokens'] for r in recovered),
        reason_counts={kind: sum(r['interruption_type'] == kind for r in records)
                       for kind in sorted({r['interruption_type'] for r in records})},
        records=records, source_files=sources,
        interpretation='Terminal usage recovery only. Request status, accepted decisions, images, '
            'episode outcomes and clocks remain unchanged. Remaining unknown usage is not zero.')


def apply_recoveries(episodes, requests, audit):
    by_source = {r['source']: r for r in audit['records'] if r['recovered']}
    for row in requests:
        row['decision_completed_in_episode'] = row['status'] == 'completed'
        row['token_usage_recovered'] = row['source'] in by_source
        if row['token_usage_recovered']:
            row.update(by_source[row['source']]['usage'])
            row['usage_complete'] = True
    for episode in episodes:
        key = (episode['task'], episode['seed'], episode['arm'])
        selected = [r for r in requests if (r['task'], r['seed'], r['arm']) == key]
        recovered = [r for r in selected if r['token_usage_recovered']]
        episode['receipt_total_tokens'] = episode['total_tokens']
        episode['recovered_interrupted_requests'] = len(recovered)
        episode['recovered_interrupted_tokens'] = sum(r['total_tokens'] for r in recovered)
        episode['unknown_token_requests'] = sum(not r['usage_complete'] for r in selected)
        for k in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'total_tokens',
                  'cache_write_input_tokens', 'reasoning_output_tokens'):
            if k in episode:
                episode[k] += sum(r.get(k, 0) for r in recovered)
        episode['usage_complete'] = not episode['unknown_token_requests']
