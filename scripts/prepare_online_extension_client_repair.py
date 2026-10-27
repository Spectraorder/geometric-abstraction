"""Register a one-time repair of the extension's incomplete client packaging.

The four initial attempts reached a complete model response but failed the actor
audit before any submitted action. Keep these artifacts/costs and use new actor
contexts for the same fixed slots. Disable the unrelated Code Mode host; do not
relax the actor event audit or change task policies. This is not a retry daemon.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time

import run_online_extension_study as study
from verified_token_usage import turn_usage


def diagnose(root, protocol):
    attempts, evidence = [], {}
    for slot in protocol['slots']:
        episode = study.episode_path(root, slot)
        if not episode.exists():
            continue
        result = study.read(episode / 'result.json')
        study.require(slot['index'] < 4 and result['status'] == 'infrastructure_error',
                      'Repair is restricted to the first four startup failures')
        study.require(result['model_requests_completed'] == 0
                      and result.get('dispatched_operations', result.get('operations')) == 0,
                      'An attempt progressed beyond the first response')
        turns = list(episode.glob('context_*/decision_*'))
        study.require(len(turns) == 1, 'Unexpected model interaction history')
        turn = turns[0]
        events = [json.loads(s) for s in (turn / 'model.jsonl').read_text().splitlines()]
        errors = [e['item']['message'] for e in events
                  if e.get('type') == 'item.completed' and e.get('item', {}).get('type') == 'error']
        expected = ('Code Mode is unavailable because failed to spawn code-mode host '
                    + str(root / 'bin/codex-code-mode-host')
                    + ': host executable was not found. Code mode will fail closed; enable '
                    '`features.code_mode_host` and install `codex-code-mode-host`.')
        study.require(errors == [expected], 'Failure is not the diagnosed missing-host error')
        study.require(not any(e.get('item', {}).get('type') not in ('error', 'agent_message', 'reasoning')
                              for e in events if e.get('type') == 'item.completed'),
                      'Unexpected actor tool use')
        raw = [e['usage'] for e in events if e.get('type') == 'turn.completed']
        study.require(len(raw) == 1, 'Missing complete request usage')
        transcripts = list((turn.parent / 'actor_home/sessions').rglob('*.jsonl'))
        study.require(len(transcripts) == 1, 'Unexpected context count')
        transcript = transcripts[0]
        rows = [json.loads(s) for s in transcript.read_text().splitlines()]
        counts = [e['payload']['info'] for e in rows if e.get('type') == 'event_msg'
                  and e.get('payload', {}).get('type') == 'token_count' and e['payload'].get('info')]
        usage = turn_usage({}, counts[-1]['total_token_usage'], raw[0], counts)
        for path in (episode / 'result.json', turn / 'model.jsonl', transcript,
                     turn / 'attempt_failure.json', turn / 'invocation.json', turn / 'prompt.txt'):
            evidence[str(path)] = study.digest(path)
        attempts.append(dict(slot, status='infrastructure_error', source=str(episode),
                             recovered_recorded_usage=usage, policy_decision_executed=False))
    study.require(len(attempts) == 4, 'Expected exactly four diagnosed startup attempts')
    return dict(attempts=attempts,
                total_recorded_tokens=sum(a['recovered_recorded_usage']['total_tokens'] for a in attempts),
                interpretation='Infrastructure overhead, separate from episode-policy costs; not unknown usage '
                               'because complete CLI counters agree with saved session totals'), evidence


def prepare(root):
    protocol = study.check(root)
    study.require((root / 'STOP.json').exists(), 'Original coordinator has not stopped')
    lifecycle = study.read(root / 'lifecycle.json')
    study.require(lifecycle['status'] == 'stopped', 'Original coordinator is still active')
    diagnosis, failure_hashes = diagnose(root, protocol)
    target = root / 'client_repair_01'
    target.mkdir(exist_ok=False)
    for directory in ('sources', 'tasks', 'bin'):
        (target / directory).mkdir()
    for name in protocol['source_sha256']:
        path = target / 'sources' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / 'sources' / name, path)
    own = Path(__file__).resolve().relative_to(study.ROOT)
    shutil.copy2(study.ROOT / own, target / 'sources' / own)
    protocol['source_sha256'][str(own)] = study.digest(study.ROOT / own)
    for name in protocol['artifact_sha256']:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, path)
    for task in study.STUDIES:
        (target / 'tasks' / task / 'sources').symlink_to('../../sources', target_is_directory=True)
    native = target / 'bin/codex.native'
    shutil.copy2(root / 'bin/codex', native)
    wrapper = target / 'bin/codex'
    wrapper.write_text('#!/bin/sh\nexec ' + shlex.quote(str(native))
                       + ' --disable code_mode --disable code_mode_host "$@"\n')
    wrapper.chmod(0o755)
    # Offline configuration check: no credentials, requests, or scene resets.
    with tempfile.TemporaryDirectory(prefix='client_check_', dir=target) as home:
        response = subprocess.run([str(wrapper), 'features', 'list'], capture_output=True, text=True,
                                  check=True, env=dict(os.environ, CODEX_HOME=home))
        features = {line.split()[0]: line.split()[-1] for line in response.stdout.splitlines()}
        study.require(features['code_mode'] == features['code_mode_host'] == 'false',
                      'Code Mode was not disabled')
    study.write(target / 'infrastructure_attempts.json', diagnosis)
    for name in ('progress.json', 'lifecycle.json', 'STOP.json', 'protocol.json'):
        shutil.copy2(root / name, target / ('original_' + name))
    added = ['bin/codex.native', 'infrastructure_attempts.json',
             'original_progress.json', 'original_lifecycle.json', 'original_STOP.json', 'original_protocol.json']
    protocol['artifact_sha256'].update({name: study.digest(target / name) for name in added})
    protocol['legacy_evidence_sha256'].update(failure_hashes)
    protocol.update(registered_at=time.time(), client_sha256=study.digest(wrapper),
                    client_repair=dict(original_root=str(root), flags=['--disable code_mode', '--disable code_mode_host'],
                        cause='Copied client omitted its default-enabled Code Mode helper',
                        original_attempts_preserved=4, replacement_attempts_for_same_slots=4,
                        episode_slots_unchanged=270, total_attempts_including_startup_failures=274,
                        old_contexts_reused=False, audit_relaxed=False,
                        future_failure_policy='Stop, preserve evidence, no automatic retries'))
    study.write(target / 'protocol.json', protocol)
    (target / 'protocol.sha256').write_text(study.digest(target / 'protocol.json') + '\n')
    study.write(target / 'progress.json', dict(status='registered', completed=0, total=270,
                                             legacy_completed=30, pooled_total=300,
                                             prior_infrastructure_attempts=4))
    study.check(target)
    study.write(root / 'continuation.json', dict(root=str(target), progress=str(target / 'progress.json'),
                                               reason=protocol['client_repair']))
    print(json.dumps(dict(root=str(target), recovered_infrastructure_tokens=diagnosis['total_recorded_tokens'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    prepare(parser.parse_args().root.resolve())
