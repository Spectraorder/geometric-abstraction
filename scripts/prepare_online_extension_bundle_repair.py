"""Freeze the complete client bundle after two preserved startup failures.

Code Mode remains disabled, but the newer client requires its host component for
clean initialization even when no code tool is used. Check actual exec startup
against an unreachable localhost provider before admitting any live episodes.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import tempfile
import time

import run_online_extension_study as study
from prepare_online_extension_client_repair import diagnose
from geometry_actor_session import DISABLED_TOOLS


def offline_check(wrapper, root):
    provider = ('{name="Offline startup check",base_url="http://127.0.0.1:1",'
                'wire_api="responses",requires_openai_auth=false}')
    with tempfile.TemporaryDirectory(prefix='offline_check_', dir=root) as home:
        environment = dict(os.environ, CODEX_HOME=home)
        features = subprocess.check_output([str(wrapper), 'features', 'list'], text=True, env=environment)
        values = {line.split()[0]: line.split()[-1] for line in features.splitlines()}
        study.require(values['code_mode'] == 'false' and values['code_mode_host'] == 'true',
                      'Unexpected client feature configuration')
        command = [str(wrapper), 'exec', '--ignore-user-config', '--skip-git-repo-check',
                   '--sandbox', 'read-only', '--json', '--model', 'gpt-6-astra',
                   '--enable', 'skip_host_skill_discovery', '-c', 'suppress_unstable_features_warning=true',
                   '-c', 'model_provider="offline-check"', '-c', 'model_providers.offline-check=' + provider]
        for feature in DISABLED_TOOLS:
            command += ['--disable', feature]
        process = subprocess.Popen(command + ['-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, env=environment, cwd=home,
                                   start_new_session=True)
        try:
            out, err = process.communicate('Offline configuration check.', timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            out, err = process.communicate(timeout=3)
        events = [json.loads(line) for line in out.splitlines() if line.startswith('{')]
        study.require(any(e['type'] == 'turn.started' for e in events), 'Initialization did not finish')
        study.require(not any(e.get('item', {}).get('type') == 'error' for e in events),
                      'Client still emits an initialization error')
        study.require(not any(e['type'] == 'turn.completed' for e in events),
                      'Offline preflight unexpectedly completed a model request')
        return dict(status='passed', events=events, stderr=err, credentials_loaded=False,
                    model_endpoint='http://127.0.0.1:1', completed_model_requests=0,
                    code_mode=False, code_mode_host=True)


def prepare(root):
    protocol = study.check(root)
    old_roots = (root, root / 'client_repair_01')
    attempts, failure_hashes = [], {}
    for old in old_roots:
        prior = study.check(old)
        study.require(study.read(old / 'lifecycle.json')['status'] == 'stopped', 'Coordinator still running')
        diagnosis, hashes = diagnose(old, prior)
        attempts.extend(diagnosis['attempts'])
        failure_hashes.update(hashes)
    target = root / 'client_bundle_02'
    target.mkdir(exist_ok=False)
    for name in protocol['source_sha256']:
        path = target / 'sources' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / 'sources' / name, path)
    for script in ('scripts/prepare_online_extension_client_repair.py',
                   'scripts/prepare_online_extension_bundle_repair.py'):
        shutil.copy2(study.ROOT / script, target / 'sources' / script)
        protocol['source_sha256'][script] = study.digest(study.ROOT / script)
    for name in protocol['artifact_sha256']:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, path)
    for task in study.STUDIES:
        (target / 'tasks' / task / 'sources').symlink_to('../../sources', target_is_directory=True)
    original_client = Path(protocol['client_original_path'])
    study.require(study.digest(original_client) == protocol['client_sha256'], 'Installed client changed')
    shutil.copytree(original_client.parent, target / 'client')
    (target / 'bin').mkdir()
    wrapper = target / 'bin/codex'
    wrapper.write_text('#!/bin/sh\nexec ' + shlex.quote(str(target / 'client/codex'))
                       + ' "$@" --disable code_mode\n')
    wrapper.chmod(0o755)
    study.write(target / 'offline_client_check.json', offline_check(wrapper, target))
    overhead = dict(attempts=attempts,
                    total_recorded_tokens=sum(a['recovered_recorded_usage']['total_tokens'] for a in attempts),
                    interpretation='Preserved infrastructure attempts before policy execution; '
                                   'reported separately from valid-episode costs and included in campaign overhead')
    study.write(target / 'infrastructure_attempts.json', overhead)
    for index, old in enumerate(old_roots):
        for name in ('progress.json', 'lifecycle.json', 'STOP.json', 'protocol.json'):
            shutil.copy2(old / name, target / f'failed_{index}_{name}')
    copied = [p for p in (target / 'client').rglob('*') if p.is_file()]
    copied += [target / name for name in ('offline_client_check.json', 'infrastructure_attempts.json')]
    copied += list(target.glob('failed_*.json'))
    protocol['artifact_sha256'].update({str(p.relative_to(target)): study.digest(p) for p in copied})
    protocol['legacy_evidence_sha256'].update(failure_hashes)
    protocol.update(registered_at=time.time(), client_sha256=study.digest(wrapper),
                    client_repair=dict(original_root=str(root),
                        complete_client_bundle=True, code_mode_tool=False, host_component_installed=True,
                        cause='Missing mandatory host component; disabling host still emits an initialization error',
                        original_attempts_preserved=8, affected_slots=4, episode_slots_unchanged=270,
                        total_attempts_including_startup_failures=278, old_contexts_reused=False,
                        audit_relaxed=False, future_failure_policy='Stop; no automatic retries'))
    study.write(target / 'protocol.json', protocol)
    (target / 'protocol.sha256').write_text(study.digest(target / 'protocol.json') + '\n')
    study.write(target / 'progress.json', dict(status='registered', completed=0, total=270,
                                             legacy_completed=30, pooled_total=300))
    study.check(target)
    study.write(root / 'continuation.json', dict(root=str(target), progress=str(target / 'progress.json'),
                                               reason=protocol['client_repair']))
    print(json.dumps(dict(root=str(target), infrastructure_tokens=overhead['total_recorded_tokens'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    prepare(parser.parse_args().root.resolve())
