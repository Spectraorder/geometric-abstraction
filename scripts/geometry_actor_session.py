"""Persistent, arm-isolated Codex actor sessions with archived turn provenance."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone
from uuid import UUID

from run_geometry_ablation import SCHEMA, audit_events, propose as propose_fresh

DISABLED_TOOLS = ('shell_tool', 'unified_exec', 'apps', 'browser_use',
                  'computer_use', 'skill_search', 'multi_agent')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def session_file(home, session_id):
    UUID(session_id)
    matches = list((home / 'sessions').rglob(f'*{session_id}.jsonl'))
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one persisted actor session')
    return matches[0]


def session_events(path, session_id):
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not events or events[0].get('type') != 'session_meta' or events[0].get('payload', {}).get('id') != session_id:
        raise ValueError('Actor session transcript identity mismatch')
    return events


def propose(directory, prompt, images, manifest):
    """Resume only this arm's verified previous turn; incomplete calls need inspection."""
    mode = manifest.get('actor_context_mode', 'fresh')
    if mode == 'fresh':
        return propose_fresh(directory, prompt, images, manifest)
    if mode != 'persistent':
        raise ValueError('Unsupported actor context mode')
    directory = Path(directory).resolve()
    arm = directory.parent.name
    root = directory.parents[1]
    revision = int(directory.name.removeprefix('round_'))
    if arm not in manifest['arms'] or revision < 1:
        raise ValueError('Invalid actor chain location')
    directory.parent.mkdir(parents=True, exist_ok=True)
    state = directory.parent / '.actor_session'
    state.mkdir(mode=0o700, exist_ok=True)
    with (state / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _propose_locked(directory, prompt, images, manifest, root, arm, revision, state)


def _propose_locked(directory, prompt, images, manifest, root, arm, revision, state):
    home, work = state / 'codex_home', state / 'work'
    home.mkdir(mode=0o700, exist_ok=True)
    work.mkdir(mode=0o700, exist_ok=True)
    previous = None
    before = None
    receipt = state / 'state.json'
    if revision == 1:
        if receipt.exists() or any((home / 'sessions').rglob('*.jsonl')):
            raise RuntimeError('First actor turn cannot inherit a session')
    else:
        previous_dir = directory.parent / f'round_{revision - 1:03d}'
        previous = read(previous_dir / 'generation.json')
        saved = read(receipt)
        if (saved.get('arm') != arm or saved.get('revision') != revision - 1
                or saved.get('manifest_sha256') != digest(root / 'manifest.json')
                or saved.get('session_id') != previous.get('session_id')
                or saved.get('session_sha256') != previous.get('session_sha256')):
            raise RuntimeError('Actor session does not match the previous completed revision')
        before = digest(session_file(home, saved['session_id']))
        if before != previous['session_sha256'] or digest(previous_dir / 'session.jsonl') != before:
            raise RuntimeError('Actor session changed outside the registered chain')
    # Never overwrite an incomplete call or silently retry a session that advanced.
    directory.mkdir(exist_ok=False)
    (directory / 'prompt.txt').write_text(prompt)
    write(directory / 'schema.json', SCHEMA)
    command = ['codex', 'exec', '--sandbox', 'read-only', '--cd', str(work)]
    if previous is not None:
        command += ['resume', previous['session_id']]
    command += ['--ignore-user-config', '--skip-git-repo-check', '--model', manifest['model'],
                '-c', 'project_doc_max_bytes=0', '-c', 'model_reasoning_effort="high"',
                '-c', 'suppress_unstable_features_warning=true',
                '-c', 'cli_auth_credentials_store="file"', '--enable', 'skip_host_skill_discovery']
    for name in DISABLED_TOOLS:
        command += ['--disable', name]
    command += ['--output-schema', str(directory / 'schema.json'),
                '--output-last-message', str(directory / 'actor_response.json'), '--json']
    submitted_images = []
    for index, source in enumerate(images):
        source = Path(source).resolve()
        if (not source.is_relative_to(root / arm) or 'heldout' in source.parts
                or 'reserved_revisions' in source.parts):
            raise ValueError('Only own training images may enter an actor session')
        target = work / f'round_{revision:03d}_sensor_{index}.png'
        shutil.copyfile(source, target)
        command += ['--image', str(target)]
        submitted_images.append({'sha256': digest(source), 'name': str(source.relative_to(root))})
    command += ['-']
    metadata = {'started_at': datetime.now(timezone.utc).isoformat(), 'command': command,
                'prompt_sha256': digest(directory / 'prompt.txt'), 'images': submitted_images,
                'image_count': len(images), 'actor_context_mode': 'persistent',
                'fresh_context': previous is None, 'tools_disabled': True,
                'user_config_loaded': False, 'skills_loaded': False, 'repository_context': False,
                'arm': arm, 'revision': revision,
                'resumed_session_id': previous['session_id'] if previous else None,
                'session_before_sha256': before, 'manifest_sha256': digest(root / 'manifest.json')}
    write(directory / 'invocation.json', metadata)
    auth_source = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))) / 'auth.json'
    auth_copy = home / 'auth.json'
    started = time.monotonic()
    try:
        if auth_source.exists():
            shutil.copyfile(auth_source, auth_copy)
            auth_copy.chmod(0o600)
        with (directory / 'model.jsonl').open('w') as log, (directory / 'model.stderr').open('w') as err:
            subprocess.run(command, input=prompt, text=True, stdout=log, stderr=err, check=True,
                           timeout=manifest['model_timeout_s'], cwd=work,
                           env=dict(os.environ, CODEX_HOME=str(home)))
        events = [json.loads(line) for line in (directory / 'model.jsonl').read_text().splitlines()
                  if line.startswith('{')]
        usage = audit_events(events)
        ids = {e['thread_id'] for e in events if e.get('type') == 'thread.started'}
        if len(ids) != 1:
            raise RuntimeError('Expected one actor thread identity')
        session_id = ids.pop()
        if previous is not None and session_id != previous['session_id']:
            raise RuntimeError('Actor resumed a different session')
        transcript = session_file(home, session_id)
        saved_events = session_events(transcript, session_id)
        if previous is not None and not transcript.read_bytes().startswith((previous_dir / 'session.jsonl').read_bytes()):
            raise RuntimeError('Actor session history was rewritten')
        shutil.copyfile(transcript, directory / 'session.jsonl')
        snapshot_sha = digest(directory / 'session.jsonl')
        response = read(directory / 'actor_response.json')
        if set(response) != {'code', 'hypothesis'} or not all(isinstance(v, str) for v in response.values()):
            raise ValueError('Invalid model response schema')
        write(directory / 'response.json', response)
        (directory / 'candidate.py').write_text(response['code'])
        generation = {'usage': usage, 'wall_seconds': time.monotonic() - started,
                      'audit': 'passed', 'source_budget_exceeded': len(response['code']) > manifest['max_candidate_characters'],
                      'candidate_sha256': digest(directory / 'candidate.py'), 'session_id': session_id,
                      'session_sha256': snapshot_sha, 'actor_context_mode': 'persistent',
                      'compaction_records_total': sum(e.get('type') == 'compacted' for e in saved_events)}
        write(receipt, {'arm': arm, 'revision': revision, 'session_id': session_id,
                        'session_sha256': snapshot_sha, 'manifest_sha256': digest(root / 'manifest.json')})
        # Write the completion marker last. A partially advanced session is never retried automatically.
        write(directory / 'generation.json', generation)
    except Exception as error:
        write(directory / 'attempt_failure.json', {'error_type': type(error).__name__,
              'wall_seconds': time.monotonic() - started, 'usage_status': 'inspect_raw_events',
              'automatic_retry_allowed': False})
        raise
    finally:
        auth_copy.unlink(missing_ok=True)


def audit_session(directory, manifest, invocation, generation):
    """Verify persistent transcript lineage independently of the live session directory."""
    root, arm = directory.parents[1], directory.parent.name
    revision = int(directory.name.removeprefix('round_'))
    command = invocation['command']
    if ('--ephemeral' in command or '--last' in command or '--all' in command
            or invocation.get('actor_context_mode') != 'persistent'
            or generation.get('actor_context_mode') != 'persistent'
            or invocation.get('arm') != arm or invocation.get('revision') != revision
            or invocation.get('manifest_sha256') != digest(root / 'manifest.json')
            or invocation.get('fresh_context') != (revision == 1)):
        raise ValueError('Persistent actor identity or isolation mismatch')
    for name in DISABLED_TOOLS:
        if not any(command[i:i + 2] == ['--disable', name] for i in range(len(command) - 1)):
            raise ValueError('Persistent actor tool isolation mismatch')
    if '--sandbox' not in command or command[command.index('--sandbox') + 1] != 'read-only':
        raise ValueError('Persistent actor sandbox mismatch')
    session_id = generation['session_id']
    UUID(session_id)
    snapshot = directory / 'session.jsonl'
    session_events(snapshot, session_id)
    if digest(snapshot) != generation['session_sha256']:
        raise ValueError('Archived actor session changed')
    events = [json.loads(line) for line in (directory / 'model.jsonl').read_text().splitlines() if line.startswith('{')]
    if {e['thread_id'] for e in events if e.get('type') == 'thread.started'} != {session_id}:
        raise ValueError('Actor event session identity mismatch')
    if revision == 1:
        if 'resume' in command or invocation.get('resumed_session_id') is not None or invocation.get('session_before_sha256') is not None:
            raise ValueError('Initial actor turn inherited a session')
    else:
        previous_dir = directory.parent / f'round_{revision - 1:03d}'
        previous = read(previous_dir / 'generation.json')
        if ('resume' not in command or command[command.index('resume') + 1] != session_id
                or previous.get('session_id') != session_id
                or invocation.get('resumed_session_id') != session_id
                or invocation.get('session_before_sha256') != previous.get('session_sha256')
                or digest(previous_dir / 'session.jsonl') != previous.get('session_sha256')
                or not snapshot.read_bytes().startswith((previous_dir / 'session.jsonl').read_bytes())):
            raise ValueError('Persistent actor history chain mismatch')
    for other in manifest['arms']:
        if other == arm:
            continue
        for path in (root / other).glob('round_*/generation.json'):
            if read(path).get('session_id') == session_id:
                raise ValueError('Actor session shared across experimental arms')
