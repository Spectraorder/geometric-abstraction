"""Persistent Astra/Luna decision sessions with physics-aware waiting and usage logs."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from geometry_actor_session import (DISABLED_TOOLS, audit_events, digest,
                                    session_file, session_events, write)
from verified_token_usage import turn_usage

SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['action', 'skill', 'node', 'plan', 'reason'],
    'properties': {
        'action': {'type': 'string', 'enum': ['execute', 'clear', 'observe', 'release', 'replan', 'wait']},
        'skill': {'type': 'string', 'enum': ['bridge', 'stack', 'none']},
        'node': {'type': 'string'},
        'plan': {'type': 'array', 'items': {'type': 'string'}},
        'reason': {'type': 'string'},
    },
}


def audit_decision_events(events, model, resumed):
    # Codex emits a typed error item for an intentional model change even when
    # the turn succeeds. Permit only this exact informational warning.
    allowed_warnings = {
        'This session was recorded with model `'+previous+'` but is resuming with `'
        +model+'`. Consider switching back to `'+previous+'` as it may affect Codex performance.'
        for previous in ('gpt-6-astra', 'gpt-6-luna') if previous != model
    } if resumed else set()
    filtered = [event for event in events if not (
        event.get('type') == 'item.completed' and event.get('item', {}).get('type') == 'error'
        and event['item'].get('message') in allowed_warnings)]
    return audit_events(filtered)


def validate_response(response):
    if set(response) != set(SCHEMA['required']):
        raise ValueError('Unexpected decision fields')
    for key in ('action', 'skill'):
        if response[key] not in SCHEMA['properties'][key]['enum']:
            raise ValueError('Invalid decision '+key)
    if (not isinstance(response['node'], str) or not isinstance(response['reason'], str)
            or not isinstance(response['plan'], list)
            or not all(isinstance(value, str) for value in response['plan'])):
        raise ValueError('Invalid decision types')
    if len(json.dumps(response)) > 6000:
        raise ValueError('Decision exceeded the registered response size')
    if response['action'] == 'execute' and response['skill'] == 'none':
        raise ValueError('Execute requires a skill')
    return response


class OnlineActor:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.home = self.output/'actor_home'
        self.work = self.output/'actor_work'
        self.home.mkdir(mode=0o700, exist_ok=False)
        self.work.mkdir(mode=0o700, exist_ok=False)
        self.session_id = None
        self.previous_counts = []
        self.previous_totals = {}
        self.previous_transcript = b''
        self.turn = 0
        self.receipts = []

    def request(self, prompt, images, *, slow, advance, deadline):
        """The owner advances physics while waiting; deadlines are never extended."""
        directory = self.output/f'decision_{self.turn:03d}'
        directory.mkdir(exist_ok=False)
        self.turn += 1
        model = 'gpt-6-astra' if slow else 'gpt-6-luna'
        effort = 'medium' if slow else 'low'
        (directory/'prompt.txt').write_text(prompt)
        write(directory/'schema.json', SCHEMA)
        command = ['codex', 'exec', '--sandbox', 'read-only', '--cd', str(self.work)]
        if self.session_id is not None:
            command += ['resume', self.session_id]
        command += ['--ignore-user-config', '--skip-git-repo-check', '--model', model,
                    '-c', 'project_doc_max_bytes=0', '-c', 'model_reasoning_effort="'+effort+'"',
                    '-c', 'web_search="disabled"',
                    '-c', 'suppress_unstable_features_warning=true',
                    '-c', 'cli_auth_credentials_store="file"', '--enable', 'skip_host_skill_discovery']
        for name in DISABLED_TOOLS:
            command += ['--disable', name]
        command += ['--output-schema', str(directory/'schema.json'), '--output-last-message',
                    str(directory/'response.json'), '--json']
        attachments = []
        for index, source in enumerate(images):
            target = self.work/f'turn_{self.turn:03d}_{index}.png'
            shutil.copyfile(source, target)
            command += ['--image', str(target)]
            attachments.append({'source': str(source), 'sha256': digest(source)})
        command.append('-')
        write(directory/'invocation.json', {'command': command, 'model': model,
              'effort': effort, 'images': attachments, 'session_before': self.session_id,
              'tools_disabled': list(DISABLED_TOOLS), 'repository_context': False})
        auth_source = Path(os.environ.get('CODEX_HOME', str(Path.home()/'.codex')))/'auth.json'
        auth_copy = self.home/'auth.json'
        started = time.monotonic()
        process = None
        try:
            shutil.copyfile(auth_source, auth_copy)
            auth_copy.chmod(0o600)
            with (directory/'model.jsonl').open('w') as out, (directory/'model.stderr').open('w') as err:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out,
                    stderr=err, text=True, cwd=self.work, start_new_session=True,
                    env=dict(os.environ, CODEX_HOME=str(self.home)))
                process.stdin.write(prompt)
                process.stdin.close()
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Episode budget exhausted during model inference')
                    advance()
                if process.returncode:
                    raise RuntimeError('Model process failed; see archived model.stderr')
            events = [json.loads(line) for line in (directory/'model.jsonl').read_text().splitlines()
                      if line.startswith('{')]
            raw_usage = audit_decision_events(events, model, self.session_id is not None)
            ids = {e['thread_id'] for e in events if e.get('type') == 'thread.started'}
            if len(ids) != 1 or (self.session_id is not None and ids != {self.session_id}):
                raise ValueError('Persistent model session identity changed')
            self.session_id = ids.pop()
            path = session_file(self.home, self.session_id)
            transcript = path.read_bytes()
            if not transcript.startswith(self.previous_transcript):
                raise ValueError('Model session history was rewritten')
            archived = session_events(path, self.session_id)
            counts = [e['payload']['info'] for e in archived
                      if e.get('type') == 'event_msg' and e.get('payload', {}).get('type') == 'token_count'
                      and e['payload'].get('info') is not None]
            if not counts or counts[:len(self.previous_counts)] != self.previous_counts:
                raise ValueError('Token history changed')
            totals = counts[-1]['total_token_usage']
            usage = turn_usage(self.previous_totals, totals, raw_usage,
                               counts[len(self.previous_counts):])
            response = validate_response(json.loads((directory/'response.json').read_text()))
            shutil.copyfile(path, directory/'session.jsonl')
            receipt = {'model': model, 'effort': effort, 'session_id': self.session_id,
                       'usage': usage, 'raw_cumulative_usage': raw_usage,
                       'wall_seconds': time.monotonic()-started, 'image_count': len(images)}
            write(directory/'receipt.json', receipt)
            self.previous_counts = counts
            self.previous_totals = totals
            self.previous_transcript = transcript
            self.receipts.append(receipt)
            return response
        except Exception as error:
            write(directory/'attempt_failure.json', {'type': type(error).__name__,
                  'error': str(error), 'wall_seconds': time.monotonic()-started,
                  'usage_status': 'inspect_raw_events', 'automatic_retry': False})
            raise
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            auth_copy.unlink(missing_ok=True)
