"""One explicitly authorized recovery of the interrupted paired ring study."""
import argparse
import ast
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import run_ring_peg_comparison_study as study

REPAIR = 'resume_01'
FAILED = 'replica_03/basic/round_002'
read, write, digest, require = study.read, study.write_json, study.digest, study.require


@contextmanager
def exclusive(path):
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def token_counts(path):
    return [e['payload']['info'] for e in map(json.loads, path.read_text().splitlines())
            if e.get('type') == 'event_msg' and e['payload'].get('type') == 'token_count'
            and e['payload'].get('info')]


def inspect_interrupted(previous, live):
    prefix = previous.read_bytes()
    require(live.read_bytes().startswith(prefix), 'Interrupted session rewrote completed history')
    extra = [json.loads(line) for line in live.read_bytes()[len(prefix):].splitlines()]
    require(not any(e.get('type') == 'response_item' and e.get('payload', {}).get('role') == 'assistant'
                    for e in extra), 'An assistant response exists; inspect rather than regenerate')
    before, after = token_counts(previous), token_counts(live)
    require(after[:len(before)] == before and before, 'Token history changed')
    delta = study.turn_usage(before[-1]['total_token_usage'], after[-1]['total_token_usage'],
                             {}, after[len(before):])
    return {'recorded_token_delta': delta, 'usage_complete': False,
            'usage_status': 'Unknown final usage: interrupted request has no completion receipt.',
            'note': 'Unchanged cumulative broadcasts are not new usage; zero recorded delta does not mean zero consumption.'}


def reusable_rollout(root, directory, arm, revision, seed, reserved=False, workers=24):
    stage = study.stage_path(directory, arm, revision)
    path = (root/'evaluation'/directory.name/arm/f'revision_{revision:03d}'/f'seed_{seed}'
            if reserved else stage/'trials'/f'seed_{seed}')
    if path.exists():
        require((path/'result.json').exists(), 'Incomplete trial cannot be retried')
        return study.audit_trial(path, stage/'candidate.py', arm, seed)
    require(not path.with_suffix('.log').exists(), 'Unfinished trial log requires inspection')
    return study.rollout(root, directory, arm, revision, seed, reserved, workers)


def reuse_or_propose(stage, prompt, images, manifest):
    if not stage.exists():
        return study.actor.propose(stage, prompt, images, manifest)
    require((stage/'generation.json').exists(), 'Incomplete model request cannot be retried')
    invocation, generation = read(stage/'invocation.json'), read(stage/'generation.json')
    study.actor.audit_session(stage, manifest, invocation, generation)
    require((stage/'prompt.txt').read_text() == prompt, 'Saved prompt differs from reconstructed prompt')
    require(digest(stage/'candidate.py') == generation['candidate_sha256'], 'Saved candidate changed')
    require([digest(path) for path in images] == [row['sha256'] for row in invocation['images']],
            'Saved training images differ')


def corrected_functions():
    """Keep the registered training and evaluation algorithms, changing dispatch only."""
    source = Path(study.__file__).read_text()
    replacements = {
        'target.mkdir(exist_ok=False)': 'target.mkdir(exist_ok=True)',
        '            stage.mkdir()\n            shutil.copyfile(root/\'starter.py\', stage/\'candidate.py\')':
            "            if stage.exists():\n                require(digest(stage/'candidate.py') == digest(root/'starter.py'), 'R0 changed')\n            else:\n                stage.mkdir()\n                shutil.copyfile(root/'starter.py', stage/'candidate.py')",
        "                actor.propose(stage, prompt, images, manifest)":
            "                reuse_or_propose(stage, prompt, images, manifest)",
        "            with lease(root, 'model', protocol['limits']['model'], f'{name}/{arm}/R{revision}'):":
            "            with model_slot(root, stage, protocol, name, arm, revision):",
        "(root/name/f'{arm}_campaign.log').open('x')":
            "(root/REPAIR/f'{name}_{arm}_campaign.log').open('x')",
        "             'started_at': time.time(), 'rollout_workers': 24}":
            "             'started_at': time.time(), 'rollout_workers': 24, 'recovery': REPAIR,\n             'prior_elapsed_seconds': read(root/REPAIR/'lifecycle.json')['elapsed_seconds'],\n             'incomplete_model_attempts': 1, 'model_attempt_budget': 31}",
    }
    for old, new in replacements.items():
        require(source.count(old) == 1, 'Frozen coordinator no longer matches recovery transform: '+old)
        source = source.replace(old, new)
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('chain', 'pipeline')]
    require(len(functions) == 2, 'Missing coordinator functions')
    namespace = dict(vars(study), __file__=__file__, REPAIR=REPAIR,
                     reuse_or_propose=reuse_or_propose, model_slot=model_slot,
                     rollout=reusable_rollout, freeze_training=freeze_training, summarize=summarize)
    exec(compile(ast.Module(body=functions, type_ignores=[]), __file__, 'exec'), namespace)
    return namespace['chain'], namespace['pipeline'], source


@contextmanager
def model_slot(root, stage, protocol, name, arm, revision):
    if (stage/'generation.json').exists():
        yield
    else:
        with study.lease(root, 'model', protocol['limits']['model'], f'{name}/{arm}/R{revision}'):
            yield


def register(root):
    protocol = study.check(root)
    require(read(root/'lifecycle.json')['status'] == 'failed', 'Campaign is not stopped')
    pid = read(root/'lifecycle.json')['pid']
    require(not Path(f'/proc/{pid}').exists(), 'Prior coordinator is still alive')
    require('timed out after 900 seconds' in read(root/'STOP.json')['error'], 'Unexpected stop cause')
    require(not (root/'all_candidates_frozen.json').exists() and not (root/'evaluation').exists(),
            'Unexpected reserved execution')
    destination = root/REPAIR
    require(not destination.exists(), 'Recovery already registered')
    stage = root/FAILED
    require(read(stage/'attempt_failure.json')['error_type'] == 'TimeoutExpired', 'Unexpected failed attempt')
    require(not any((stage/name).exists() for name in ('candidate.py', 'actor_response.json', 'generation.json')),
            'Failed turn has a response; do not regenerate')
    previous = stage.parent/'round_001/session.jsonl'
    state = read(stage.parent/'.actor_session/state.json')
    require(state['revision'] == 1 and state['session_sha256'] == digest(previous), 'Unexpected actor checkpoint')
    live = study.actor.session_file(stage.parent/'.actor_session/codex_home', state['session_id'])
    usage = inspect_interrupted(previous, live)
    actor_home = stage.parent/'.actor_session/codex_home'
    require(not (actor_home/'auth.json').exists(), 'Unexpected retained actor credentials')
    home_hashes = {str(p.relative_to(actor_home)): digest(p) for p in actor_home.rglob('*')
                   if p.is_file() and not p.is_symlink()}
    preserved, failed_leases = {}, {}
    for resource in ('model', 'rollout'):
        for path in (root/'scheduler'/resource/'events').glob('*.json'):
            row = read(path)
            require('finished_at' in row, 'Live or incomplete resource lease')
            if row['status'] != 'completed':
                require(resource == 'model' and row['status'] == 'failed' and row['label'] == 'replica_03/basic/R2',
                        'Unexpected failed lease')
                failed_leases[str(path.relative_to(root))] = digest(path)
    require(len(failed_leases) == 1, 'Expected exactly one interrupted request')
    for replica in protocol['replicas']:
        directory = root/replica['name']
        manifest = read(directory/'manifest.json')
        for arm in study.ARMS:
            for revision in range(4):
                candidate_stage = study.stage_path(directory, arm, revision)
                if not candidate_stage.exists() or candidate_stage == stage:
                    continue
                if revision:
                    invocation, generation = read(candidate_stage/'invocation.json'), read(candidate_stage/'generation.json')
                    study.actor.audit_session(candidate_stage, manifest, invocation, generation)
                    require(digest(candidate_stage/'candidate.py') == generation['candidate_sha256'], 'Candidate changed')
                for trial in (candidate_stage/'trials').glob('seed_*'):
                    if not trial.is_dir():
                        continue
                    study.audit_trial(trial, candidate_stage/'candidate.py', arm, int(trial.name.removeprefix('seed_')))
            for path in (directory/arm).rglob('*'):
                if path.is_file() and '.actor_session' not in path.parts and not path.is_relative_to(stage):
                    if path.name not in ('progress.json', 'training_curve.json'):
                        preserved[str(path.relative_to(root))] = digest(path)
    require(len(list(root.glob('replica_*/*/round_*/generation.json'))) == 22, 'Unexpected generation inventory')
    require(len(list(root.glob('replica_*/*/*/trials/*/result.json'))) == 140, 'Unexpected training inventory')
    destination.mkdir()
    for path in root.glob('replica_*/*/training_curve.json'):
        target = destination/'prior_curves'/path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copytree(stage, destination/'interrupted_attempt')
    shutil.copyfile(live, destination/'interrupted_session.jsonl')
    for name in ('STOP.json', 'lifecycle.json', 'progress.json', 'launch.json', 'campaign.log'):
        if (root/name).exists():
            shutil.copyfile(root/name, destination/name)
    shutil.copyfile(Path(__file__), destination/Path(__file__).name)
    _, _, source = corrected_functions()
    (destination/'corrected_runner.py').write_text(source)
    usage.update(wall_seconds=read(stage/'attempt_failure.json')['wall_seconds'],
                 image_count=read(stage/'invocation.json')['image_count'])
    write(destination/'interrupted_resources.json', usage)
    archived = {str(p.relative_to(destination)): digest(p) for p in destination.rglob('*') if p.is_file()}
    write(destination/'registration.json', {
        'time': time.time(), 'authorization': 'User explicitly requested repair and resume.',
        'scope': 'Reuse completed candidates and trials; restore the failed actor to its completed R1 checkpoint and retry R2 once. No runtime, prompts, seeds, selection, timeouts or model settings change.',
        'protocol_sha256': digest(root/'protocol.json'), 'adapter_sha256': digest(Path(__file__)),
        'corrected_runner_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'preserved_files': preserved, 'archive_sha256': archived, 'failed_leases': failed_leases,
        'interrupted_stage': FAILED, 'live_session': str(live.relative_to(root)),
        'actor_home_sha256': home_hashes,
        'session_before_restore_sha256': digest(live), 'restore_sha256': digest(previous),
        'completed_generations': 22, 'remaining_generations': 8, 'successful_generation_budget': 30,
        'model_attempt_budget': 31, 'incomplete_attempts': 1,
        'remaining_training_slots': 60, 'reserved_slots': 4000,
        'incomplete_usage': usage, 'future_failure_rule': 'Stop and preserve; no further automatic retries.'})
    return check(root)


def check(root):
    study.check(root)
    destination = root/REPAIR
    receipt = read(destination/'registration.json')
    require(digest(root/'protocol.json') == receipt['protocol_sha256'], 'Recovery protocol changed')
    require(digest(Path(__file__)) == receipt['adapter_sha256'], 'Recovery adapter changed')
    _, _, source = corrected_functions()
    require(hashlib.sha256(source.encode()).hexdigest() == receipt['corrected_runner_sha256'], 'Recovery transform changed')
    for relative, sha in receipt['archive_sha256'].items():
        require(digest(destination/relative) == sha, 'Recovery archive changed: '+relative)
    for relative, sha in receipt['preserved_files'].items():
        require(digest(root/relative) == sha, 'Preserved evidence changed: '+relative)
    for relative, sha in receipt['failed_leases'].items():
        require(digest(root/relative) == sha, 'Interrupted lease changed')
    archived_home = destination/'original_codex_home'
    if archived_home.exists():
        for relative, sha in receipt['actor_home_sha256'].items():
            require(digest(archived_home/relative) == sha, 'Archived actor home changed')
    return receipt


def recovered_lease_audit(root, limits):
    receipt = read(root/REPAIR/'registration.json')
    report = {}
    for resource, limit in limits.items():
        events = []
        failed = 0
        for path in (root/'scheduler'/resource/'events').glob('*.json'):
            row = read(path)
            if row['status'] != 'completed':
                require(receipt['failed_leases'].get(str(path.relative_to(root))) == digest(path),
                        'Unregistered failed resource lease')
                failed += 1
            require('finished_at' in row, 'Incomplete resource lease')
            events.append(row)
        timeline = sorted((t, delta) for row in events
                          for t, delta in ((row['started_at'], 1), (row['finished_at'], -1)))
        active = maximum = 0
        for _, delta in timeline:
            active += delta
            maximum = max(maximum, active)
        require(maximum <= limit, 'Concurrency limit exceeded')
        report[resource] = {'jobs': len(events), 'completed': len(events)-failed, 'interrupted': failed,
                            'max_concurrent': maximum, 'limit': limit,
                            'queue_seconds': sum(row['started_at']-row['queued_at'] for row in events)}
    require(report['model']['jobs'] <= receipt['model_attempt_budget'], 'Model attempt budget exceeded')
    return report


def freeze_training(root):
    receipt = check(root)
    recovered_lease_audit(root, study.check(root)['limits'])
    accounting = study.resources.summarize
    def with_incomplete(directory, rounds):
        result = accounting(directory, rounds)
        result['incomplete_attempts'] = ([dict(receipt['incomplete_usage'], arm='basic', revision=2)]
                                         if directory.name == 'replica_03' else [])
        result['all_attempt_usage_complete'] = not result['incomplete_attempts']
        result['notes'].append('Completed-turn totals exclude separately recorded interrupted attempts; unknown consumption is not zero.')
        return result
    with patch.object(study.resources, 'summarize', with_incomplete):
        return study.freeze_training(root)


def summarize(root):
    receipt = check(root)
    with patch.object(study, 'audit_leases', recovered_lease_audit):
        result = study.summarize(root)
    result['recovery'] = REPAIR
    result['incomplete_model_attempts'] = [receipt['incomplete_usage']]
    result['token_curve_scope'] = 'Verified completed-turn tokens only. One interrupted Base R2 request has unknown final token usage; total consumption is incomplete.'
    write(root/'summary.json', result)
    return result


def run(root):
    # Activation is one-shot. Original STOP and lifecycle bytes remain in the archive.
    with exclusive(root/'coordinator.lock'):
        receipt = check(root)
        destination = root/REPAIR
        require(not (destination/'activation.json').exists(), 'Recovery already attempted; inspect new failure')
        stage = root/FAILED
        live = root/receipt['live_session']
        require(digest(live) == receipt['session_before_restore_sha256'], 'Live session changed before activation')
        for name in ('STOP.json', 'lifecycle.json'):
            require(digest(root/name) == receipt['archive_sha256'][name], 'Stop state changed')
        require({p.name: digest(p) for p in stage.iterdir()} ==
                {p.name: digest(p) for p in (destination/'interrupted_attempt').iterdir()}, 'Failed attempt changed')
        write(destination/'activation.json', {'time': time.time(), 'pid': os.getpid(),
                                             'registration_sha256': digest(destination/'registration.json')})
        actor_home = stage.parent/'.actor_session/codex_home'
        for relative, sha in receipt['actor_home_sha256'].items():
            require(digest(actor_home/relative) == sha, 'Actor home changed before activation')
        stage.rename(destination/'original_failed_stage')
        # Rebuild local history indexes from the completed transcript only.
        # The interrupted databases and their WAL files remain archived unchanged.
        actor_home.rename(destination/'original_codex_home')
        live.parent.mkdir(parents=True, mode=0o700)
        shutil.copyfile(stage.parent/'round_001/session.jsonl', live)
        require(digest(live) == receipt['restore_sha256'], 'Checkpoint restore failed')
        (root/'STOP.json').rename(destination/'original_STOP.json')
        (root/'lifecycle.json').rename(destination/'original_lifecycle.json')
    _, pipeline, _ = corrected_functions()
    pipeline(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('register', 'check', 'run', 'chain', 'summarize'), required=True)
    parser.add_argument('--replica')
    parser.add_argument('--arm', choices=study.ARMS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == 'register':
        with exclusive(root/'coordinator.lock'):
            receipt = register(root)
            print('Registered recovery:', receipt['remaining_generations'], 'remaining generations')
    elif args.phase == 'check':
        print('Recovery verified:', check(root)['model_attempt_budget'], 'maximum attempts')
    elif args.phase == 'chain':
        try:
            check(root)
            chain, _, _ = corrected_functions()
            chain(root, args.replica, args.arm)
        except BaseException as error:
            study.stop(root, error)
            raise
    elif args.phase == 'summarize':
        summarize(root)
    else:
        run(root)
