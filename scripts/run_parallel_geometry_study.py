"""Isolated 50k policy chains with global model/rollout limits and selected testing."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import run_expanded_budget_geometry_experiment as variant
import evaluate_selected_interface_policies as selected_audit
from experiment_parallel import audit_leases, bounded_map, check_stop, lease, stop, write_json
from summarize_geometry_session_resources import summarize
from summarize_mixed_scene_learning import require

ROOT = variant.ROOT
base = variant.base
neutral = variant.interface.neutral
read = lambda path: json.loads(path.read_text())
DEPENDENCIES = [Path(__file__), ROOT/'scripts/experiment_parallel.py',
                Path(selected_audit.__file__), ROOT/'scripts/summarize_geometry_session_resources.py']


@contextmanager
def exclusive(path):
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prepare(root, replicas, rounds, train_start, test_start, layouts, models, rollouts):
    require(not root.exists(), 'Use a new study directory')
    require(1 <= replicas <= 5 and 1 <= rounds <= 5 and layouts > 0, 'Invalid study budget')
    require(1 <= models <= 4 and 1 <= rollouts <= 8, 'Invalid concurrency budget')
    tests = list(range(test_start, test_start+layouts))
    trains = list(range(train_start, train_start+5*replicas))
    require(not set(trains).intersection(tests), 'Training/test overlap')
    inspected = set()
    for pattern in ('**/manifest.json', '**/registration.json', '**/protocol.json'):
        for path in (ROOT/'outputs').glob(pattern):
            record = read(path)
            for key in ('train_seeds', 'heldout_seeds', 'reserved_seeds', 'test_seeds'):
                inspected.update(record.get(key, []))
    require(not set(trains+tests).intersection(inspected), 'New study overlaps registered layouts')
    root.mkdir(parents=True)
    rows = []
    for index in range(replicas):
        directory = root/f'replica_{index+1:02d}'
        manifest = variant.prepare(directory, trains[index*5:(index+1)*5], tests, rounds)
        rows.append({'directory': directory.name, 'manifest_sha256': base.digest(directory/'manifest.json'),
                     'train_seeds': manifest['train_seeds']})
    protocol = {'schema': 'parallel_geometry_selected_study_v1', 'registered_at': time.time(),
                'variant': variant.VERSION, 'replicas': rows, 'rounds': rounds, 'heldout_seeds': tests,
                'limits': {'model': models, 'rollout': rollouts}, 'training_slots': replicas*2*(rounds+1)*5,
                'model_calls': replicas*2*rounds, 'evaluation_slots': replicas*2*layouts,
                'minimum_free_disk_bytes': 15*1024**3,
                'source_sha256': {str(p.relative_to(ROOT)): base.digest(p) for p in DEPENDENCIES},
                'independence': 'Independent persistent sessions; disjoint five-layout training sets between replicas, shared within each arm pair. Common reserved split across all replicas.',
                'selection': 'Training-only strict success-set improvement; selected policy only at test.',
                'freeze_rule': 'Every replica must pass training audit before any reserved rollout.',
                'failure_rule': 'Stop new dispatch on an infrastructure error; drain active jobs. No automatic model or incomplete-trial retries.',
                'analysis': 'Report every paired replica and mean/range of paired differences. Shared test layouts and policies are not independent Bernoulli replications.'}
    for relative in protocol['source_sha256']:
        destination = root/'scheduler_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, destination)
    write_json(root/'protocol.json', protocol)
    check(root)
    return protocol


def check(root):
    protocol = read(root/'protocol.json')
    require(protocol['schema'] == 'parallel_geometry_selected_study_v1', 'Wrong study schema')
    for relative, digest in protocol['source_sha256'].items():
        require(base.digest(ROOT/relative) == digest == base.digest(root/'scheduler_snapshot'/relative), 'Scheduler source changed')
    for row in protocol['replicas']:
        directory = root/row['directory']
        manifest = read(directory/'manifest.json')
        require(base.digest(directory/'manifest.json') == row['manifest_sha256'], 'Replica manifest changed')
        require(manifest['heldout_seeds'] == protocol['heldout_seeds'] and manifest['train_seeds'] == row['train_seeds'], 'Replica split changed')
        variant.check(directory, manifest)
    return protocol


def train_replica(root, name):
    protocol = check(root)
    require(name in [row['directory'] for row in protocol['replicas']], 'Unknown replica')
    directory = root/name
    with exclusive(directory/'training_coordinator.lock'):
        check_stop(root)
        manifest = read(directory/'manifest.json')
        raw_execute = neutral.execute
        def one_trial(job):
            arm, candidate, target, seed, error = job
            output = target/f'seed_{seed}'
            if (output/'result.json').exists():
                return raw_execute(directory, arm, candidate, target, [seed], manifest, error)
            with lease(root, 'rollout', protocol['limits']['rollout'], f'{name}/{arm}/{target.name}/{seed}'):
                require(shutil.disk_usage(root).free >= protocol['minimum_free_disk_bytes'], 'Disk reserve reached')
                return raw_execute(directory, arm, candidate, target, [seed], manifest, error)
        def parallel_trials(study, arm, candidate, target, seeds, current, error=None):
            require(study == directory and current == manifest, 'Unexpected training context')
            jobs = [(arm, candidate, target, seed, error) for seed in seeds]
            results = {job[3]: result for job, result in bounded_map(one_trial, jobs, protocol['limits']['rollout'])}
            return [row for seed in seeds for row in results[seed]]
        if not (directory/'training_summary.json').exists():
            with patch.object(neutral, 'execute', parallel_trials):
                if not (directory/'starter_parity.json').exists():
                    variant.bootstrap(directory, manifest)
                with variant.configured():
                    original_propose = base.propose
                    def propose(stage, prompt, images, current):
                        with lease(root, 'model', protocol['limits']['model'], str(stage.relative_to(root))):
                            return original_propose(stage, prompt, images, current)
                    with patch.object(base, 'propose', propose):
                        base.campaign(directory, manifest, 'train')
        report = variant.audit(directory, allow_partial=True)
        require(report['exact_actor_contexts_verified'] == 2*protocol['rounds'], 'Incomplete actor audit')
        write_json(directory/'audit_training.json', report)
        write_json(directory/'resource_accounting.json', summarize(directory, protocol['rounds']))
        write_json(directory/'training_complete.json', {'status': 'completed', 'time': time.time(),
                   'training_sha256': base.digest(directory/'training_summary.json')})


def freeze(root, protocol):
    records = {}
    for row in protocol['replicas']:
        directory = root/row['directory']
        require((directory/'training_complete.json').exists(), 'All training must finish before test')
        variant.audit(directory, allow_partial=True)
        trained = read(directory/'training_summary.json')
        records[row['directory']] = {'selected': trained['frozen'],
            'evidence': {str(path.relative_to(directory)): base.digest(path)
                         for path in [directory/'training_summary.json', directory/'all_candidates_frozen.json',
                                      directory/'audit_training.json', directory/'resource_accounting.json'] +
                         [directory/arm/name for arm in trained['frozen'] for name in ('frozen.py', '.actor_session/state.json')]}}
    gate = {'protocol_sha256': base.digest(root/'protocol.json'), 'replicas': records}
    path = root/'all_candidates_frozen.json'
    if path.exists():
        require(read(path) == gate, 'Study freeze changed')
    else:
        write_json(path, gate)
    return gate


def check_gate(root, protocol):
    gate = read(root/'all_candidates_frozen.json')
    require(gate['protocol_sha256'] == base.digest(root/'protocol.json'), 'Study protocol changed')
    require(set(gate['replicas']) == {row['directory'] for row in protocol['replicas']}, 'Incomplete global freeze')
    for name, record in gate['replicas'].items():
        for relative, digest in record['evidence'].items():
            require(base.digest(root/name/relative) == digest, 'Frozen evidence changed')
    return gate


def register_evaluation(root, protocol):
    gate = check_gate(root, protocol)
    for row in protocol['replicas']:
        name = row['directory']
        output = root/name/'selected_evaluation'
        registration = {'reserved_seeds': protocol['heldout_seeds'], 'workers': protocol['limits']['rollout'],
            'required_rollouts': 2*len(protocol['heldout_seeds']), 'selected': gate['replicas'][name]['selected'],
            'global_freeze_sha256': base.digest(root/'all_candidates_frozen.json'),
            'limitations': ['One learned chain per arm in this replica; analyze all replica pairs.',
                            'Layouts are shared across replicas; do not pool as independent policy repetitions.']}
        path = output/'registration.json'
        if path.exists():
            require(read(path) == registration, 'Evaluation registration changed')
        else:
            write_json(path, registration)


def check_evaluation(root, directory, output):
    protocol = check(root)
    gate = check_gate(root, protocol)
    registration = read(output/'registration.json')
    require(registration['global_freeze_sha256'] == base.digest(root/'all_candidates_frozen.json'), 'Global freeze changed')
    require(registration['selected'] == gate['replicas'][directory.name]['selected'], 'Test selection changed')
    require(registration['reserved_seeds'] == protocol['heldout_seeds'], 'Test split changed')
    return registration, read(directory/'manifest.json')


def evaluate(root, protocol):
    register_evaluation(root, protocol)
    jobs = []
    completed = 0
    for seed in protocol['heldout_seeds']:
        for index, row in enumerate(protocol['replicas']):
            arms = ('basic', 'symbolic') if (index+seed)%2 else ('symbolic', 'basic')
            for arm in arms:
                jobs.append((row['directory'], arm, seed))
    manifests = {row['directory']: read(root/row['directory']/'manifest.json') for row in protocol['replicas']}
    def execute(job):
        name, arm, seed = job
        directory = root/name
        target = directory/'selected_evaluation'/arm/'heldout'
        if (target/f'seed_{seed}'/'result.json').exists():
            return base.execute(directory, arm, directory/arm/'frozen.py', target, [seed], manifests[name])
        with lease(root, 'rollout', protocol['limits']['rollout'], f'{name}/{arm}/test/{seed}'):
            require(shutil.disk_usage(root).free >= protocol['minimum_free_disk_bytes'], 'Disk reserve reached')
            return base.execute(directory, arm, directory/arm/'frozen.py', target, [seed], manifests[name])
    with variant.configured():
        for job, _ in bounded_map(execute, jobs, protocol['limits']['rollout'], on_error=lambda error: stop(root, error)):
            completed += 1
            write_json(root/'progress.json', {'status': 'evaluating', 'pid': os.getpid(),
                       'completed': completed, 'total': len(jobs), 'last_job': job, 'time': time.time()})
    return audit(root, protocol)


def audit(root, protocol):
    check_gate(root, protocol)
    reports = {}
    adapter = SimpleNamespace(neutral=neutral, audit=variant.audit)
    for row in protocol['replicas']:
        directory = root/row['directory']
        output = directory/'selected_evaluation'
        # Reuse the frozen result auditor; audit calls are sequential in this process.
        with patch.multiple(selected_audit, experiment=adapter,
                            check=lambda directory, output: check_evaluation(root, directory, output)):
            report = selected_audit.audit(directory, output)
        write_json(output/'audit.json', report)
        reports[row['directory']] = report
    first_report = next(iter(reports.values()))
    for report in reports.values():
        for relative, digest in first_report['result_hashes'].items():
            if relative.endswith('/initial.png'):
                require(report['result_hashes'][relative] == digest, 'Initial layout differs between replicas')
    sessions = []
    for row in protocol['replicas']:
        for arm in ('basic', 'symbolic'):
            sessions.append(read(root/row['directory']/arm/'.actor_session/state.json')['session_id'])
    require(len(set(sessions)) == len(sessions), 'Actor sessions were shared between chains')
    differences = [report['paired']['second_minus_first_sr'] for report in reports.values()]
    resources = audit_leases(root, protocol['limits'])
    require(resources['model']['jobs'] == protocol['model_calls'], 'Unexpected number of model attempts')
    report = {'status': 'completed', 'protocol_sha256': base.digest(root/'protocol.json'),
              'replicas': {name: {key: value for key, value in value.items() if key != 'result_hashes'} for name, value in reports.items()},
              'mean_success_rate': {arm: statistics.mean(r['arms'][arm]['success_rate'] for r in reports.values())
                                    for arm in ('basic', 'symbolic')},
              'paired_differences': differences, 'mean_paired_difference': statistics.mean(differences),
              'paired_difference_range': [min(differences), max(differences)], 'scheduler': resources,
              'evaluation_rollouts': sum(r['valid_rollouts'] for r in reports.values()),
              'analysis': protocol['analysis'], 'resource_accounting': {row['directory']: read(root/row['directory']/'resource_accounting.json') for row in protocol['replicas']}}
    require(report['evaluation_rollouts'] == protocol['evaluation_slots'], 'Incomplete test budget')
    write_json(root/'summary.json', report)
    write_json(root/'progress.json', {'status': 'completed', 'time': time.time(), 'pid': os.getpid(),
               'completed': report['evaluation_rollouts'], 'total': protocol['evaluation_slots']})
    return report


@contextmanager
def training_progress(root, protocol):
    finished = threading.Event()
    def update():
        while not finished.is_set():
            replicas = {}
            for row in protocol['replicas']:
                directory = root/row['directory']
                path = directory/'progress.json'
                try:
                    progress = read(path) if path.exists() else {}
                except json.JSONDecodeError:
                    progress = {}
                replicas[directory.name] = {
                    'status': 'completed' if (directory/'training_complete.json').exists() else progress.get('status', 'starting'),
                    'revision': progress.get('revision'),
                    'generations': len(list(directory.glob('*/round_*/generation.json'))),
                    'training_slots': len(list(directory.glob('*/round_*/trials/seed_*/result.json'))) +
                                      len(list(directory.glob('*/revision_000/trials/seed_*/result.json')))}
            active = {}
            for resource in ('model', 'rollout'):
                events = [read(path) for path in (root/'scheduler'/resource/'events').glob('*.json')]
                active[resource] = sum(event['status'] == 'running' for event in events)
            write_json(root/'progress.json', {'status': 'training', 'pid': os.getpid(), 'time': time.time(),
                       'limits': protocol['limits'], 'active': active, 'replicas': replicas})
            finished.wait(5)
    thread = threading.Thread(target=update, daemon=True)
    thread.start()
    try:
        yield
    finally:
        finished.set()
        thread.join()


def run(root, phase):
    with exclusive(root/'coordinator.lock'):
        protocol = check(root)
        check_stop(root)
        if (root/'summary.json').exists():
            return audit(root, protocol)
        started = time.time()
        write_json(root/'lifecycle.json', {'status': 'running', 'pid': os.getpid(), 'started_at': started})
        try:
            if phase != 'evaluate':
                def train(row):
                    check_stop(root)
                    directory = root/row['directory']
                    with (directory/'training.log').open('a') as log:
                        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--output', str(root),
                                        '--phase', 'worker', '--replica', row['directory']],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    return row['directory']
                write_json(root/'progress.json', {'status': 'training', 'pid': os.getpid(), 'total_replicas': len(protocol['replicas'])})
                with training_progress(root, protocol):
                    for _, name in bounded_map(train, protocol['replicas'], len(protocol['replicas']), on_error=lambda error: stop(root, error)):
                        print('TRAINING COMPLETE '+name, flush=True)
                freeze(root, protocol)
            else:
                check_gate(root, protocol)
            if phase == 'train':
                write_json(root/'progress.json', {'status': 'trained'})
            else:
                evaluate(root, protocol)
            write_json(root/'lifecycle.json', {'status': 'trained' if phase == 'train' else 'completed',
                       'pid': os.getpid(), 'elapsed_seconds': time.time()-started})
        except BaseException as error:
            stop(root, error)
            write_json(root/'progress.json', {'status': 'failed', 'error': str(error), 'time': time.time()})
            write_json(root/'lifecycle.json', {'status': 'failed', 'pid': os.getpid(), 'error': str(error),
                       'elapsed_seconds': time.time()-started})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'all', 'train', 'evaluate', 'audit', 'worker'), default='all')
    parser.add_argument('--replica')
    parser.add_argument('--replicas', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--train-start', type=int, default=2027020101)
    parser.add_argument('--test-start', type=int, default=2027021001)
    parser.add_argument('--layouts', type=int, default=100)
    parser.add_argument('--model-workers', type=int, default=4)
    parser.add_argument('--rollout-workers', type=int, default=4)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.phase == 'prepare':
        result = prepare(root, args.replicas, args.rounds, args.train_start, args.test_start,
                         args.layouts, args.model_workers, args.rollout_workers)
        print({key: result[key] for key in ('training_slots', 'model_calls', 'evaluation_slots', 'limits')})
    elif args.phase == 'worker':
        try:
            train_replica(root, args.replica)
        except BaseException as error:
            stop(root, error)
            raise
    elif args.phase == 'audit':
        with exclusive(root/'coordinator.lock'):
            print(audit(root, check(root))['status'])
    else:
        def interrupt(signum, frame):
            stop(root, RuntimeError(f'Supervisor received signal {signum}; draining active work'))
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        run(root, args.phase)
