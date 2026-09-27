"""Run the complete three-cube Base/Full study with bounded global parallelism."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import signal
import threading
import time
from unittest.mock import patch

import run_parallel_geometry_study as study
import three_cube_geometry_experiment as variant
import evaluate_three_cube_curves as planned_curves
from experiment_parallel import lease as original_lease, stop, write_json
from summarize_mixed_scene_learning import require

ROOT = study.ROOT
base = study.base
read = study.read
DEPENDENCIES = study.DEPENDENCIES + [Path(__file__), Path(variant.__file__), Path(planned_curves.__file__),
                                    ROOT/'scripts/evaluate_geometry_learning_curves.py']


def resource_workers(sample, requested):
    if planned_curves.curves.pilot_workers([sample], requested) < requested:
        return min(requested, 8)
    return min(requested, 16) if sample['memory']['MemAvailable'] < 12*1024**3 else requested


@contextmanager
def monitored_resources(root, requested):
    finished = threading.Event()
    def update():
        while not finished.is_set():
            try:
                sample = planned_curves.curves.hardware()
                record = dict(sample, effective_rollout_workers=resource_workers(sample, requested))
                write_json(root/'resource_status.json', record)
                with (root/'hardware.jsonl').open('a') as handle:
                    import json
                    handle.write(json.dumps(record)+'\n')
            except Exception as error:
                write_json(root/'resource_status.json', {'effective_rollout_workers': min(requested, 8), 'error': str(error)})
            finished.wait(3)
    thread = threading.Thread(target=update, daemon=True)
    thread.start()
    try:
        yield
    finally:
        finished.set()
        thread.join()


@contextmanager
def resource_lease(root, resource, slots, label):
    status = root/'resource_status.json'
    if resource == 'rollout' and status.exists():
        slots = min(slots, read(status)['effective_rollout_workers'])
    with original_lease(root, resource, slots, label):
        yield


@contextmanager
def installed():
    with variant.execution_adapter(), patch.object(study, 'variant', variant), \
         patch.object(study, '__file__', str(Path(__file__).resolve())), \
         patch.object(study, 'lease', resource_lease), planned_curves.installed():
        yield


def prepare(root, replicas=5, rounds=3, train_start=2027090101, test_start=2027091001,
            layouts=100, models=4, rollouts=24):
    require(not root.exists(), 'Use a new study directory')
    require(replicas == 5 and rounds == 3 and layouts == 100, 'This complete study registers five pairs, R0-R3 and 100 test layouts')
    require(1 <= models <= 4 and rollouts in (8, 16, 24), 'Invalid concurrency budget')
    trains = list(range(train_start, train_start+5*replicas))
    tests = list(range(test_start, test_start+layouts))
    require(not set(trains).intersection(tests), 'Training/test overlap')
    inspected = set()
    for pattern in ('**/manifest.json', '**/registration.json', '**/protocol.json'):
        for path in (ROOT/'outputs').glob(pattern):
            record = read(path)
            for key in ('train_seeds', 'heldout_seeds', 'reserved_seeds', 'test_seeds'):
                inspected.update(record.get(key, []))
    require(not set(trains+tests).intersection(inspected), 'Study overlaps registered layouts')
    root.mkdir(parents=True)
    rows = []
    for index in range(replicas):
        directory = root/f'replica_{index+1:02d}'
        manifest = variant.prepare(directory, trains[index*5:(index+1)*5], tests, rounds)
        rows.append({'directory': directory.name, 'manifest_sha256': base.digest(directory/'manifest.json'),
                     'train_seeds': manifest['train_seeds']})
    protocol = {'schema': 'parallel_geometry_selected_study_v1', 'registered_at': time.time(),
        'variant': variant.VERSION, 'task': 'three_cube_vertical_stack', 'replicas': rows, 'rounds': rounds,
        'heldout_seeds': tests, 'limits': {'model': models, 'rollout': rollouts},
        'training_slots': 200, 'model_calls': 30, 'evaluation_slots': 1000,
        'learning_curve_plan': {'revisions': [0, 1, 2, 3], 'total_slots': 4000, 'reuse_selected_slots': 1000,
                               'additional_slots': 3000, 'registered_before_training': True,
                               'scope': planned_curves.SCOPE},
        'minimum_free_disk_bytes': 15*1024**3,
        'source_sha256': {str(p.relative_to(ROOT)): base.digest(p) for p in DEPENDENCIES},
        'hardware_at_registration': planned_curves.curves.hardware(),
        'resource_rule': 'Global cap 24; new training/selected-test leases reduce to 16 below 12 GiB available RAM, or eight below 6 GiB or above 90 percent GPU memory. Active leases drain. Curve stage also uses a 48-slot action-policy pilot.',
        'independence': 'Disjoint five-layout training sets between replicas; shared within each Base/Full pair. All replicas use 100 common reserved layouts.',
        'selection': 'Training-only strict success-set improvement; ties retain incumbent.',
        'freeze_rule': 'Every replica must pass training audit before any reserved rollout. Evaluate every frozen revision.',
        'failure_rule': 'Stop dispatch and drain on infrastructure error. Retain incomplete attempts; no silent retry.',
        'analysis': 'Report raw and training-retained SR by iteration and cumulative tokens, paired-chain variability and rollout costs. Shared test layouts are not independent policy replications.'}
    for relative in protocol['source_sha256']:
        destination = root/'scheduler_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, destination)
    write_json(root/'protocol.json', protocol)
    study.check(root)
    write_json(root/'progress.json', {'status': 'registered', 'training_slots': 200, 'test_slots': 4000})
    return protocol


def run(root, phase):
    with study.exclusive(root/'pipeline.lock'):
        protocol = study.check(root)
        try:
            if not (root/'summary.json').exists():
                with monitored_resources(root, protocol['limits']['rollout']):
                    study.run(root, 'train' if phase == 'train' else 'all')
            if phase == 'train':
                return
            output = root/'learning_curves'
            if not (output/'registration.json').exists():
                planned_curves.prepare(root, output, root/'curve_data', protocol['limits']['rollout'])
            write_json(root/'lifecycle.json', {'status': 'learning_curves', 'pid': os.getpid(), 'time': time.time()})
            write_json(root/'progress.json', {'status': 'learning_curves', 'details': 'learning_curves/progress.json',
                                             'selected_test_slots_completed': 1000, 'all_revision_test_slots': 4000})
            planned_curves.curves.run(root, output)
            write_json(root/'lifecycle.json', {'status': 'completed', 'pid': os.getpid(), 'time': time.time()})
            write_json(root/'progress.json', {'status': 'completed', 'training_slots': 200, 'test_slots': 4000,
                                             'model_calls': 30, 'curves': 'learning_curves/summary.json'})
        except BaseException as error:
            stop(root, error)
            write_json(root/'lifecycle.json', {'status': 'failed', 'pid': os.getpid(), 'error': str(error), 'time': time.time()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'all', 'train', 'audit', 'worker'), default='all')
    parser.add_argument('--replica')
    parser.add_argument('--train-start', type=int, default=2027090101)
    parser.add_argument('--test-start', type=int, default=2027091001)
    parser.add_argument('--model-workers', type=int, default=4)
    parser.add_argument('--rollout-workers', type=int, default=24)
    args = parser.parse_args()
    root = args.output.resolve()
    def interrupt(signum, frame):
        stop(root, RuntimeError(f'Supervisor received signal {signum}; draining active work'))
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with installed():
        if args.phase == 'prepare':
            report = prepare(root, train_start=args.train_start, test_start=args.test_start,
                             models=args.model_workers, rollouts=args.rollout_workers)
            print({key: report[key] for key in ('training_slots', 'model_calls', 'evaluation_slots', 'learning_curve_plan', 'limits')})
        elif args.phase == 'worker':
            try:
                study.train_replica(root, args.replica)
            except BaseException as error:
                stop(root, error)
                raise
        elif args.phase == 'audit':
            with study.exclusive(root/'pipeline.lock'), study.exclusive(root/'coordinator.lock'):
                study.audit(root, study.check(root))
                planned_curves.curves.audit(root, root/'learning_curves')
            print('Completed study audit passed')
        else:
            run(root, args.phase)


if __name__ == '__main__':
    main()
