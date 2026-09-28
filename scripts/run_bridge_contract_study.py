"""Run five paired bridge chains and all 4,000 registered reserved slots."""
import argparse
import hashlib
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import signal
import time
from unittest.mock import patch

import run_three_cube_geometry_study as parent
import bridge_contract_experiment as variant
import bridge_actor_transport as transport
import summarize_geometry_session_resources as resources
from verified_token_usage import turn_usage
from experiment_parallel import stop, write_json
from summarize_mixed_scene_learning import require

ROOT = parent.ROOT
base = parent.base
read = parent.read
study = parent.study
planned_curves = parent.planned_curves
TRAIN_START = 2032100101
TEST_START = 2032101001
DEPENDENCIES = list(dict.fromkeys(parent.DEPENDENCIES + [Path(__file__), Path(variant.__file__),
    Path(transport.__file__), ROOT/'scripts/evaluate_bridge_contract.py',
    ROOT/'scripts/three_cube_contract_experiment.py', ROOT/'scripts/verified_token_usage.py',
    ROOT/'scripts/geometry_actor_session.py']))


@contextmanager
def installed():
    with variant.installed(), transport.installed(), \
         patch.object(study, 'variant', variant), \
         patch.object(study, 'lease', parent.resource_lease), \
         patch.object(resources, 'turn_usage', turn_usage), \
         patch.object(study, '__file__', str(Path(__file__).resolve())):
        yield


def check(root):
    protocol = study.check(root)
    require(protocol['variant'] == variant.VERSION and protocol['executor_source_disclosed'] is False,
            'Incorrect source disclosure protocol')
    require(len(protocol['replicas']) == 5 and protocol['rounds'] == 3 and
            len(protocol['heldout_seeds']) == 100 and protocol['evaluation_slots'] == 4000 and
            protocol['learning_curve_plan']['total_slots'] == 4000, 'Study budget changed')
    require(protocol['actor_transport'] == 'openai-http' and
            protocol['http_actor_sha256'] == hashlib.sha256(transport.actor_source().encode()).hexdigest(),
            'Actor transport changed')
    require(base.digest(root/'scheduler_snapshot/http_actor.py') == protocol['http_actor_sha256'],
            'Frozen actor transport changed')
    return protocol


def prepare(root):
    require(not root.exists(), 'Use a new study directory')
    trains = list(range(TRAIN_START, TRAIN_START+25))
    tests = list(range(TEST_START, TEST_START+100))
    inspected = set()
    for pattern in ('**/manifest.json', '**/registration.json', '**/protocol.json'):
        for path in (ROOT/'outputs').glob(pattern):
            record = read(path)
            for key in ('train_seeds', 'heldout_seeds', 'reserved_seeds', 'test_seeds'):
                inspected.update(record.get(key, []))
    require(not set(trains).intersection(tests), 'Training/test overlap')
    require(not set(trains+tests).intersection(inspected), 'Layouts overlap previous registrations')
    require(shutil.disk_usage(root.parent).free >= 15*1024**3, 'Insufficient disk reserve')
    root.mkdir(parents=True)
    rows = []
    for index in range(5):
        directory = root/f'replica_{index+1:02d}'
        manifest = variant.prepare(directory, trains[index*5:index*5+5], tests, 3)
        rows.append({'directory': directory.name, 'manifest_sha256': base.digest(directory/'manifest.json'),
                     'train_seeds': manifest['train_seeds']})
    protocol = {
        'schema': 'parallel_geometry_selected_study_v1', 'registered_at': time.time(),
        'variant': variant.VERSION, 'task': 'thin_beam_bridge',
        'replicas': rows, 'rounds': 3, 'heldout_seeds': tests,
        'limits': {'model': 4, 'rollout': 24}, 'training_slots': 200, 'model_calls': 30,
        'evaluation_slots': 4000, 'executor_source_disclosed': False,
        'learning_curve_plan': {'revisions': [0, 1, 2, 3], 'total_slots': 4000,
            'registered_before_training': True, 'reuse_selected_slots': 0,
            'scope': 'Every frozen revision on all 100 shared fresh layouts after all training freezes.'},
        'minimum_free_disk_bytes': 15*1024**3,
        'source_sha256': {str(p.relative_to(ROOT)): base.digest(p) for p in DEPENDENCIES},
        'actor_transport': 'openai-http',
        'http_actor_sha256': hashlib.sha256(transport.actor_source().encode()).hexdigest(),
        'hardware_at_registration': planned_curves.curves.hardware(),
        'selection': 'Preserve all incumbent training successes and strictly improve; ties retain incumbent.',
        'freeze_rule': 'All five pairs must finish training and pass exact-context audits before testing.',
        'failure_rule': 'Stop dispatch on infrastructure failures. Preserve incomplete attempts; no automatic retries.',
        'resource_rule': 'Four model requests and 24 rollouts globally; training falls to 16 or eight under resource pressure. Evaluation starts with 48 trials and falls to eight if required.',
        'analysis': 'Raw and training-retained test curves, cumulative tokens and per-chain variability. One hundred shared layouts are not 500 independent layouts.',
        'contrast': 'Same sensors, actuator, goal, feedback rules and policy budget; neither arm sees executor source. Full additionally receives the supplied geometry API.',
        'historical_comparison': 'Fresh chains and layouts; comparison with the old source-disclosed study is descriptive, not a paired causal ablation.'}
    for relative in protocol['source_sha256']:
        destination = root/'scheduler_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, destination)
    (root/'scheduler_snapshot/http_actor.py').write_text(transport.actor_source())
    write_json(root/'protocol.json', protocol)
    check(root)
    write_json(root/'progress.json', {'status': 'registered', 'training_slots': 200, 'test_slots': 4000})
    return protocol


def run(root, phase):
    check(root)
    try:
        if phase in ('all', 'train'):
            with parent.monitored_resources(root, 24):
                study.run(root, 'train')
        if phase == 'train':
            return
        import evaluate_bridge_contract as evaluation
        if not (root/evaluation.DIRECTORY/'registration.json').exists():
            evaluation.prepare(root)
        write_json(root/'lifecycle.json', {'status': 'evaluating', 'pid': os.getpid(), 'time': time.time()})
        evaluation.run(root)
        write_json(root/'lifecycle.json', {'status': 'completed', 'pid': os.getpid(), 'time': time.time()})
    except BaseException as error:
        stop(root, error)
        write_json(root/'lifecycle.json', {'status': 'failed', 'error': str(error), 'time': time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'all', 'train', 'evaluate', 'audit', 'check', 'worker'), default='all')
    parser.add_argument('--replica')
    args = parser.parse_args()
    root = args.output.resolve()
    def interrupt(signum, frame):
        stop(root, RuntimeError(f'Supervisor received signal {signum}'))
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with installed():
        if args.phase == 'prepare':
            report = prepare(root)
            print({k: report[k] for k in ('variant', 'training_slots', 'model_calls', 'evaluation_slots')})
        elif args.phase == 'check':
            print(check(root)['variant'])
        elif args.phase == 'worker':
            try:
                study.train_replica(root, args.replica)
            except BaseException as error:
                stop(root, error)
                raise
        elif args.phase == 'audit':
            import evaluate_bridge_contract as evaluation
            with study.exclusive(root/'pipeline.lock'), study.exclusive(root/'coordinator.lock'):
                print(evaluation.audit(root)['mean_curves'])
        else:
            run(root, args.phase)


if __name__ == '__main__':
    main()
