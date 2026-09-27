"""Resume the registered three-cube study with a recorded auditor-only correction.

Frozen sources stay unchanged. The historical auditor is compiled with one exact
replacement, and both its original and corrected source hashes are recorded.
"""
import argparse
import ast
from contextlib import contextmanager, ExitStack
import hashlib
import os
from pathlib import Path
import shutil
import signal
import time
from unittest.mock import patch

import run_three_cube_geometry_study as runner
import summarize_mixed_scene_learning as mixed
import summarize_visual_input_experiment as visual
from experiment_parallel import stop, write_json
from summarize_mixed_scene_learning import require

ROOT = runner.ROOT
REPAIR = 'audit_object_count_repair'
OLD = "result['scene']['object_count'] == 4"
NEW = "result['scene']['object_count'] == manifest['object_count']"
AUDITOR = Path(mixed.__file__)
EXPECTED_AUDITOR_SHA256 = '1bd95db843d1a73ccc1605c595c1485b6846764fea969d28793b59cf29b5d352'


def corrected_source():
    require(runner.base.digest(AUDITOR) == EXPECTED_AUDITOR_SHA256,
            'Auditor differs from the source inspected for this repair')
    source = AUDITOR.read_text()
    require(source.count(OLD) == 1, 'Expected exactly one legacy object-count check')
    return source.replace(OLD, NEW)


def corrected_auditor():
    # Compile only the audit function; retain every other check and its globals.
    tree = ast.parse(corrected_source(), filename=str(AUDITOR))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'audit']
    require(len(functions) == 1, 'Expected exactly one historical audit function')
    namespace = dict(vars(mixed))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(AUDITOR), 'exec'), namespace)
    original = namespace['audit']

    def audit(root, allow_partial=False):
        report = original(root, allow_partial)
        report['auditor_correction'] = {
            'kind': 'registered_object_count', 'before': OLD, 'after': NEW,
            'original_source_sha256': EXPECTED_AUDITOR_SHA256,
            'corrected_source_sha256': hashlib.sha256(corrected_source().encode()).hexdigest(),
            'adapter_sha256': runner.base.digest(Path(__file__)),
        }
        return report
    return audit


@contextmanager
def installed():
    auditor = corrected_auditor()
    with runner.installed(), patch.object(mixed, 'audit', auditor), \
         patch.object(visual, 'base_audit', auditor), \
         patch.object(runner.study, '__file__', str(Path(__file__).resolve())):
        # Worker subprocesses must enter this same audited repair wrapper.
        yield


def require_complete_attempts(root):
    for path in root.glob('replica_*/*/round_*/invocation.json'):
        require((path.parent/'generation.json').exists(), 'Incomplete model request retained: '+str(path))
    for path in root.glob('replica_*/*/*/trials/seed_*'):
        if path.is_dir():
            require((path/'result.json').exists(), 'Incomplete rollout retained: '+str(path))


def register(root):
    directory = root/REPAIR
    require(not directory.exists(), 'Repair already registered; use --phase check or all')
    with ExitStack() as locks:
        for path in [root/'pipeline.lock', root/'coordinator.lock'] + sorted(root.glob('replica_*/training_coordinator.lock')):
            locks.enter_context(runner.study.exclusive(path))
        protocol = runner.study.check(root)
        require(protocol['variant'] == runner.variant.VERSION, 'Wrong study variant')
        require(runner.read(root/'STOP.json')['error'] == 'Scene mismatch', 'Unexpected stop reason')
        require(not (root/'all_candidates_frozen.json').exists(), 'Repair expected before global training freeze')
        require_complete_attempts(root)
        reports = {row['directory']: runner.variant.audit(root/row['directory'], allow_partial=True)
                   for row in protocol['replicas']}
        # Preserve immutable completed evidence, not progress or actor session state.
        patterns = ('protocol.json', 'replica_*/manifest.json',
                    'replica_*/*/*/trials/seed_*/result.json',
                    'replica_*/*/*/trials/seed_*/actor_feedback.json',
                    'replica_*/*/*/trials/seed_*/initial.png')
        paths = {path for pattern in patterns for path in root.glob(pattern)}
        for name in ('candidate.py', 'bundled.py', 'generation.json', 'model.jsonl',
                     'invocation.json', 'prompt.txt', 'response.json'):
            paths.update(root.glob('replica_*/*/round_*/'+name))
        receipt = {
            'schema': 'three_cube_auditor_repair_v1', 'registered_at': time.time(),
            'authorization': 'User requested repairing the auditor and continuing the registered study.',
            'scope': 'Audit object count against the manifest; unchanged training, runtime, seeds and budgets.',
            'before': OLD, 'after': NEW,
            'original_auditor_sha256': EXPECTED_AUDITOR_SHA256,
            'corrected_auditor_sha256': hashlib.sha256(corrected_source().encode()).hexdigest(),
            'adapter_sha256': runner.base.digest(Path(__file__)),
            'stop_sha256': runner.base.digest(root/'STOP.json'),
            'preserved_files': {str(path.relative_to(root)): runner.base.digest(path) for path in sorted(paths)},
            'training_slots_before_resume': sum(r['valid_rollouts']+r['invalid_slots'] for r in reports.values()),
            'model_generations_before_resume': sum(len(r['actors']) for r in reports.values()),
        }
        directory.mkdir()
        shutil.copyfile(Path(__file__), directory/Path(__file__).name)
        (directory/'corrected_auditor.py').write_text(corrected_source())
        for name in ('STOP.json', 'progress.json', 'lifecycle.json', 'launch.json'):
            if (root/name).exists():
                shutil.copyfile(root/name, directory/(name+'.before_resume'))
        for name, report in reports.items():
            write_json(directory/'pre_resume_audits'/(name+'.json'), report)
        write_json(directory/'registration.json', receipt)
        return receipt


def check(root):
    receipt = runner.read(root/REPAIR/'registration.json')
    require(receipt['schema'] == 'three_cube_auditor_repair_v1', 'Wrong repair schema')
    require(receipt['before'] == OLD and receipt['after'] == NEW, 'Repair scope changed')
    require(receipt['adapter_sha256'] == runner.base.digest(Path(__file__)) ==
            runner.base.digest(root/REPAIR/Path(__file__).name), 'Repair adapter changed')
    require(receipt['original_auditor_sha256'] == EXPECTED_AUDITOR_SHA256, 'Original auditor changed')
    require(receipt['corrected_auditor_sha256'] == hashlib.sha256(corrected_source().encode()).hexdigest() ==
            runner.base.digest(root/REPAIR/'corrected_auditor.py'), 'Corrected audit source changed')
    for relative, digest in receipt['preserved_files'].items():
        require(runner.base.digest(root/relative) == digest, 'Pre-repair evidence changed: '+relative)
    runner.study.check(root)
    return receipt


def release_original_stop(root, receipt):
    with runner.study.exclusive(root/'pipeline.lock'), runner.study.exclusive(root/'coordinator.lock'):
        path = root/'STOP.json'
        if path.exists():
            require(runner.base.digest(path) == receipt['stop_sha256'], 'New stop requires separate inspection')
            require_complete_attempts(root)
            destination = root/REPAIR/'STOP.resumed.json'
            require(not destination.exists(), 'Original stop already resumed')
            path.rename(destination)
            write_json(root/REPAIR/'resume.json', {'pid': os.getpid(), 'time': time.time(),
                                                  'registration_sha256': runner.base.digest(root/REPAIR/'registration.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('register', 'check', 'all', 'train', 'audit', 'worker'), required=True)
    parser.add_argument('--replica')
    args = parser.parse_args()
    root = args.output.resolve()

    def interrupt(signum, frame):
        stop(root, RuntimeError(f'Supervisor received signal {signum}; draining active work'))
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with installed():
        receipt = register(root) if args.phase == 'register' else check(root)
        if args.phase in ('register', 'check'):
            print({key: receipt[key] for key in ('schema', 'training_slots_before_resume', 'model_generations_before_resume')})
        elif args.phase == 'worker':
            try:
                runner.study.train_replica(root, args.replica)
            except BaseException as error:
                stop(root, error)
                raise
        elif args.phase == 'audit':
            with runner.study.exclusive(root/'pipeline.lock'), runner.study.exclusive(root/'coordinator.lock'):
                runner.study.audit(root, runner.study.check(root))
                runner.planned_curves.curves.audit(root, root/'learning_curves')
            print('Completed repaired study audit passed')
        else:
            release_original_stop(root, receipt)
            runner.run(root, args.phase)


if __name__ == '__main__':
    main()
