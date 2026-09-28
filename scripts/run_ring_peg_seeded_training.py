"""Train from a disclosed, sensor-grounded capable R0 in a separate campaign."""
import argparse
import shutil
from contextlib import contextmanager
from pathlib import Path
import time
from unittest.mock import patch
import run_ring_peg_repaired_training as repaired

original = repaired.original
ROOT = original.ROOT
STARTER_PATH = ROOT/'policies/controllers/ring_peg_starter.py'
VALIDATION = ROOT/'outputs/ring_peg_dev/capable_starter_training_01'


def contract():
    text = repaired.CONTRACT
    text = text.replace('No task-solving starter or executor/perception source is provided. R0 only observes.',
                        'A task-specific capable R0 is supplied as disclosed prior experience. Executor and perception source remain hidden.')
    return text + '''
This is training from a capable initial policy, not learning from scratch.
The supplied R0 reuses grasp-search logic from an earlier Full training proposal
and hand-edited sensor-driven transport/release logic. Its development layouts
and the five training layouts have been used during development. No reserved
outcomes, simulator object coordinates, or privileged oracle are supplied.
Improve the supplied policy using training feedback. R0 success and cost must be
reported alongside R1-R3; do not attribute inherited capability to these revisions.
'''


def sources():
    result = repaired.sources()
    for path in (STARTER_PATH, Path(__file__)):
        result[str(path.relative_to(ROOT))] = original.digest(path)
    return result


def starter_evidence():
    registration = original.read(VALIDATION/'registration.json')
    if registration['seeds'] != original.SEEDS:
        raise RuntimeError('Require a complete five-layout development validation')
    sha = original.digest(STARTER_PATH)
    if original.digest(VALIDATION/'candidate.py') != sha or registration['candidate_sha256'] != sha:
        raise RuntimeError('Validate this exact starter before registration')
    results = []
    for seed in original.SEEDS:
        path = VALIDATION/f'seed_{seed}/result.json'
        row = original.read(path)
        receipt = original.read(path.with_name('repair_receipt.json'))
        if receipt['source_sha256'] != original.digest(ROOT/'scripts/ring_peg_development.py'):
            raise RuntimeError('Starter validation used a different sensor adapter')
        if row['seed'] != seed or row['candidate_sha256'] != sha or not row.get('rollout_executed'):
            raise RuntimeError('Missing or mismatched starter validation')
        results.append({'seed': seed, 'success': row['success'], 'result_sha256': original.digest(path)})
    if not any(row['success'] for row in results):
        raise RuntimeError('Starting policy has not demonstrated complete task execution')
    return results


@contextmanager
def installed():
    # All legacy entry points and their frozen campaigns stay unchanged.
    with patch.object(original, 'STARTER', STARTER_PATH.read_text()), \
            patch.object(original, 'CONTRACT', contract()), \
            patch.object(original, 'sources', sources), \
            patch.object(original, 'rollout', repaired.rollout):
        yield


def check(root):
    manifest = original.check(root)
    if manifest['schema'] != 'ring_peg_capable_start_v1':
        raise RuntimeError('Not a capable-start campaign')
    if original.digest(root/'starter.py') != manifest['starter_sha256']:
        raise RuntimeError('Registered starter changed')
    if original.digest(root/'actor_contract.txt') != manifest['contract_sha256']:
        raise RuntimeError('Registered contract changed')
    return manifest


def prepare(root):
    evidence = starter_evidence()
    manifest = repaired.prepare(root)
    manifest.update(schema='ring_peg_capable_start_v1',
                    preceding_training='outputs/ring_peg_full_train_02',
                    starter_sha256=original.digest(STARTER_PATH),
                    contract_sha256=original.digest(root/'actor_contract.txt'),
                    starter_provenance='Grasp logic adapted from Full pilot 02 R3; sensor-driven transport/release hand-edited. No privileged oracle source or truth coordinates.',
                    starter_validation=evidence,
                    development_outcome='Exact capable starter validated on all five reused training layouts; see per-layout starter_validation. This is development evidence, not a held-out success estimate.',
                    scope='One Full chain improving a disclosed capable R0; no neutral-start, Base comparison or reserved success-rate claim.')
    original.write(root/'manifest.json', manifest)
    original.write(root/'starter_validation.json', evidence)
    for seed in original.SEEDS:
        destination = root/'development_evidence'/'capable_starter'/f'seed_{seed}'
        destination.mkdir(parents=True)
        for name in ('result.json', 'repair_receipt.json'):
            shutil.copyfile(VALIDATION/f'seed_{seed}'/name, destination/name)
    original.write(root/'progress.json', {'status': 'registered', 'training_slots': 20, 'model_calls': 3,
                                         'starting_point': 'capable_sensor_policy', 'model_calls_dispatched': 0})
    check(root)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'train', 'check'), required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    try:
        with installed():
            if args.phase == 'prepare':
                print(prepare(root)['schema'])
            elif args.phase == 'check':
                print(check(root)['schema'])
            else:
                check(root)
                original.train(root)
                check(root)
    except BaseException as error:
        if root.exists():
            original.write(root/'STOP.json', {'error': type(error).__name__+': '+str(error), 'time': time.time()})
        raise
