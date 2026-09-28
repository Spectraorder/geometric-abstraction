"""A separate Full scaffold variant with disclosed qualitative geometry guidance."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import time
from unittest.mock import patch
import run_ring_peg_repaired_training as repaired
import run_ring_peg_scaffold_training as scaffold

original = repaired.original
ROOT = original.ROOT
STARTER_PATH = ROOT/'policies/controllers/ring_peg_guided_api_starter.py'
_original_check = original.check


GUIDANCE_PATH = ROOT/'docs/RING_PEG_STARTER_GUIDANCE.md'


def contract():
    return scaffold.contract()+'\nAdditional starter guidance:\n'+GUIDANCE_PATH.read_text()


def sources():
    result = repaired.sources()
    for path in (STARTER_PATH, GUIDANCE_PATH, Path(scaffold.__file__), Path(__file__)):
        result[str(path.relative_to(ROOT))] = original.digest(path)
    return result


def check(root):
    manifest = _original_check(root)
    if manifest['schema'] != 'ring_peg_guided_scaffold_v1':
        raise RuntimeError('Not a guided API-scaffold campaign')
    if original.digest(root/'starter.py') != manifest['starter_sha256']:
        raise RuntimeError('Registered starter changed')
    if original.digest(root/'actor_contract.txt') != manifest['contract_sha256']:
        raise RuntimeError('Registered contract changed')
    return manifest


@contextmanager
def installed():
    with patch.object(original, 'STARTER', STARTER_PATH.read_text()), \
            patch.object(original, 'CONTRACT', contract()), \
            patch.object(original, 'sources', sources), \
            patch.object(original, 'rollout', repaired.rollout):
        yield


def prepare(root):
    original.validate_policy(STARTER_PATH.read_text())
    manifest = repaired.prepare(root)
    manifest.update(schema='ring_peg_guided_scaffold_v1',
                    preceding_training='outputs/ring_peg_scaffold_train_01',
                    starter_sha256=original.digest(STARTER_PATH),
                    contract_sha256=original.digest(root/'actor_contract.txt'),
                    starter_provenance='Same non-acting API scaffold with hand-authored qualitative geometry/robot-interface explanations. No prior trained policy source.',
                    guidance_sha256=original.digest(GUIDANCE_PATH),
                    actor_exclusions=['capable ring starter', 'previous trained candidates', 'oracle source', 'private evaluator state'],
                    scope='One Full chain from a non-acting scaffold with disclosed qualitative guidance; reused development/training layouts, no Base comparison or reserved evaluation.')
    original.write(root/'manifest.json', manifest)
    original.write(root/'progress.json', {'status': 'registered', 'training_slots': 20, 'model_calls': 3,
                                         'starting_point': 'guided_api_scaffold', 'model_calls_dispatched': 0})
    check(root)
    return manifest


def train(root):
    check(root)
    if (root/'STOP.json').exists() or (root/'symbolic/revision_000').exists():
        raise RuntimeError('Preserve existing attempts; no automatic resume or retry')
    # Every rollout and revision revalidates both runtime and actor inputs.
    with patch.object(original, 'check', check):
        original.train(root)
    check(root)


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
                train(root)
    except BaseException as error:
        if root.exists():
            original.write(root/'STOP.json', {'error': type(error).__name__+': '+str(error), 'time': time.time()})
        raise
