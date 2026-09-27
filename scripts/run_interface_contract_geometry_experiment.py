"""Neutral-start comparison with explicit, shared interface guidance and adapter."""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
from unittest.mock import patch

import run_unseeded_geometry_experiment as neutral
from summarize_mixed_scene_learning import require

ROOT = neutral.ROOT
base = neutral.base
NEUTRAL_SOURCES = neutral.sources
VERSION = 'interface_contract_geometry_actor_sessions_v1'
STARTER = ROOT/'policies/interface_only_starter.py'
CONTRACT = ROOT/'docs/POLICY_INTERFACE_CONTRACT.md'
COMMON = neutral.COMMON.replace(
    'After action calls, record checkpoint(color,step,action_status=returned_status).',
    "After action calls, pass result['status'] (a string) to checkpoint's action_status,\n"
    'or use the shared checkpoint_action helper; never pass the result dictionary.')
COMMON += '\n' + CONTRACT.read_text()


def sources():
    result = NEUTRAL_SOURCES()
    for path in (STARTER, CONTRACT, Path(__file__)):
        result[str(path.relative_to(ROOT))] = base.digest(path)
    return result


@contextmanager
def installed():
    with patch.multiple(neutral, VERSION=VERSION, STARTER=STARTER, COMMON=COMMON, sources=sources):
        yield


def prepare(root, train, reserved, rounds=3):
    with installed():
        manifest = neutral.prepare(root, train, reserved, rounds)
    manifest['interface_contract_sha256'] = base.digest(CONTRACT)
    manifest['shared_interface_adapter'] = 'checkpoint_action: pass the action status string unchanged'
    manifest['limitations'].append('Shared interface documentation and adapter may reduce integration errors; reduction is not yet measured.')
    base.write(root/'manifest.json', manifest)
    return manifest


def check(root, manifest):
    require(manifest.get('interface_contract_sha256') == base.digest(CONTRACT), 'Interface contract changed')
    with installed():
        neutral.check(root, manifest)


def audit(root, allow_partial=False):
    manifest = json.loads((root/'manifest.json').read_text())
    check(root, manifest)
    with installed():
        report = neutral.audit(root, allow_partial)
    report['shared_interface_contract_audit'] = 'passed'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--train-seeds', type=int, nargs='+')
    parser.add_argument('--reserved-seeds', type=int, nargs='+')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--phase', choices=('prepare', 'bootstrap', 'train', 'evaluate', 'audit'), default='prepare')
    args = parser.parse_args(); root = args.output.resolve()
    if not args.resume and (not args.train_seeds or not args.reserved_seeds):
        parser.error('Explicit seeds required for a new registration')
    manifest = json.loads((root/'manifest.json').read_text()) if args.resume else prepare(
        root, args.train_seeds, args.reserved_seeds, args.rounds)
    with (root/'training_coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check(root, manifest)
        with installed():
            if args.phase == 'bootstrap':
                neutral.bootstrap(root, manifest)
            elif args.phase in ('train', 'evaluate'):
                if not (root/'starter_parity.json').exists():
                    neutral.bootstrap(root, manifest)
                with neutral.configured():
                    base.campaign(root, manifest, args.phase)
        if args.phase == 'audit':
            report = audit(root, allow_partial=True)
            base.write(root/'audit.json', report)
            print({key: report[key] for key in ('status', 'valid_rollouts', 'invalid_slots',
                                               'exact_actor_contexts_verified', 'shared_interface_contract_audit')})
