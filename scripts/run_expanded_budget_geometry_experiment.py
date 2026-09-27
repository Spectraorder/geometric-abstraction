"""Neutral-start interface comparison with a shared 50000-character source budget."""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
from unittest.mock import patch

import run_interface_contract_geometry_experiment as interface
from summarize_mixed_scene_learning import require

ROOT = interface.ROOT
base = interface.base
INTERFACE_SOURCES = interface.sources
VERSION = 'interface_contract_geometry_50000_actor_sessions_v1'
LIMIT = 50000
CONTRACT = ROOT/'docs/POLICY_INTERFACE_CONTRACT_50K.md'
COMMON = interface.COMMON.replace(interface.CONTRACT.read_text(), CONTRACT.read_text()).replace(
    'Maximum source length is 30000 characters.', f'Maximum source length is {LIMIT} characters.')


def sources():
    result = INTERFACE_SOURCES()
    result[str(Path(__file__).relative_to(ROOT))] = base.digest(Path(__file__))
    return result


@contextmanager
def installed():
    # Scope the larger budget to this variant, including nested validation/audits.
    with patch.multiple(interface, VERSION=VERSION, CONTRACT=CONTRACT, COMMON=COMMON, sources=sources), \
         patch.object(base, 'LIMIT', LIMIT):
        yield


@contextmanager
def configured():
    with installed(), interface.installed(), interface.neutral.configured():
        yield


def prepare(root, train, reserved, rounds=3):
    with installed():
        return interface.prepare(root, train, reserved, rounds)


def check(root, manifest):
    require(manifest.get('max_candidate_characters') == LIMIT, 'Incorrect source budget for expanded-budget variant')
    with installed():
        interface.check(root, manifest)


def bootstrap(root, manifest):
    check(root, manifest)
    with installed(), interface.installed():
        interface.neutral.bootstrap(root, manifest)


def audit(root, allow_partial=False):
    manifest = json.loads((root/'manifest.json').read_text())
    check(root, manifest)
    with installed():
        report = interface.audit(root, allow_partial)
    report['source_budget_characters'] = LIMIT
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--train-seeds', type=int, nargs='+')
    parser.add_argument('--reserved-seeds', type=int, nargs='+')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--phase', choices=('prepare', 'bootstrap', 'train', 'evaluate', 'audit'), default='prepare')
    args = parser.parse_args()
    root = args.output.resolve()
    if not args.resume and (not args.train_seeds or not args.reserved_seeds):
        parser.error('Explicit seeds required for a new registration')
    manifest = json.loads((root/'manifest.json').read_text()) if args.resume else prepare(
        root, args.train_seeds, args.reserved_seeds, args.rounds)
    with (root/'training_coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check(root, manifest)
        if args.phase == 'bootstrap':
            bootstrap(root, manifest)
        elif args.phase in ('train', 'evaluate'):
            if not (root/'starter_parity.json').exists():
                bootstrap(root, manifest)
            with configured():
                base.campaign(root, manifest, args.phase)
        elif args.phase == 'audit':
            report = audit(root, allow_partial=True)
            base.write(root/'audit.json', report)
            print({key: report[key] for key in ('status', 'valid_rollouts', 'invalid_slots',
                                               'exact_actor_contexts_verified', 'shared_interface_contract_audit',
                                               'source_budget_characters')})


if __name__ == '__main__':
    main()
