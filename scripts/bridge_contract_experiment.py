"""Bridge acquisition with public actuator contracts and hidden executor source."""
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

import run_expanded_budget_geometry_experiment as original
from three_cube_contract_experiment import strip_executor
from summarize_mixed_scene_learning import require

ROOT = original.ROOT
base = original.base
interface = original.interface
neutral = interface.neutral
constraint = neutral.parent
VERSION = 'bridge_contract_only_50000_v1'
PARENT_CONTEXT = constraint.context
PARENT_SOURCES = original.sources
EXTRA = ('bridge_contract_experiment.py', 'run_bridge_contract_study.py',
         'evaluate_bridge_contract.py', 'bridge_actor_transport.py',
         'three_cube_contract_experiment.py', 'verified_token_usage.py')


def context(root, arm, revision, manifest):
    prompt, images = PARENT_CONTEXT(root, arm, revision, manifest)
    return strip_executor(prompt, (root/'executor.py').read_text(),
                          revision == 1 or manifest['actor_context_mode'] != 'persistent'), images


def sources():
    result = PARENT_SOURCES()
    for name in EXTRA:
        path = ROOT/'scripts'/name
        result[str(path.relative_to(ROOT))] = base.digest(path)
    return result


@contextmanager
def installed():
    with patch.multiple(original, VERSION=VERSION, sources=sources), \
         patch.object(constraint, 'context', context):
        yield


@contextmanager
def configured():
    with installed(), original.configured():
        yield


def prepare(root, train, reserved, rounds=3):
    with installed():
        manifest = original.prepare(root, train, reserved, rounds)
    manifest.update(executor_source_disclosed=False, prompt_variant=VERSION,
                    actor_transport='openai-http')
    manifest['limitations'].append(
        'Both arms receive public actuator contracts without executor source. '
        'The unchanged shared actuator still performs visual grasp localization. '
        'Historical bridge comparisons use different chains and layouts.')
    base.write(root/'manifest.json', manifest)
    check(root, manifest)
    return manifest


def check(root, manifest):
    require(manifest.get('executor_source_disclosed') is False and
            manifest.get('prompt_variant') == VERSION and
            manifest.get('actor_transport') == 'openai-http', 'Wrong bridge interface or transport')
    require(manifest['recipe'] == 'thin_beam_bridge' and manifest['object_count'] == 4,
            'Bridge task changed')
    with installed():
        original.check(root, manifest)


def bootstrap(root, manifest):
    with installed():
        original.bootstrap(root, manifest)


def audit(root, allow_partial=False):
    check(root, json.loads((root/'manifest.json').read_text()))
    with installed():
        report = original.audit(root, allow_partial)
    report['executor_source_disclosed'] = False
    return report


execute = neutral.execute
