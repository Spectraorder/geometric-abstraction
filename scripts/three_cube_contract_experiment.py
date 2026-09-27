"""Three-cube comparison with actuator contracts and no supplied executor source."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import three_cube_geometry_experiment as original
from summarize_mixed_scene_learning import require

ROOT = original.ROOT
base = original.base
interface = original.interface
neutral = original.neutral
constraint = original.constraint
VERSION = 'three_cube_contract_only_50000_v1'
PARENT_CONTEXT = constraint.context
PARENT_SOURCES = original.sources
NOTICE = ('\nShared actuator implementation is hidden in both arms. Use only the public\n'
          'action contract above. No perception implementation is supplied.\n')


def strip_executor(prompt, source, expected=True):
    block = '\nFrozen common motion loop:\n' + source
    require(prompt.count(block) == int(expected), 'Unexpected executor disclosure boundary')
    if expected:
        prompt = prompt.replace(block, NOTICE, 1)
    prompt = prompt.replace(
        'The executor source describes general motion primitives, not a task controller.',
        'The shared actuator implements generic motion primitives; its source is not supplied.')
    require('Frozen common motion loop:' not in prompt and source not in prompt,
            'Executor implementation remains in actor context')
    return prompt


def context(root, arm, revision, manifest):
    prompt, images = PARENT_CONTEXT(root, arm, revision, manifest)
    return strip_executor(prompt, (root/'executor.py').read_text(),
                          revision == 1 or manifest['actor_context_mode'] != 'persistent'), images


def sources():
    result = PARENT_SOURCES()
    for name in ('three_cube_contract_experiment.py', 'run_three_cube_contract_study.py',
                 'evaluate_three_cube_contract.py', 'verified_token_usage.py',
                 'resume_three_cube_geometry_study.py'):
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


@contextmanager
def execution_adapter():
    with installed(), original.execution_adapter():
        yield


def prepare(root, train, reserved, rounds=3):
    with installed():
        manifest = original.prepare(root, train, reserved, rounds)
    manifest.update(executor_source_disclosed=False, prompt_variant=VERSION)
    manifest['limitations'].append(
        'Both arms receive public actuator contracts without executor source. Base may independently '
        'implement geometry from calibrated RGB-D. The shared actuator still performs visual grasping.')
    base.write(root/'manifest.json', manifest)
    return manifest


def check(root, manifest):
    require(manifest.get('executor_source_disclosed') is False and
            manifest.get('prompt_variant') == VERSION, 'Wrong source disclosure contract')
    with installed():
        original.check(root, manifest)


def bootstrap(root, manifest):
    with installed():
        original.bootstrap(root, manifest)


def audit(root, allow_partial=False):
    with installed():
        report = original.audit(root, allow_partial)
    report['executor_source_disclosed'] = False
    return report


execute = original.execute
