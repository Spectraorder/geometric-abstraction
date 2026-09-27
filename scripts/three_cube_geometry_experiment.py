"""Scoped adapter for neutral-start three-cube experiments with current geometry.

Historical experiment sources remain unchanged. The support-graph evaluator,
actor session isolation, training retention and input boundary are reused.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import run_expanded_budget_geometry_experiment as expanded
from build_polygon_prism_session import library as point_library
from run_three_cube_trial import PHYSICS, RECIPE, cube_recipe
from summarize_mixed_scene_learning import require

ROOT = expanded.ROOT
base = expanded.base
interface = expanded.interface
neutral = interface.neutral
constraint = neutral.parent
VERSION = 'three_cube_point_geometry_50000_v1'
DRIVER = ROOT/'scripts/run_three_cube_trial.py'
GRASP_FRACTION = .4
ORIGINAL_PREPARE = base.prepare
ORIGINAL_SOURCES = expanded.sources
EXTRA = ('scripts/run_three_cube_trial.py', 'scripts/three_cube_geometry_experiment.py',
         'scripts/run_three_cube_geometry_study.py', 'scripts/evaluate_three_cube_curves.py',
         'scripts/build_polygon_prism_session.py', 'prototypes/point_geometry.py',
         'prototypes/polygon_prism/geometry.py', 'prototypes/polygon_prism/partial_boundary.py')


def library(arm):
    return point_library(arm, grasp_fraction=GRASP_FRACTION)


def sources():
    result = ORIGINAL_SOURCES()
    result.update({name: base.digest(ROOT/name) for name in EXTRA})
    return result


def task_prepare(root, rounds=3, train=None, heldout=None, arms=None):
    manifest = ORIGINAL_PREPARE(root, rounds, train, heldout, arms)
    executor = (root/'executor.py').read_text()
    old = 'grasp_fraction=0.70'
    require(executor.count(old) == 1, 'Expected one shared grasp default')
    (root/'executor.py').write_text(executor.replace(old, 'grasp_fraction=0.40'))
    manifest.update(recipe=RECIPE, object_count=3, physics=PHYSICS,
                    executor_sha256=base.digest(root/'executor.py'),
                    grasp_fraction_override=GRASP_FRACTION,
                    geometry_variant='point_geometry_extension_with_legacy_cube_backend')
    base.write(root/'manifest.json', manifest)
    return manifest


@contextmanager
def installed():
    with patch.multiple(expanded, VERSION=VERSION, sources=sources), \
         patch.multiple(neutral, DRIVER=DRIVER, structure_recipe=cube_recipe), \
         patch.object(constraint, 'library', library), \
         patch.object(base, 'prepare', task_prepare):
        yield


@contextmanager
def configured():
    with installed(), expanded.configured():
        yield


def prepare(root, train, reserved, rounds=3):
    with installed():
        manifest = expanded.prepare(root, train, reserved, rounds)
    manifest['limitations'].append(
        'The updated bundle retains the validated cube observer; this task does not measure novel-shape perception. '
        'The shared 0.40 grasp and libccd settings define a new variant, not a direct replication of the bridge study.')
    base.write(root/'manifest.json', manifest)
    return manifest


def check(root, manifest):
    require(manifest['recipe'] == RECIPE and manifest['object_count'] == 3 and
            manifest['physics'] == PHYSICS and manifest.get('grasp_fraction_override') == GRASP_FRACTION,
            'Three-cube configuration changed')
    with installed():
        expanded.check(root, manifest)


def bootstrap(root, manifest):
    with installed():
        expanded.bootstrap(root, manifest)


def audit(root, allow_partial=False):
    with installed():
        return expanded.audit(root, allow_partial)


def execute(root, arm, candidate, directory, seeds, manifest, error=None):
    """Run the scoped driver and validate every result against its registered task."""
    rows = []
    for seed in seeds:
        base.assert_frozen(root, manifest)
        output = directory/f'seed_{seed}'
        if not error and not (output/'result.json').exists():
            require(not output.exists(), 'Incomplete trial retained: '+str(output))
            require(shutil.disk_usage(root).free >= manifest['minimum_free_disk_bytes'], 'Insufficient disk')
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(DRIVER), '--candidate', str(candidate), '--output', str(output),
                       '--seed', str(seed), '--task', manifest['task'], '--layout', 'sampled',
                       '--structure-recipe', manifest['recipe'], '--structure-color-map',
                       json.dumps(manifest['color_maps'][str(seed)]), '--object-count', '3', '--wrist-depth',
                       '--record-geometry', '--timeout', '120', '--geometry-library', str(root/(arm+'_library.py'))]
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='egl')
            with output.with_suffix('.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=240)
        if error and not (output/'result.json').exists():
            require(not output.exists(), 'Incomplete trial retained: '+str(output))
            output.mkdir(parents=True)
            result = {'seed': seed, 'success': False, 'policy_error': error, 'invalid_candidate': True,
                      'rollout_executed': False, 'candidate_sha256': base.digest(candidate), 'notes': []}
            base.write(output/'result.json', result)
            base.write(output/'actor_feedback.json', result)
        result = json.loads((output/'result.json').read_text())
        require(result['seed'] == seed and result['candidate_sha256'] == base.digest(candidate) and
                type(result['success']) is bool, 'Trial source/seed/success mismatch')
        if result.get('invalid_candidate'):
            require(bool(error) and not result['success'] and not result['rollout_executed'],
                    'Invalid candidate accounting changed')
        else:
            require(not error and result['geometry_library_sha256'] == manifest['library_sha256'][arm] and
                    result['task'] == manifest['task'] and result['layout'] == 'sampled' and
                    result['scene']['seed'] == seed and result['scene']['recipe'] == manifest['recipe'] and
                    result['scene']['color_map'] == manifest['color_maps'][str(seed)] and
                    result['scene']['object_count'] == manifest['object_count'] and
                    result['physics'] == manifest['physics'] and result['sensor'] == manifest['sensor'] and
                    not result['sensor_blackout'] and result['wrist_depth'], 'Trial settings differ from protocol')
            neutral.check_goal(output, result, manifest)
        row = {key: result.get(key) for key in ('seed', 'success', 'policy_error', 'camera_calls', 'cpu_profile')}
        row['rollout_executed'] = not result.get('invalid_candidate', False)
        rows.append(row)
        print(str(directory.relative_to(root))+' seed='+str(seed)+' success='+str(row['success'])+
              ' error='+str(row['policy_error']), flush=True)
    return rows


@contextmanager
def execution_adapter():
    # Install once per coordinator/worker; nested configuration must preserve
    # the scheduler's temporary parallel wrapper around neutral.execute.
    with installed(), patch.object(neutral, 'execute', execute):
        yield
