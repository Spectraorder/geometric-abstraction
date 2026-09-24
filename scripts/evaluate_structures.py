"""Freeze and evaluate CPU policies on low-rise support-graph recipes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from aspire_so100.structure_tasks import STRUCTURE_RECIPES, structure_recipe
from evaluate_columns import COLUMN_SENSOR_PROFILE, seeded_color_permutation


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCES = ('scripts/run_vision_trial.py', 'view_scene.py',
                   'src/aspire_so100/structure_tasks.py', 'src/aspire_so100/inventory_structure.py',
                   'src/aspire_so100/structure_geometry.py', 'src/aspire_so100/block_catalog.py',
                   'src/aspire_so100/rgbd.py', 'src/aspire_so100/stacking_env.py')


def runtime_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in RUNTIME_SOURCES}


def plan_trials(recipes, seeds, distractors, permute_colors=False):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be nonempty and unique')
    if not recipes or len(set(recipes)) != len(recipes):
        raise ValueError('Recipes must be nonempty and unique')
    if not distractors or len(set(distractors)) != len(distractors) or any(n < 0 for n in distractors):
        raise ValueError('Distractor counts must be distinct nonnegative integers')
    trials = []
    for recipe_name in recipes:
        for extra in distractors:
            case = f'{recipe_name}_d{extra}'+('_colors' if permute_colors else '')
            for seed in seeds:
                color_map = seeded_color_permutation(seed) if permute_colors else None
                recipe = structure_recipe(recipe_name, color_map=color_map)
                count = len(recipe.nodes)+extra
                if count > 20:
                    raise ValueError('Requested scene exceeds the inventory')
                trials.append({'case': case, 'seed': seed, 'recipe': recipe_name,
                               'distractors': extra, 'object_count': count, 'color_map': color_map,
                               'order': list(recipe.order), 'goal': recipe.public_goal(),
                               'sensor': dict(COLUMN_SENSOR_PROFILE)})
    return trials


def validate_result(result, trial, digest):
    scene = result.get('scene', {})
    metrics = result.get('metrics', {})
    if (result.get('candidate_sha256') != digest or result.get('seed') != trial['seed']
            or result.get('task') != 'inventory_structure' or result.get('layout') != 'sampled'
            or result.get('goal') != trial['goal'] or result.get('sensor') != trial['sensor']
            or result.get('sensor_blackout', False)
            or any(scene.get(k) != trial[k] for k in ('seed', 'recipe', 'object_count', 'order', 'color_map', 'goal'))):
        raise ValueError('Result provenance differs from the frozen structure trial')
    success = result.get('success')
    required = len(trial['order'])
    if (type(success) is not bool or type(metrics.get('success')) is not bool
            or success != metrics['success'] or metrics.get('required_placements') != required
            or (success and (metrics.get('placed_blocks') != required or metrics.get('score') != 1.
                             or metrics.get('structure_stable') is not True
                             or metrics.get('failure') is not None or result.get('policy_error') is not None))):
        raise ValueError('Result contradicts strict whole-structure success')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=Path('policies/assemble_structure_cpu.py'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipes', nargs='+', choices=tuple(STRUCTURE_RECIPES), default=list(STRUCTURE_RECIPES))
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--distractors', nargs='+', type=int, default=[0])
    parser.add_argument('--permute-colors', action='store_true')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--split', choices=('development', 'heldout'), default='development')
    parser.add_argument('--frozen-sha256')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('Jobs must be positive')
    source = args.candidate.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if ((args.split == 'heldout' and args.frozen_sha256 is None)
            or (args.frozen_sha256 is not None and args.frozen_sha256 != digest)):
        parser.error('Frozen source hash is required and must match the candidate')
    try:
        trials = plan_trials(args.recipes, args.seeds, args.distractors, args.permute_colors)
    except ValueError as error:
        parser.error(str(error))
    manifest = {'schema_version': 2, 'split': args.split, 'candidate_sha256': digest,
                'runtime_sources_sha256': runtime_hashes(),
                'task': 'inventory_structure', 'trials': trials, 'trial_count': len(trials)}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        parser.error('Output belongs to a different source or trial matrix')
    candidate = (args.output/'candidate.py').resolve()
    if candidate.exists() and candidate.read_bytes() != source:
        parser.error('Frozen output candidate has changed')
    candidate.write_bytes(source)
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    if args.plan_only:
        print(f'Planned {len(trials)} {args.split} trials; source {digest}')
        return

    def run(trial):
        directory = args.output/trial['case']/f'seed_{trial["seed"]}'
        result_path = directory/'result.json'
        if not result_path.exists():
            if directory.exists():
                raise RuntimeError('Incomplete trial retained; choose a new output or inspect it: '+str(directory))
            directory.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(ROOT/'scripts/run_vision_trial.py'), '--task', 'inventory_structure',
                       '--layout', 'sampled', '--candidate', str(candidate), '--output', str(directory),
                       '--structure-recipe', trial['recipe'], '--seed', str(trial['seed']),
                       '--object-count', str(trial['object_count'])]
            if trial['color_map'] is not None:
                command += ['--structure-color-map', json.dumps(trial['color_map'])]
            env = dict(os.environ, MUJOCO_GL='egl', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            with (directory.parent/f'seed_{trial["seed"]}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=240)
        result = json.loads(result_path.read_text())
        validate_result(result, trial, digest)
        result['declared_trial'] = trial
        return result

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run, trial): trial for trial in trials}
        for future in as_completed(futures):
            trial = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(trial['case'], trial['seed'], result['success'], flush=True)
            except Exception as error:
                errors.append({'declared_trial': trial, 'success': False, 'worker_error': str(error)})
                print(trial['case'], trial['seed'], 'ERROR', error, flush=True)
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest or runtime_hashes() != manifest['runtime_sources_sha256']:
        raise RuntimeError('Frozen candidate or runtime sources changed during evaluation')
    order = {(t['case'], t['seed']): i for i, t in enumerate(trials)}
    results.sort(key=lambda r: order[(r['declared_trial']['case'], r['seed'])])
    summary = {'split': args.split, 'candidate_sha256': digest, 'trials': len(trials),
               'successes': sum(r['success'] for r in results), 'results': results, 'worker_errors': errors}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Total: {summary["successes"]}/{len(trials)}; worker errors: {len(errors)}')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
