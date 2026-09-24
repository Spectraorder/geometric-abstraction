"""Evaluate one frozen CPU policy across single-column recipes and layout seeds."""
import argparse
import hashlib
import json
from pathlib import Path

from learn_randomized_vision import evaluate
from aspire_so100.tower_tasks import RECIPES, TowerRecipe, tower_recipe


COLUMN_SENSOR_PROFILE = {'depth_sampling': 'pixel_center', 'offscreen_samples': 0}


def seeded_color_permutation(seed):
    """A stable bijection independent of Python hash randomization or RNG versions."""
    colors = ('red', 'yellow', 'blue', 'green')
    shuffled = sorted(colors, key=lambda color: hashlib.sha256(
        f'column_colors_v1:{seed}:{color}'.encode('ascii')).digest())
    return dict(zip(colors, shuffled))


def plan_cases(recipes, seeds, *, levels=None, distractors=(0,), permute_colors=False):
    """Declare every goal and scene size before evaluating a candidate."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be nonempty and unique')
    if not recipes or len(set(recipes)) != len(recipes):
        raise ValueError('Recipes must be nonempty and unique')
    if not distractors or len(set(distractors)) != len(distractors) or any(d < 0 for d in distractors):
        raise ValueError('Distractor counts must be nonempty, unique and nonnegative')
    if levels is not None and ('random' not in recipes or not levels or len(set(levels)) != len(levels)):
        raise ValueError('Distinct level counts require the random recipe')
    cases = []
    for name in recipes:
        for count in (levels or [None]) if name == 'random' else [None]:
            for extra in distractors:
                key = name
                if count is not None:
                    key += f'_l{count}'
                if extra or len(distractors) > 1:
                    key += f'_d{extra}'
                if permute_colors:
                    key += '_colors'
                trials = []
                for seed in seeds:
                    recipe = tower_recipe(name, seed=seed, levels=count)
                    source_order = list(recipe.order)
                    permutation = seeded_color_permutation(seed) if permute_colors else None
                    if permutation is not None:
                        order = tuple(permutation[color]+'_'+shape for color, shape in
                                      (identity.split('_', 1) for identity in recipe.order))
                        recipe = TowerRecipe('custom', order)
                    object_count = len(recipe.order) + extra
                    if object_count > 20:
                        raise ValueError('Levels plus distractors cannot exceed the 20-object inventory')
                    trials.append({'seed': seed, 'task': 'inventory_tower', 'layout': 'sampled',
                                   'recipe': recipe.name, 'tower_levels': count,
                                   'level_count': len(recipe.order), 'distractors': extra,
                                   'object_count': object_count, 'order': list(recipe.order),
                                   'goal': recipe.public_goal(), 'sensor': dict(COLUMN_SENSOR_PROFILE)})
                    if permutation is not None:
                        trials[-1].update(source_recipe=name, source_order=source_order,
                                          color_permutation=permutation, permutation_algorithm='sha256_sort_v1')
                case = {'name': key, 'recipe': name, 'levels': count,
                        'distractors': extra, 'trials': trials}
                if permute_colors:
                    case['permute_colors'] = True
                cases.append(case)
    return cases


def prepare_manifest(candidate, output, cases, *, split='development', frozen_sha256=None):
    """Freeze source and reject accidental reuse of a different experiment."""
    candidate = candidate.resolve()
    source = candidate.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if split == 'heldout' and frozen_sha256 is None:
        raise ValueError('Held-out evaluation requires --frozen-sha256 from the frozen candidate')
    if frozen_sha256 is not None and frozen_sha256 != digest:
        raise ValueError('Candidate does not match --frozen-sha256')
    manifest = {'schema_version': 2, 'split': split, 'source_candidate': str(candidate),
                'sensor': dict(COLUMN_SENSOR_PROFILE),
                'candidate_sha256': digest, 'frozen_candidate': 'candidate.py',
                'trial_count': sum(len(case['trials']) for case in cases), 'cases': cases}
    output.mkdir(parents=True, exist_ok=True)
    path = output/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Output manifest belongs to a different source, split, or trial matrix, including the sensor profile')
    frozen = output/'candidate.py'
    if frozen.exists() and frozen.read_bytes() != source:
        raise ValueError('Output candidate differs from the requested frozen source')
    frozen.write_bytes(source)
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest, frozen


def validate_sensor(result, expected, path):
    """Reject legacy, mismatched or blacked-out observations before cache use."""
    if result.get('sensor') != expected or result.get('sensor_blackout', False):
        raise RuntimeError(f'Trial sensor profile differs from the declared experiment: {path}')


def validate_column_result(result, trial, digest, path):
    """Verify cache provenance and strict whole-column success before counting."""
    validate_sensor(result, trial['sensor'], path)
    if result.get('candidate_sha256') != digest:
        raise RuntimeError(f'Trial candidate hash differs from the frozen experiment: {path}')
    scene = result.get('scene', {})
    if (result.get('seed') != trial['seed'] or result.get('task') != trial['task']
            or result.get('layout') != trial['layout'] or result.get('goal') != trial['goal']
            or scene.get('object_count') != trial['object_count']
            or scene.get('seed') != trial['seed'] or scene.get('recipe') != trial['recipe']
            or scene.get('order') != trial['order']):
        raise RuntimeError(f'Trial result does not match its declared goal and scene size/order: {path}')
    metrics = result.get('metrics', {})
    success = result.get('success')
    required = trial['level_count'] - 1
    if (type(success) is not bool or type(metrics.get('success')) is not bool
            or success != metrics['success'] or metrics.get('required_placements') != required
            or (success and (metrics.get('placed_blocks') != required
                             or metrics.get('score') != 1.0
                             or metrics.get('structure_stable') is not True
                             or metrics.get('failure') is not None
                             or result.get('policy_error') is not None))):
        raise RuntimeError(f'Trial success contradicts whole-column evaluation metrics: {path}')


def evaluate_case(candidate, directory, case, jobs, *, expected_sha256=None):
    """Group compatible scene sizes while retaining each declared trial."""
    groups = {}
    for trial in case['trials']:
        groups.setdefault(trial['object_count'], []).append(trial)
    results = []
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError('Frozen candidate no longer matches the manifest')
    for object_count, trials in groups.items():
        group_dir = directory if len(groups) == 1 else directory/f'objects_{object_count}'
        for trial in trials:
            cached = group_dir/f'seed_{trial["seed"]}'/'result.json'
            if cached.exists():
                validate_column_result(json.loads(cached.read_text()), trial, digest, cached)
        goal_options = ({'tower_orders': {trial['seed']: trial['order'] for trial in trials}}
                        if case.get('permute_colors', False) else
                        {'tower_recipe': case['recipe'], 'tower_levels': case['levels']})
        summary = evaluate(candidate, group_dir, [trial['seed'] for trial in trials], jobs,
                           task='inventory_tower', layout='sampled', object_count=object_count,
                           **goal_options)
        if summary['sha256'] != digest:
            raise RuntimeError('Frozen candidate changed during evaluation')
        declared = {trial['seed']: trial for trial in trials}
        for result in summary['results']:
            trial = declared.pop(result['seed'])
            validate_column_result(result, trial, digest, group_dir/f'seed_{result["seed"]}')
            results.append(result | {'declared_trial': trial})
        if declared:
            raise RuntimeError('Evaluation omitted declared seeds')
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen candidate changed during evaluation')
    order = {trial['seed']: index for index, trial in enumerate(case['trials'])}
    results.sort(key=lambda result: order[result['seed']])
    summary = {'candidate': str(candidate.resolve()), 'sha256': digest,
               'task': 'inventory_tower', 'layout': 'sampled', 'recipe': case['recipe'],
               'sensor': dict(COLUMN_SENSOR_PROFILE),
               'tower_levels': case['levels'], 'distractors': case['distractors'],
               'permute_colors': case.get('permute_colors', False),
               'successes': sum(result['success'] for result in results), 'trials': len(results),
               'score': sum(result['metrics']['score'] for result in results)/len(results),
               'results': results}
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=Path('policies/stack_column_cpu.py'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipes', nargs='+', choices=(*RECIPES, 'random'), default=['cube_pair', 'cube_triple', 'cube_quad', 'mixed_triple', 'slender_top', 'cylinder_base'])
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--levels', nargs='+', type=int, choices=(2, 3, 4),
                        help='Explicit random-recipe level matrix; default: 3')
    parser.add_argument('--distractors', nargs='+', type=int, default=[0],
                        help='Distractor-count matrix; each setting has independent layouts')
    parser.add_argument('--permute-colors', action='store_true',
                        help='Apply a seed-stable color bijection to each goal, preserving shapes and levels')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--split', choices=('development', 'heldout'), default='development')
    parser.add_argument('--frozen-sha256', help='Required source hash for held-out evaluation')
    parser.add_argument('--plan-only', action='store_true', help='Freeze the matrix and source without running trials')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('Jobs must be positive')
    try:
        cases = plan_cases(args.recipes, args.seeds, levels=args.levels, distractors=args.distractors,
                           permute_colors=args.permute_colors)
        manifest, candidate = prepare_manifest(args.candidate, args.output, cases,
                                               split=args.split, frozen_sha256=args.frozen_sha256)
    except ValueError as exc:
        parser.error(str(exc))
    if args.plan_only:
        print(f'Planned {manifest["trial_count"]} {args.split} trials; source {manifest["candidate_sha256"]}')
        return
    summary = {}
    for case in cases:
        summary[case['name']] = evaluate_case(candidate, args.output/case['name'], case, args.jobs,
                                               expected_sha256=manifest['candidate_sha256'])
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for name, result in summary.items():
        print(name, result['successes'], '/', result['trials'])
    successes = sum(result['successes'] for result in summary.values())
    print(f'Total: {successes}/{manifest["trial_count"]} ({successes/manifest["trial_count"]:.1%})')


if __name__ == '__main__':
    main()
