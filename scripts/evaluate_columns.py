"""Evaluate one frozen CPU policy across single-column recipes and layout seeds."""
import argparse
import json
from pathlib import Path

from learn_randomized_vision import evaluate
from aspire_so100.tower_tasks import RECIPES, tower_recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=Path('policies/stack_column_cpu.py'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipes', nargs='+', choices=(*RECIPES, 'random'), default=['cube_pair', 'cube_triple', 'cube_quad', 'mixed_triple', 'slender_top', 'cylinder_base'])
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--levels', type=int, help='Number of levels for the random recipe')
    parser.add_argument('--distractors', type=int, default=0)
    parser.add_argument('--jobs', type=int, default=3)
    args = parser.parse_args()
    if args.distractors < 0 or args.jobs < 1 or (args.levels is not None and 'random' not in args.recipes):
        parser.error('Invalid distractor count, jobs, or random levels')
    summary = {}
    for name in args.recipes:
        levels = args.levels if name == 'random' else None
        recipe = tower_recipe(name, seed=args.seeds[0], levels=levels)
        summary[name] = evaluate(args.candidate, args.output/name, args.seeds, args.jobs,
                                 task='inventory_tower', layout='sampled',
                                 object_count=len(recipe.order)+args.distractors,
                                 tower_recipe=name, tower_levels=levels)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for name, result in summary.items():
        print(name, result['successes'], '/', result['trials'])


if __name__ == '__main__':
    main()
