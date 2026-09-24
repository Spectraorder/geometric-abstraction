"""Color generalization matrices preserve task structure and exact cache identity."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from aspire_so100.tower_tasks import TowerRecipe, tower_recipe
from test_evaluate_columns import COLUMNS
from test_randomized_vision_resume import LEARNER


class ColumnColorOrderTests(unittest.TestCase):
    def test_seeded_bijection_preserves_shapes_levels_and_goal_provenance(self):
        cases = COLUMNS.plan_cases(['mixed_triple', 'random'], list(range(901, 911)),
                                   levels=[2, 3, 4], distractors=[0, 2], permute_colors=True)
        self.assertEqual(len(cases), 8)
        mappings = set()
        for case in cases:
            self.assertTrue(case['name'].endswith('_colors'))
            for trial in case['trials']:
                mapping = trial['color_permutation']
                self.assertEqual(set(mapping), {'red', 'yellow', 'blue', 'green'})
                self.assertEqual(set(mapping.values()), set(mapping))
                self.assertEqual(mapping, COLUMNS.seeded_color_permutation(trial['seed']))
                mappings.add(tuple(mapping.values()))
                source = tower_recipe(trial['source_recipe'], seed=trial['seed'], levels=trial['tower_levels'])
                self.assertEqual(trial['source_order'], list(source.order))
                self.assertEqual([name.split('_', 1)[1] for name in trial['order']],
                                 [name.split('_', 1)[1] for name in source.order])
                self.assertEqual(trial['recipe'], 'custom')
                self.assertEqual(len(set(trial['order'])), trial['level_count'])
                self.assertEqual(trial['goal'], TowerRecipe('custom', tuple(trial['order'])).public_goal())
                self.assertEqual(trial['permutation_algorithm'], 'sha256_sort_v1')
        self.assertGreater(len(mappings), 4)

    def test_matrix_dispatches_declared_custom_orders_without_recipe_flags(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root/'candidate.py'
            candidate.write_text('def run(api): pass\n')
            case = COLUMNS.plan_cases(['mixed_triple'], [901, 902], permute_colors=True)[0]
            directory = root/'trials'
            directory.mkdir()
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            results = [dict(seed=trial['seed'], task=trial['task'], layout=trial['layout'],
                            candidate_sha256=digest, success=True,
                            metrics=dict(success=True, score=1., placed_blocks=2, required_placements=2,
                                         structure_stable=True, failure=None), goal=trial['goal'],
                            scene=dict(seed=trial['seed'], object_count=3, recipe='custom', order=trial['order']),
                            sensor=dict(COLUMNS.COLUMN_SENSOR_PROFILE)) for trial in case['trials']]
            with patch.object(COLUMNS, 'evaluate', return_value=dict(sha256=digest, results=results)) as evaluate:
                summary = COLUMNS.evaluate_case(candidate, directory, case, 2)
            self.assertEqual(evaluate.call_args.kwargs['tower_orders'],
                             {trial['seed']: trial['order'] for trial in case['trials']})
            self.assertNotIn('tower_recipe', evaluate.call_args.kwargs)
            self.assertNotIn('tower_levels', evaluate.call_args.kwargs)
            self.assertTrue(summary['permute_colors'])
            self.assertEqual(summary['results'][0]['declared_trial'], case['trials'][0])

    def test_parallel_custom_trials_execute_exact_orders_and_reject_wrong_order_cache(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root/'candidate.py'
            candidate.write_text('def run(api): pass\n')
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            orders = {901: ['green_cube', 'red_prism', 'blue_cylinder'],
                      902: ['yellow_cube', 'blue_prism', 'red_cylinder']}
            commands = []
            def run_trial(command, **kwargs):
                commands.append(command)
                seed = int(command[command.index('--seed')+1])
                order = command[command.index('--tower-order')+1:]
                self.assertEqual(order, orders[seed])
                self.assertNotIn('--tower-recipe', command)
                self.assertNotIn('--tower-levels', command)
                output = Path(command[command.index('--output')+1])
                output.mkdir(parents=True, exist_ok=False)
                result = dict(candidate_sha256=digest, task='inventory_tower', layout='sampled', seed=seed,
                              scene=dict(object_count=3, recipe='custom', order=order),
                              goal=TowerRecipe('custom', tuple(order)).public_goal(), success=True,
                              metrics=dict(score=1.), policy_error=None)
                (output/'result.json').write_text(json.dumps(result))
            kwargs = dict(task='inventory_tower', layout='sampled', object_count=3, tower_orders=orders)
            with patch.object(LEARNER.subprocess, 'run', side_effect=run_trial):
                summary = LEARNER.evaluate(candidate, root/'trials', [901, 902], 2, **kwargs)
            self.assertEqual(summary['successes'], 2)
            self.assertEqual(len(commands), 2)
            with patch.object(LEARNER.subprocess, 'run') as run:
                LEARNER.evaluate(candidate, root/'trials', [901, 902], 2, **kwargs)
                changed = {901: list(reversed(orders[901])), 902: orders[902]}
                with self.assertRaisesRegex(RuntimeError, 'different tower goal'):
                    LEARNER.evaluate(candidate, root/'trials', [901, 902], 2, **(kwargs | dict(tower_orders=changed)))
                path = root/'trials/seed_901/result.json'
                result = json.loads(path.read_text())
                result['scene']['order'].reverse()
                path.write_text(json.dumps(result))
                with self.assertRaisesRegex(RuntimeError, 'different tower goal'):
                    LEARNER.evaluate(candidate, root/'trials', [901, 902], 2, **kwargs)
                run.assert_not_called()

    def test_custom_orders_are_validated_before_any_trial_runs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root/'candidate.py'
            candidate.write_text('def run(api): pass\n')
            base = dict(task='inventory_tower', layout='sampled', object_count=3,
                        tower_orders={901: ['blue_cube', 'red_cube', 'green_cube']})
            invalid = [dict(task='inventory_prism'), dict(tower_recipe='cube_triple'),
                       dict(tower_levels=3), dict(tower_orders={}), dict(object_count=2),
                       dict(tower_orders={901: ['blue_cube', 'blue_cube']})]
            with patch.object(LEARNER.subprocess, 'run') as run:
                for options in invalid:
                    with self.subTest(options=options), self.assertRaises(ValueError):
                        LEARNER.evaluate(candidate, root/'trials', [901], 1, **(base | options))
                run.assert_not_called()
            self.assertFalse((root/'trials').exists())


if __name__ == '__main__':
    unittest.main()
