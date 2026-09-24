"""Evaluation provenance and strict success checks for support graphs."""
import importlib.util
from copy import deepcopy
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import evaluate_structures as EVALUATOR


class StructureEvaluationTests(unittest.TestCase):
    def test_declared_matrix_retains_colors_shapes_and_every_distractor_cell(self):
        trials = EVALUATOR.plan_trials(['cube_bridge', 'staircase'], [1201, 1202], [0, 2], True)
        self.assertEqual(len(trials), 8)
        for trial in trials:
            self.assertEqual(trial['object_count'], len(trial['goal']['nodes'])+trial['distractors'])
            self.assertEqual({node['shape'] for node in trial['goal']['nodes']},
                             {'cube'} if trial['recipe'] == 'cube_bridge' else {'cube', 'prism'})
            self.assertEqual(len(set(trial['order'])), len(trial['order']))
            self.assertEqual(set(trial['color_map']), set(trial['color_map'].values()))
        for recipes, seeds, distractors in [([], [1], [0]), (['cube_bridge'], [1, 1], [0]),
                                            (['cube_bridge'], [1], [-1]), (['staircase'], [1], [18])]:
            with self.assertRaises(ValueError):
                EVALUATOR.plan_trials(recipes, seeds, distractors)

    def test_wrong_goal_sensor_source_scene_and_partial_success_are_rejected(self):
        trial = EVALUATOR.plan_trials(['cube_bridge'], [1201], [2], True)[0]
        result = {'candidate_sha256': 'frozen', 'seed': trial['seed'], 'task': 'inventory_structure',
                  'layout': 'sampled', 'goal': trial['goal'], 'sensor': trial['sensor'],
                  'scene': {key: trial[key] for key in ('seed', 'recipe', 'object_count', 'order', 'color_map', 'goal')},
                  'success': True, 'policy_error': None,
                  'metrics': {'success': True, 'required_placements': 3, 'placed_blocks': 3,
                              'score': 1., 'structure_stable': True, 'failure': None}}
        EVALUATOR.validate_result(result, trial, 'frozen')
        for key, value in [('seed', 1), ('sensor_blackout', True), ('candidate_sha256', 'changed'),
                           ('goal', {}), ('success', 'true'), ('policy_error', 'failure')]:
            bad = deepcopy(result);bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                EVALUATOR.validate_result(bad, trial, 'frozen')
        for key, value in [('structure_stable', False), ('placed_blocks', 2), ('required_placements', 2)]:
            bad = deepcopy(result);bad['metrics'][key] = value
            with self.subTest(metric=key), self.assertRaises(ValueError):
                EVALUATOR.validate_result(bad, trial, 'frozen')
        bad = deepcopy(result);bad['scene']['order'].reverse()
        with self.assertRaises(ValueError):
            EVALUATOR.validate_result(bad, trial, 'frozen')
