"""Verify removal of supplied task solutions without changing task objectives."""
import ast
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_unseeded_geometry_experiment as neutral
from run_unseeded_vision_trial import declarative_goal, PUBLIC_GOAL
from aspire_so100.structure_tasks import STRUCTURE_RECIPES, structure_recipe


class NeutralStartTests(unittest.TestCase):
    def test_goal_relabeling_preserves_geometry_and_supports_across_recipes(self):
        for name in STRUCTURE_RECIPES:
            for offset in range(4):
                recipe = structure_recipe(name, color_map=neutral.base.color_map(offset))
                original = PUBLIC_GOAL(recipe); before = repr(recipe)
                if len({node['color'] for node in original['nodes']}) != len(original['nodes']):
                    with self.assertRaisesRegex(ValueError, 'unique object colors'):
                        declarative_goal(recipe)
                    continue
                goal = declarative_goal(recipe)
                colors = {node['key']: node['color'] for node in original['nodes']}
                old = {node['color']: node for node in original['nodes']}
                self.assertEqual([node['color'] for node in goal['nodes']], sorted(old))
                self.assertEqual(goal['length_unit']['observed_cube'], colors[original['length_unit']['observed_cube']])
                for node in goal['nodes']:
                    self.assertEqual(node['key'], node['color'])
                    self.assertEqual(node['supports'], [colors[key] for key in old[node['color']]['supports']])
                    for key in ('shape', 'grid', 'orientation'):
                        self.assertEqual(node[key], old[node['color']][key])
                self.assertEqual(repr(recipe), before)
                self.assertEqual(PUBLIC_GOAL(recipe), original)

    def test_color_order_does_not_encode_a_construction_order(self):
        goal = declarative_goal(structure_recipe('thin_beam_bridge'))
        seen = set(); violations = []
        for node in goal['nodes']:
            violations.extend(key for key in node['supports'] if key not in seen)
            seen.add(node['key'])
        self.assertTrue(violations)

    def test_baseline_only_reads_images_and_notes(self):
        source = neutral.STARTER.read_text()
        neutral.parent.parent.validate_planner(source)
        methods = [node.func.attr for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertEqual(methods, ['camera', 'camera', 'note'])
        self.assertNotIn('Session', source)

    def test_new_registration_and_prompts_exclude_the_old_planner(self):
        old_common = neutral.parent.COMMON
        old_version = neutral.parent.VERSION
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/'experiment'
            manifest = neutral.prepare(root, [101], [201], rounds=3)
            neutral.check(root, manifest)
            self.assertEqual(manifest['rounds'], 3)
            self.assertFalse(manifest['supplied_task_planner'])
            self.assertEqual((root/'starter.py').read_text(), neutral.STARTER.read_text())
            curve = [{'revision': 0, 'results': [{'seed': 101, 'success': False}]}]
            with neutral.installed(), \
                 patch.object(neutral.parent.base, 'feedback', return_value={'success': False, 'policy_error': None, 'notes': []}), \
                 patch.object(neutral.parent, 'training_curve', return_value=curve), \
                 patch.object(neutral.parent, 'image_feedback', return_value=('', [])), \
                 patch.object(neutral.parent, 'event_feedback', return_value=[]):
                for arm in manifest['arms']:
                    prompt, _ = neutral.parent.context(root, arm, 1, manifest)
                    for text in ('def fit_table(api):', 'def locate(api, plane,', 'def masks(rgb):', 'low-rise bridge',
                                 'marginal beam', 'two\ncube foundations', 'extra correction',
                                 'common starter\'s visible-top estimator'):
                        self.assertNotIn(text, prompt)
                    self.assertIn('NOT construction order', prompt)
                    self.assertIn('Observation-only baseline', prompt)
            self.assertEqual(neutral.parent.COMMON, old_common)
            self.assertEqual(neutral.parent.VERSION, old_version)
            with self.assertRaises(ValueError):
                neutral.parent.check(root, manifest)


if __name__ == '__main__':
    unittest.main()
