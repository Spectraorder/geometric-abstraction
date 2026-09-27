"""Keep source validation, registered budgets and actor prompts consistent."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_expanded_budget_geometry_experiment as variant


def padded_policy(size):
    source = 'def run(api):\n    pass\n# '
    return source + 'x' * (size - len(source))


class ExpandedBudgetTests(unittest.TestCase):
    def test_real_validator_accepts_boundary_and_restores_legacy_limit(self):
        with variant.configured():
            for size in (30001, 50000):
                variant.base.validate_planner(padded_policy(size))
            with self.assertRaisesRegex(ValueError, 'source budget'):
                variant.base.validate_planner(padded_policy(50001))
            with self.assertRaisesRegex(ValueError, 'module level'):
                variant.base.validate_planner('CONSTANT = 1\ndef run(api):\n    pass\n')
        self.assertEqual(variant.base.LIMIT, 30000)
        with variant.interface.installed(), variant.interface.neutral.configured():
            with self.assertRaisesRegex(ValueError, 'source budget'):
                variant.base.validate_planner(padded_policy(30001))

    def test_registration_prompts_and_snapshot_share_the_new_budget(self):
        old_common = variant.interface.COMMON
        old_contract = variant.interface.CONTRACT
        parent = variant.interface.neutral.parent
        with TemporaryDirectory() as temp:
            root = Path(temp)/'experiment'
            manifest = variant.prepare(root, [101], [201])
            variant.check(root, manifest)
            self.assertEqual(manifest['variant'], variant.VERSION)
            self.assertEqual(manifest['max_candidate_characters'], 50000)
            self.assertEqual(manifest['rounds'], 3)
            snapshot = root/'source_snapshot'/variant.CONTRACT.relative_to(variant.ROOT)
            self.assertEqual(snapshot.read_text(), variant.CONTRACT.read_text())
            self.assertIn('scripts/run_expanded_budget_geometry_experiment.py', manifest['source_sha256'])
            curve = [{'revision': 0, 'results': [{'seed': 101, 'success': False}]}]
            with variant.configured(), \
                 patch.object(variant.base, 'feedback', return_value={'success': False, 'policy_error': None, 'notes': []}), \
                 patch.object(parent, 'training_curve', return_value=curve), \
                 patch.object(parent, 'image_feedback', return_value=('', [])), \
                 patch.object(parent, 'event_feedback', return_value=[]):
                for arm in manifest['arms']:
                    prompt, _ = variant.base.context(root, arm, 1, manifest)
                    self.assertIn('Maximum source length is 50000 characters.', prompt)
                    self.assertIn(variant.CONTRACT.read_text(), prompt)
                    self.assertNotIn('30000-character', prompt)
                    self.assertNotIn('Maximum source length is 30000', prompt)
            with self.assertRaises(ValueError):
                variant.interface.check(root, manifest)
            with self.assertRaisesRegex(ValueError, 'Incorrect source budget'):
                variant.check(root, dict(manifest, max_candidate_characters=30000))
        self.assertEqual(variant.interface.COMMON, old_common)
        self.assertEqual(variant.interface.CONTRACT, old_contract)
        self.assertEqual(variant.base.LIMIT, 30000)

    def test_legacy_registration_cannot_resume_with_the_new_budget(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)/'legacy'
            manifest = variant.interface.prepare(root, [101], [201])
            self.assertEqual(manifest['max_candidate_characters'], 30000)
            with self.assertRaisesRegex(ValueError, 'Incorrect source budget'):
                variant.check(root, manifest)
            variant.interface.check(root, manifest)

    def test_cli_preparation_uses_expanded_registration(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)/'experiment'
            argv = ['run_expanded_budget_geometry_experiment.py', '--output', str(root),
                    '--train-seeds', '101', '--reserved-seeds', '201', '--phase', 'prepare']
            with patch.object(sys, 'argv', argv):
                variant.main()
            manifest = json.loads((root/'manifest.json').read_text())
            variant.check(root, manifest)
            self.assertEqual(manifest['max_candidate_characters'], 50000)
            self.assertFalse(list(root.glob('*/round_*/invocation.json')))


if __name__ == '__main__':
    unittest.main()
