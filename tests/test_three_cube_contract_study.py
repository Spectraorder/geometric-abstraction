"""Guard prompt disclosure and preserve the existing sensor/action boundary."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import three_cube_contract_experiment as variant
import run_three_cube_contract_study as runner


class ContractStudyTests(unittest.TestCase):
    def test_only_executor_disclosure_changes(self):
        source = 'class RGBDPickPlace:\n    pass\n'
        original = 'public sensor API\nFrozen common motion loop:\n'+source+'\nIdentical starter planner:\nobservations'
        result = variant.strip_executor(original, source)
        self.assertEqual(result, 'public sensor API'+variant.NOTICE+'\nIdentical starter planner:\nobservations')
        self.assertNotIn(source, result)
        with self.assertRaisesRegex(ValueError, 'boundary'):
            variant.strip_executor(original+original, source)
        with self.assertRaisesRegex(ValueError, 'boundary'):
            variant.strip_executor('missing source', source)

    def test_persistent_followup_is_unchanged(self):
        prompt = 'Own policy implementation and training feedback'
        self.assertEqual(variant.strip_executor(prompt, 'class RGBDPickPlace: pass', False), prompt)

    def test_both_arms_keep_images_and_feedback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'executor.py').write_text('executor implementation')
            prompt = 'contract\nFrozen common motion loop:\nexecutor implementation\nfeedback'
            images = ['side.png']
            with patch.object(variant, 'PARENT_CONTEXT', return_value=(prompt, images)):
                for arm in ('basic', 'symbolic'):
                    text, attached = variant.context(root, arm, 1, {'actor_context_mode': 'persistent'})
                    self.assertIs(attached, images)
                    self.assertTrue(text.endswith('\nfeedback'))
                    self.assertNotIn('executor implementation', text)

    def test_configuration_reaches_nested_context_and_restores(self):
        old_context = variant.constraint.context
        with runner.installed(), variant.configured():
            self.assertIs(variant.base.context, variant.context)
            self.assertIs(runner.study.variant, variant)
            self.assertEqual(runner.study.__file__, str(Path(runner.__file__).resolve()))
            self.assertIs(runner.resources.turn_usage, runner.turn_usage)
        self.assertIs(variant.constraint.context, old_context)

    def test_historical_initial_prompts_remove_only_host_source(self):
        root = runner.ROOT/'outputs/geometry_cube_stack_01/replica_01'
        if not root.exists():
            self.skipTest('Historical artifacts are not installed')
        for arm in ('basic', 'symbolic'):
            original = (root/arm/'round_001/prompt.txt').read_text()
            source = (root/'executor.py').read_text()
            transformed = variant.strip_executor(original, source)
            self.assertNotIn('class RGBDPickPlace:', transformed)
            self.assertNotIn('def locate(self, name, model=None, near=None)', transformed)
            self.assertIn('api.camera', transformed)
            self.assertIn('Own revision 0 feedback:', transformed)
            self.assertIn('Identical starter planner:', transformed)
            with runner.installed(), variant.configured():
                import json
                prompt, images = variant.base.context(root, arm, 1, json.loads((root/'manifest.json').read_text()))
            self.assertEqual(prompt, transformed)
            self.assertTrue(images)


if __name__ == '__main__':
    unittest.main()
