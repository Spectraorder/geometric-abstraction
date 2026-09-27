"""Shared API guidance must prevent type ambiguity without supplying a task plan."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_interface_contract_geometry_experiment as variant


class InterfaceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = {}
        exec(variant.neutral.parent.library('symbolic'), cls.namespace)
        cls.starter = {}
        exec(variant.STARTER.read_text(), cls.starter)

    def sessions(self):
        records = []
        full = object.__new__(self.namespace['GeometryStepSession'])
        full.colors = {'blue'}
        full.ledger = SimpleNamespace(targets={})
        full.api = SimpleNamespace(record_geometry=records.append)
        full.get = lambda color: None
        full.time = lambda: 1.0
        def no_refresh(color):
            raise AssertionError('Adapter must not refresh geometry')
        full.observe = no_refresh
        return [self.namespace['VisualInputSession'](full), full], records

    def test_adapter_uses_real_base_and_full_checkpoint_type_contracts(self):
        sessions, records = self.sessions()
        for session in sessions:
            for status, held in [('grasp_unconfirmed', False), ('lift_visually_confirmed', True),
                                 ('placement_unconfirmed', True), ('released', False)]:
                result = {'status': status, 'held': held, 'observation': {'private_to_full': True}}
                before = dict(result)
                record = self.starter['checkpoint_action'](session, 'blue', 'action', result)
                self.assertEqual(record['action_status'], status)
                self.assertNotIn('observation', record)
                self.assertEqual(result, before)
        self.assertEqual(len(records), 8)

    def test_bad_status_is_rejected_without_logging_or_coercion(self):
        sessions, records = self.sessions()
        for session in sessions:
            for bad in ({'status': {'status': 'released'}}, {'status': True}, {'status': None}):
                with self.assertRaisesRegex(ValueError, 'must be a string'):
                    self.starter['checkpoint_action'](session, 'blue', 'action', bad)
        self.assertEqual(records, [])

    def test_starter_remains_nonacting_and_passes_the_actual_validator(self):
        variant.neutral.parent.parent.validate_planner(variant.STARTER.read_text())
        with self.assertRaisesRegex(ValueError, 'module level'):
            variant.neutral.parent.parent.validate_planner("HYPOTHESIS = 'test'\ndef run(api):\n    pass\n")
        calls = []
        api = SimpleNamespace(camera=lambda view: calls.append(('camera', view)),
                              note=lambda text: calls.append(('note', text)))
        self.starter['run'](api)
        self.assertEqual([call[0] for call in calls], ['camera', 'camera', 'note'])

    def test_contract_examples_pass_source_validation(self):
        parts = variant.CONTRACT.read_text().split('```python\n')[1:]
        for part in parts[:2]:
            source = part.split('```', 1)[0] + '\ndef run(api):\n    pass\n'
            variant.neutral.parent.parent.validate_planner(source)

    def test_registration_and_both_prompts_include_the_same_contract(self):
        old_common = variant.neutral.COMMON
        old_starter = variant.neutral.STARTER
        with TemporaryDirectory() as temp:
            root = Path(temp)/'experiment'
            manifest = variant.prepare(root, [101], [201])
            variant.check(root, manifest)
            self.assertEqual(manifest['rounds'], 3)
            self.assertEqual(manifest['variant'], variant.VERSION)
            self.assertEqual((root/'starter.py').read_text(), variant.STARTER.read_text())
            self.assertEqual((root/'source_snapshot/docs/POLICY_INTERFACE_CONTRACT.md').read_text(), variant.CONTRACT.read_text())
            curve = [{'revision': 0, 'results': [{'seed': 101, 'success': False}]}]
            with variant.installed(), variant.neutral.installed(), \
                 patch.object(variant.neutral.parent.base, 'feedback', return_value={'success': False, 'policy_error': None, 'notes': []}), \
                 patch.object(variant.neutral.parent, 'training_curve', return_value=curve), \
                 patch.object(variant.neutral.parent, 'image_feedback', return_value=('', [])), \
                 patch.object(variant.neutral.parent, 'event_feedback', return_value=[]):
                for arm in manifest['arms']:
                    prompt, _ = variant.neutral.parent.context(root, arm, 1, manifest)
                    self.assertIn(variant.CONTRACT.read_text(), prompt)
                    self.assertIn("status = result['status']", prompt)
                    self.assertNotIn('action_status=returned_status', prompt)
                    self.assertNotIn('def fit_table(api):', prompt)
            with self.assertRaises(ValueError):
                variant.neutral.check(root, manifest)
        self.assertEqual(variant.neutral.COMMON, old_common)
        self.assertEqual(variant.neutral.STARTER, old_starter)


if __name__ == '__main__':
    unittest.main()
