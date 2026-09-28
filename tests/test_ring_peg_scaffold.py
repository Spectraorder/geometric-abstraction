"""Verify non-acting R0 and explicit command semantics before model training."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_ring_peg_scaffold_training as training
from run_ring_peg_trial import validate_policy
spec = importlib.util.spec_from_file_location('api_starter', training.STARTER_PATH)
scaffold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaffold)


class ScaffoldTests(unittest.TestCase):
    def test_r0_only_reads_public_objects_even_after_unavailable_observation(self):
        api = Mock()
        api.task.return_value = {'objects': [{'id': 'first'}, {'id': 'second'}]}
        api.observe.side_effect = [RuntimeError('occluded'), {'status': 'current_estimate'}]
        scaffold.run(api)
        self.assertEqual([row.args[0] for row in api.observe.call_args_list], ['first', 'second'])
        api.move_pad.assert_not_called()
        api.gripper.assert_not_called()
        api.plan_grasp.assert_not_called()

    def test_increment_retains_command_height_despite_robot_tracking_error(self):
        api = Mock()
        api.robot.return_value = {'jaw': .2, 'fixed_pad_m': [.1, -.2, .45]}
        api.plan_pad.return_value = {'fixed_pad_m': [.12, -.2, .5]}
        api.move_pad.return_value = {'fixed_pad_m': [.119, -.2, .495]}
        result = scaffold.move_increment(api, [.1, -.2, .5], [.02, 0, 0], .3)
        self.assertAlmostEqual(api.move_pad.call_args.args[0][2], .5)
        self.assertEqual(result['commanded_pad_m'][2], .5)
        self.assertEqual(result['robot']['fixed_pad_m'][2], .495)

    def test_failed_preview_does_not_execute_motion(self):
        api = Mock()
        api.robot.return_value = {'jaw': .2}
        api.plan_pad.side_effect = RuntimeError('unreachable')
        self.assertIsNone(scaffold.move_target(api, [.1, -.2, .5], .3))
        api.move_pad.assert_not_called()

    def test_sandbox_and_variant_isolation(self):
        validate_policy(training.STARTER_PATH.read_text())
        before = training.original.STARTER
        with training.installed():
            self.assertEqual(training.original.STARTER, training.STARTER_PATH.read_text())
            self.assertNotIn('policies/controllers/ring_peg_starter.py', training.sources())
            self.assertNotIn('releasing above peg', training.original.STARTER)
        self.assertEqual(training.original.STARTER, before)


if __name__ == '__main__':
    unittest.main()
