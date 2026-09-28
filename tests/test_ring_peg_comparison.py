"""Matched sensors, Base isolation, and qualitative contract parity."""
import base64
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from ring_peg_comparison_runtime import ComparisonSensors
import ring_peg_comparison_experiment as experiment
from run_ring_peg_trial import validate_policy


class ComparisonTests(unittest.TestCase):
    def sensor(self, arm):
        api = object.__new__(ComparisonSensors)
        api.arm = arm
        api.records = []
        api.colors = {'ring': 'red', 'peg': 'blue'}
        return api

    def test_base_rejects_geometry_before_any_observation(self):
        api = self.sensor('basic')
        api.camera = Mock(side_effect=AssertionError('Unexpected sensor fitting'))
        with self.assertRaisesRegex(ValueError, 'unavailable in Base'):
            api.dispatch('observe', {'name': 'ring'})
        api.camera.assert_not_called()
        self.assertEqual(api.records, [])
        for method in ('reset', 'private_ring_state', 'evaluate_on_peg'):
            with self.assertRaises(ValueError):
                api.dispatch(method, {})

    def test_same_raw_arrays_in_both_arms_and_private_fields_excluded(self):
        frame = {'rgb': np.zeros((4, 5, 3), np.uint8), 'depth_m': np.ones((4, 5)),
                 'valid_depth': np.ones((4, 5), bool), 'intrinsics': np.eye(3),
                 'camera_to_world': np.eye(4), 'time': .5, 'private_pose': [1, 2, 3]}
        for arm in experiment.ARMS:
            api = self.sensor(arm)
            api.camera = Mock(return_value=frame)
            packet = api.dispatch('camera', {'view': 'side'})
            with np.load(io.BytesIO(base64.b64decode(packet['rgbd_npz_base64'])), allow_pickle=False) as decoded:
                self.assertEqual(set(decoded.files), set(frame)-{'private_pose'})
                for key in decoded.files:
                    np.testing.assert_array_equal(decoded[key], frame[key])
            self.assertEqual(api.records, [])

    def test_goal_and_guidance_match_except_geometry_capability(self):
        tasks = [self.sensor(arm).task() for arm in experiment.ARMS]
        self.assertFalse(tasks[0].pop('geometry_available'))
        self.assertTrue(tasks[1].pop('geometry_available'))
        self.assertEqual(tasks[0], tasks[1])
        guide = experiment.GUIDANCE_PATH.read_text()
        for arm in experiment.ARMS:
            self.assertIn(guide, experiment.contract(arm))
            self.assertIn('api.camera', experiment.contract(arm))
        self.assertNotIn("api.observe('ring'):", experiment.contract('basic'))
        self.assertIn("api.observe('ring'):", experiment.contract('symbolic'))
        validate_policy(experiment.STARTER_PATH.read_text())

class StudyTests(unittest.TestCase):
    def test_chain_retains_training_success_set_and_uses_own_feedback(self):
        import tempfile
        import json
        import run_ring_peg_comparison_study as study
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root/'replica_01'
            directory.mkdir()
            seeds = [11, 12, 13, 14, 15]
            study.write_json(directory/'manifest.json', {'arms': ['basic', 'symbolic'], 'train_seeds': seeds})
            (root/'starter.py').write_text('def run(api):\n    pass\n')
            (root/'basic_contract.txt').write_text('Shared contract with Base sensor access.')
            protocol = {'replicas': [{'name': 'replica_01'}], 'limits': {'model': 4}}
            success_sets = [set(), {11, 12}, {11, 13}, {11, 12, 13}]
            prompts = []
            def proposal(stage, prompt, images, manifest):
                prompts.append(prompt)
                stage.mkdir()
                (stage/'candidate.py').write_text('def run(api):\n    api.note('+repr(stage.name)+')\n')
                for name in ('invocation.json', 'generation.json'):
                    study.write_json(stage/name, {})
            def rollout(root, directory, arm, revision, seed):
                result = {'seed': seed, 'success': seed in success_sets[revision], 'status': 'completed'}
                study.write_json(study.stage_path(directory, arm, revision)/'trials'/f'seed_{seed}'/'actor_feedback.json', result)
                return result
            with patch.object(study, 'check', return_value=protocol), \
                    patch.object(study.actor, 'propose', side_effect=proposal), \
                    patch.object(study.actor, 'audit_session'), patch.object(study, 'rollout', side_effect=rollout):
                study.chain(root, 'replica_01', 'basic')
            rows = study.read(directory/'basic/training_curve.json')
            self.assertEqual([r['selected_revision'] for r in rows], [0, 1, 1, 3])
            self.assertIn('Retained policy R1', prompts[2])
            self.assertEqual(len(prompts), 3)
            self.assertEqual(study.read(directory/'basic/summary.json')['selected_revision'], 3)

    def test_audit_rejects_base_geometry_feedback(self):
        import tempfile
        import run_ring_peg_comparison_study as study
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            candidate = path/'candidate.py'
            candidate.write_text('def run(api):\n    pass\n')
            result = {'arm': 'basic', 'seed': 1, 'candidate_sha256': study.digest(candidate),
                      'status': 'completed', 'rollout_executed': True, 'success': False,
                      'private_evaluation': {'success': False, 'hold_s': 2.}}
            study.write_json(path/'result.json', result)
            study.write_json(path/'sandbox.json', {'command': ['bwrap', '--unshare-net']})
            study.write_json(path/'actor_feedback.json', {'seed': 1, 'success': False,
                             'observations': [{'object': 'ring', 'center_m': [0, 0, 0]}]})
            with self.assertRaisesRegex(RuntimeError, 'Base geometry leak'):
                study.audit_trial(path, candidate, 'basic', 1)


if __name__ == '__main__':
    unittest.main()
