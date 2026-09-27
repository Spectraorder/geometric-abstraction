"""Guard against skill leakage, mutable treatments, and unsafe model-tool use."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from unittest.mock import patch
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from run_geometry_ablation import audit_events, build_geometry_library, context_for, feedback, prepare, trial, campaign, digest, write
from test_cube_geometry import cube_frame


class GeometryAblationTests(unittest.TestCase):
    def test_library_measures_sensor_geometry_and_is_read_only(self):
        namespace = {}
        exec(build_geometry_library(), namespace)
        geometry = namespace['GEOMETRY']
        self.assertEqual({x for x in dir(geometry) if not x.startswith('_')},
                         {'cube', 'table', 'gripper', 'margin'})
        frame, center = cube_frame(np.eye(3))
        # The shared ray fixture suspends the cube; raise its rendered table
        # to the bottom face before testing initial dimension measurement.
        v, u = np.indices(frame['depth_m'].shape)
        rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(frame['intrinsics']).T
        rays = rays @ frame['camera_to_world'][:3, :3].T
        table_depth = (.06-frame['camera_to_world'][2, 3])/rays[..., 2]
        frame['depth_m'] = np.minimum(frame['depth_m'], table_depth)
        fit = geometry.cube(frame, 'red', table_plane=geometry.table(frame))
        self.assertAlmostEqual(fit['height'], .04, delta=.001)
        np.testing.assert_allclose(fit['center'], center, atol=.001)
        self.assertEqual(fit['vertices'].shape, (8, 3))
        with self.assertRaises(AttributeError):
            geometry.cube = lambda *args: None
        self.assertNotIn('run', namespace)
        self.assertNotIn('ColumnTracker', namespace)

    def test_actor_tool_use_is_rejected_before_execution(self):
        completion = {'type': 'turn.completed', 'usage': {'input_tokens': 123}}
        self.assertEqual(audit_events([completion]), {'input_tokens': 123})
        for kind in ('command_execution', 'mcp_tool_call', 'web_search', 'file_change', 'unknown'):
            with self.assertRaisesRegex(RuntimeError, 'isolation'):
                audit_events([{'type': 'item.completed', 'item': {'type': kind}}, completion])
        with self.assertRaises(RuntimeError):
            audit_events([])

    def test_feedback_projection_drops_private_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'actor_feedback.json').write_text(json.dumps({
                'success': False, 'policy_error': None, 'notes': ['sensor note'],
                'positions': 'PRIVATE', 'metrics': {'score': 999}, 'seed': 999}))
            self.assertEqual(feedback(out), {'success': False, 'policy_error': None, 'notes': ['sensor note']})

    def test_manifest_mismatches_and_seed_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = prepare(root, 3, [3101], [3201], 'gpt-6-astra')
            self.assertEqual(manifest, prepare(root, 3, [3101], [3201], 'gpt-6-astra'))
            with self.assertRaises(ValueError):
                prepare(root, 4, [3101], [3201], 'gpt-6-astra')
            with self.assertRaises(ValueError):
                prepare(root, 3, [3101], [3101], 'gpt-6-astra')
            prompt, _ = context_for(root, 'raw', 1, manifest)
            self.assertNotIn('geometry.cube', prompt)
            self.assertIn(str(manifest['max_candidate_characters']), prompt)
            self.assertIn(str(manifest['target_candidate_characters']), prompt)
            self.assertTrue((root/'source_snapshot/scripts/run_geometry_ablation.py').exists())
            self.assertNotIn('3101', prompt)
            self.assertNotIn('3201', prompt)
            self.assertNotIn('stack_column_cpu', prompt)
            prompt, _ = context_for(root, 'geometry', 1, manifest)
            self.assertIn('geometry.cube', prompt)
            self.assertIn('model_valid', prompt)
            self.assertIn('Retain a', prompt)

    def test_modified_geometry_is_rejected_before_trial_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = prepare(root, 1, [3101], [3201], 'gpt-6-astra')
            candidate = root/'policy.py'
            candidate.write_text('def run(api):\n    pass\n')
            library = root/'geometry.py'
            library.write_text(library.read_text() + '# modified after freeze\n')
            with self.assertRaisesRegex(RuntimeError, 'frozen treatment'):
                trial(candidate, root/'trial', 3101, manifest, library)
            self.assertFalse((root/'trial').exists())

    def test_oversize_candidate_is_preserved_and_never_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = prepare(root, 1, [3301], [3401], 'gpt-6-astra')
            candidate = root/'candidate.py'
            source = 'def run(api):\n    pass\n' + '#' * manifest['max_candidate_characters']
            candidate.write_text(source)
            result = trial(candidate, root/'trial', 3301, manifest)
            self.assertFalse(result['rollout_executed'])
            self.assertEqual(result['failure_stage'], 'candidate_validation')
            self.assertFalse((root/'trial/rollout.npz').exists())
            self.assertEqual(candidate.read_text(), source)
            self.assertIn(str(manifest['max_candidate_characters']), feedback(root/'trial')['policy_error'])
            self.assertEqual(trial(candidate, root/'trial', 3301, manifest)['failure_stage'],
                             'candidate_validation')

    def test_partial_cube_pose_retains_model_and_does_not_claim_full_observation(self):
        namespace = {}
        exec(build_geometry_library(), namespace)
        geometry = namespace['GEOMETRY']
        frame, center = cube_frame(np.eye(3), hide_top=True)
        model = {'height': .04, 'dims': np.array([.04, .04])}
        fit = geometry.cube(frame, 'red', model=model, table_plane=np.zeros(3))
        self.assertTrue(fit['model_valid'])
        self.assertFalse(fit['position_fully_observed'])
        self.assertFalse(fit['geometry_reliable'])
        self.assertAlmostEqual(fit['height'], .04)
        prior_center = center + [.001, -.002, .01]
        predicted = geometry.cube(frame, 'red', model=model, table_plane=np.zeros(3),
                                  prior={'center': prior_center})
        axes = predicted['rotation'][:, ~predicted['constrained_axes']]
        np.testing.assert_allclose(axes.T @ predicted['center'], axes.T @ prior_center, atol=1e-9)
        self.assertEqual(predicted['position_source'], 'prior_and_visible_planes')
        with self.assertRaises(ValueError):
            geometry.cube(frame, 'red', model={'height': float('nan')}, table_plane=np.zeros(3))

    def test_recorded_initial_view_excludes_blue_background_and_keeps_partial_red(self):
        namespace = {}
        exec(build_geometry_library(), namespace)
        with np.load(ROOT/'tests/fixtures/ablation_cube_initial_rgbd.npz') as archive:
            frame = dict(archive)
        observations = {color: namespace['GEOMETRY'].cube(frame, color)
                        for color in ('blue', 'red', 'green')}
        for fit in observations.values():
            self.assertTrue(fit['model_valid'])
            self.assertTrue(np.isfinite(fit['center']).all())
            self.assertAlmostEqual(fit['height'], .03, delta=.001)
        self.assertTrue(observations['blue']['geometry_reliable'])
        self.assertFalse(observations['red']['position_fully_observed'])

    def test_parallel_proposals_keep_histories_separate_and_freeze_before_heldout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = prepare(root, 2, [3301], [3401], 'gpt-6-astra')
            barrier = threading.Barrier(2)
            prompts = []
            calls = []

            def fake_propose(directory, prompt, images, settings):
                arm = directory.parent.name
                barrier.wait(timeout=5)
                prompts.append((arm, prompt))
                directory.mkdir(parents=True)
                source = f'def run(api):\n    marker = "{arm}_private"\n'
                write(directory/'response.json', {'hypothesis': arm+'_private', 'code': source})
                (directory/'candidate.py').write_text(source)
                write(directory/'generation.json', {'candidate_sha256': digest(directory/'candidate.py'),
                                                    'usage': {}, 'wall_seconds': 0., 'audit': 'passed'})

            def fake_trial(candidate, out, seed, settings, library=None):
                calls.append((str(out.relative_to(root)), seed))
                if seed == 3401:
                    self.assertTrue((root/'frozen.json').exists())
                    self.assertTrue(all((root/arm/'frozen.py').exists() for arm in ('raw', 'geometry')))
                out.mkdir(parents=True)
                (out/'initial.png').write_bytes(b'fixture')
                write(out/'actor_feedback.json', {'success': False, 'policy_error': None, 'notes': []})
                return {'success': False, 'policy_error': None, 'rollout_executed': True,
                        'failure_stage': 'task_evaluation'}

            with patch('run_geometry_ablation.propose', side_effect=fake_propose), \
                 patch('run_geometry_ablation.trial', side_effect=fake_trial):
                result = campaign(root, manifest)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(len(calls), 7)  # One bootstrap, four training, two reserved.
            self.assertEqual(len(prompts), 4)
            for arm, prompt in prompts:
                self.assertNotIn(('geometry' if arm == 'raw' else 'raw')+'_private', prompt)
            self.assertEqual(result['frozen']['raw']['revision'], 1)
            self.assertEqual(result['frozen']['geometry']['revision'], 1)

    def test_worker_exports_library_only_when_explicitly_supplied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root/'geometry.py'
            library.write_text(build_geometry_library())
            candidate = root/'candidate.py'
            candidate.write_text('def run(api):\n    assert geometry is not None\n')
            command = [sys.executable, str(ROOT/'scripts/vision_policy_worker.py'), str(candidate)]
            absent = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(absent.returncode, 0)
            self.assertIn('NameError', absent.stdout)
            supplied = subprocess.run(command+[str(library)], capture_output=True, text=True, timeout=30)
            self.assertEqual(supplied.returncode, 0, supplied.stderr)
            self.assertEqual(json.loads(supplied.stdout)['method'], 'done')


if __name__ == '__main__':
    unittest.main()
