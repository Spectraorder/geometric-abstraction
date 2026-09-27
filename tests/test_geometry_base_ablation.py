"""Shared baseline identity, revision-zero selection, and treatment boundaries."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from run_geometry_ablation import prepare, context_for, campaign, digest, write, build_geometry_library

BASE = ROOT/'policies/stack_three_blocks_vision.py'


class GeometryBaseAblationTests(unittest.TestCase):
    def test_shared_source_sensor_contract_and_target_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = prepare(root, 1, [3701], [3801], 'gpt-6-astra', BASE)
            self.assertEqual((root/'base_policy.py').read_bytes(), BASE.read_bytes())
            self.assertEqual(manifest['base_policy_sha256'], digest(BASE))
            self.assertTrue(manifest['wrist_depth'])
            self.assertEqual(manifest['version'], 4)
            folder = root/'bootstrap/seed_3701'
            folder.mkdir(parents=True)
            write(folder/'actor_feedback.json', {'success': False, 'policy_error': None, 'notes': ['shared failure']})
            prompts = {}
            for arm in ('raw', 'geometry'):
                prompt, images = context_for(root, arm, 1, manifest)
                prompts[arm] = prompt
                self.assertIn(BASE.read_text(), prompt)
                self.assertIn('shared failure', prompt)
                self.assertIn('ideal simulated wrist depth', prompt)
                self.assertIn('api.gripper_geometry()', prompt)
                self.assertNotIn('NO wrist depth', prompt)
                self.assertNotIn('3801', prompt)
            self.assertNotIn('geometry.targets', prompts['raw'])
            self.assertIn('geometry.targets', prompts['geometry'])
            (root/'base_policy.py').write_text('def run(api): pass\n')
            with self.assertRaisesRegex(RuntimeError, 'base policy'):
                context_for(root, 'raw', 1, manifest)
        namespace = {}
        exec(build_geometry_library(include_targets=True), namespace)
        api = namespace['GEOMETRY']
        self.assertEqual({x for x in dir(api) if not x.startswith('_')},
                         {'cube', 'table', 'gripper', 'margin', 'targets', 'error', 'relation'})
        with self.assertRaises(AttributeError):
            api.targets = None
        self.assertNotIn('run', namespace)

    def test_privileged_or_treatment_dependent_base_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for body in ('api.get_observation()', 'geometry.cube(api.camera(), "red")'):
                source = root/'base.py'
                source.write_text('def run(api):\n    '+body+'\n')
                with self.assertRaises(ValueError):
                    prepare(root/'experiment', 1, [3701], [3801], 'gpt-6-astra', source)

    def test_revision_zero_can_win_and_is_executed_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = prepare(root, 1, [3701], [3801], 'gpt-6-astra', BASE)
            calls = []
            def propose(directory, prompt, images, settings):
                directory.mkdir(parents=True)
                (directory/'candidate.py').write_text('def run(api):\n    pass\n')
                write(directory/'generation.json', {'candidate_sha256': digest(directory/'candidate.py'), 'usage': {}, 'wall_seconds': 0.})
            def trial(candidate, out, seed, settings, library=None):
                calls.append(str(out.relative_to(root)))
                success = digest(candidate)==digest(BASE)
                if seed == 3801:
                    self.assertTrue((root/'frozen.json').exists())
                    self.assertTrue(all((root/arm/'frozen.py').exists() for arm in ('raw','geometry')))
                out.mkdir(parents=True)
                write(out/'actor_feedback.json', {'success': success, 'policy_error': None, 'notes': []})
                return {'success': success, 'policy_error': None, 'rollout_executed': True, 'failure_stage': None if success else 'task_evaluation'}
            with patch('run_geometry_ablation.propose', side_effect=propose), patch('run_geometry_ablation.trial', side_effect=trial):
                result = campaign(root, manifest)
            self.assertEqual(len(calls), 5)
            self.assertEqual(sum(x.startswith('bootstrap/') for x in calls), 1)
            for arm in ('raw', 'geometry'):
                self.assertEqual(result['frozen'][arm]['revision'], 0)
                self.assertEqual((root/arm/'frozen.py').read_bytes(), BASE.read_bytes())
                self.assertEqual([r['revision'] for r in result['curves'][arm]], [0,1])


if __name__ == '__main__':
    unittest.main()
