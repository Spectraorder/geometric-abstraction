"""Task neutrality, variant isolation, resource limits and planned curve accounting."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_three_cube_geometry_study as runner
from run_three_cube_trial import cube_recipe, PHYSICS, apply_physics, mujoco


class ThreeCubeStudyTests(unittest.TestCase):
    def test_goal_is_a_color_sorted_chain_without_metric_truth(self):
        for seed in range(4):
            mapping = runner.base.color_map(seed)
            recipe = cube_recipe(color_map=mapping)
            goal = recipe.public_goal()
            nodes = goal['nodes']
            self.assertEqual([node['color'] for node in nodes], sorted(node['color'] for node in nodes))
            self.assertEqual(len(nodes), 3)
            self.assertTrue(all(node['shape'] == 'cube' and node['grid'] == [0., 0.] for node in nodes))
            self.assertEqual({node['key']: node['supports'] for node in nodes}, {
                mapping['blue']: [], mapping['red']: [mapping['blue']], mapping['green']: [mapping['red']],
            })
            self.assertTrue(all(set(node) == {'key', 'color', 'shape', 'grid', 'supports', 'orientation'} for node in nodes))
        with self.assertRaises(ValueError):
            cube_recipe(color_map={'blue': 'blue'})

    def test_physics_changes_only_the_registered_collision_flag(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="box" size=".01 .01 .01"/></worldbody></mujoco>')
        friction = model.geom_friction.copy()
        timestep = model.opt.timestep
        self.assertEqual(apply_physics(model, {'noslip_iterations': 0}), PHYSICS)
        self.assertEqual(model.opt.timestep, timestep)
        self.assertTrue((model.geom_friction == friction).all())
        with self.assertRaises(ValueError):
            apply_physics(model, {'noslip_iterations': 10})

    def test_variant_registration_preserves_projection_parity_and_restores_hooks(self):
        previous_variant = runner.study.variant
        previous_driver = runner.study.neutral.DRIVER
        with TemporaryDirectory() as folder, runner.installed():
            root = Path(folder)/'replica'
            manifest = runner.variant.prepare(root, [2027100201], [2027101201], 1)
            runner.variant.check(root, manifest)
            self.assertEqual((manifest['object_count'], manifest['recipe']), (3, 'cube_triple'))
            self.assertEqual(manifest['physics'], PHYSICS)
            self.assertEqual(manifest['grasp_fraction_override'], .4)
            self.assertEqual(manifest['initial_policy'], 'observation_only')
            self.assertEqual((root/'basic_library.py').read_text().replace('SURFACE_ONLY = True', 'SURFACE_ONLY = False'),
                             (root/'symbolic_library.py').read_text())
            with runner.variant.configured():
                runner.base.assert_frozen(root, manifest)
        self.assertIs(runner.study.variant, previous_variant)
        self.assertEqual(runner.study.neutral.DRIVER, previous_driver)

    def test_nested_configuration_keeps_the_scheduler_execution_wrapper(self):
        def scheduled(*args, **kwargs):
            raise AssertionError('No rollout should run in this configuration test')
        with runner.installed(), patch.object(runner.variant.neutral, 'execute', scheduled):
            with runner.variant.configured():
                self.assertIs(runner.base.execute, scheduled)

    def test_result_ingestion_uses_the_registered_object_count(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            candidate = root/'candidate.py'
            candidate.write_text('def run(api):\n    pass\n')
            target = root/'basic'/'trials'
            output = target/'seed_1'
            manifest = {'library_sha256': {'basic': 'library'}, 'task': 'inventory_structure',
                        'recipe': 'cube_triple', 'object_count': 3, 'color_maps': {'1': {}},
                        'physics': PHYSICS, 'sensor': {'depth_sampling': 'pixel_center'}}
            result = {'seed': 1, 'success': False, 'candidate_sha256': runner.base.digest(candidate),
                      'geometry_library_sha256': 'library', 'task': manifest['task'], 'layout': 'sampled',
                      'scene': {'seed': 1, 'recipe': 'cube_triple', 'object_count': 3, 'color_map': {}},
                      'physics': PHYSICS, 'sensor': manifest['sensor'], 'sensor_blackout': False, 'wrist_depth': True}
            runner.write_json(output/'result.json', result)
            with patch.object(runner.base, 'assert_frozen'), patch.object(runner.variant.neutral, 'check_goal') as goal:
                rows = runner.variant.execute(root, 'basic', candidate, target, [1], manifest)
                self.assertTrue(rows[0]['rollout_executed'])
                goal.assert_called_once()
                result['scene']['object_count'] = 4
                runner.write_json(output/'result.json', result)
                with self.assertRaisesRegex(ValueError, 'Trial settings differ'):
                    runner.variant.execute(root, 'basic', candidate, target, [1], manifest)

    def test_resource_limits_fall_back_without_exceeding_the_requested_cap(self):
        sample = {'memory': {'MemAvailable': 20*1024**3}, 'gpu_query_returncode': 0,
                  'gpu_csv': 'RTX 3090, 24576, 6000, 50'}
        self.assertEqual(runner.resource_workers(sample, 24), 24)
        self.assertEqual(runner.resource_workers(sample, 8), 8)
        self.assertEqual(runner.resource_workers(dict(sample, memory={'MemAvailable': 10*1024**3}), 24), 16)
        self.assertEqual(runner.resource_workers(dict(sample, memory={'MemAvailable': 5*1024**3}), 24), 8)
        self.assertEqual(runner.resource_workers(dict(sample, gpu_csv='RTX 3090, 24576, 23000, 90'), 24), 8)

    def test_curve_materialization_reuses_only_training_selected_revisions(self):
        curve = runner.planned_curves
        with TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = {'rounds': 3, 'limits': {'rollout': 24}, 'registered_at': 1.,
                        'replicas': [{'directory': 'replica_01'}], 'heldout_seeds': [10, 11],
                        'learning_curve_plan': {'revisions': [0, 1, 2, 3], 'total_slots': 16}}
            write = runner.write_json
            write(root/'protocol.json', protocol)
            write(root/'all_candidates_frozen.json', {})
            write(root/'summary.json', {'status': 'completed'})
            replica = root/'replica_01'
            training = {'revision_frozen': {}, 'frozen': {'basic': {'revision': 1}, 'symbolic': {'revision': 2}}}
            for arm in ('basic', 'symbolic'):
                entries = []
                for revision in range(4):
                    candidate = replica/arm/'frozen_revisions'/f'revision_{revision:03d}.py'
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    candidate.write_text('def run(api):\n    pass\n')
                    entries.append({'revision': revision, 'sha256': runner.base.digest(candidate)})
                training['revision_frozen'][arm] = entries
            write(replica/'training_summary.json', training)
            write(replica/'selected_evaluation/audit.json', {'result_hashes': {}})
            with curve.installed(), patch.object(curve.curves.study, 'check', return_value=protocol), \
                 patch.object(curve.curves.study, 'check_gate') as gate:
                registration = curve.prepare(root, root/'learning_curves', root/'curve_data', 24)
            self.assertGreaterEqual(gate.call_count, 1)
            self.assertEqual((registration['slots'], registration['reused_slots'], registration['new_slots']), (16, 4, 12))
            self.assertEqual(registration['plan_registered_at'], 1.)
            for row in registration['records']:
                self.assertEqual(row['reused'], row['revision'] == training['frozen'][row['arm']]['revision'])
                self.assertFalse((root/row['trial']/'result.json').exists())


if __name__ == '__main__':
    unittest.main()
