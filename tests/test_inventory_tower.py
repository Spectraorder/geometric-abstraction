"""Symbolic task boundaries, shape-aware support, and ordered-column scoring."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from aspire_so100.block_catalog import generate_block_layout
from aspire_so100.inventory_tower import InventoryTowerEnv
from aspire_so100.playback import recorded_env_for_result
from aspire_so100.tower_tasks import RECIPES, TowerRecipe, tower_recipe
from test_randomized_vision_resume import LEARNER
from test_vision_boundary import runner

ROOT = Path(__file__).resolve().parents[1]


class TowerTaskTests(unittest.TestCase):
    def test_random_goals_are_seeded_unique_and_pose_free(self):
        identities, orders = set(), set()
        for seed in range(20):
            for levels in (2, 3, 4):
                recipe = tower_recipe('random', seed=seed, levels=levels)
                self.assertEqual(recipe, tower_recipe('random', seed=seed, levels=levels))
                self.assertEqual(len(set(recipe.order)), levels)
                identities.update(recipe.order)
                orders.add(recipe.order)
                public = recipe.public_goal()
                self.assertEqual(set(public), {'kind', 'levels'})
                for item in public['levels']:
                    self.assertEqual(set(item), {'color', 'shape'})
                layout = generate_block_layout(seed, levels+2, 'single_column', required_objects=recipe.order)
                self.assertTrue(set(recipe.order) <= set(layout.positions))
                self.assertEqual(tuple(layout.roles.values()), recipe.order)
        self.assertGreater(len(orders), 40)
        self.assertEqual({n.split('_', 1)[0] for n in identities}, {'red', 'yellow', 'blue', 'green'})
        self.assertEqual({n.split('_', 1)[1] for n in identities}, {'cube', 'prism', 'thin_prism', 'cylinder'})

    def test_invalid_recipes_and_missing_required_objects_are_rejected(self):
        for order in [('blue_cube',), ('blue_cube', 'blue_cube'), ('blue_cube', 'pink_cube'),
                      ('red_triangular_prism', 'blue_cube')]:
            with self.subTest(order=order), self.assertRaises(ValueError):
                TowerRecipe('invalid', order)
        for levels in (1, 5, True, 2.5):
            with self.assertRaises(ValueError):
                tower_recipe('random', levels=levels)
        with self.assertRaises(ValueError):
            tower_recipe('cube_pair', levels=3)
        with self.assertRaises(ValueError):
            generate_block_layout(1, 2, 'single_column', required_objects=RECIPES['cube_triple'])

    def test_three_and_four_level_physical_columns_pass_but_wrong_order_fails(self):
        for recipe in ('cube_triple', 'cube_quad'):
            with self.subTest(recipe=recipe), TemporaryDirectory() as temp:
                env = InventoryTowerEnv(Path(temp), recipe=recipe, object_count=len(RECIPES[recipe]))
                try:
                    env.reset(51)
                    self.assertFalse(env.evaluate().success)
                    for index, name in enumerate(env.order):
                        env._set_freejoint_pose(name, (-.04, -.23, .415+.03*index), 0.)
                    mujoco.mj_forward(env.model, env.data)
                    env._step_for(3.)
                    self.assertTrue(env.check_structure(list(env.order), 5.)['stable'])
                    self.assertTrue(env.evaluate().success)
                    first, second = env.order[1:3]
                    for name, z in ((first, .475), (second, .445)):
                        env._set_freejoint_pose(name, (-.04, -.23, z), 0.)
                    mujoco.mj_forward(env.model, env.data)
                    self.assertFalse(env.evaluate().success)
                finally:
                    env.close()

    def test_cylinder_bounds_and_circular_support_exclude_square_corners(self):
        with TemporaryDirectory() as temp:
            env = InventoryTowerEnv(Path(temp), recipe='cylinder_base', object_count=3)
            try:
                env.reset(51)
                name = env.order[0]
                env._set_freejoint_pose(name, (-.04, -.23, .43), 0.)
                mujoco.mj_forward(env.model, env.data)
                low, high = env.geometry_bounds(name)
                np.testing.assert_allclose(high-low, [.03, .03, .06], atol=1e-7)
                self.assertGreater(env.support_margin(name, [-.04, -.23, .46]), .014)
                # This point lies inside the bounding square but outside the disk.
                self.assertLess(env.support_margin(name, [-.026, -.216, .46]), 0)
                for n, z in zip(env.order[1:], (.475, .505)):
                    env._set_freejoint_pose(n, (-.04, -.23, z), 0.)
                mujoco.mj_forward(env.model, env.data)
                env._step_for(3.)
                self.assertTrue(env.check_structure(list(env.order), 5.)['stable'])
                self.assertTrue(env.evaluate().success)
            finally:
                env.close()

    def test_triangular_cap_is_representable_but_not_an_interior_support(self):
        recipe = tower_recipe('wedge_cap')
        self.assertTrue(recipe.order[-1].endswith('triangular_prism'))
        with TemporaryDirectory() as temp:
            env = InventoryTowerEnv(Path(temp), recipe='wedge_cap', object_count=3)
            try:
                env.reset(51)
                for i, name in enumerate(env.order):
                    env._set_freejoint_pose(name, (-.04, -.23, .415+.03*i), 0.)
                mujoco.mj_forward(env.model, env.data)
                low, high = env.geometry_bounds(env.order[-1])
                np.testing.assert_allclose(high-low, [.03]*3, atol=1e-7)
                self.assertAlmostEqual(low[2], .46, places=7)
                self.assertEqual(env.support_margin(env.order[-1], [-.04, -.23, .49]), -np.inf)
                # Representability is not evidence that an unvalidated mesh cap
                # or grasp will remain stable over the physical time window.
                self.assertFalse(env.evaluate().success)
            finally:
                env.close()

    def test_recorded_recipe_order_is_validated_on_replay(self):
        with TemporaryDirectory() as temp:
            env = InventoryTowerEnv(Path(temp), recipe='random', levels=4, scene_seed=812, object_count=6)
            try:
                result = dict(task='inventory_tower', seed=812, scene=env.scene_metadata)
                replay = recorded_env_for_result(result, Path(temp))
                try:
                    self.assertEqual(replay.order, env.order)
                    self.assertEqual(replay.model.nq, env.model.nq)
                finally:
                    replay.close()
                bad = deepcopy(result)
                bad['scene']['objects'].reverse()
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    recorded_env_for_result(bad, Path(temp))
            finally:
                env.close()

    def test_cached_trial_cannot_be_reused_for_a_different_recipe(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root/'candidate.py'
            candidate.write_text('def run(api): pass\n')
            out = root/'trials/seed_801'
            out.mkdir(parents=True)
            result = dict(candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                          task='inventory_tower', layout='sampled', seed=801,
                          scene=dict(object_count=3, recipe='cube_triple'),
                          goal=tower_recipe('cube_triple').public_goal(), success=False,
                          metrics=dict(score=0.), policy_error=None)
            (out/'result.json').write_text(json.dumps(result))
            with patch.object(LEARNER.subprocess, 'run') as run:
                LEARNER.evaluate(candidate, out.parent, [801], 1, task='inventory_tower', layout='sampled',
                                 object_count=3, tower_recipe='cube_triple')
                with self.assertRaisesRegex(RuntimeError, 'different tower goal'):
                    LEARNER.evaluate(candidate, out.parent, [801], 1, task='inventory_tower', layout='sampled',
                                     object_count=3, tower_recipe='slender_top')
                run.assert_not_called()

    def test_generated_policy_is_reproducible_and_rejects_unsupported_cap_before_motion(self):
        spec = importlib.util.spec_from_file_location('builder', ROOT/'scripts/build_inventory_policy.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        source = (ROOT/'policies/stack_column_cpu.py').read_text()
        self.assertEqual(source, builder.build_policy('column_policy'))
        runner.validate_source(source)
        namespace = {}
        exec(source, namespace)
        class API:
            def task(self):
                return tower_recipe('wedge_cap').public_goal()
            def note(self, text):
                self.message = text
        api = API()
        namespace['run'](api)
        self.assertIn('not implemented', api.message)
