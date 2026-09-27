"""Support graph goals, shared load feasibility and private physical checks."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from aspire_so100.block_catalog import generate_block_layout
from aspire_so100.inventory_structure import InventoryStructureEnv, configure_structure_initial_state
from aspire_so100.playback import recorded_env_for_result
from aspire_so100.structure_geometry import clip_polygon, model_footprint, support_equilibrium
from aspire_so100.structure_tasks import StructureNode, StructureRecipe, STRUCTURE_RECIPES, structure_recipe

ROOT = Path(__file__).resolve().parents[1]


def box(center, side=.03, mass=1.):
    return {'shape': 'cube', 'center': np.asarray(center, float), 'dims': np.array([side, side]), 'mass': mass}


class StructureTaskTests(unittest.TestCase):
    def test_public_goal_has_only_identities_and_desired_relative_relationships(self):
        for name in STRUCTURE_RECIPES:
            recipe = structure_recipe(name)
            goal = recipe.public_goal()
            self.assertEqual(set(goal), {'kind', 'length_unit', 'axes', 'nodes'})
            self.assertEqual(goal['kind'], 'support_graph')
            for node in goal['nodes']:
                self.assertEqual(set(node), {'key', 'color', 'shape', 'grid', 'supports', 'orientation'})
            layout = generate_block_layout(31, len(recipe.nodes)+2, 'support_graph', required_objects=recipe.order)
            self.assertTrue(set(recipe.order) <= set(layout.positions))
            self.assertEqual(len(layout.positions), len(recipe.nodes)+2)
            self.assertEqual(tuple(layout.roles.values()), recipe.order)
        mapping = dict(red='blue', blue='green', green='yellow', yellow='red')
        changed = structure_recipe('cube_bridge', color_map=mapping)
        self.assertEqual(changed.order, ('green_cube', 'blue_cube', 'yellow_cube'))
        self.assertEqual(changed.nodes[-1].supports, ('left', 'right'))

    def test_flat_bridge_goal_and_private_faces_use_the_short_vertical_axis(self):
        recipe = structure_recipe('thin_beam_bridge')
        self.assertEqual(recipe.nodes[1].grid[0]-recipe.nodes[0].grid[0], 2.)
        self.assertEqual(recipe.nodes[2].orientation, 'flat_x')
        self.assertEqual(recipe.nodes[2].supports, ('left', 'right'))
        mapping = dict(red='blue', blue='green', green='yellow', yellow='red')
        self.assertEqual(structure_recipe('thin_beam_bridge', color_map=mapping).nodes[2].orientation, 'flat_x')
        with TemporaryDirectory() as directory:
            env = InventoryStructureEnv(Path(directory), scene_seed=1401,
                                        recipe='thin_beam_bridge', object_count=4)
            try:
                env.reset(1401)
                model = env.private_model('green_thin_prism')
                self.assertAlmostEqual(model['height'], .015)
                np.testing.assert_allclose(np.ptp(model['top_vertices'], axis=0), [.06, .03, 0.], atol=1e-5)
                np.testing.assert_allclose(model['normal'], [0., 0., 1.], atol=1e-5)
                self.assertEqual(env.scene_metadata['initial_orientations'], {'green_thin_prism': 'flat_x'})
                self.assertFalse(env.evaluate().success)
                self.assertAlmostEqual(env.geometry_bounds('green_thin_prism')[0][2], .4, delta=.0002)
                before = env.data.qpos.copy()
                configure_structure_initial_state(env.model, recipe)
                env.reset(1401)
                np.testing.assert_allclose(env.data.qpos, before, atol=1e-9)
            finally:
                env.close()

    def test_invalid_graphs_cannot_reach_scene_generation(self):
        recipe = structure_recipe('cube_bridge')
        with self.assertRaises(ValueError):
            StructureRecipe('cycle', (recipe.nodes[-1], *recipe.nodes[:-1]), 'left')
        with self.assertRaises(ValueError):
            StructureRecipe('duplicate', (recipe.nodes[0], recipe.nodes[0], recipe.nodes[-1]), 'left')
        with self.assertRaises(ValueError):
            StructureRecipe('unknown', (*recipe.nodes[:-1], StructureNode('span', 'green_cube', (0., 0.), ('missing',))), 'left')
        with self.assertRaises(ValueError):
            StructureRecipe('bad_grid', (*recipe.nodes[:-1], StructureNode('span', 'green_cube', (np.nan, 0.), ('left',))), 'left')
        with self.assertRaises(ValueError):
            structure_recipe('cube_bridge', color_map=dict(red='red', yellow='red', blue='blue', green='green'))
        with self.assertRaises(ValueError):
            generate_block_layout(1, 4, 'support_graph', required_objects=structure_recipe('staircase').order)
        with self.assertRaises(ValueError):
            generate_block_layout(1, 5, 'single_column', required_objects=structure_recipe('staircase').order)

    def test_shared_load_is_balanced_across_two_real_support_patches(self):
        models = {'left': box([-.0165, 0., .015]), 'right': box([.0165, 0., .015]),
                  'span': box([0., 0., .045])}
        parents = {'left': [], 'right': [], 'span': ['left', 'right']}
        result = support_equilibrium(models, parents)
        self.assertTrue(result['feasible'], result)
        loads = {(a, b): load for a, b, load in result['loads']}
        self.assertGreater(loads['left', 'span'], 0.)
        self.assertGreater(loads['right', 'span'], 0.)
        self.assertAlmostEqual(loads['left', 'span']+loads['right', 'span'], 1/3)
        # The span COM is outside each individual footprint. Its union supports it.
        self.assertGreater(.0165, .015)
        moved = deepcopy(models)
        moved['span']['center'][0] = .03
        self.assertFalse(support_equilibrium(moved, parents)['feasible'])
        # Force both loads through only the left support: the top COM tips it.
        self.assertFalse(support_equilibrium(models, dict(parents, span=['left']))['feasible'])

    def test_patch_margin_and_cylinder_disk_are_geometric(self):
        polygon = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])*.015
        inside = clip_polygon(polygon, polygon, margin=.002)
        np.testing.assert_allclose(inside.min(axis=0), [-.013, -.013])
        np.testing.assert_allclose(inside.max(axis=0), [.013, .013])
        self.assertEqual(len(clip_polygon(polygon, polygon, margin=.02)), 0)
        models = {'base': dict(box([0., 0., 0.]), shape='cylinder'),
                  'top': box([.013, .013, .03], side=.008)}
        self.assertFalse(support_equilibrium(models, {'base': [], 'top': ['base']})['feasible'])

    def test_tilted_top_and_bottom_faces_remain_distinct(self):
        angle = np.deg2rad(6.)
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                             [-np.sin(angle), 0., np.cos(angle)]])
        corners = np.array([[x, y, z] for x in (-.015, .015) for y in (-.015, .015) for z in (-.03, .03)])
        model = dict(box([0., 0., .03]), height=.06, normal=rotation[:, 2], vertices=corners @ rotation.T+[0., 0., .03])
        for shape in ('prism', 'cylinder'):
            model['shape'] = shape
            top = model_footprint(model, 'top').mean(axis=0)
            bottom = model_footprint(model, 'bottom').mean(axis=0)
            np.testing.assert_allclose(top-bottom, [.06*np.sin(angle), 0.], atol=1e-10)

    def test_viewer_forwards_the_structure_goal_and_color_mapping(self):
        from aspire_so100.playback import watch_vision_policy
        mapping = dict(red='blue', blue='green', green='yellow', yellow='red')
        with TemporaryDirectory() as directory:
            output = Path(directory)/'trial'
            (Path(directory)/'candidate.py').write_text('def run(api): pass\n')
            with patch('aspire_so100.playback.subprocess.run') as run, patch('aspire_so100.playback.watch_vision_rollout') as watch:
                watch_vision_policy(Path(directory)/'candidate.py', output=output, seed=1101,
                                    layout='sampled', task='inventory_structure', object_count=5,
                                    structure_recipe='staircase', structure_color_map=mapping)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index('--structure-recipe')+1], 'staircase')
                self.assertIn('--structure-color-map', command)
                self.assertEqual(command[command.index('--task')+1], 'inventory_structure')
                watch.assert_called_once_with(output)

    def test_physical_fixtures_require_all_graph_edges_and_keep_replay_identity(self):
        for name in STRUCTURE_RECIPES:
            with self.subTest(recipe=name), TemporaryDirectory() as directory:
                recipe = structure_recipe(name)
                env = InventoryStructureEnv(Path(directory), scene_seed=51, recipe=name, object_count=len(recipe.nodes))
                try:
                    env.reset(51)
                    self.assertFalse(env.evaluate().success)
                    heights = {}
                    for node in recipe.nodes:
                        geom = env.model.geom(f'{node.object_name}/{node.object_name}_geom')
                        half = float(geom.size[0] if node.orientation == 'flat_x' else geom.size[1] if node.object_name.endswith('_cylinder') else geom.size[2])
                        bottom = max([heights[k] for k in node.supports], default=.4)
                        xy = np.array([-.05, -.38])+.03*np.asarray(node.grid)
                        env._set_freejoint_pose(node.object_name, (*xy, bottom+half), 0.)
                        if node.orientation == 'flat_x':
                            address = int(env.model.joint(f'{node.object_name}/{node.object_name}_joint').qposadr[0])
                            env.data.qpos[address+3:address+7] = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
                        heights[node.key] = bottom+2*half
                    mujoco.mj_forward(env.model, env.data)
                    env._step_for(2.)
                    check = env.check_structure(list(env.order), 5.)
                    self.assertTrue(check['stable'], check)
                    self.assertTrue(env.evaluate().success)
                    self.assertEqual(env.evaluate().required_placements, len(recipe.nodes))
                    result = {'task': 'inventory_structure', 'seed': 51, 'scene': env.scene_metadata}
                    replay = recorded_env_for_result(result, Path(directory)/'replay')
                    try:
                        self.assertEqual(replay.scene_metadata, env.scene_metadata)
                    finally:
                        replay.close()
                    if name == 'thin_beam_bridge':
                        # Keep the beam center at its correct target but turn it upright.
                        center = env._object_state('green_thin_prism')['position']
                        env._set_freejoint_pose('green_thin_prism', tuple(center), 0.)
                        mujoco.mj_forward(env.model, env.data)
                        reasons = env.structure_status(list(env.order))['reasons']
                        self.assertTrue(any('broad face is not flat' in reason for reason in reasons))
                    if name == 'cube_bridge':
                        env._set_freejoint_pose('red_cube', (.08, -.25, .415), 0.)
                        mujoco.mj_forward(env.model, env.data)
                        status = env.structure_status(list(env.order))
                        self.assertFalse(status['stable'])
                        self.assertTrue(any('missing required support red_cube' in reason for reason in status['reasons']))
                finally:
                    env.close()
