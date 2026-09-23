"""Inventory geometry, bounded scene sampling, and task-specific requirements."""
import unittest

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from aspire_so100.block_catalog import (
    BLOCK_COLORS, CATALOG_OBJECTS, SHAPE_DIMENSIONS, generate_block_layout,
)
from aspire_so100.stacking_env import _SCENE_MODULE as scene


def free_objects(model):
    return {model.body(model.jnt_bodyid[j]).name.split('/')[0]
            for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}


class BlockCatalogTests(unittest.TestCase):
    def test_triangular_prism_is_exactly_half_a_cube(self):
        model = scene.get_triangular_prism_spec('wedge', .03, BLOCK_COLORS['yellow']).compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        vertices = model.mesh_vert @ data.geom_xmat[0].reshape(3, 3).T + data.geom_xpos[0]
        self.assertEqual(len(vertices), 6)
        np.testing.assert_allclose(np.ptp(vertices, axis=0), [.03]*3, atol=1e-8)
        self.assertAlmostEqual(ConvexHull(vertices).volume, .03**3/2, places=11)
        np.testing.assert_allclose(np.mean(vertices, axis=0), [-.005, 0., -.005], atol=1e-8)
        self.assertEqual(model.geom_type[0], mujoco.mjtGeom.mjGEOM_MESH)
        self.assertAlmostEqual(model.body_mass[1], .015)

    def test_catalog_has_exactly_twenty_colored_objects_and_settles(self):
        _, model, data = scene.build_scene(scene_task='catalog')
        self.assertEqual(free_objects(model), set(CATALOG_OBJECTS))
        for color, rgba in BLOCK_COLORS.items():
            for shape, dimensions in SHAPE_DIMENSIONS.items():
                name = f'{color}_{shape}'
                geom = model.geom(f'{name}/{name}_geom')
                np.testing.assert_allclose(geom.rgba, rgba, atol=1e-7)
                if shape == 'cylinder':
                    np.testing.assert_allclose(geom.size[:2], [.015, .03])
                elif shape != 'triangular_prism':
                    np.testing.assert_allclose(geom.size, np.asarray(dimensions)/2)
        mujoco.mj_step(model, data, nstep=int(2/model.opt.timestep))
        table = model.geom('table/table_top').id
        touching = {int(c.geom[1] if c.geom[0] == table else c.geom[0])
                    for c in data.contact if table in c.geom}
        for name in CATALOG_OBJECTS:
            self.assertIn(model.geom(f'{name}/{name}_geom').id, touching)
        self.assertTrue(np.isfinite(data.qpos).all())

    def test_sampling_is_seeded_unique_and_separated(self):
        seen = set()
        for seed in range(30):
            for count in (1, 6, 12, 20):
                layout = generate_block_layout(seed, count)
                self.assertEqual(layout, generate_block_layout(seed, count))
                self.assertEqual(len(layout.positions), count)
                self.assertTrue(set(layout.positions) <= set(CATALOG_OBJECTS))
                seen.update(layout.positions)
                names = list(layout.positions)
                for i, a in enumerate(names):
                    for b in names[i+1:]:
                        size_a = np.array(SHAPE_DIMENSIONS[a.split('_', 1)[1]][:2])/2
                        size_b = np.array(SHAPE_DIMENSIONS[b.split('_', 1)[1]][:2])/2
                        distance = abs(np.array(layout.positions[a][:2])-layout.positions[b][:2])
                        self.assertTrue(np.any(distance >= size_a+size_b+.02))
        self.assertEqual(seen, set(CATALOG_OBJECTS))

    def test_upright_task_reserves_target_and_base_in_central_slots(self):
        for seed in range(30):
            layout = generate_block_layout(seed, 6, 'upright_thin_prism')
            self.assertEqual(layout.roles, {'target': 'red_thin_prism', 'base': 'blue_cube'})
            self.assertEqual(len(layout.positions), 6)
            for name in layout.roles.values():
                x, y, _ = layout.positions[name]
                self.assertLessEqual(abs(x), .106)
                self.assertTrue(-.366 <= y <= -.194)
        _, model, _ = scene.build_scene(scene_task='upright_thin_prism', seed=17, object_count=4)
        self.assertEqual(free_objects(model), set(generate_block_layout(17, 4, 'upright_thin_prism').positions))
        for count in (0, 21, True, 2.5):
            with self.assertRaises(ValueError):
                generate_block_layout(count=count)
        with self.assertRaises(ValueError):
            generate_block_layout(count=1, task='upright_thin_prism')
        with self.assertRaises(ValueError):
            generate_block_layout(task='unknown')
