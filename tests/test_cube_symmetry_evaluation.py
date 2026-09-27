"""Unmarked cube faces are equivalent; tilt, missing support and noncubes are not."""
import itertools
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from aspire_so100.inventory_tower import InventoryTowerEnv
from aspire_so100.inventory_structure import InventoryStructureEnv


class CubeSymmetryEvaluationTests(unittest.TestCase):
    def test_all_24_cube_rotations_preserve_support_geometry(self):
        with TemporaryDirectory() as tmp:
            env = InventoryTowerEnv(Path(tmp), object_count=3, recipe='cube_triple')
            try:
                env.reset(51)
                name = env.order[0]
                address = int(env.model.joint(f'{name}/{name}_joint').qposadr[0])
                env.data.qpos[address:address+3] = [-.04, -.23, .415]
                count = 0
                for perm in itertools.permutations(range(3)):
                    for signs in itertools.product((-1., 1.), repeat=3):
                        rotation = np.eye(3)[:, perm] @ np.diag(signs)
                        if np.linalg.det(rotation) < 0:
                            continue
                        q = Rotation.from_matrix(rotation).as_quat()
                        env.data.qpos[address+3:address+7] = q[[3, 0, 1, 2]]
                        mujoco.mj_forward(env.model, env.data)
                        normal, top, _ = env.support_frame(name)
                        np.testing.assert_allclose(normal, [0, 0, 1], atol=1e-12)
                        np.testing.assert_allclose(top, [-.04, -.23, .43], atol=1e-12)
                        self.assertAlmostEqual(env.support_margin(name, [-.03, -.23, .43]), .005)
                        self.assertLess(env.support_margin(name, [-.02, -.23, .43]), 0.)
                        count += 1
                self.assertEqual(count, 24)
            finally:
                env.close()

    def test_rotated_physical_stack_passes_but_tilt_still_fails(self):
        with TemporaryDirectory() as tmp:
            env = InventoryTowerEnv(Path(tmp), object_count=3, recipe='cube_triple')
            try:
                env.reset(51)
                rotations = [Rotation.from_euler('y', 90, degrees=True),
                             Rotation.from_euler('x', 180, degrees=True),
                             Rotation.from_euler('x', 90, degrees=True)]
                for i, (name, rot) in enumerate(zip(env.order, rotations)):
                    env._set_freejoint_pose(name, [-.04, -.23, .415 + .03*i], 0.)
                    address = int(env.model.joint(f'{name}/{name}_joint').qposadr[0])
                    env.data.qpos[address+3:address+7] = rot.as_quat()[[3, 0, 1, 2]]
                mujoco.mj_forward(env.model, env.data)
                env._step_for(3.)
                self.assertTrue(env.check_structure(list(env.order), 5.)['stable'])
                self.assertTrue(env.evaluate().success)
                name = env.order[-1]
                address = int(env.model.joint(f'{name}/{name}_joint').qposadr[0])
                env.data.qpos[address+3:address+7] = Rotation.from_euler('x', 12, degrees=True).as_quat()[[3, 0, 1, 2]]
                mujoco.mj_forward(env.model, env.data)
                self.assertIn(name+': no cube face is upright', env.structure_status(env.order)['reasons'])
            finally:
                env.close()

    def test_cylinder_cannot_use_a_side_as_an_upright_face(self):
        with TemporaryDirectory() as tmp:
            env = InventoryTowerEnv(Path(tmp), object_count=3, recipe='cylinder_base')
            try:
                env.reset(51)
                name = env.order[0]
                address = int(env.model.joint(f'{name}/{name}_joint').qposadr[0])
                env.data.qpos[address+3:address+7] = Rotation.from_euler('x', 90, degrees=True).as_quat()[[3, 0, 1, 2]]
                mujoco.mj_forward(env.model, env.data)
                self.assertIn(name+': local Z axis is not upright', env.structure_status(env.order)['reasons'])
            finally:
                env.close()


class SupportGraphCubeSymmetryTests(unittest.TestCase):
    def test_cube_support_faces_are_equivalent_under_all_24_rotations(self):
        with TemporaryDirectory() as tmp:
            env = InventoryStructureEnv(Path(tmp), object_count=4, recipe='thin_beam_bridge')
            try:
                env.reset(51)
                name = env.order[0]
                address = int(env.model.joint(f'{name}/{name}_joint').qposadr[0])
                env.data.qpos[address:address+3] = [-.04, -.23, .415]
                count = 0
                for perm in itertools.permutations(range(3)):
                    for signs in itertools.product((-1., 1.), repeat=3):
                        rotation = np.eye(3)[:, perm] @ np.diag(signs)
                        if np.linalg.det(rotation) < 0:
                            continue
                        q = Rotation.from_matrix(rotation).as_quat()
                        env.data.qpos[address+3:address+7] = q[[3, 0, 1, 2]]
                        mujoco.mj_forward(env.model, env.data)
                        model = env.private_model(name)
                        np.testing.assert_allclose(model['normal'], [0, 0, 1], atol=1e-12)
                        np.testing.assert_allclose(model['top_vertices'][:, 2], .43, atol=1e-12)
                        np.testing.assert_allclose(model['bottom_vertices'][:, 2], .40, atol=1e-12)
                        np.testing.assert_allclose(model['dims'], [.03, .03], atol=1e-12)
                        reasons = env.structure_status(env.order)['reasons']
                        self.assertNotIn(name+': local Z axis is not upright', reasons)
                        self.assertNotIn(name+': no cube face is upright', reasons)
                        count += 1
                self.assertEqual(count, 24)
                env.data.qpos[address+3:address+7] = Rotation.from_euler('x', 12, degrees=True).as_quat()[[3, 0, 1, 2]]
                mujoco.mj_forward(env.model, env.data)
                self.assertIn(name+': no cube face is upright', env.structure_status(env.order)['reasons'])
            finally:
                env.close()

    def test_physical_bridge_accepts_cube_symmetry_but_rejects_wrong_non_cube_axes(self):
        with TemporaryDirectory() as tmp:
            env = InventoryStructureEnv(Path(tmp), object_count=4, recipe='thin_beam_bridge')
            try:
                env.reset(51)
                # Two foundations, a flat beam and a cylinder cap; private test setup only.
                poses = [([-.07, -.26, .415], Rotation.from_euler('y', 90, degrees=True)),
                         ([-.01, -.26, .415], Rotation.from_euler('x', 180, degrees=True)),
                         ([-.04, -.26, .4375], Rotation.from_euler('y', 90, degrees=True)),
                         ([-.04, -.26, .475], Rotation.identity())]
                for name, (center, rot) in zip(env.order, poses):
                    address = int(env.model.joint(f'{name}/{name}_joint').qposadr[0])
                    env.data.qpos[address:address+3] = center
                    env.data.qpos[address+3:address+7] = rot.as_quat()[[3, 0, 1, 2]]
                mujoco.mj_forward(env.model, env.data)
                env._step_for(3.)
                self.assertTrue(env.check_structure(list(env.order), 5.)['stable'])
                self.assertTrue(env.evaluate().success)
                for name, rot, reason in [
                    (env.order[2], Rotation.identity(), ': broad face is not flat with the long axis along X'),
                    (env.order[3], Rotation.from_euler('x', 90, degrees=True), ': local Z axis is not upright'),
                ]:
                    address = int(env.model.joint(f'{name}/{name}_joint').qposadr[0])
                    saved = env.data.qpos[address+3:address+7].copy()
                    env.data.qpos[address+3:address+7] = rot.as_quat()[[3, 0, 1, 2]]
                    mujoco.mj_forward(env.model, env.data)
                    self.assertIn(name+reason, env.structure_status(env.order)['reasons'])
                    env.data.qpos[address+3:address+7] = saved
            finally:
                env.close()
