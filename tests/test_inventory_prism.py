"""Identity-aware private evaluation and reconstruction of sampled scenes."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import mujoco
import numpy as np

from aspire_so100.block_catalog import generate_block_layout
from aspire_so100.inventory_prism import InventoryPrismEnv
from aspire_so100.playback import recorded_env_for_result


class InventoryPrismTests(unittest.TestCase):
    def test_stacking_a_same_color_cylinder_does_not_satisfy_the_task(self):
        with TemporaryDirectory() as temp:
            env = InventoryPrismEnv(Path(temp), scene_seed=604, object_count=6)
            try:
                env.reset(604)
                env._set_freejoint_pose('blue_cube', (-.04, -.23, .415), 0.)
                env._set_freejoint_pose('red_cylinder', (-.04, -.23, .46), 0.)
                env._set_freejoint_pose('red_thin_prism', (.12, -.36, .43), 0.)
                mujoco.mj_forward(env.model, env.data)
                env._step_for(12.)
                self.assertTrue(env._objects_in_contact('red_cylinder', 'blue_cube'))
                self.assertFalse(env.check_structure(list(env.order), 5.)['stable'])
                self.assertFalse(env.evaluate().success)
            finally:
                env.close()

    def test_evaluator_checks_the_required_object_and_its_long_axis(self):
        with TemporaryDirectory() as temp:
            env = InventoryPrismEnv(Path(temp), scene_seed=601, object_count=2)
            try:
                obs = env.reset(601)
                self.assertEqual(set(obs['objects']), {'red_thin_prism', 'blue_cube'})
                self.assertFalse(env.structure_status(list(env.order))['stable'])
                env._set_freejoint_pose('blue_cube', (-.04, -.23, .415), 0.)
                env._set_freejoint_pose('red_thin_prism', (-.04, -.23, .46), 0.)
                mujoco.mj_forward(env.model, env.data)
                env._step_for(12.)
                self.assertTrue(env.check_structure(list(env.order), 5.)['stable'])
                self.assertTrue(env.evaluate().success)
                joint = env.model.joint('red_thin_prism/red_thin_prism_joint')
                address = int(joint.qposadr[0])
                env.data.qpos[address+2] = .4375
                env.data.qpos[address+3:address+7] = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
                env.data.qvel[:] = 0.
                mujoco.mj_forward(env.model, env.data)
                self.assertIn('red_thin_prism: long axis is not vertical',
                              env.structure_status(list(env.order))['reasons'])
                self.assertFalse(env.evaluate().success)
                with self.assertRaisesRegex(ValueError, 'different seed'):
                    env.reset(602)
            finally:
                env.close()

    def test_replay_validates_identity_order_not_only_state_width(self):
        with TemporaryDirectory() as temp:
            env = InventoryPrismEnv(Path(temp), scene_seed=601, object_count=6)
            try:
                obs = env.reset(601)
                self.assertEqual(set(obs['objects']), set(generate_block_layout(601, 6, 'upright_thin_prism').positions))
                result = {'task': 'inventory_prism', 'seed': 601, 'scene': env.scene_metadata}
                replay = recorded_env_for_result(result, Path(temp))
                try:
                    self.assertEqual(replay.scene_metadata, env.scene_metadata)
                    self.assertEqual(replay.model.nq, env.model.nq)
                finally:
                    replay.close()
                broken = deepcopy(result)
                broken['scene']['objects'].reverse()
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    recorded_env_for_result(broken, Path(temp))
            finally:
                env.close()
