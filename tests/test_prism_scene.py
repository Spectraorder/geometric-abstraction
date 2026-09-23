"""Six-object scene, deterministic layouts, and task-specific upright success."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import mujoco
import numpy as np

from aspire_so100.prism import PrismEnv
from aspire_so100.prism_layouts import PRISM_OBJECTS, PRISM_HALF_SIZES, PRISM_ORDER, prism_positions


class PrismSceneTests(unittest.TestCase):
    def test_random_layouts_repeat_with_clearance(self):
        for seed in range(20):
            positions = prism_positions(seed)
            self.assertEqual(set(positions), set(PRISM_OBJECTS))
            repeated = prism_positions(seed)
            for name, xy in positions.items():
                np.testing.assert_array_equal(xy, repeated[name])
                for other in positions:
                    if name != other:
                        margin = np.array(PRISM_HALF_SIZES[name][:2])+PRISM_HALF_SIZES[other][:2]+.03
                        self.assertTrue(np.any(abs(xy-positions[other]) >= margin))

    def test_scene_contains_exactly_six_task_objects_with_correct_sizes(self):
        with TemporaryDirectory() as tmp:
            env = PrismEnv(Path(tmp))
            try:
                observed = env.reset(1, 'nominal')
                free_bodies = {env.model.body(env.model.jnt_bodyid[j]).name.split('/')[0]
                               for j in range(env.model.njnt)
                               if env.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
                self.assertEqual(free_bodies, set(PRISM_OBJECTS))
                self.assertEqual(set(observed['objects']), set(PRISM_OBJECTS))
                for name in PRISM_OBJECTS:
                    expected = (.015, .03, 0.) if name == 'cyan_cylinder' else PRISM_HALF_SIZES[name]
                    np.testing.assert_allclose(env.model.geom(f'{name}/{name}_geom').size, expected)
            finally:
                env.close()

    def test_evaluator_accepts_upright_stack_and_rejects_horizontal_prism(self):
        with TemporaryDirectory() as tmp:
            env = PrismEnv(Path(tmp))
            try:
                env.reset(1, 'nominal')
                # Initialize both bodies as an exactly aligned stack; do not
                # teleport the prism onto an already slightly tilted base.
                env._set_freejoint_pose(PRISM_ORDER[0], (-.04, -.23, .415), 0.)
                env._set_freejoint_pose(PRISM_ORDER[1], (-.04, -.23, .46), 0.)
                mujoco.mj_forward(env.model, env.data)
                env._step_for(12.)  # Let insertion/contact transients settle.
                self.assertTrue(env.check_structure(list(PRISM_ORDER), duration=5.)['stable'])
                self.assertTrue(env.evaluate().success)
                joint = env.model.joint(f'{PRISM_ORDER[1]}/{PRISM_ORDER[1]}_joint')
                address = int(joint.qposadr[0])
                env.data.qpos[address+2] = .4375
                env.data.qpos[address+3:address+7] = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
                env.data.qvel[:] = 0.
                mujoco.mj_forward(env.model, env.data)
                status = env.structure_status(list(PRISM_ORDER))
                self.assertFalse(status['stable'])
                self.assertIn('purple_thin_prism: long axis is not vertical', status['reasons'])
                self.assertFalse(env.evaluate().success)
            finally:
                env.close()
