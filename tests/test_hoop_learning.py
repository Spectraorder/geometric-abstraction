"""Private placement evaluator, scene split and replay checks."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import mujoco
import numpy as np
from aspire_so100.hoop_insertion import HoopPoleEnv
from aspire_so100.hoop_pole import BASE_HEIGHT,POLE_HEIGHT,learning_layout
from aspire_so100.playback import recorded_env_for_result


class HoopLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.root=Path(self.tmp.name)
        self.e=HoopPoleEnv(self.root/'env');self.e.reset(6101)

    def tearDown(self):
        self.e.close();self.tmp.cleanup()

    def place(self,offset):
        e=self.e;e._set_freejoint_pose('red_hoop',e.data.xpos[e.pole]+offset,0.)
        mujoco.mj_forward(e.model,e.data)

    def test_initial_scene_cannot_pass_and_goal_has_no_geometry(self):
        self.assertFalse(self.e.evaluate().success)
        self.assertEqual(set(self.e.public_goal()),{'kind','hoop','pole','instruction'})
        self.assertEqual(self.e.model.neq,0)
        self.assertEqual(learning_layout(6101),learning_layout(6101))
        self.assertNotEqual(learning_layout(6101),learning_layout(6201))

    def test_released_seated_hoop_passes_only_after_hold(self):
        e=self.e;e.set_gripper(.8,.2)
        self.place([0,0,BASE_HEIGHT+.0001])
        e.wait(.5);self.assertFalse(e.evaluate().success)
        e.wait(3.)
        self.assertTrue(e.evaluate().success)
        self.assertEqual(e.evaluate().score,100)
        e.failure='test policy error'
        self.assertFalse(e.evaluate().success)

    def test_misalignment_hover_and_closed_gripper_cannot_pass(self):
        e=self.e;e.set_gripper(.8,.2)
        for offset in ([0,0,POLE_HEIGHT+.01],[.03,0,BASE_HEIGHT]):
            self.place(offset);e.stable_seconds=3.
            self.assertFalse(e.evaluate().success)
        self.place([0,0,BASE_HEIGHT+.0001]);e.wait(3.)
        self.assertTrue(e.evaluate().success)
        e.data.qpos[e.jaw_address]=0.
        self.assertFalse(e.evaluate().success)
        e.data.qpos[e.jaw_address]=.8
        self.place([.04,0,0]);e.wait(.1)
        self.assertEqual(e.stable_seconds,0.)
        self.assertFalse(e.evaluate().success)

    def test_replay_preserves_layout_and_rejects_changed_metadata(self):
        result=dict(task='hoop_pole',seed=6101,scene=dict(self.e.scene_metadata))
        other=recorded_env_for_result(result,self.root/'replay')
        try:
            self.assertEqual(other.scene_metadata,self.e.scene_metadata)
            np.testing.assert_array_equal(other.model.qpos0,self.e.model.qpos0)
        finally:other.close()
        result['scene']['generator']='wrong'
        with self.assertRaisesRegex(ValueError,'differs'):
            recorded_env_for_result(result,self.root/'bad')


class TightHoopLearningTests(HoopLearningTests):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.root=Path(self.tmp.name)
        self.e=HoopPoleEnv(self.root/'env',hoop_variant='tight');self.e.reset(6101)

    def test_tighter_center_tolerance_and_saved_geometry(self):
        e=self.e;e.set_gripper(.8,.2)
        self.assertEqual(e.scene_metadata['hoop_inner_radius_m'],.010)
        self.assertEqual(e.scene_metadata['generator'],'hoop_pole_learning_tight_v1')
        self.place([.0035,0,BASE_HEIGHT+.0001]);e.wait(.5)
        self.assertGreater(e.placement_status()['center_error_m'],.003)
        self.assertFalse(e.placement_status()['seated'])
        self.place([.002,0,BASE_HEIGHT+.0001]);e.wait(3.)
        self.assertTrue(e.evaluate().success)
        result=dict(task='hoop_pole',seed=6101,scene=dict(e.scene_metadata))
        other=recorded_env_for_result(result,self.root/'tight_replay')
        try:
            np.testing.assert_array_equal(other.model.mesh_vert,e.model.mesh_vert)
        finally:other.close()


if __name__=='__main__':unittest.main()
