"""Five-object identity, observation boundaries and non-resetting episode state."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from online_bridge_scene import OnlineBridgeTowerEnv,scene_catalog,recipe,COLORS,catalog
from build_online_bridge_geometry import online_library


class OnlineSceneTests(unittest.TestCase):
    def test_success_requires_an_uninterrupted_low_drift_interval(self):
        env=OnlineBridgeTowerEnv.__new__(OnlineBridgeTowerEnv)
        env.order=('object',)
        env.success_time=None;env.success_evidence=None
        env.last_success_check=-float('inf')
        env.stable_since=None;env.stable_positions=None
        env.episode_started=time.monotonic()
        position=np.zeros(3)
        env._object_state=lambda name: {'position':position}
        env.structure_status=lambda names: {'stable':True}
        env.data=SimpleNamespace(time=0.)
        env._monitor_success()
        env.data.time=4.9;env._monitor_success()
        self.assertIsNone(env.success_time)
        position[0]=.004
        env.data.time=5.;env._monitor_success()
        self.assertIsNone(env.success_time)
        self.assertIsNone(env.stable_since)
        env.data.time=5.1;env._monitor_success()
        env.data.time=10.2;env._monitor_success()
        self.assertIsNotNone(env.success_time)
        self.assertGreaterEqual(env.success_evidence['duration_sim_seconds'],5.)

    def test_five_object_goal_has_no_metric_truth_and_catalog_is_scoped(self):
        original=catalog.CATALOG_OBJECTS
        with scene_catalog(2028100101):
            task=recipe(2028100101)
            self.assertEqual(len(task.nodes),5)
            self.assertEqual({n.object_name.split('_')[0] for n in task.nodes},set(COLORS))
            self.assertEqual(task.nodes[-1].supports,('lower',))
            self.assertEqual(task.nodes[2].supports,('left','right'))
            self.assertTrue(all(set(n)=={'key','color','shape','grid','supports','orientation'} for n in task.public_goal()['nodes']))
        self.assertIs(catalog.CATALOG_OBJECTS,original)

    def test_added_color_mask_does_not_overlap_existing_colors(self):
        namespace={};exec(online_library(),namespace)
        pixels=np.array([[[220,20,220],[220,20,20],[20,220,20],[20,50,220],[220,180,10]]],dtype=np.uint8)
        expected=('magenta','red','green','blue','yellow')
        for index,color in enumerate(expected):
            mask=namespace['inventory_color_mask'](pixels,color)
            self.assertEqual(np.flatnonzero(mask).tolist(),[index])

    def test_reset_is_forbidden_and_physics_advances_during_wait(self):
        with TemporaryDirectory() as folder:
            env=OnlineBridgeTowerEnv(Path(folder),scene_seed=2028100101)
            try:
                env.reset(2028100101)
                with self.assertRaisesRegex(RuntimeError,'cannot reset'):
                    env.reset(2028100101)
                env.start_episode(1.,realtime=False)
                start=float(env.data.time)
                time.sleep(.03)
                env.advance_while_waiting()
                self.assertGreater(float(env.data.time),start+.02)
                self.assertEqual(env.reset_count,1)
                env.deadline=time.monotonic()-1
                with self.assertRaises(TimeoutError):env.advance_while_waiting()
            finally:env.close()


if __name__=='__main__':unittest.main()
