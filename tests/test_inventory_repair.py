"""Curriculum checks must distinguish a correction from collapse or no-op."""
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from aspire_so100.inventory_repair import InventoryRepairEnv
from aspire_so100.playback import recorded_env_for_result

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from learn_inventory_repair import apply_edits


class InventoryRepairTests(unittest.TestCase):
    def test_milestones_require_private_geometry_and_matching_ik(self):
        with TemporaryDirectory() as tmp:
            env=InventoryRepairEnv(Path(tmp),scene_seed=1501)
            try:
                env.reset(1501)
                bad={'stage':'identify','values':{'base_center':[0,0,0],
                    'target_center':[0,0,0],'target_dimensions':[0,0,0]}}
                self.assertIsNone(env.record_report(bad))
                self.assertFalse(env.verified['identify'])
                target=env._object_state(env.order[1])['position'].copy()
                good={'stage':'identify','values':{
                    'base_center':env.base_anchor.tolist(),'target_center':target.tolist(),
                    'target_dimensions':(2*env.extent(env.order[1])).tolist()}}
                self.assertIsNone(env.record_report(good))
                self.assertTrue(env.verified['identify'])
                position=(target+[0,0,.02]).tolist()
                report={'stage':'plan','values':{'role':'grasp','position':position}}
                self.assertIsNone(env.record_report(report))
                self.assertFalse(env.verified['plan'])
                preview={'params':{'position':position},'observation':{
                    'position_error':.01,'tool_rotation':[[0,0,1],[1,0,0],[0,1,0]]}}
                env.record_report(report,preview)
                self.assertFalse(env.progress_reports[-1]['accepted'])
                preview['observation']['position_error']=.001
                for role,dz in (('grasp',.02),('approach',.04),('lift',.035)):
                    position=(target+[0,0,dz]).tolist()
                    preview['params']['position']=position
                    self.assertIsNone(env.record_report({'stage':'plan','values':{
                        'role':role,'position':position}},preview))
                self.assertTrue(env.verified['plan'])
                # Closing away from the object is not a grasp milestone.
                self.assertEqual(env.contact_fingers(),set())
                env.set_gripper(-.174,.2)
                env.wait(.2)
                self.assertFalse(env.verified['grasp'])
                self.assertFalse(env.evaluate().success)
                env.reset(1501)
                self.assertFalse(any(env.verified.values()))
            finally:
                env.close()

    def test_grasp_monitor_requires_two_jaws_lift_and_hold(self):
        with TemporaryDirectory() as tmp:
            env=InventoryRepairEnv(Path(tmp),scene_seed=1501)
            try:
                env.reset(1501)
                env.verified.update(identify=True,plan=True)
                # Synthetic contact/state fixtures test the private monitor;
                # these are not a policy rollout or evidence of learned grasping.
                target=env.target_anchor.copy()
                def state(name):
                    return {'position':env.base_anchor if name==env.order[0] else target}
                with patch('aspire_so100.prism.PrismEnv._step'), \
                     patch.object(env,'_object_state',side_effect=state), \
                     patch.object(env,'contact_fingers',return_value={1,2}) as contacts:
                    for _ in range(100): env._step()
                    self.assertFalse(env.verified['grasp'])  # not lifted
                    target[2]+=.006
                    contacts.return_value={1}
                    for _ in range(100): env._step()
                    self.assertFalse(env.verified['grasp'])  # one jaw
                    contacts.return_value={1,2}
                    env._step()
                    self.assertFalse(env.verified['grasp'])  # too brief
                    for _ in range(int(.11/env.model.opt.timestep)): env._step()
                    self.assertTrue(env.verified['grasp'])
            finally:
                env.close()

    def test_noop_passes_centered_but_fails_every_offset_direction(self):
        with TemporaryDirectory() as tmp:
            for seed in (1501,1502,1503,1504):
                for case in ('centered','offset'):
                    with self.subTest(seed=seed,case=case):
                        env=InventoryRepairEnv(Path(tmp),scene_seed=seed,repair_case=case)
                        try:
                            env.reset(seed)
                            self.assertEqual(len(env.objects),6)
                            if case == 'offset':
                                self.assertGreater(np.linalg.norm(env.initial_offset),.0055)
                                self.assertLess(np.linalg.norm(env.initial_offset),.009)
                            env.wait(20.)
                            env.check_structure(list(env.order),5.)
                            self.assertEqual(env.evaluate().success,case=='centered')
                            self.assertFalse(env.collapse_seen)
                        finally:
                            env.close()

    def test_centered_repair_passes_but_moving_foundation_is_permanent_failure(self):
        with TemporaryDirectory() as tmp:
            env=InventoryRepairEnv(Path(tmp),scene_seed=1501)
            try:
                env.reset(1501)
                base=env.base_anchor.copy()
                # Test fixture only: the online worker has no pose-setting API.
                env._set_freejoint_pose(env.order[1],(*base[:2],.46),0.)
                mujoco.mj_forward(env.model,env.data)
                env.check_structure(list(env.order),5.)
                self.assertTrue(env.evaluate().success)
                env._set_freejoint_pose(env.order[0],(*(base[:2]+[.01,0.]),base[2]),0.)
                mujoco.mj_forward(env.model,env.data)
                env._step()
                env._set_freejoint_pose(env.order[0],base,0.)
                mujoco.mj_forward(env.model,env.data)
                env.check_structure(list(env.order),5.)
                self.assertFalse(env.evaluate().success)
                self.assertGreater(env.max_base_drift,.003)
            finally:
                env.close()

    def test_drop_is_failure_and_replay_preserves_case(self):
        with TemporaryDirectory() as tmp:
            env=InventoryRepairEnv(Path(tmp),scene_seed=1502,repair_case='centered')
            try:
                env.reset(1502)
                result={'task':env.task_id,'scene':env.scene_metadata,'seed':1502}
                replay=recorded_env_for_result(result,Path(tmp))
                try:
                    self.assertEqual(replay.scene_metadata,env.scene_metadata)
                finally:
                    replay.close()
                env._set_freejoint_pose(env.order[1],(.15,-.24,.43),0.)
                mujoco.mj_forward(env.model,env.data)
                env._step()
                self.assertTrue(env.collapse_seen)
                self.assertFalse(env.evaluate().success)
            finally:
                env.close()

    def test_learner_edits_are_exact_and_validated(self):
        source='def run(api):\n    api.wait(1)\n'
        self.assertIn('wait(6)',apply_edits(source,[{'old':'wait(1)','new':'wait(6)'}]))
        with self.assertRaises(ValueError):
            apply_edits(source,[{'old':'missing','new':''}])
        with self.assertRaises(ValueError):
            apply_edits(source,[{'old':'api.wait(1)','new':'open("secret")'}])

    def test_duplicate_edit_needs_unique_neighboring_anchors(self):
        source='def run(api):\n    api.note("before")\n    api.wait(1)\n    api.note("after")\n    api.wait(1)\n'
        with self.assertRaises(ValueError):
            apply_edits(source,[{'old':'api.wait(1)','new':'api.wait(6)'}])
        edits=[{'old':'api.note("before")','new':'api.note("start")'},
               {'old':'api.wait(1)','new':'api.wait(6)'},
               {'old':'api.note("after")','new':'api.note("end")'}]
        result=apply_edits(source,edits)
        self.assertEqual(result.count('api.wait(6)'),1)
        self.assertTrue(result.endswith('    api.wait(1)\n'))


if __name__=='__main__':
    unittest.main()
