"""Private physics/evaluator checks before spending model tokens."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from aspire_so100.fasten_screws import (FastenScrewsEnv, benchmark_score, ENTRY_Z,
                                       REQUIRED_DEPTH, THREAD_HOLD_TORQUE)
from aspire_so100.fasteners import SCENE_VERSION


class FastenLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.env=FastenScrewsEnv(Path(self.tmp.name))
        self.env.reset(5101)

    def tearDown(self):
        self.env.close();self.tmp.cleanup()

    def place_at_entry(self,bolt='red'):
        e=self.env
        point=e.data.xpos[e.bolts[bolt]].copy()+[0,0,ENTRY_Z]
        e._set_freejoint_pose('red_nut',point,0.)
        mujoco.mj_forward(e.model,e.data)

    def engage_fixture(self,bolt='red'):
        # Test fixture represents the nut already held by both jaws at entry.
        # All subsequent rotation/depth evolution is integrated physics.
        self.place_at_entry(bolt)
        with patch.object(self.env,'_contacts',return_value=self.env.jaws):
            self.env._thread_forces()
        self.assertIn('red',self.env.threads)

    def test_scores_and_unmanipulated_scene(self):
        self.assertEqual(self.env.scene_metadata['generator'],SCENE_VERSION)
        self.assertEqual([benchmark_score(n,True,True) for n in range(4)],[0,20,50,100])
        self.assertEqual(benchmark_score(3,True,False),50)
        self.assertEqual(benchmark_score(3,False,True),0)
        self.assertEqual(self.env.evaluate().score,0)
        for color in self.env.bolts:
            self.assertEqual(self.env.model.body(self.env.bolts[color]).jntnum[0],0)

    def test_engagement_targets_matching_mesh_phase_at_nonzero_yaw(self):
        e=self.env
        point=e.data.xpos[e.bolts['red']].copy()+[0,0,ENTRY_Z]
        for yaw in (.7,-1.2,2.8):
            e.threads.clear()
            e._set_freejoint_pose('red_nut',point,yaw)
            mujoco.mj_forward(e.model,e.data)
            with patch.object(e,'_contacts',return_value=e.jaws):
                e._thread_forces()
            entry=e.threads['red']['entry_z']
            phase=(entry-e.thread_phase)/e.thread_pitch-yaw/(2*np.pi)
            self.assertAlmostEqual(phase,round(phase))
            self.assertLessEqual(abs(entry-ENTRY_Z),e.thread_pitch/2)

    def test_alignment_without_grasp_does_not_engage_and_drop_does_not_score(self):
        e=self.env;self.place_at_entry()
        e._thread_forces()
        self.assertNotIn('red',e.threads)
        point=e.data.xpos[e.bolts['red']].copy()+[0,0,ENTRY_Z-REQUIRED_DEPTH]
        e._set_freejoint_pose('red_nut',point,0.)
        mujoco.mj_forward(e.model,e.data)
        self.assertFalse(e.pair_status('red')['valid'])
        self.assertEqual(e.evaluate().score,0)

    def test_axial_push_is_resisted_and_wrong_color_cannot_score(self):
        e=self.env;self.engage_fixture('blue')
        nut=e.nuts['red']
        for _ in range(500):
            e._thread_forces()
            e.data.xfrc_applied[nut,2]-=1.
            mujoco.mj_step(e.model,e.data)
        delta,_,_=e._pose('red','blue')
        self.assertLess(ENTRY_Z-delta[2],.004)
        self.assertFalse(e.pair_status('red')['matching'])
        self.assertEqual(e.evaluate().score,0)

    def test_clockwise_torque_advances_thread_and_release_can_score(self):
        e=self.env;self.engage_fixture()
        nut=e.nuts['red']
        for _ in range(7000):
            e._thread_forces()
            e.data.xfrc_applied[nut,5]-=.02
            mujoco.mj_step(e.model,e.data)
            if e.pair_status('red')['turns']>=3.15: break
        self.assertGreaterEqual(e.pair_status('red')['turns'],3.)
        e.set_gripper(.8,.5)
        e.wait(3.)
        self.assertTrue(e.pair_status('red')['valid'])
        self.assertEqual(e.evaluate().score,20)
        before=e.pair_status('red')
        e._step_for(30.)
        after=e.pair_status('red')
        self.assertLess(abs(after['turns']-before['turns']),.001)
        self.assertLess(abs(after['depth_m']-before['depth_m']),.00001)
        self.assertTrue(after['valid'])

    def test_turn_reversal_cancels_instead_of_accumulating(self):
        e=self.env;self.engage_fixture()
        point=e.data.xpos[e.nuts['red']].copy()
        for yaw in (-1.,0.,-1.,0.):
            e._set_freejoint_pose('red_nut',point,yaw)
            mujoco.mj_forward(e.model,e.data)
            e._thread_forces()
        self.assertAlmostEqual(e.pair_status('red')['turns'],0.)
        self.assertFalse(e.pair_status('red')['valid'])

    def test_released_nut_does_not_creep_under_gravity(self):
        e=self.env;self.engage_fixture()
        e.wait(1.)  # Allow the compliant axial spring to settle under weight.
        before=e.pair_status('red')
        e._step_for(30.)
        after=e.pair_status('red')
        self.assertTrue(after['engaged'])
        self.assertFalse(after['touching_gripper'])
        self.assertLess(abs(after['turns']-before['turns']),.001)
        self.assertLess(abs(after['depth_m']-before['depth_m']),.00001)
        self.assertEqual(e.evaluate().score,0)

    def test_waiting_near_completion_cannot_earn_fastening_credit(self):
        e=self.env;self.engage_fixture()
        # Private fixture: put a partially threaded nut just below the turn
        # requirement. No actor receives these simulator values.
        twist=-2*np.pi*2.9
        entry=e.threads['red']['entry_z']
        point=e.data.xpos[e.bolts['red']].copy()+[0,0,entry-2.9*e.thread_pitch]
        e._set_freejoint_pose('red_nut',point,twist)
        mujoco.mj_forward(e.model,e.data)
        e.threads['red'].update(twist=twist,yaw=e._pose('red','red')[2])
        if e.thread_constraints is not None:
            # This private fixture relocates an already-threaded nut; initialize
            # its auxiliary guide consistently, rather than storing spring energy.
            jz,jyaw=e.thread_constraints.joints['red']
            e.data.qpos[e.model.jnt_qposadr[jz]]=entry+e.thread_lead*twist
            e.data.qpos[e.model.jnt_qposadr[jyaw]]=twist
            mujoco.mj_forward(e.model,e.data)
        e._step_for(30.)
        self.assertLess(e.pair_status('red')['turns'],2.901)
        self.assertFalse(e.pair_status('red')['valid'])
        self.assertEqual(e.evaluate().score,0)

    def test_friction_is_only_active_while_engaged_and_clears_on_reset(self):
        e=self.env
        dof=e.spin_dofs['red']
        self.assertEqual(e.model.dof_frictionloss[dof],0.)
        self.engage_fixture()
        self.assertEqual(e.model.dof_frictionloss[dof],THREAD_HOLD_TORQUE)
        # Other nuts and translation are not artificially locked.
        self.assertEqual(e.model.dof_frictionloss[e.spin_dofs['blue']],0.)
        np.testing.assert_array_equal(e.model.dof_frictionloss[dof-5:dof],0.)
        point=e.data.xpos[e.nuts['red']].copy()+[.03,0,0]
        e._set_freejoint_pose('red_nut',point,0.)
        mujoco.mj_forward(e.model,e.data)
        e._thread_forces()
        self.assertNotIn('red',e.threads)
        self.assertEqual(e.model.dof_frictionloss[dof],0.)
        self.assertEqual(e.model.dof_damping[dof],0.)
        self.engage_fixture()
        e.reset(5101)
        self.assertEqual(e.model.dof_frictionloss[dof],0.)
        self.assertEqual(e.model.dof_damping[dof],0.)

    def test_applied_torque_can_reverse_and_unthread(self):
        e=self.env;self.engage_fixture()
        nut=e.nuts['red']
        def drive(torque,seconds):
            for _ in range(round(seconds/e.model.opt.timestep)):
                e._thread_forces()
                e.data.xfrc_applied[nut,5]+=torque
                mujoco.mj_step(e.model,e.data)
                if 'red' not in e.threads: break
        drive(-.02,1.)
        before=e.pair_status('red')
        self.assertGreater(before['turns'],.1)
        drive(.02,.3)
        after=e.pair_status('red')
        self.assertLess(after['turns'],before['turns']-.05)
        self.assertLess(after['depth_m'],before['depth_m'])
        drive(.02,2.)
        self.assertNotIn('red',e.threads)
        self.assertEqual(e.model.dof_frictionloss[e.spin_dofs['red']],0.)

    def test_subthreshold_torque_does_not_start_turning(self):
        e=self.env;self.engage_fixture()
        nut=e.nuts['red']
        for sign in (-1,1):
            for _ in range(round(2/e.model.opt.timestep)):
                e._thread_forces()
                e.data.xfrc_applied[nut,5]+=sign*THREAD_HOLD_TORQUE*.4
                mujoco.mj_step(e.model,e.data)
        self.assertLess(abs(e.pair_status('red')['turns']),.001)
