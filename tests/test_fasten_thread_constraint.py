"""Load/release regressions for the solver-resolved screw surrogate."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import math
import unittest
import mujoco
import numpy as np
from aspire_so100.fasten_screws import FastenScrewsEnv


class ThreadConstraintTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.e=FastenScrewsEnv(Path(self.tmp.name))
        self.e.reset(5101)

    def tearDown(self):
        self.e.close()
        self.tmp.cleanup()

    def place(self, z):
        e=self.e
        e._set_freejoint_pose('red_nut',e.data.xpos[e.bolts['red']]+[0,0,z],0.)
        mujoco.mj_forward(e.model,e.data)

    def engage(self):
        e=self.e;self.place(e.entry_z)
        q=e.data.qpos.copy();v=e.data.qvel.copy()
        with patch.object(e,'_contacts',return_value=e.jaws):
            e._thread_forces()
        # Only auxiliary guide states may change at capture, never any object
        # or robot pose/velocity. The final six coordinates are the 3 guides.
        np.testing.assert_array_equal(e.data.qpos[:-6],q[:-6])
        np.testing.assert_array_equal(e.data.qvel[:-6],v[:-6])
        self.assertIn('red',e.threads)
        e._step_for(1.)

    def test_load_does_not_compress_tilt_or_rebound_by_millimetres(self):
        e=self.e;self.engage();errors=[]
        for _ in range(500):
            e._thread_forces()
            e.data.xfrc_applied[e.nuts['red'],:3]+=[3,0,-6]
            e.data.xfrc_applied[e.nuts['red'],4]+=.025
            mujoco.mj_step(e.model,e.data)
            delta,rotation,_=e._pose('red','red');t=e.threads['red']
            errors.append([abs(delta[2]-t['entry_z']-e.thread_lead*t['twist']),
                           np.linalg.norm(delta[:2]),math.acos(np.clip(rotation[2,2],-1,1))])
        worst=np.max(errors,axis=0)
        self.assertLess(worst[0],.0003)
        self.assertLess(worst[1],.0001)
        self.assertLess(worst[2],math.radians(1))
        loaded_z=e.data.xpos[e.nuts['red'],2]
        e._step_for(2.)
        self.assertLess(abs(e.data.xpos[e.nuts['red'],2]-loaded_z),.0003)
        self.assertFalse(np.any(e.data.warning.number))

    def test_unengaged_nut_stops_at_entry_instead_of_falling_down_shaft(self):
        e=self.e;self.place(e.entry_z+.01)
        e._step_for(3.)
        delta,_,_=e._pose('red','red')
        self.assertNotIn('red',e.threads)
        self.assertGreater(delta[2],e.entry_z-.0003)
        self.assertLess(delta[2],e.entry_z+.001)
        self.assertEqual(e.evaluate().score,0)

    def test_reset_disables_constraints_and_restores_entry_contact(self):
        e=self.e;self.engage();c=e.thread_constraints
        self.assertTrue(e.data.eq_active[c.weld['red']])
        self.assertTrue(e.data.eq_active[c.helix['red']])
        self.assertEqual(e.model.geom_contype[c.entry['red']],0)
        e.reset(5101)
        self.assertFalse(np.any(e.data.eq_active))
        self.assertEqual(e.model.geom_contype[c.entry['red']],2)
        self.assertEqual(e.threads,{})

    def test_unthreading_releases_constraint_and_restores_entry_contact(self):
        e=self.e;self.engage();c=e.thread_constraints
        for _ in range(1000):
            e._thread_forces()
            e.data.xfrc_applied[e.nuts['red'],5]+=.02
            mujoco.mj_step(e.model,e.data)
            if 'red' not in e.threads:
                break
        self.assertNotIn('red',e.threads)
        self.assertFalse(e.data.eq_active[c.weld['red']])
        self.assertFalse(e.data.eq_active[c.helix['red']])
        self.assertEqual(e.model.geom_contype[c.entry['red']],2)


if __name__=='__main__':
    unittest.main()
