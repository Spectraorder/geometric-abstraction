"""Opening is a shared action, not a geometric or success certificate."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from prototype_open_gripper_session import library
from build_geometry_session import library as base_library


class OpenGripperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns={};exec(library(base_library('symbolic')),cls.ns)

    def session(self, fail=False):
        class API:
            def __init__(self):self.commands=[]
            def gripper(self, angle, seconds):
                self.commands.append((angle,seconds))
                if fail:raise RuntimeError('command not acknowledged')
        pnp=self.ns['ObservedRGBDPickPlace'].__new__(self.ns['ObservedRGBDPickPlace'])
        pnp.api=API();pnp.held={'color':'red','confirmed':True}
        session=self.ns['GeometryStepSession'].__new__(self.ns['GeometryStepSession']);session.pnp=pnp
        return session

    def test_both_interfaces_issue_identical_explicit_command_without_sensing(self):
        full=self.session();other=self.session();base=self.ns['VisualInputSession'](other)
        a,b=full.open_gripper(),base.open_gripper()
        self.assertEqual(a,b);self.assertEqual(a,{'status':'opened_unverified','held':False})
        self.assertEqual(full.pnp.api.commands,[(.8,.6)])
        self.assertEqual(full.pnp.api.commands,other.pnp.api.commands)
        self.assertIsNone(full.pnp.held);self.assertIsNone(other.pnp.held)

    def test_failed_open_does_not_fabricate_a_cleared_state(self):
        session=self.session(fail=True);before=session.pnp.held.copy()
        with self.assertRaisesRegex(RuntimeError,'not acknowledged'):session.open_gripper()
        self.assertEqual(session.pnp.held,before)

    def test_explicit_open_allows_pick_to_reach_fresh_perception_again(self):
        session=self.session()
        def unavailable(color):raise ValueError('fresh perception required')
        session.pnp.observe=unavailable
        with self.assertRaisesRegex(RuntimeError,'Finish or recover'):session.pick('blue')
        session.open_gripper()
        with self.assertRaisesRegex(ValueError,'fresh perception required'):session.pick('blue')
        self.assertIsNone(session.pnp.held)


if __name__=='__main__':unittest.main()
