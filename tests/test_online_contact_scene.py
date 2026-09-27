"""Unnamed collision geoms must participate in private robot-contact checks."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from online_contact_scene import ContactCompleteEnv


class ContactTests(unittest.TestCase):
    def environment(self, pair):
        env=object.__new__(ContactCompleteEnv)
        bodies=['world','robot','robot/Fixed_Jaw',None,'cube/cube','robotic/part']
        geoms=[None,None,'cube/geom',None]
        env.model=SimpleNamespace(ngeom=4,geom_bodyid=[2,3,4,5],body_parentid=[0,0,1,2,0,0],
                                  geom=lambda i:SimpleNamespace(name=geoms[i]),
                                  body=lambda i:SimpleNamespace(name=bodies[i]))
        env.data=SimpleNamespace(contact=[SimpleNamespace(geom1=pair[0],geom2=pair[1])])
        return env

    def test_unnamed_robot_geometry_is_detected_in_both_contact_orders(self):
        for pair in ((0,2),(2,0)):
            self.assertTrue(self.environment(pair)._objects_in_contact('cube','robot'))

    def test_unnamed_child_body_inherits_its_robot_namespace(self):
        self.assertTrue(self.environment((1,2))._objects_in_contact('cube','robot'))

    def test_prefix_does_not_match_a_different_namespace(self):
        self.assertFalse(self.environment((3,2))._objects_in_contact('cube','robot'))

if __name__=='__main__':unittest.main()
