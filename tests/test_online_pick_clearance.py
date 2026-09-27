"""Conditional planning bounds and jaw-envelope collision screening."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from online_pick_clearance import overlaps,conditional_center_box

class ClearanceTests(unittest.TestCase):
    def test_upper_structure_can_intersect_jaw_without_source_object_overlap(self):
        housing=np.array([[[-.02,-.07,.42],[.02,-.01,.49]]])
        self.assertTrue(overlaps([[-.015,-.045,.45],[.015,-.015,.48]],housing))
        self.assertFalse(overlaps([[.10,-.045,.45],[.13,-.015,.48]],housing))

    def test_cap_samples_bound_horizontal_center_without_promoting_pose(self):
        points=np.array([[x,y,.02] for x in np.linspace(-.019,.019,4) for y in np.linspace(-.009,.009,4)])
        center,radius,evidence=conditional_center_box(points,np.eye(3),[.04,.02,.04],[0,0,0],[False,False,True])
        np.testing.assert_allclose(center,[0,0,0],atol=1e-12)
        self.assertTrue(np.all(radius<.004))
        self.assertFalse(evidence['pose_available']);self.assertTrue(evidence['planning_only'])

    def test_contradictory_shape_samples_are_rejected(self):
        points=np.array([[x,y,.02] for x in (-.04,.04) for y in (-.04,.04) for _ in range(2)])
        with self.assertRaises(ValueError):conditional_center_box(points,np.eye(3),[.04,.02,.04],[0,0,0],[False,False,True])

if __name__=='__main__':unittest.main()
