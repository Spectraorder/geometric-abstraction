"""Known robot geometry must not erase foreground samples or mutate sensor input."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robot_visual_mask_prototype import exclude_robot_spheres


class RobotMaskTests(unittest.TestCase):
    def frame(self):
        return {'time':1.,'depth_m':np.ones((1,3)),'valid_depth':np.ones((1,3),bool),
                'intrinsics':np.eye(3),'camera_to_world':np.eye(4)}

    def test_same_time_3d_exclusion_preserves_input_and_other_points(self):
        frame=self.frame();result,event=exclude_robot_spheres(frame,[{'center_m':[1,0,1],'radius_m':.1}],1.)
        np.testing.assert_array_equal(result['valid_depth'],[[True,False,True]])
        self.assertTrue(frame['valid_depth'].all());self.assertEqual(event['removed_depth_samples'],1)

    def test_projected_overlap_at_different_depth_is_not_excluded(self):
        result,_=exclude_robot_spheres(self.frame(),[{'center_m':[2,0,2],'radius_m':.1}],1.)
        self.assertTrue(result['valid_depth'].all())

    def test_stale_robot_geometry_is_rejected(self):
        with self.assertRaises(ValueError):exclude_robot_spheres(self.frame(),[],.9)


if __name__=='__main__':unittest.main()
