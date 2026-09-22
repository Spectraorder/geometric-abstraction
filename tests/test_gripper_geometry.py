import unittest
import numpy as np

from aspire_so100.gripper_geometry import estimate_gripper_geometry


class GripperGeometryTests(unittest.TestCase):
    def frame(self):
        rgb = np.full((64, 64, 3), [20, 170, 40], dtype=np.uint8)
        rgb[20:44, 6:16] = 180
        rgb[20:44, 48:58] = 180
        return {'rgb': rgb, 'depth_m': np.full((64, 64), .48),
                'valid_depth': np.ones((64, 64), dtype=bool),
                'intrinsics': np.array([[1000., 0, 31.5], [0, 1000., 31.5], [0, 0, 1]]),
                'camera_to_world': np.eye(4)}

    def test_observed_jaws_and_clearance_exclude_colored_payload(self):
        result = estimate_gripper_geometry(self.frame(), np.array([0., 0., .5]), .4)
        self.assertEqual(result['pixel_count'], 480)
        self.assertAlmostEqual(result['fingertip_drop_m'], .02)
        self.assertAlmostEqual(result['visible_clearance_m'], .08)
        self.assertAlmostEqual(result['observed_opening_m'], 33*.00048)
        self.assertEqual(result['jaw_centers'].shape, (2, 3))
        self.assertTrue(result['visible_only'])

    def test_hidden_jaw_does_not_invent_an_opening(self):
        frame = self.frame()
        frame['valid_depth'][:, 32:] = False
        result = estimate_gripper_geometry(frame, np.array([0., 0., .5]))
        self.assertIsNone(result['observed_opening_m'])
        self.assertIsNone(result['jaw_centers'])
        self.assertTrue(result['visible_only'])

    def test_blackout_rejected(self):
        frame = self.frame()
        frame['valid_depth'][:] = False
        with self.assertRaises(ValueError):
            estimate_gripper_geometry(frame, np.array([0., 0., .5]))


if __name__ == '__main__':
    unittest.main()
