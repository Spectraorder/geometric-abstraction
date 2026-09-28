"""Regression checks for color coverage, projection, and final pad orientation."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import ring_peg_development as repair


class RepairTests(unittest.TestCase):
    def test_magenta_is_distinct_from_red_blue_and_background(self):
        rgb = np.array([[[180, 30, 160], [180, 20, 20], [20, 30, 180], [150, 100, 60], [200, 200, 200]]])
        np.testing.assert_array_equal(repair.color_mask(rgb, 'magenta'), [[True, False, False, False, False]])

    def test_projection_uses_camera_rays_not_background_depth(self):
        frame = {'depth_m': np.full((2, 2), 5.), 'intrinsics': np.eye(3), 'camera_to_world': np.eye(4)}
        projected = repair.project_pixels_to_plane(frame, np.array([0, 0, 1.]), 2.)
        np.testing.assert_allclose(projected[1, 1], [2, 2, 2])
        frame['depth_m'][:] = 10.
        np.testing.assert_allclose(repair.project_pixels_to_plane(frame, np.array([0, 0, 1.]), 2.), projected)

    def test_corrected_target_must_still_align(self):
        env = SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(1)), arm_qpos_addresses=[0])
        robot = repair.CorrectedPadActuator(env)
        aligned = {'closing_axis': np.array([1., 0, 0])}
        misaligned = {'closing_axis': np.array([0., 1, 0]), 'roll': 0., 'fixed_pad_m': np.zeros(3)}
        with patch.object(robot, 'preview', return_value=aligned), patch.object(robot, 'plan_pad', return_value=misaligned):
            with self.assertRaisesRegex(ValueError, 'No reachable roll'):
                robot.plan_grasp([0, 0, 0], [1, 0, 0], .15)

    def test_zero_direction_is_rejected(self):
        robot = repair.CorrectedPadActuator(None)
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            robot.plan_grasp([0, 0, 0], [0, 0, 0], .15)

    def test_missing_current_pixels_cannot_be_replaced_by_shape(self):
        frame = {'rgb': np.zeros((32, 32, 3), np.uint8), 'depth_m': np.ones((32, 32)),
                 'valid_depth': np.ones((32, 32), bool), 'intrinsics': np.eye(3),
                 'camera_to_world': np.eye(4), 'time': 1.}
        with repair.installed(), self.assertRaises(ValueError):
            repair.register_ring(frame, 'magenta', np.zeros(3), {})


if __name__ == '__main__':
    unittest.main()
