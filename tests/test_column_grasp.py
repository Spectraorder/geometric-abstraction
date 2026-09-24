"""Payload orientation constraints and bounded robot-only planning calls."""
import unittest
import numpy as np
from aspire_so100.column_grasp import (orientation_for_payload, grasp_orientation_for_target,
                                       _grasp_align, _grasp_yaw)


class ExactRobot:
    def __init__(self):
        self.calls = []

    def plan_tool(self, position, orientation):
        self.calls.append((position, orientation))
        return dict(tool_position=position, tool_rotation=orientation, position_error=0.)


class ColumnGraspTests(unittest.TestCase):
    def test_actual_grasp_normal_is_corrected_and_yaw_is_free(self):
        api = ExactRobot()
        local_up = np.array([.2, .3, .932737905])
        local_up /= np.linalg.norm(local_up)
        seed = _grasp_yaw(.7)
        result = orientation_for_payload(api, [0., -.25, .55], local_up, seed)
        np.testing.assert_allclose(result['tool_rotation'] @ local_up, [0., 0., 1.], atol=1e-8)
        self.assertTrue(result['reachable'])
        self.assertEqual(result['calls'], 1)
        self.assertEqual(len(api.calls), 1)
        for yaw in (-1., 0., 2.):
            np.testing.assert_allclose(_grasp_yaw(yaw) @ result['tool_rotation'] @ local_up,
                                       [0., 0., 1.], atol=1e-8)

    def test_free_yaw_search_avoids_an_unreachable_orientation(self):
        class RestrictedRobot(ExactRobot):
            def plan_tool(self, position, orientation):
                result = super().plan_tool(position, orientation)
                angle = np.arctan2(orientation[1][0], orientation[0][0])
                result['position_error'] = .01 if abs(angle-np.pi/4) > .01 else 0.
                return result
        api = RestrictedRobot()
        result = orientation_for_payload(api, [0., -.25, .55], [0., 0., 1.], np.eye(3))
        self.assertTrue(result['reachable'])
        self.assertAlmostEqual(result['yaw'], np.pi/4)
        self.assertEqual(result['calls'], 2)

    def test_actual_robot_rotation_defines_tilt_and_rpc_budget(self):
        class TiltedRobot(ExactRobot):
            def plan_tool(self, position, orientation):
                result = super().plan_tool(position, orientation)
                yaw = np.arctan2(orientation[1][0], orientation[0][0])
                result['tool_rotation'] = _grasp_align([0., 0., 1.], [.1, 0., 1.]) @ _grasp_yaw(yaw)
                return result
        api = TiltedRobot()
        result = orientation_for_payload(api, [0., -.25, .55], [0., 0., 1.], np.eye(3), max_calls=11)
        self.assertAlmostEqual(result['up_error'], np.arctan(.1))
        self.assertEqual(result['calls'], 11)
        self.assertEqual(len(api.calls), 11)
        np.testing.assert_allclose(result['orientation'] @ [0., 0., 1.], [0., 0., 1.], atol=1e-8)

    def test_soft_ik_orientation_bias_can_be_compensated_within_budget(self):
        class BiasedRobot(ExactRobot):
            def plan_tool(self, position, orientation):
                result = super().plan_tool(position, orientation)
                result['tool_rotation'] = _grasp_align([0., 0., 1.], [.1, 0., 1.]) @ np.asarray(orientation)
                return result
        result = orientation_for_payload(BiasedRobot(), [0., -.25, .55], [0., 0., 1.], np.eye(3))
        self.assertLess(result['up_error'], 1e-7)
        self.assertLessEqual(result['calls'], 12)

    def test_source_match_preserves_upright_payload_at_pitched_destination(self):
        api = ExactRobot()
        target = _grasp_align([0., 1., 0.], [.2, .1, 1.])
        result = grasp_orientation_for_target(api, [.1, -.3, .44], target)
        np.testing.assert_allclose(result['transfer_up'], [0., 0., 1.], atol=1e-8)
        self.assertLess(result['transfer_up_error'], 1e-7)

    def test_opposite_axis_and_unreachable_targets_remain_explicit(self):
        aligned = _grasp_align([0., 0., -1.], [0., 0., 1.])
        np.testing.assert_allclose(aligned @ [0., 0., -1.], [0., 0., 1.], atol=1e-8)
        self.assertAlmostEqual(np.linalg.det(aligned), 1.)
        class UnreachableRobot(ExactRobot):
            def plan_tool(self, position, orientation):
                result = super().plan_tool(position, orientation)
                result['position_error'] = .02
                return result
        result = orientation_for_payload(UnreachableRobot(), [0., -.25, .8], [0., 0., 1.], np.eye(3))
        self.assertFalse(result['reachable'])
        self.assertLessEqual(result['calls'], 12)
        self.assertGreaterEqual(result['calls'], 8)
        with self.assertRaises(ValueError):
            orientation_for_payload(ExactRobot(), [0., -.25, .5], [0., 0., 0.], np.eye(3))
        with self.assertRaises(ValueError):
            orientation_for_payload(ExactRobot(), [0., -.25, .5], [0., 0., 1.], np.eye(3), max_calls=13)


if __name__ == '__main__':
    unittest.main()
