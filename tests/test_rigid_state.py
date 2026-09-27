"""Rigid targets, cube symmetry, observability and the robot-only boundary."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aspire_so100.rigid_state import cube_vertices, stack_targets, state_error, support_relation


def observation(center, side=.03, yaw=0., reliable=True):
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return {'center': np.array(center), 'rotation': rotation, 'height': side,
            'vertices': cube_vertices(center, rotation, side), 'geometry_reliable': reliable,
            'position_fully_observed': reliable, 'inferred_orientation': not reliable}


class RigidStateTests(unittest.TestCase):
    def test_targets_preserve_rigidity_and_measured_sizes(self):
        objects = {'blue': observation([.12, -.2, .4], .029, .43),
                   'red': observation([0., 0., .4], .031),
                   'green': observation([.2, .1, .4], .032)}
        targets = stack_targets(objects, ['blue', 'red', 'green'], [0., 0., .3855])
        np.testing.assert_allclose(targets['red']['center'], [.12, -.2, .43])
        np.testing.assert_allclose(targets['green']['center'], [.12, -.2, .4615])
        for color, target in targets.items():
            distances = np.linalg.norm(target['vertices'][:, None] - target['vertices'][None, :], axis=2)
            nearest = np.sort(distances, axis=1)[:, 1:4]
            np.testing.assert_allclose(nearest, objects[color]['height'])
        self.assertEqual(targets['red']['support'], 'blue')

    def test_target_anchor_requires_an_explicit_observation_schema(self):
        objects = {'blue': observation([0., 0., .415]), 'red': observation([.1, 0., .415])}
        for invalid in (np.array([0., 0., .415]), {'center': [0., 0., .415]}):
            with self.assertRaisesRegex(ValueError, 'anchor must be an observation dict'):
                stack_targets(objects, ['blue', 'red'], [0., 0., .4], anchor=invalid)
        anchor = observation([.02, .03, .415])
        targets = stack_targets(objects, ['blue', 'red'], [0., 0., .4], anchor=anchor)
        np.testing.assert_allclose(targets['red']['center'], [.02, .03, .445])

    def test_cube_symmetries_are_not_false_rotation_errors(self):
        target = observation([0., 0., .4])
        rotated = observation([0., 0., .4], yaw=np.pi/2)
        error = state_error(rotated, target)
        self.assertLess(error['rotation_error_rad'], 1e-7)
        self.assertLess(error['vertex_rms_m'], 1e-8)
        turned = state_error(observation([0., 0., .4], yaw=.2), target)
        self.assertAlmostEqual(turned['rotation_error_rad'], .2)

    def test_partial_observation_does_not_certify_a_perfect_pose(self):
        current = observation([.01, 0., .4], reliable=False)
        error = state_error(current, observation([0., 0., .4]))
        np.testing.assert_allclose(error['translation_to_goal_m'], [-.01, 0., 0.])
        self.assertAlmostEqual(error['center_error_m'], .01)
        self.assertFalse(error['pose_fully_observed'])

    def test_gap_is_signed_geometric_distance_not_contact(self):
        lower = observation([0., 0., .4])
        upper = observation([.002, 0., .432])
        relation = support_relation(upper, lower)
        self.assertAlmostEqual(relation['lowest_vertex_gap_m'], .002)
        self.assertAlmostEqual(relation['projected_center_margin_m'], .013)
        self.assertFalse(relation['contact_or_stability_certified'])

    def test_base_relocation_reanchors_instance_ids_without_shape_suffix(self):
        from aspire_so100.rigid_state_demo import RigidStateObserver
        class API:
            def task(self):
                return {'levels': [{'color': c} for c in ('blue', 'red', 'green')]}
            def note(self, text):
                pass
        class Observer(RigidStateObserver):
            def snapshot(self, label, reanchor=False):
                self.last_reanchor = reanchor
        observer = Observer(API())
        observer.note('blue_1 after withdrawal center=[0, 0, .4]')
        self.assertTrue(observer.last_reanchor)
        observer.note('red_0 after withdrawal center=[0, 0, .43]')
        self.assertFalse(observer.last_reanchor)

    def test_invalid_rigid_transform_is_rejected(self):
        with self.assertRaises(ValueError):
            cube_vertices([0., 0., 0.], np.diag([-1., 1., 1.]), .03)

    def test_gripper_fk_does_not_depend_on_object_state(self):
        import mujoco
        from aspire_so100.playback import RecordedInventoryTowerEnv
        from aspire_so100.robot_geometry import robot_gripper_geometry
        with tempfile.TemporaryDirectory() as tmp:
            env = RecordedInventoryTowerEnv(Path(tmp), scene_seed=3501, object_count=3, recipe='cube_triple')
            try:
                env.reset(3501, layout='sampled')
                before = robot_gripper_geometry(env.model, env.data)
                for joint in range(env.model.njnt):
                    if env.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                        address = env.model.jnt_qposadr[joint]
                        env.data.qpos[address:address+3] += [10., -7., 3.]
                mujoco.mj_forward(env.model, env.data)
                after = robot_gripper_geometry(env.model, env.data)
                for a, b in zip(before['pads'], after['pads']):
                    np.testing.assert_array_equal(a['vertices'], b['vertices'])
                self.assertEqual(before['visual_geometry_coverage'], 'ee_site_marker_only')
                self.assertAlmostEqual(before['visual_spheres'][0]['radius_m'], .004)
                np.testing.assert_array_equal(before['visual_spheres'][0]['center_m'], after['visual_spheres'][0]['center_m'])
                jaw = int(env.model.joint('robot/Jaw').qposadr[0])
                env.data.qpos[jaw] += .1
                moved = robot_gripper_geometry(env.model, env.data)
                self.assertGreater(np.linalg.norm(moved['pads'][4]['vertices']-after['pads'][4]['vertices']), .001)
                np.testing.assert_array_equal(moved['pads'][0]['vertices'], after['pads'][0]['vertices'])
            finally:
                env.close()


if __name__ == '__main__':
    unittest.main()
