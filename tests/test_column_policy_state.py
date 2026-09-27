"""Controller state regressions with synthetic sensors and a robot API mock."""
import inspect
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('column_controller', Path(__file__).resolve().parents[1]/'policies/controllers/column_policy.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
run = controller.run
from aspire_so100.column_geometry import ColumnGeometryConflict, estimate_column_geometry
from aspire_so100.instance_geometry import inventory_points
from test_column_tracking import primitive_scene


class PolicyInitialized(Exception):
    """End setup when the controller reaches its first workspace IK preview."""


class SensorAPI:
    def __init__(self, objects):
        self.objects = objects
        self.frame, _ = primitive_scene(objects)
        self.robot = {'tool_position': np.array([0., -.12, .15]),
                      'tool_rotation': np.eye(3), 'time': 0.}
        self.helpers = None

    def task(self):
        return {'kind': 'single_column', 'levels': [
            {'color': 'blue', 'shape': 'prism'},
            {'color': 'blue', 'shape': 'cylinder'}]}

    def camera(self, view='side'):
        return self.frame

    def proprioception(self):
        return {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self.robot.items()}

    def move_tool(self, position, seconds=1., orientation=None):
        self.robot['tool_position'] = np.asarray(position).copy()
        self.robot['time'] += seconds

    def gripper(self, angle, seconds=.5):
        pass

    def note(self, message):
        pass

    def plan_tool(self, position, orientation=None):
        # Exercise the real setup and retain its live helper closures. No
        # controller source is copied, rewritten, or evaluated independently.
        self.helpers = dict(inspect.currentframe().f_back.f_locals)
        raise PolicyInitialized()

    def observe_lifted_cylinder(self, center, rotation):
        self.objects[0] = dict(self.objects[0], center=center, rotation=rotation)
        self.frame, labels = primitive_scene(self.objects)
        points, _ = inventory_points(self.frame)
        local = (points-center) @ rotation
        top = (labels == 0) & (abs(local[..., 2]-self.objects[0]['height']/2) < 1e-7)
        self.frame['rgb'][top] = 0


class ColumnPolicyStateTests(unittest.TestCase):
    def test_visual_width_refinement_persists_after_resolution_improves(self):
        objects = [
            {'shape': 'cylinder', 'center': np.array([-.055, -.02, .0275]),
             'dims': np.array([.032, .032]), 'height': .055},
            {'shape': 'thin_prism', 'center': np.array([.023, -.012, .027]),
             'dims': np.array([.017, .031]), 'height': .054},
        ]
        api = SensorAPI(objects)
        api.task = lambda: {'kind': 'single_column', 'levels': [
            {'color': 'blue', 'shape': 'cylinder'},
            {'color': 'blue', 'shape': 'thin_prism'}]}
        # The physical objects stay identical. Coarser initial rays miss more
        # of the silhouette; later full-resolution observations resolve it.
        for field in ('rgb', 'depth_m', 'valid_depth'):
            api.frame[field] = api.frame[field][::2, ::2]
        api.frame['intrinsics'] = api.frame['intrinsics'].copy()
        api.frame['intrinsics'][:2] /= 2
        with self.assertRaises(PolicyInitialized):
            run(api)
        state = api.helpers
        name = state['tracker'].select('blue', 'thin_prism')
        initial = state['blocks'][name]['model']
        initial_dims = initial['dims'].copy()
        initial_location = state['blocks'][name]['center'].copy()
        api.frame, _ = primitive_scene(objects)
        first = state['perceive'](name)
        self.assertGreater(first['dims'][1] - initial_dims[1], .0003)
        self.assertEqual(first['model']['dimension_refinement_count'], 1)
        np.testing.assert_array_equal(state['blocks'][name]['model']['dims'], first['dims'])
        np.testing.assert_array_equal(state['blocks'][name]['center'], initial_location)
        second = state['perceive'](name)
        np.testing.assert_array_equal(second['dims'], first['dims'])
        self.assertEqual(second['model']['dimension_refinement_count'], 1)
        np.testing.assert_array_equal(initial['dims'], initial_dims)

    def test_conflicting_finite_surfaces_cannot_fall_back_to_legacy_pose_or_stale_support(self):
        objects = [
            {'shape': 'cylinder', 'center': np.array([-.035, -.02, .0275]),
             'dims': np.array([.032, .032]), 'height': .055},
            {'shape': 'prism', 'center': np.array([.055, -.02, .032]),
             'dims': np.array([.032, .032]), 'height': .064},
        ]
        api = SensorAPI(objects)
        with self.assertRaises(PolicyInitialized):
            run(api)
        state = api.helpers
        name = state['tracker'].select('blue', 'prism')
        prior = state['visual_models'][name]
        wrong = dict(prior['model'], dims=prior['model']['dims'] - .005,
                     height=prior['model']['height'] - .005)
        state['capture']()
        frame = state['tracker'].isolated_frame(state['last_frame'][0], name)
        # These are actual finite depth surfaces, not a mocked fitter failure.
        # The legacy controller previously accepted this model with a 4.3 mm
        # center error despite the new fitter rejecting its visible extent.
        with self.assertRaisesRegex(ColumnGeometryConflict, 'contradicts learned'):
            estimate_column_geometry(frame, 'blue', 'prism', wrong, state['plane'])
        with self.assertRaisesRegex(ColumnGeometryConflict, 'contradicts learned'):
            state['locate'](name, wrong)
        state['blocks'][name]['model'] = wrong
        with self.assertRaisesRegex(ColumnGeometryConflict, 'contradicts learned'):
            state['perceive'](name)
        conflicting_prior = dict(prior, model=wrong, dims=wrong['dims'],
                                 height=wrong['height'],
                                 axis_dimensions=prior['axis_dimensions'] - .005)
        with self.assertRaisesRegex(ColumnGeometryConflict, 'contradicts learned'):
            state['refresh_support'](name, conflicting_prior)

    def test_occluded_first_grasp_keeps_original_pose_across_multiple_captures(self):
        objects = [
            {'shape': 'cylinder', 'center': np.array([-.035, -.02, .0275]),
             'dims': np.array([.032, .032]), 'height': .055},
            {'shape': 'prism', 'center': np.array([.055, -.02, .032]),
             'dims': np.array([.032, .032]), 'height': .064},
        ]
        api = SensorAPI(objects)
        with self.assertRaises(PolicyInitialized):
            run(api)
        state = api.helpers
        name = state['tracker'].select('blue', 'cylinder')
        initial_visual = state['visual_models'][name]
        original_center = initial_visual['center'].copy()
        actual_center = objects[0]['center'].copy()
        original_tool = original_center + [0., -.01, .03]
        api.robot = {'tool_position': original_tool.copy(), 'tool_rotation': np.eye(3), 'time': 10.}
        state['remember_grasp'](name)

        # The top is occluded after a 60 mm lift. Association anchors should
        # advance with each visible frame, but the warm geometry reference must
        # remain paired with the robot pose at the original grasp.
        api.robot = {'tool_position': original_tool+[0., 0., .06],
                     'tool_rotation': np.eye(3), 'time': 12.}
        api.observe_lifted_cylinder(actual_center+[0., 0., .06], np.eye(3))
        for _ in range(3):
            state['capture']()
        np.testing.assert_allclose(state['grasp_anchors'][name][1]['tool_position'], api.robot['tool_position'])
        np.testing.assert_allclose(state['warm_grasps'][name][1]['tool_position'], original_tool)
        fit = state['perceive'](name, api.robot['tool_position'])
        self.assertFalse(fit['observed_top'])
        np.testing.assert_allclose(fit['center'], original_center+[0., 0., .06], atol=.0005)

        # A later observation has already replaced visual_models. The original
        # warm reference still must not be paired with the newer association
        # pose, even when the robot rotates before the first held fit is stored.
        rotation = Rotation.from_euler('xyz', [.06, -.04, .08]).as_matrix()
        api.robot = {'tool_position': original_tool+[.002, -.001, .072],
                     'tool_rotation': rotation, 'time': 13.}
        expected = api.robot['tool_position'] + rotation @ (actual_center-original_tool)
        api.observe_lifted_cylinder(expected, rotation)
        for _ in range(2):
            state['capture']()
        fit = state['perceive'](name, api.robot['tool_position'])
        self.assertFalse(fit['observed_top'])
        np.testing.assert_allclose(fit['center'], expected, atol=.0005)
        np.testing.assert_allclose(fit['normal'], rotation[:, 2], atol=.0005)
        np.testing.assert_allclose(state['warm_grasps'][name][0]['center'], original_center)
        np.testing.assert_allclose(state['warm_grasps'][name][1]['tool_position'], original_tool)
        state['release_track'](name)
        self.assertNotIn(name, state['warm_grasps'])
        self.assertNotIn(name, state['grasp_anchors'])


if __name__ == '__main__':
    unittest.main()
