"""Evidence, identity and action prediction boundaries for geometric text state."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from aspire_so100.geometric_state import GeometricState
from aspire_so100.ablation_geometry import observe_cube
from aspire_so100.instance_geometry import inventory_table


def fit(center=(0., 0., .415), rotation=None, axes=(True, True, True)):
    return {'shape': 'cube', 'center': np.array(center), 'rotation': np.eye(3) if rotation is None else rotation,
            'height': .03, 'constrained_axes': np.array(axes), 'inferred_orientation': sum(axes)<2,
            'surface_rms_m': .0005, 'position_fully_observed': all(axes)}


class GeometricStateTests(unittest.TestCase):
    def test_partial_update_keeps_hidden_coordinates_and_uncertainty(self):
        state = GeometricState()
        state.observe('red', fit(), 0.)
        # The top is visible, but a clipped color patch biases its XY centroid.
        updated = state.observe('red', fit((.02, -.04, .425), axes=(False, False, True)), 1.)
        np.testing.assert_allclose(updated['center'], [0., 0., .425])
        self.assertFalse(updated['position_fully_observed'])
        self.assertEqual(updated['observed_translation_rank'], 1)
        self.assertGreater(updated['covariance_m2'][0, 0], updated['covariance_m2'][2, 2])

    def test_constraint_axes_are_in_fit_frame_not_world(self):
        state = GeometricState()
        state.observe('blue', fit(), 0.)
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        result = state.observe('blue', fit((.1, .003, .5), rotation, (True, False, False)), 1.)
        np.testing.assert_allclose(result['center'], [0., .003, .415])

    def test_occlusion_preserves_object_but_not_current_visibility(self):
        state = GeometricState()
        state.observe('red', fit(), 1.)
        state.missing('red', 2.)
        result = state.estimate('red', 3.)
        self.assertEqual(result['source'], 'remembered')
        self.assertEqual(result['age_s'], 2.)
        self.assertFalse(result['position_fully_observed'])
        self.assertFalse(result['geometry_reliable'])
        self.assertGreater(result['covariance_m2'][0, 0], .001**2)

    def test_rigid_prediction_is_conditional_and_observation_can_reject_it(self):
        state = GeometricState()
        state.observe('red', fit(), 0.)
        state.set_target('red', [0., 0., .445])
        transform = np.eye(4)
        transform[:3, 3] = [.02, 0., .04]
        predicted = state.predict('red', transform, 1., 'if the visual grasp hypothesis remains valid')
        np.testing.assert_allclose(predicted['center'], [.02, 0., .455])
        self.assertEqual(predicted['source'], 'conditional_prediction')
        self.assertFalse(predicted['geometry_reliable'])
        np.testing.assert_allclose(state.objects['red']['center'], [0., 0., .415])
        np.testing.assert_allclose(state.targets['red']['center'], [0., 0., .445])
        updated = state.observe('red', fit(), 2.)
        np.testing.assert_allclose(updated['center'], [0., 0., .415])
        packet = state.packet()
        self.assertTrue(packet['objects']['red']['prediction_contradicted'])
        self.assertEqual(packet['recent_events'][-1]['kind'], 'prediction_contradicted')
        self.assertFalse(packet['task_success_certified'])

    def test_cube_axis_permutations_preserve_landmark_labels(self):
        state = GeometricState()
        state.observe('blue', fit(), 0.)
        before = state.landmarks('blue')
        rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        state.observe('blue', fit(rotation=rotation), 1.)
        after = state.landmarks('blue')
        for key in before:
            np.testing.assert_allclose(before[key], after[key], atol=1e-12)

    def test_same_time_views_can_supply_complementary_evidence(self):
        state = GeometricState()
        state.observe('red', fit(axes=(True, False, True)), 0., 'side')
        state.observe('red', fit(axes=(False, True, True)), 0., 'wrist')
        self.assertEqual(state.estimate('red')['observed_translation_rank'], 3)
        self.assertTrue(state.estimate('red')['geometry_reliable'])
        # Old complementary measurements remain memory, not fresh observation.
        state.observe('red', fit(axes=(True, False, True)), 1., 'side')
        self.assertEqual(state.estimate('red')['observed_translation_rank'], 2)

    def test_nearly_parallel_views_do_not_invent_a_measured_direction(self):
        state = GeometricState()
        state.observe('red', fit(axes=(True, False, True)), 0., 'side')
        theta = .001
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
        state.observe('red', fit(rotation=rotation, axes=(True, False, True)), 0., 'wrist')
        self.assertEqual(state.estimate('red')['observed_translation_rank'], 2)

    def test_bad_timestamps_transforms_and_unknown_shape_are_rejected(self):
        state = GeometricState()
        state.observe('red', fit(), 1.)
        for time in [0., np.nan]:
            with self.assertRaises(ValueError):
                state.observe('red', fit(), time)
        with self.assertRaises(ValueError):
            state.predict('red', np.eye(4), 2., '')
        with self.assertRaises(ValueError):
            state.observe('blue', dict(fit(), shape='cylinder'), 2.)
        self.assertEqual(state.time, 1.)

    def test_real_rgbd_packet_is_finite_and_contains_no_privileged_contact(self):
        frame = dict(np.load(Path(__file__).parent/'fixtures/ablation_cube_initial_rgbd.npz'))
        plane = inventory_table(frame)
        state = GeometricState()
        for color in ('blue', 'red', 'green'):
            state.observe(color, observe_cube(frame, color, table_plane=plane), 0.)
        packet = state.packet(details=True)
        json.dumps(packet, allow_nan=False)
        self.assertEqual(set(packet['objects']), {'blue', 'red', 'green'})
        self.assertTrue(all(not relation['contact_confirmed'] for relation in packet['relations']))
        self.assertIn('corner_7', packet['objects']['red']['landmarks_mm'])
        self.assertLess(len(state.text()), 7000)

    def test_state_is_available_as_action_free_sandbox_compatible_library(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
        from run_geometry_ablation import build_geometry_library
        namespace = {}
        exec(build_geometry_library(include_state=True), namespace)
        state = namespace['GEOMETRY'].state()
        state.observe('red', fit(), 0.)
        self.assertEqual(state.packet()['schema'], 'geometric_state_v1')
        self.assertNotIn('api', namespace)


if __name__ == '__main__':
    unittest.main()
