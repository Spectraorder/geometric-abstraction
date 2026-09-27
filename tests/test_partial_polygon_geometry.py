"""Occlusion updates require current evidence and preserve conditional provenance."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_polygon_prism_session import library
from test_polygon_prism_geometry import prism_frame, footprint


class PartialPolygonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = {}
        exec(library('symbolic'), cls.ns)

    def fit(self, frame, model=None, plane=None):
        return self.ns['observe_polygon_prism'](frame, 'red', model, np.zeros(3) if plane is None else plane)

    def obscure(self, frame, kind='occluder', fraction=.35):
        frame = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in frame.items()}
        mask = self.ns['inventory_color_mask'](frame['rgb'], 'red')
        columns = np.where(mask)[1]
        cutoff = np.quantile(columns, 1-fraction)
        cut = np.indices(mask.shape)[1] >= cutoff
        if kind == 'missing':
            frame['valid_depth'][cut] = False
        else:
            frame['rgb'][cut] = [30, 40, 210]
            frame['depth_m'][cut] *= .90 if kind == 'occluder' else 1.05
        return frame

    def recorded(self, key):
        with np.load(ROOT/'tests/fixtures/partial_polygon_rgbd.npz') as record:
            frame = {field: record[key+'_'+field] for field in
                     ('rgb', 'depth_m', 'valid_depth', 'intrinsics', 'camera_to_world', 'time')}
            expected = record[key+'_expected_center_m']
        return frame, expected

    def test_partial_update_tracks_new_pose_without_a_previous_pose(self):
        polygon = footprint('trapezoid')
        model = self.fit(prism_frame(polygon, resolution_scale=3))['model']
        rotation = Rotation.from_euler('xyz', [.08, -.04, .38]).as_matrix()
        center = np.array([-.025, .015, .075])
        frame = self.obscure(prism_frame(polygon, center=center, rotation=rotation, stamp=2., resolution_scale=3))
        measured = self.fit(frame, model)
        _, centroid, _ = self.ns['point_polygon_moments'](polygon)
        np.testing.assert_allclose(measured['center'], center+rotation@np.r_[centroid, 0.], atol=.0015)
        np.testing.assert_allclose(measured['normal'], rotation[:, 2], atol=.001)
        self.assertIn('partial_edges', measured['pose_update_mode'])
        self.assertGreaterEqual(measured['registration_supported_edges'], 2)
        self.assertEqual(measured['observation_time'], 2.)
        # Irrelevant historical-pose hints must not influence the measurement.
        poisoned = dict(model, previous_center=[100., 100., 100.], previous_rotation=np.eye(3))
        np.testing.assert_array_equal(self.fit(frame, poisoned)['center'], measured['center'])

    def test_missing_depth_and_free_space_do_not_explain_a_hidden_boundary(self):
        initial = prism_frame(footprint('trapezoid'), resolution_scale=3)
        model = self.fit(initial)['model']
        for kind in ('missing', 'background'):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.fit(self.obscure(initial, kind), model)
        with self.assertRaises(ValueError):
            self.fit(self.obscure(initial), None)

    def test_one_visible_direction_and_heavy_occlusion_remain_unavailable(self):
        boundary = np.column_stack((np.linspace(-.02, .02, 20), np.zeros(20)))
        with self.assertRaisesRegex(ValueError, 'two independent edges'):
            self.ns['prism_visible_lines'](boundary, .001)
        initial = prism_frame(resolution_scale=3)
        model = self.fit(initial)['model']
        with self.assertRaises(ValueError):
            self.fit(self.obscure(initial, fraction=.90), model)

    def test_recorded_held_frames_have_bounded_center_error(self):
        for key in ('a', 'b'):
            initial, _ = self.recorded(key+'_initial')
            held, expected = self.recorded(key+'_held')
            plane = self.ns['inventory_table'](initial)
            model = self.fit(initial, plane=plane)['model']
            measured = self.fit(held, model, plane)
            np.testing.assert_allclose(measured['center'], expected, atol=.0015)
            self.assertIn('partial_edges', measured['pose_update_mode'])
            self.ns['MetricScene']().observe('red', measured)
        ambiguous, _ = self.recorded('b_ambiguous')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.fit(ambiguous, model, plane)

    def test_complementary_current_views_resolve_correspondence_but_reject_stale_or_inconsistent_data(self):
        initial, _ = self.recorded('b_initial')
        side, expected = self.recorded('b_ambiguous')
        wrist, _ = self.recorded('b_wrist')
        plane = self.ns['inventory_table'](initial)
        model = self.fit(initial, plane=plane)['model']
        observer = self.ns['MixedPnPObservation']([{'color': 'red', 'shape': 'polygon_prism'}])
        measured = observer.observe_views([('side', side), ('wrist', wrist)], 'red', model, plane)
        np.testing.assert_allclose(measured['center'], expected, atol=.0015)
        self.assertEqual(measured['view'], 'joint_current_polygon_edges')
        self.assertEqual(measured['registration_views'], ['side', 'wrist'])
        stale = dict(wrist, time=float(wrist['time'])-1.)
        with self.assertRaisesRegex(ValueError, 'same finite timestamp'):
            observer.observe_views([('side', side), ('wrist', stale)], 'red', model, plane)
        displaced = dict(wrist, camera_to_world=wrist['camera_to_world'].copy())
        displaced['camera_to_world'][0, 3] += .04
        with self.assertRaisesRegex(ValueError, 'No usable current view'):
            observer.observe_views([('side', side), ('wrist', displaced)], 'red', model, plane)
        with self.assertRaisesRegex(ValueError, 'Repeated camera geometry'):
            self.ns['observe_partial_polygon_views']([('side', side), ('copy', side)], 'red', model, plane)

    def test_partial_references_and_checkpoint_preserve_inference(self):
        initial, _ = self.recorded('a_initial')
        held, _ = self.recorded('a_held')
        plane = self.ns['inventory_table'](initial)
        model = self.fit(initial, plane=plane)['model']
        measured = self.fit(held, model, plane)
        scene = self.ns['MetricScene']()
        scene.observe('red', measured)
        symbolic = self.ns['SymbolicScene'](scene)
        roles = ('vertex_'+str(len(measured['top_polygon_m'])), 'top_face', 'bottom_face')
        references = [symbolic.reference('red.'+role) for role in roles]
        for ref in references:
            self.assertTrue(all(source.startswith('inferred_') for source in ref['vertex_sources']))
        self.assertIn('occluded_boundary', references[1]['semantics'])
        state = self.ns['GeometryStepState']('red')
        record = state.update('held', measured, measured['observation_time'], references)
        self.assertIn('two_independent_edges', record['geometry']['observability_condition'])
        for point in record['requested_points'].values():
            self.assertTrue(all(source.startswith('inferred_') for source in point['evidence_sources']))
            self.assertNotIn('vertices_m', point)
        for field, value in (('registration_supported_edges', 1), ('registration_edge_rms_m', .01),
                             ('registration_center_spread_m', .05), ('registration_condition', np.inf)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                scene.observe('red', dict(measured, **{field:value}))

    def test_incompatible_initialized_shape_cannot_force_a_partial_pose(self):
        initial, _ = self.recorded('a_initial')
        held, _ = self.recorded('a_held')
        plane = self.ns['inventory_table'](initial)
        model = self.fit(initial, plane=plane)['model']
        with self.assertRaises(ValueError):
            self.fit(held, dict(model, footprint_local_m=np.asarray(model['footprint_local_m'])*1.5), plane)


if __name__ == '__main__':
    unittest.main()
