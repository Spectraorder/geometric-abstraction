"""Flat bridge beams must be identified from dimensions and oriented surfaces."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from aspire_so100.beam_geometry import flat_beam_aligned, select_structure_nodes, estimate_flat_beam_geometry
from aspire_so100.column_geometry import estimate_column_geometry
from aspire_so100.column_tracking import ColumnTracker
from test_column_tracking import primitive_scene


class BeamGeometryTests(unittest.TestCase):
    def test_flat_shape_is_measured_at_different_metric_scales(self):
        for scale in (.9, 1.2):
            with self.subTest(scale=scale):
                obj = {'shape': 'thin_prism', 'center': np.array([.01, -.02, .0075*scale]),
                       'dims': np.array([.06, .03])*scale, 'height': .015*scale}
                frame, _ = primitive_scene([obj])
                tracker = ColumnTracker(plane=np.zeros(3))
                tracker.update(frame)
                nodes = [{'color': 'blue', 'shape': 'thin_prism', 'orientation': 'flat_x'}]
                selected = select_structure_nodes(tracker, nodes, frame)
                self.assertEqual(len(selected), 1)
                fit = estimate_flat_beam_geometry(tracker.isolated_frame(frame, selected[0]), 'blue', tracker.plane)
                np.testing.assert_allclose(np.r_[fit['dims'], fit['height']],
                                           [.06*scale, .03*scale, .015*scale], atol=.001)
                self.assertTrue(flat_beam_aligned(fit))

    def test_goal_color_does_not_turn_an_upright_prism_or_cube_into_a_beam(self):
        for dims, height in (([.03, .03], .03), ([.015, .03], .06), ([.06, .03], .03)):
            obj = {'shape': 'prism', 'center': np.array([.01, -.02, height/2]),
                   'dims': np.array(dims), 'height': height}
            frame, _ = primitive_scene([obj])
            tracker = ColumnTracker(plane=np.zeros(3))
            tracker.update(frame)
            with self.assertRaisesRegex(ValueError, 'No unique visually supported flat'):
                select_structure_nodes(tracker, [{'color': 'blue', 'shape': 'thin_prism',
                                                 'orientation': 'flat_x'}], frame)

    def test_supported_beam_keeps_learned_thickness_above_a_raised_surface(self):
        obj = {'shape': 'thin_prism', 'center': np.array([.01, -.02, .0075]),
               'dims': np.array([.06, .03]), 'height': .015}
        initial, _ = primitive_scene([obj])
        learned = estimate_flat_beam_geometry(initial, 'blue', np.zeros(3))
        obj['center'] = obj['center'] + [0., 0., .03]
        supported, _ = primitive_scene([obj])
        fitted = estimate_flat_beam_geometry(supported, 'blue', np.zeros(3), learned['model'])
        self.assertAlmostEqual(fitted['height'], .015, delta=.0001)
        np.testing.assert_allclose(fitted['center'], obj['center'], atol=.0005)
        self.assertAlmostEqual(fitted['top'], .045, delta=.0001)
        self.assertAlmostEqual(fitted['bottom'], .03, delta=.0001)

    def test_initial_measurement_rejects_missing_pixels_and_nonrectangular_top(self):
        obj = {'shape': 'thin_prism', 'center': np.array([.01, -.02, .0075]),
               'dims': np.array([.06, .03]), 'height': .015}
        frame, labels = primitive_scene([obj])
        blank = dict(frame, valid_depth=np.zeros_like(frame['valid_depth']))
        with self.assertRaises(ValueError):
            estimate_flat_beam_geometry(blank, 'blue', np.zeros(3))
        from aspire_so100.instance_geometry import inventory_points
        points, _ = inventory_points(frame)
        missing_corner = (points[..., 0] > .015) & (points[..., 1] > -.02) & (labels == 0)
        frame['valid_depth'][missing_corner] = False
        with self.assertRaisesRegex(ValueError, 'clipped|corners'):
            estimate_flat_beam_geometry(frame, 'blue', np.zeros(3))

    def test_orientation_tracks_physical_axis_order_and_rejects_wrong_face(self):
        dimensions = np.array([.015, .03, .06])
        rotation = Rotation.from_euler('y', 90, degrees=True).as_matrix()
        self.assertTrue(flat_beam_aligned({'axis_dimensions': dimensions, 'rotation': rotation}))
        # Fitting may reorder or reverse axes without changing the object pose.
        order = [2, 0, 1]
        self.assertTrue(flat_beam_aligned({'axis_dimensions': dimensions[order],
                                          'rotation': rotation[:, order]}))
        for wrong in (np.eye(3), Rotation.from_euler('x', 90, degrees=True).as_matrix(),
                      Rotation.from_euler('z', 12, degrees=True).as_matrix() @ rotation):
            self.assertFalse(flat_beam_aligned({'axis_dimensions': dimensions, 'rotation': wrong}))
        self.assertFalse(flat_beam_aligned({'normal': np.array([0., 0., 1.])}))
