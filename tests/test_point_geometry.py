"""Reference geometry is independent of perception and rejects invalid topology."""
from pathlib import Path
import unittest

import numpy as np


class PointGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = {}
        path = Path(__file__).resolve().parents[1]/'prototypes/point_geometry.py'
        exec(compile(path.read_text(), str(path), 'exec'), cls.namespace)

    def reference(self, fit, role):
        return self.namespace['vertex_geometry_reference'](fit, role)

    def geometry(self):
        return {'vertices': np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                'edge_vertex_indices': [[0, 1]], 'face_vertex_indices': [[0, 1, 2]],
                'vertex_sources': ['fitted_from_current_depth']*3}

    def test_connectivity_requires_distinct_in_range_integer_indices(self):
        invalid = [('edge', [-1, 1]), ('edge', [0, 3]), ('edge', [0., 1.]),
                   ('edge', [False, True]), ('edge', [[0, 1]]), ('edge', [0, 0]),
                   ('edge', [0]), ('face', [0, 1]), ('face', [0, 1, 1]), ('face', [])]
        for kind, indices in invalid:
            fit = self.geometry()
            fit[kind+'_vertex_indices'] = [indices]
            with self.subTest(kind=kind, indices=indices), self.assertRaises(ValueError):
                self.reference(fit, kind+'_0')
        for role in ('vertex', 'vertex_-1', 'edge_x', 'face_0_extra', 'vertex_²', 'vertex_99', None):
            with self.subTest(role=role), self.assertRaises(ValueError):
                self.reference(self.geometry(), role)

    def test_coordinates_and_per_point_evidence_are_required(self):
        fit = self.geometry()
        invalid = [('vertices', [[0., 0.]]), ('vertices', [[0., 0., np.nan]]),
                   ('vertex_sources', ['observed']), ('vertex_sources', ['observed', '', 'observed'])]
        for field, value in invalid:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.reference(dict(fit, **{field: value}), 'vertex_0')
        fit['vertices'][1] = fit['vertices'][0]
        with self.assertRaisesRegex(ValueError, 'geometrically distinct'):
            self.reference(fit, 'edge_0')
        with self.assertRaisesRegex(ValueError, 'geometrically distinct'):
            self.reference(fit, 'face_0')

    def test_declared_concave_face_winding_and_centroid_are_consistent(self):
        # This supplied boundary tests representation, not concave-object sensing.
        points = np.array([[0., 0., 0.], [2., 0., 0.], [2., 1., 0.],
                           [1., 1., 0.], [1., 2., 0.], [0., 2., 0.]])
        fit = dict(self.geometry(), vertices=points, vertex_sources=['fitted_from_current_depth']*6)
        for offset in range(6):
            for reverse in (False, True):
                indices = np.roll(np.arange(6), offset)
                if reverse:
                    indices = indices[::-1]
                fit['face_vertex_indices'] = [indices]
                ref = self.reference(fit, 'face_0')
                np.testing.assert_allclose(ref['center_m'], [5/6, 5/6, 0.])
                np.testing.assert_allclose(ref['normal'], [0., 0., -1. if reverse else 1.])

    def test_self_intersecting_boundary_does_not_define_a_face(self):
        points = np.array([[0., 0., 0.], [2., 2., 0.], [0., 2., 0.], [2., 0., 0.], [3., -1., 0.]])
        fit = dict(self.geometry(), vertices=points, vertex_sources=['fitted_from_current_depth']*5,
                   face_vertex_indices=[[0, 1, 2, 3, 4]])
        with self.assertRaisesRegex(ValueError, 'intersects itself'):
            self.reference(fit, 'face_0')

    def test_mixed_evidence_labels_do_not_become_observed_geometry(self):
        fit = self.geometry()
        fit['vertex_sources'][0] = 'inferred_hidden_point_under_model'
        for role in ('vertex_0', 'edge_0', 'face_0'):
            reference = self.reference(fit, role)
            self.assertIn('model_inferred', reference['semantics'])
            self.assertIn('inferred_hidden_point_under_model', reference['vertex_sources'])
        self.assertNotIn('inferred', self.reference(fit, 'vertex_1')['semantics'])


if __name__ == '__main__':
    unittest.main()
