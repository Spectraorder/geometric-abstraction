"""Unequal rectangular-prism dimensions inferred from analytic RGB-D."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from aspire_so100.cuboid_geometry import color_mask, estimate_cuboid_geometry
from test_cube_geometry import cube_frame


class CuboidGeometryTests(unittest.TestCase):
    def test_shadowed_orange_is_not_red_and_blue_is_not_cyan(self):
        # Observed shaded colors in the six-object simulation scene.
        colors = np.array([[[160, 31, 27], [154, 65, 8], [25, 85, 180], [25, 160, 190]]])
        np.testing.assert_array_equal(color_mask(colors, 'red'), [[True, False, False, False]])
        np.testing.assert_array_equal(color_mask(colors, 'orange'), [[False, True, False, False]])
        np.testing.assert_array_equal(color_mask(colors, 'blue'), [[False, False, True, False]])
        np.testing.assert_array_equal(color_mask(colors, 'cyan'), [[False, False, False, True]])

    def test_slender_prism_centroid_and_vertices(self):
        dimensions = np.array([.015, .03, .06])
        rotation = Rotation.from_euler('xyz', [.06, -.04, .13]).as_matrix()
        frame, center = cube_frame(rotation, half_size=dimensions/2)
        mask = frame['rgb'][:,:,1] == 30
        frame['rgb'][mask] = [160, 30, 210]
        result = estimate_cuboid_geometry(frame, 'purple', dimensions)
        self.assertTrue(result['reliable'])
        np.testing.assert_allclose(result['centroid'], center, atol=.0007)
        signs = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        expected = center+(signs*dimensions/2) @ rotation.T
        distances = np.linalg.norm(result['vertices'][:,None]-expected[None],axis=2)
        self.assertLess(float(np.max(np.min(distances,axis=1))),.0007)
        self.assertEqual(result['vertices_kind'], 'inferred_cuboid_model')

    def test_hidden_top_is_not_a_reliable_full_fit(self):
        dims = [.015, .03, .06]
        frame, _ = cube_frame(np.eye(3), hide_top=True, half_size=np.asarray(dims)/2)
        result = estimate_cuboid_geometry(frame, 'red', dims)
        self.assertFalse(result['reliable'])
        self.assertLess(result['observed_face_count'],3)

    def test_invalid_dimensions_rejected(self):
        for dims in [[.015,.03], [.015,0,.06], [.015,np.nan,.06]]:
            with self.assertRaisesRegex(ValueError, 'Invalid visual cuboid dimensions'):
                estimate_cuboid_geometry({}, 'purple', dims)
