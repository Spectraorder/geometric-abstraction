"""Sensor geometry checks with analytic ray/cube intersections, no simulator."""
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from aspire_so100.cube_geometry import estimate_cube_geometry, support_margin


def cube_frame(rotation, hide_top=False, half_size=None):
    """Ray-cast a rotated box over a table, returning calibrated optical depth."""
    half_size = np.array([.02, .02, .02] if half_size is None else half_size)
    width, height = 240, 180
    center = np.array([.01, -.02, .08])
    camera = np.array([.24, -.28, .38])
    forward = center - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, down, forward))
    pose[:3, 3] = camera
    k = np.array([[240., 0., (width-1)/2], [0., 240., (height-1)/2], [0., 0., 1.]])
    v, u = np.indices((height, width))
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(k).T @ pose[:3, :3].T
    direction = rays @ rotation
    origin = (camera - center) @ rotation
    with np.errstate(divide='ignore', invalid='ignore'):
        first = (-half_size - origin) / direction
        second = (half_size - origin) / direction
        entry = np.max(np.minimum(first, second), axis=2)
        leave = np.min(np.maximum(first, second), axis=2)
        table_depth = -camera[2] / rays[..., 2]
    hit = (entry > 0) & (entry < leave) & (entry < table_depth)
    depth = np.where(hit, entry, table_depth)
    rgb = np.full((height, width, 3), [160, 120, 70], dtype=np.uint8)
    rgb[hit] = [210, 30, 30]
    if hide_top:
        local = origin + entry[..., None] * direction
        rgb[hit & (abs(local[..., 2] - half_size[2]) < 1e-7)] = 0
    return {'rgb': rgb, 'depth_m': depth, 'valid_depth': np.isfinite(depth) & (depth > 0),
            'intrinsics': k, 'camera_to_world': pose}, center


class CubeGeometryTests(unittest.TestCase):
    def test_occluded_face_rejects_fit_instead_of_returning_nan(self):
        # A 14x15 RGB-D crop from a real policy observation; no object state.
        # The old fit took the median of an empty normal-consistency subset.
        path = Path(__file__).parent / 'fixtures/occluded_red_depth.npz'
        with np.load(path, allow_pickle=False) as archive:
            frame = dict(archive)
        with self.assertRaisesRegex(ValueError, 'insufficient consistent depth'):
            estimate_cube_geometry(frame, 'red', .0299, table_plane=np.array([0., 0., .4]))

    def test_rotated_cube_centroid_and_unordered_corners(self):
        rotation = Rotation.from_euler('xyz', [.08, -.06, .23]).as_matrix()
        frame, center = cube_frame(rotation)
        result = estimate_cube_geometry(frame, 'red', .04)
        self.assertTrue(result['reliable'])
        self.assertEqual(result['observed_face_count'], 3)
        np.testing.assert_allclose(result['centroid'], center, atol=.0005)
        signs = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        expected = center + .02 * signs @ rotation.T
        distances = np.linalg.norm(result['vertices'][:,None] - expected[None], axis=2)
        self.assertLess(float(np.max(np.min(distances, axis=1))), .0005)
        self.assertEqual(result['vertices_kind'], 'inferred_cube_model')
        # Averaging visible samples is not a solid object's centroid.
        self.assertGreater(float(np.linalg.norm(result['surface_centroid'] - center)), .005)

    def test_hidden_face_marks_missing_axis_as_unreliable(self):
        frame, _ = cube_frame(np.eye(3), hide_top=True)
        result = estimate_cube_geometry(frame, 'red', .04)
        self.assertFalse(result['reliable'])
        self.assertLess(result['observed_face_count'], 3)
        self.assertFalse(np.all(result['constrained_axes']))

    def test_missing_depth_and_invalid_size_are_rejected(self):
        frame, _ = cube_frame(np.eye(3))
        with self.assertRaises(ValueError):
            estimate_cube_geometry(frame, 'red', 0)
        frame['valid_depth'][:] = False
        with self.assertRaises(ValueError):
            estimate_cube_geometry(frame, 'red', .04)

    def test_support_margin_uses_rotated_edges(self):
        rotation = Rotation.from_euler('z', .7).as_matrix()
        corners = np.array([[x,y,.02] for x in (-.02,.02) for y in (-.02,.02)]) @ rotation.T
        self.assertAlmostEqual(support_margin([0,0,.04], corners, [0,0,1]), .02)
        point = np.array([.03,0,.04]) @ rotation.T
        self.assertAlmostEqual(support_margin(point, corners, [0,0,1]), -.01)
