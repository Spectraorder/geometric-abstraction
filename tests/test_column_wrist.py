"""Known-size wrist RGB silhouettes constrain translation absent from side depth."""
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw

from aspire_so100.column_geometry import estimate_column_geometry
from aspire_so100.column_wrist import refine_column_with_wrist
from test_cube_geometry import cube_frame
from test_column_geometry import cylinder_frame

FIXTURES = Path(__file__).parent / 'fixtures'


def cuboid_block(center, rotation, dimensions, hidden_axis=1):
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    vertices = center + (signs * dimensions / 2) @ rotation.T
    return {'shape': 'thin_prism', 'center': center, 'rotation': rotation,
            'normal': rotation[:, 2], 'constrained_axes': np.arange(3) != hidden_axis,
            'vertices': vertices, 'top_vertices': vertices[signs[:, 2] > 0],
            'top': float((center + rotation[:, 2] * dimensions[2] / 2)[2]),
            'bottom': float(vertices[:, 2].min()), 'dims': dimensions[:2], 'height': dimensions[2]}


def rgb_only(frame):
    return {key: frame[key] for key in ('rgb', 'intrinsics', 'camera_to_world')}


class ColumnWristTests(unittest.TestCase):
    def test_real_wrist_resolves_thirteen_mm_grasp_slip_without_changing_depth_axes(self):
        with np.load(FIXTURES / 'column_wrist_cube_rgb.npz') as archive:
            frame = dict(archive)
        self.assertNotIn('depth_m', frame)
        block = json.loads((FIXTURES / 'column_wrist_cube_prior.json').read_text())
        result = refine_column_with_wrist(frame, 'red', block)
        # Expected center is an offline replay oracle, never a fitting input.
        expected = np.array([-.005303437171588823, -.2646518869387033, .47062915495660146])
        self.assertGreater(np.linalg.norm(np.asarray(block['center']) - expected), .013)
        self.assertLess(np.linalg.norm(result['center'] - expected), .0005)
        observed = np.asarray(block['rotation'])[:, np.asarray(block['constrained_axes'])]
        correction = result['center'] - block['center']
        np.testing.assert_allclose(observed.T @ correction, 0, atol=1e-10)
        self.assertGreater(result['wrist_fit']['silhouette_coverage'], .95)
        self.assertLess(result['wrist_fit']['bbox_rms_px'], .5)
        np.testing.assert_allclose(result['vertices'] - block['vertices'], np.broadcast_to(correction, (8, 3)))

    def test_non_catalog_tilted_cuboid_uses_rgb_and_learned_size(self):
        dimensions = np.array([.021, .033, .045])
        rotation = Rotation.from_euler('xyz', [.08, -.05, .12]).as_matrix()
        frame, center = cube_frame(rotation, half_size=dimensions / 2)
        prior = cuboid_block(center + .012 * rotation[:, 1], rotation, dimensions)
        result = refine_column_with_wrist(rgb_only(frame), 'red', prior)
        np.testing.assert_allclose(result['center'], center, atol=.001)
        np.testing.assert_allclose(result['normal'], prior['normal'])

    def test_cylinder_silhouette_uses_learned_radius_and_preserves_cap_height(self):
        frame, center = cylinder_frame(radius=.019, height=.052)
        block = estimate_column_geometry(frame, 'red', 'cylinder', table_plane=np.zeros(3))
        displacement = np.array([0., .010, 0.])
        block['center'] += displacement
        block['vertices'] += displacement
        block['top_vertices'] += displacement
        block['constrained_axes'] = np.array([True, False, True])
        result = refine_column_with_wrist(rgb_only(frame), 'red', block)
        np.testing.assert_allclose(result['center'], center, atol=.001)
        self.assertEqual(result['top'], block['top'])

    def test_partial_or_missing_silhouette_cannot_become_translation_evidence(self):
        with np.load(FIXTURES / 'column_wrist_cube_rgb.npz') as archive:
            frame = dict(archive)
        block = json.loads((FIXTURES / 'column_wrist_cube_prior.json').read_text())
        for partial in (True, False):
            hidden = dict(frame, rgb=frame['rgb'].copy())
            if partial:
                hidden['rgb'][:110] = 0
            else:
                hidden['rgb'][:] = 0
            with self.assertRaisesRegex(ValueError, 'unique complete wrist silhouette'):
                refine_column_with_wrist(hidden, 'red', block)

    def test_fixed_side_depth_axes_reject_an_incompatible_wrist_pose(self):
        with np.load(FIXTURES / 'column_wrist_cube_rgb.npz') as archive:
            frame = dict(archive)
        block = json.loads((FIXTURES / 'column_wrist_cube_prior.json').read_text())
        block['constrained_axes'] = [True, True, True]
        with self.assertRaises(ValueError):
            refine_column_with_wrist(frame, 'red', block)
        block['constrained_axes'] = [True, True, False]
        with self.assertRaises(ValueError):
            refine_column_with_wrist(frame, 'red', block, max_correction_m=.008)

    def test_two_individually_valid_silhouettes_are_rejected_when_ambiguous(self):
        with np.load(FIXTURES / 'column_wrist_cube_rgb.npz') as archive:
            frame = dict(archive)
        block = json.loads((FIXTURES / 'column_wrist_cube_prior.json').read_text())
        block = refine_column_with_wrist(frame, 'red', block)
        pose = frame['camera_to_world']
        vertices = block['vertices'].copy()
        shift = -.020 * pose[:3, 0]
        for field in ('center', 'vertices', 'top_vertices'):
            block[field] += shift
        masks = []
        for distance in (0., -.040):
            image = Image.new('RGB', (320, 240))
            local = (vertices + distance * pose[:3, 0] - pose[:3, 3]) @ pose[:3, :3]
            projected = local @ frame['intrinsics'].T
            projected = projected[:, :2] / projected[:, 2, None]
            polygon = projected[ConvexHull(projected).vertices]
            ImageDraw.Draw(image).polygon([tuple(point) for point in polygon], fill=(210, 30, 30))
            pixels = np.asarray(image)
            masks.append(pixels)
            refine_column_with_wrist(dict(frame, rgb=pixels), 'red', block,
                                     max_correction_m=.030, constrained_directions=np.empty((3, 0)))
        with self.assertRaisesRegex(ValueError, 'unique complete wrist silhouette'):
            refine_column_with_wrist(dict(frame, rgb=np.maximum(*masks)), 'red', block,
                                     max_correction_m=.030, constrained_directions=np.empty((3, 0)))

    def test_finite_side_bounds_are_hard_constraints_in_the_fitted_world_basis(self):
        with np.load(FIXTURES / 'column_wrist_cube_rgb.npz') as archive:
            frame = dict(archive)
        block = json.loads((FIXTURES / 'column_wrist_cube_prior.json').read_text())
        unconstrained = refine_column_with_wrist(frame, 'red', block)
        rotation = np.asarray(block['rotation'])
        projected_prior = np.asarray(block['center']) @ rotation
        projected_fit = unconstrained['center'] @ rotation
        # These absolute rotated coordinates differ from both world XYZ and
        # center-relative coordinates. Tight finite edges can already resolve
        # a plane-null axis and must not be overwritten by another sensor.
        tight = dict(block, finite_center_bounds_m=np.column_stack(
            (projected_prior - .0001, projected_prior + .0001)))
        with self.assertRaisesRegex(ValueError, 'finite side-camera center bounds'):
            refine_column_with_wrist(frame, 'red', tight)
        wide = dict(block, finite_center_bounds_m=np.column_stack(
            (np.minimum(projected_prior, projected_fit) - .0001,
             np.maximum(projected_prior, projected_fit) + .0001)))
        accepted = refine_column_with_wrist(frame, 'red', wide)
        np.testing.assert_allclose(accepted['center'], unconstrained['center'], atol=1e-9)
        bounds = np.asarray(wide['finite_center_bounds_m'])
        self.assertTrue(np.all(accepted['center'] @ rotation >= bounds[:, 0]))
        self.assertTrue(np.all(accepted['center'] @ rotation <= bounds[:, 1]))
        invalid = dict(block, finite_center_bounds_m=np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, 'Invalid finite'):
            refine_column_with_wrist(frame, 'red', invalid)

    def test_unobserved_orientation_is_not_treated_as_translation_evidence(self):
        with np.load(FIXTURES / 'column_wrist_cube_rgb.npz') as archive:
            frame = dict(archive)
        block = json.loads((FIXTURES / 'column_wrist_cube_prior.json').read_text())
        block['inferred_orientation'] = True
        with self.assertRaisesRegex(ValueError, 'independently observed orientation'):
            refine_column_with_wrist(frame, 'red', block)
        with self.assertRaisesRegex(ValueError, 'complete learned object model'):
            refine_column_with_wrist(frame, 'red', {'center': [0., 0., 0.]})

    def test_narrow_cylinder_requires_an_explicit_radial_constraint(self):
        frame, _ = cylinder_frame()
        block = estimate_column_geometry(frame, 'red', 'cylinder', table_plane=np.zeros(3))
        block.update(angular_information=.01, constrained_axes=np.array([False, False, True]))
        with self.assertRaisesRegex(ValueError, 'radial constraint'):
            refine_column_with_wrist(rgb_only(frame), 'red', block)


if __name__ == '__main__':
    unittest.main()
