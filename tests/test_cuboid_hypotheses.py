"""Current geometric correspondence must survive rotation and view imbalance."""
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from aspire_so100.cuboid_hypotheses import cuboid_dimension_hypotheses, observe_cuboid_correspondence
from aspire_so100.instance_geometry import inventory_table
from aspire_so100.pnp_observation import MixedPnPObservation


def faces(center, rotation, sizes, selection=None, patch=False):
    clouds = []; normals = []
    for axis in range(3):
        for sign in (-1., 1.):
            if selection is not None and (axis, sign) not in selection:
                continue
            other = [i for i in range(3) if i != axis]
            width = np.minimum(sizes, .003) if patch else sizes
            for a in np.linspace(-.49, .49, 9):
                for b in np.linspace(-.49, .49, 9):
                    local = np.zeros(3); local[axis] = sign*sizes[axis]/2
                    local[other] = [a*width[other[0]], b*width[other[1]]]
                    clouds.append(center+rotation@local); normals.append(sign*rotation[:, axis])
    return np.asarray(clouds), np.asarray(normals)


class CuboidCorrespondenceTests(unittest.TestCase):
    def test_large_rotation_recovers_lengths_without_world_axis_assignment(self):
        center = np.array([.12, -.21, .48]); rotation = Rotation.from_euler('xyz', [57, 38, 71], degrees=True).as_matrix()
        initial = np.array([.047, .023, .012]); sizes = initial[[1, 0, 2]]
        cloud, normals = faces(center, rotation, sizes)
        result = cuboid_dimension_hypotheses(cloud, cloud, normals, rotation, {'dims': initial[:2], 'height': initial[2]})
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0]['center'], center, atol=1e-12)
        np.testing.assert_allclose(result[0]['axis_dimensions'], sizes, atol=1e-12)

    def test_dense_view_does_not_erase_sparse_far_face(self):
        center = np.array([0., 0., .1]); sizes = np.array([.047, .023, .012]); rotation = np.eye(3)
        sparse, sparse_normals = faces(center, rotation, sizes)
        dense, dense_normals = faces(center, rotation, sizes, [(0, 1.)])
        dense = np.repeat(dense, 100, axis=0); dense_normals = np.repeat(dense_normals, 100, axis=0)
        cloud = np.concatenate([sparse, dense]); normals = np.concatenate([sparse_normals, dense_normals])
        groups = np.r_[np.zeros(len(sparse), int), np.ones(len(dense), int)]
        fit = cuboid_dimension_hypotheses(cloud, cloud, normals, rotation, {'dims': sizes[:2], 'height': sizes[2]}, point_groups=groups)
        self.assertEqual(len(fit), 1)
        np.testing.assert_allclose(fit[0]['center'], center, atol=1e-12)
        np.testing.assert_array_equal(fit[0]['constrained_axes'], [True, True, True])

    def test_partial_plane_patches_retain_multiple_hypotheses(self):
        sizes = np.array([.047, .023, .012]); cloud = []; normals = []
        # Tiny adjacent patches near one corner reveal two plane positions,
        # but do not reveal the lengths extending behind either plane.
        for axis in (0, 1):
            for along in np.linspace(.001, .003, 9):
                for z in np.linspace(-.001, .001, 9):
                    point = np.array([sizes[0]/2, sizes[1]/2, z])
                    point[1-axis] -= along
                    cloud.append(point); normals.append(np.eye(3)[axis])
        cloud = np.asarray(cloud); normals = np.asarray(normals)
        fit = cuboid_dimension_hypotheses(cloud, cloud, normals, np.eye(3), {'dims': sizes[:2], 'height': sizes[2]})
        self.assertGreater(len(fit), 1)

    def test_recorded_regression_requires_current_joint_evidence(self):
        path = Path(__file__).parent/'fixtures/cuboid_correspondence_rgbd.npz'
        with np.load(path) as data:
            frames = {label: {k[len(label)+1:]: data[k] for k in data.files if k.startswith(label+'_')}
                      for label in ('initial', 'side', 'wrist')}
        plane = inventory_table(frames['initial'])
        objects = [{'color': 'green', 'shape': 'thin_prism', 'orientation': 'flat_x'}]
        old = MixedPnPObservation(objects); current = MixedPnPObservation(objects, correspondence=True)
        model = old.observe(frames['initial'], 'green', None, plane)['model']
        pair = [('side', frames['side']), ('wrist', frames['wrist'])]
        with self.assertRaisesRegex(ValueError, 'No usable current view'):
            old.observe_views(pair, 'green', model, plane)
        fit = current.observe_views(pair, 'green', model, plane)
        self.assertEqual(fit['source'], 'current_surface_cuboid_correspondence')
        self.assertFalse(fit['temporal_pose_prior_used'])
        self.assertFalse(fit['position_fully_observed'])
        self.assertLess(max(fit['finite_center_width_m'][~fit['constrained_axes']]), .0015)
        np.testing.assert_array_equal(fit['axis_dimension_indices'], [1, 0, 2])
        self.assertEqual(fit['correspondence_hypotheses'], 1)
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            observe_cuboid_correspondence([pair[0], ('wrist', dict(frames['wrist'], time=float(frames['wrist']['time'])-1.))], 'green', 'thin_prism', model, plane)
