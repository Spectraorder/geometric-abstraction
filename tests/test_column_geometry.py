"""Column geometry from analytic rays and an archived real RGB-D observation."""
from pathlib import Path
import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from aspire_so100.column_geometry import ColumnGeometryConflict, estimate_column_geometry, update_partial_column_geometry, select_column_levels, column_size_uncertainty
from aspire_so100.instance_geometry import InstanceTracker, inventory_points
from test_cube_geometry import cube_frame


def cylinder_frame(radius=.015, height=.06, rotation=None, center=None, hide_top=False):
    rotation = np.eye(3) if rotation is None else rotation
    center = np.array([.01, -.02, height / 2]) if center is None else np.asarray(center)
    frame, _ = cube_frame(np.eye(3))
    k, pose = frame['intrinsics'], frame['camera_to_world']
    v, u = np.indices(frame['depth_m'].shape)
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(k).T @ pose[:3, :3].T
    camera = pose[:3, 3]
    origin = (camera - center) @ rotation
    direction = rays @ rotation
    a = np.sum(direction[..., :2] ** 2, axis=2)
    b = 2 * (direction[..., :2] @ origin[:2])
    c = np.sum(origin[:2] ** 2) - radius ** 2
    discriminant = b ** 2 - 4 * a * c
    depth = np.full(a.shape, np.inf)
    for sign in (-1, 1):
        t = (-b + sign * np.sqrt(np.maximum(discriminant, 0))) / (2 * a)
        z = origin[2] + t * direction[..., 2]
        valid = (discriminant >= 0) & (abs(z) < height / 2) & (t > 0)
        depth = np.minimum(depth, np.where(valid, t, np.inf))
    top = np.zeros(a.shape, bool)
    for sign in (-1, 1):
        t = (sign * height / 2 - origin[2]) / direction[..., 2]
        xy = origin[:2] + t[..., None] * direction[..., :2]
        valid = (np.sum(xy ** 2, axis=2) < radius ** 2) & (t > 0) & (t < depth)
        if sign > 0:
            top = valid
        depth = np.where(valid, t, depth)
    table = -camera[2] / rays[..., 2]
    hit = depth < table
    frame['depth_m'] = np.minimum(depth, table)
    frame['valid_depth'] = np.isfinite(frame['depth_m']) & (frame['depth_m'] > 0)
    frame['rgb'][:] = [160, 120, 70]
    frame['rgb'][hit] = [210, 30, 30]
    if hide_top:
        frame['rgb'][hit & top] = 0
    return frame, center


class ColumnGeometryTests(unittest.TestCase):
    def test_cylinder_dimensions_are_measured_from_curved_surface(self):
        # Noncatalog sizes demonstrate that dimensions are fitted rather than
        # inferred from a fixed list of inventory measurements.
        for radius, height in ((.012, .052), (.019, .047)):
            frame, center = cylinder_frame(radius, height)
            fit = estimate_column_geometry(frame, 'red', 'cylinder', table_plane=np.zeros(3))
            np.testing.assert_allclose(fit['center'], center, atol=.0003)
            np.testing.assert_allclose(fit['dims'], [2 * radius] * 2, atol=.0003)
            self.assertAlmostEqual(fit['height'], height, delta=.0003)
            self.assertTrue(fit['geometry_reliable'])

    def test_occluded_cylinder_retains_axial_uncertainty_and_corrects_visible_axes(self):
        frame, center = cylinder_frame()
        initial = estimate_column_geometry(frame, 'red', 'cylinder', table_plane=np.zeros(3))
        rotation = Rotation.from_euler('xyz', [.06, -.04, 0]).as_matrix()
        moved_center = center + [.02, .01, .08]
        frame, _ = cylinder_frame(rotation=rotation, center=moved_center, hide_top=True)
        robot_before = {'tool_position': center, 'tool_rotation': np.eye(3)}
        robot = {'tool_position': moved_center + [.002, -.001, 0], 'tool_rotation': rotation}
        fit = update_partial_column_geometry(frame, 'red', 'cylinder', initial['model'], initial,
                                             robot_before, robot, np.zeros(3))
        self.assertFalse(fit['observed_top'])
        self.assertTrue(fit['inferred_axial_position'])
        self.assertFalse(fit['geometry_reliable'])
        np.testing.assert_allclose(fit['center'], moved_center, atol=.0004)
        np.testing.assert_allclose(fit['normal'], rotation[:, 2], atol=.0001)
        with self.assertRaisesRegex(ValueError, 'without a prior'):
            estimate_column_geometry(frame, 'red', 'cylinder', initial['model'], np.zeros(3))

    def test_narrow_cylinder_arc_updates_radius_direction_and_preserves_hidden_tangent(self):
        frame, center = cylinder_frame()
        initial = estimate_column_geometry(frame, 'red', 'cylinder', table_plane=np.zeros(3))
        points, _ = inventory_points(frame)
        radial = frame['camera_to_world'][:2, 3] - center[:2]
        radial /= np.linalg.norm(radial)
        tangent = np.array([-radial[1], radial[0]])
        relative = points[..., :2] - center[:2]
        angle = np.arctan2(relative @ tangent, relative @ radial)
        visible = (abs(angle) < .22) & (points[..., 2] < .05999)
        frame['rgb'][(frame['rgb'][..., 1] == 30) & ~visible] = 0
        offset = np.r_[.0015 * radial + .001 * tangent, 0.]
        robot_before = {'tool_position': center, 'tool_rotation': np.eye(3)}
        robot = {'tool_position': center + offset, 'tool_rotation': np.eye(3)}
        fit = update_partial_column_geometry(frame, 'red', 'cylinder', initial['model'], initial,
                                             robot_before, robot, np.zeros(3))
        error = fit['center'][:2] - center[:2]
        self.assertLess(fit['angular_information'], .025)
        self.assertFalse(fit['geometry_reliable'])
        self.assertTrue(fit['inferred_translation'])
        self.assertLess(abs(error @ radial), .0002)
        self.assertAlmostEqual(error @ tangent, .001, delta=.0002)

    def test_top_only_cylinder_corrects_axial_depth_but_retains_hidden_xy(self):
        frame, center = cylinder_frame()
        initial = estimate_column_geometry(frame, 'red', 'cylinder', table_plane=np.zeros(3))
        points, _ = inventory_points(frame)
        keep = abs(points[..., 2]-.06) < 1e-7
        frame['rgb'][(frame['rgb'][..., 1] == 30) & ~keep] = 0
        robot_before = {'tool_position': center, 'tool_rotation': np.eye(3)}
        robot = {'tool_position': center + [.0002, -.0002, .002], 'tool_rotation': np.eye(3)}
        fit = update_partial_column_geometry(frame, 'red', 'cylinder', initial['model'], initial,
                                             robot_before, robot, np.zeros(3))
        self.assertTrue(fit['top_only_update'])
        self.assertTrue(fit['observed_top'])
        self.assertFalse(fit['geometry_reliable'])
        self.assertTrue(fit['inferred_translation'])
        np.testing.assert_allclose(fit['center'][:2], robot['tool_position'][:2], atol=.0001)
        self.assertAlmostEqual(fit['center'][2], center[2], delta=.0003)
        wrong = dict(robot, tool_position=center+[.012, 0., .002])
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            update_partial_column_geometry(frame, 'red', 'cylinder', initial['model'], initial,
                                           robot_before, wrong, np.zeros(3))

    def test_visible_cylinder_side_axis_can_recover_observed_grasp_slip(self):
        frame, center = cylinder_frame()
        initial = estimate_column_geometry(frame, 'red', 'cylinder', table_plane=np.zeros(3))
        rotation = Rotation.from_euler('xyz', [.24, -.06, 0.]).as_matrix()
        moved_center = center + [.02, .01, .08]
        frame, _ = cylinder_frame(rotation=rotation, center=moved_center, hide_top=True)
        before = {'tool_position': center, 'tool_rotation': np.eye(3)}
        # The robot's unchanged attitude would predict no rotation. The image
        # independently observes the object slipping within that grasp.
        robot = {'tool_position': moved_center, 'tool_rotation': np.eye(3)}
        fit = update_partial_column_geometry(frame, 'red', 'cylinder', initial['model'], initial,
                                             before, robot, np.zeros(3))
        self.assertTrue(fit['orientation_observed'])
        self.assertEqual(fit['orientation_evidence'], 'side_normals')
        self.assertTrue(fit['grasp_slip_detected'])
        self.assertTrue(fit['inferred_axial_position'])
        self.assertFalse(fit['geometry_reliable'])
        self.assertGreater(float(fit['normal'] @ rotation[:, 2]), np.cos(.01))
        np.testing.assert_allclose(fit['center'], moved_center, atol=.0005)
        wrong = dict(robot, tool_position=moved_center+[.025, 0., 0.])
        with self.assertRaises(ValueError):
            update_partial_column_geometry(frame, 'red', 'cylinder', initial['model'], initial,
                                           before, wrong, np.zeros(3))

    def test_recorded_occluded_cylinder_observes_slip_without_object_truth(self):
        with np.load(Path(__file__).parent/'fixtures/column_cylinder_slip_rgbd.npz') as archive:
            frame = dict(archive)
        tracker = InstanceTracker()
        tracker.plane = frame['table_plane']
        tracker.update(frame)
        red = next(track for track in tracker.tracks.values() if track['color'] == 'red')
        isolated = tracker.isolated_frame(frame, red['id'])
        radius = float(frame['model_radius'])
        model = dict(radius=radius, height=float(frame['model_height']), dims=np.repeat(2*radius, 2))
        prior = dict(center=frame['prior_center'], normal=frame['prior_normal'])
        robot = dict(tool_position=np.zeros(3), tool_rotation=np.eye(3))
        fit = update_partial_column_geometry(isolated, 'red', 'cylinder', model, prior,
                                             robot, robot, frame['table_plane'])
        self.assertTrue(fit['orientation_observed'])
        self.assertEqual(fit['orientation_evidence'], 'side_normals')
        self.assertTrue(fit['grasp_slip_detected'])
        self.assertFalse(fit['observed_top'])
        self.assertLess(fit['surface_rms_m'], .0004)
        self.assertLess(np.linalg.norm(fit['center']-prior['center']), .008)

    def test_inconsistent_payload_and_blackout_are_rejected(self):
        frame, center = cylinder_frame()
        initial = estimate_column_geometry(frame, 'red', 'cylinder', table_plane=np.zeros(3))
        robot = {'tool_position': center, 'tool_rotation': np.eye(3)}
        wrong_robot = {'tool_position': center + [.025, 0, 0], 'tool_rotation': np.eye(3)}
        with self.assertRaises(ValueError):
            update_partial_column_geometry(frame, 'red', 'cylinder', initial['model'], initial,
                                           robot, wrong_robot, np.zeros(3))
        frame['valid_depth'][:] = False
        with self.assertRaises(ValueError):
            estimate_column_geometry(frame, 'red', 'cylinder', table_plane=np.zeros(3))

    def test_real_small_cylinder_is_resolved_by_curvature_not_color_alone(self):
        with np.load(Path(__file__).parent / 'fixtures/column_blue_cylinder_rgbd.npz') as archive:
            frame = dict(archive)
        tracker = InstanceTracker()
        tracker.update(frame)
        blue = next(t for t in tracker.tracks.values() if t['color'] == 'blue')
        self.assertEqual(blue['shape'], 'unknown')
        ids = select_column_levels(tracker, [{'color': 'blue', 'shape': 'cylinder'}], frame)
        self.assertEqual(ids, [blue['id']])
        fit = estimate_column_geometry(tracker.isolated_frame(frame, ids[0]), 'blue', 'cylinder',
                                       table_plane=tracker.plane)
        self.assertAlmostEqual(fit['model']['radius'], .015, delta=.0004)
        with self.assertRaisesRegex(ValueError, 'visually supported'):
            select_column_levels(tracker, [{'color': 'yellow', 'shape': 'cylinder'}], frame)
        with self.assertRaises(ValueError):
            select_column_levels(tracker, [{'color': 'red', 'shape': 'cube'}], frame)

    def test_cuboid_known_dimensions_use_prior_only_on_hidden_axes(self):
        dims = np.array([.017, .031, .053])
        frame, center = cube_frame(np.eye(3), hide_top=True, half_size=dims / 2)
        model = {'height': dims[2], 'dims': dims[:2]}
        prior = {'center': center + [.002, -.002, .001], 'normal': np.array([0., 0., 1.])}
        fit = estimate_column_geometry(frame, 'red', 'thin_prism', model, np.zeros(3), prior)
        self.assertFalse(fit['geometry_reliable'])
        np.testing.assert_allclose(fit['center'][:2], center[:2], atol=.0003)
        self.assertAlmostEqual(fit['center'][2], prior['center'][2], delta=.0003)


    def test_single_face_cuboid_retains_unobservable_tilt_from_robot_prediction(self):
        dims = np.array([.017, .031, .053])
        model = {'height': dims[2], 'dims': dims[:2]}
        initial_frame, center = cube_frame(np.eye(3), half_size=dims / 2)
        initial = estimate_column_geometry(initial_frame, 'red', 'thin_prism', model, np.zeros(3))
        rotation = Rotation.from_euler('xyz', [-.11, .04, .13]).as_matrix()
        frame, _ = cube_frame(rotation, half_size=dims / 2)
        points, _ = inventory_points(frame)
        local = (points - center) @ rotation
        # Retain only the +X side. Tilt about that side's normal cannot be
        # recovered from this observation; it must remain a propagated prior.
        visible = (frame['rgb'][..., 1] == 30) & (abs(local[..., 0] - dims[0] / 2) < 1e-7)
        frame['rgb'][(frame['rgb'][..., 1] == 30) & ~visible] = 0
        robot_before = {'tool_position': center, 'tool_rotation': np.eye(3)}
        robot = {'tool_position': center + .002 * rotation[:, 0], 'tool_rotation': rotation}
        fit = update_partial_column_geometry(frame, 'red', 'thin_prism', model, initial,
                                             robot_before, robot, np.zeros(3))
        self.assertEqual(fit['observed_face_count'], 1)
        self.assertTrue(fit['inferred_orientation'])
        self.assertFalse(fit['geometry_reliable'])
        np.testing.assert_allclose(fit['normal'], rotation[:, 2], atol=.0001)
        np.testing.assert_allclose(fit['center'], center, atol=.0003)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        expected = center + (signs * dims / 2) @ rotation.T
        distances = np.linalg.norm(fit['vertices'][:, None] - expected[None], axis=2)
        self.assertLess(float(np.max(np.min(distances, axis=1))), .0003)

    def test_full_finite_edge_corrects_large_warm_grasp_slip(self):
        frame, center = cube_frame(np.eye(3), hide_top=True, half_size=np.repeat(.015, 3))
        model = {'height': .03, 'dims': np.repeat(.03, 2)}
        # Jaw closure can move a cube before its first held visual confirmation.
        # The top plane is hidden, but the visible sides cover its full height.
        prior = {'center': center + [0., 0., .013], 'normal': np.array([0., 0., 1.])}
        fit = estimate_column_geometry(frame, 'red', 'cube', model, np.zeros(3), prior)
        np.testing.assert_allclose(fit['center'], center, atol=.0003)
        self.assertEqual(fit['observed_face_count'], 2)
        self.assertEqual(np.count_nonzero(fit['constrained_axes']), 2)
        self.assertEqual(np.count_nonzero(fit['finite_constrained_axes']), 1)
        self.assertTrue(fit['inferred_axial_position'])
        self.assertFalse(fit['geometry_reliable'])
        self.assertLessEqual(float(np.max(fit['finite_center_width_m'][~fit['constrained_axes']])), .0015)
        np.testing.assert_allclose(fit['model']['dims'], model['dims'])

    def test_narrow_finite_edge_initializes_known_cube_without_prior(self):
        frame, center = cube_frame(np.eye(3), hide_top=True, half_size=np.repeat(.015, 3))
        fit = estimate_column_geometry(frame, 'red', 'cube',
                                       {'height': .03, 'dims': np.repeat(.03, 2)}, np.zeros(3))
        np.testing.assert_allclose(fit['center'], center, atol=.0003)
        self.assertEqual(fit['observed_face_count'], 2)
        self.assertTrue(np.all(fit['constrained_axes'] | fit['finite_constrained_axes']))
        self.assertTrue(fit['inferred_axial_position'])

    def test_short_partial_edge_retains_hidden_axis_ambiguity(self):
        frame, center = cube_frame(np.eye(3), hide_top=True, half_size=np.repeat(.015, 3))
        points, _ = inventory_points(frame)
        red = frame['rgb'][..., 1] == 30
        frame['rgb'][red & (abs(points[..., 2] - center[2]) > .004)] = 0
        model = {'height': .03, 'dims': np.repeat(.03, 2)}
        prior = {'center': center + [0., 0., .004], 'normal': np.array([0., 0., 1.])}
        fit = estimate_column_geometry(frame, 'red', 'cube', model, np.zeros(3), prior)
        self.assertEqual(fit['observed_face_count'], 2)
        self.assertFalse(np.any(fit['finite_constrained_axes']))
        self.assertTrue(fit['inferred_axial_position'])
        self.assertFalse(fit['geometry_reliable'])
        np.testing.assert_allclose(fit['center'], prior['center'], atol=.0003)
        self.assertGreater(float(np.min(fit['finite_center_width_m'][~fit['constrained_axes']])), .02)
        with self.assertRaisesRegex(ValueError, 'without a prior'):
            estimate_column_geometry(frame, 'red', 'cube', model, np.zeros(3))

    def test_single_face_clips_inconsistent_prior_without_claiming_full_pose(self):
        frame, center = cube_frame(np.eye(3), half_size=np.repeat(.015, 3))
        points, _ = inventory_points(frame)
        red = frame['rgb'][..., 1] == 30
        frame['rgb'][red & (abs(points[..., 0] - center[0] - .015) > 1e-7)] = 0
        model = {'height': .03, 'dims': np.repeat(.03, 2)}
        prior = {'center': center + [0., .013, 0.], 'normal': np.array([0., 0., 1.]),
                 'rotation': np.eye(3)}
        fit = estimate_column_geometry(frame, 'red', 'cube', model, np.zeros(3), prior)
        self.assertEqual(fit['observed_face_count'], 1)
        self.assertTrue(fit['inferred_orientation'])
        self.assertTrue(fit['inferred_axial_position'])
        self.assertFalse(fit['geometry_reliable'])
        self.assertLess(abs(fit['center'][1] - center[1]), .003)
        self.assertGreater(fit['finite_center_width_m'][1], .003)
        self.assertFalse(fit['finite_constrained_axes'][1])

    def test_real_sparse_cube_frames_reject_edge_samples_without_hiding_wrong_sizes(self):
        fixtures = [('column_cube_edge_907_rgbd.npz', 1),
                    ('column_cube_edge_908_rgbd.npz', 1),
                    ('column_cube_double_edge_rgbd.npz', 2)]
        for filename, outliers in fixtures:
            with self.subTest(frame=filename):
                frame = dict(np.load(Path(__file__).parent / 'fixtures' / filename))
                height = float(frame.pop('learned_height_m'))
                plane = frame.pop('learned_table_plane')
                model = {'height': height, 'dims': np.repeat(height, 2)}
                # These real frames have two precise interior faces and only
                # 61--72 colored pixels. One or two silhouette depth samples
                # lie 1.91--5.85 mm in front of an observed outward face plane.
                fit = estimate_column_geometry(frame, 'green', 'cube', model, plane)
                self.assertEqual(fit['observed_face_count'], 2)
                self.assertEqual(fit['finite_surface_excluded_point_count'], outliers)
                self.assertLess(fit['surface_rms_m'], .000005)
                self.assertEqual(np.count_nonzero(fit['finite_constrained_axes']), 1)
                center = fit['center'] @ fit['rotation']
                bounds = fit['finite_center_bounds_m']
                self.assertTrue(np.all(center >= bounds[:, 0]))
                self.assertTrue(np.all(center <= bounds[:, 1]))
                # Robust trimming must not turn contradictory finite geometry
                # into an ordinary partial/occluded observation.
                wrong = {'height': height - .005, 'dims': np.repeat(height - .005, 2)}
                with self.assertRaisesRegex(ColumnGeometryConflict, 'contradicts learned'):
                    estimate_column_geometry(frame, 'green', 'cube', wrong, plane)

    def test_later_real_thin_prism_faces_refine_only_pixel_uncertain_widths(self):
        for seed, color in ((903, 'green'), (904, 'yellow')):
            with self.subTest(seed=seed):
                root = Path(__file__).parent / 'fixtures'
                initial = dict(np.load(root / f'column_thin_width_{seed}_initial.npz'))
                later = dict(np.load(root / f'column_thin_width_{seed}_later.npz'))
                plane = initial['learned_table_plane']
                fixed = {'height': float(initial['learned_height_m']),
                         'dims': initial['learned_dims_m'].copy()}
                prior = None
                if seed == 904:
                    prior = json.loads((root / 'column_thin_width_904_prior.json').read_text())
                    for field in ('center', 'normal', 'rotation', 'axis_dimensions'):
                        prior[field] = np.asarray(prior[field])
                # The same trusted surface must reject an immutable wrong size.
                with self.assertRaises(ColumnGeometryConflict):
                    estimate_column_geometry(later, color, 'thin_prism', fixed, plane, prior)
                model = dict(fixed, **column_size_uncertainty(initial, color, fixed, plane))
                before = model['dims'].copy()
                before_prior = None if prior is None else prior['axis_dimensions'].copy()
                fit = estimate_column_geometry(later, color, 'thin_prism', model, plane, prior)
                self.assertEqual(fit['model']['dimension_refinement_count'], 1)
                self.assertGreater(np.max(fit['dims'] - before), .00003)
                self.assertLess(np.max(fit['dims'] - before), .001)
                self.assertEqual(fit['height'], model['height'])
                self.assertTrue(np.all(fit['dims'] <= model['dims_upper_m']))
                np.testing.assert_array_equal(model['dims'], before)
                np.testing.assert_array_equal(fit['model']['dims_upper_m'], model['dims_upper_m'])
                if prior is not None:
                    np.testing.assert_array_equal(prior['axis_dimensions'], before_prior)
                expected_axes = np.r_[fit['dims'], fit['height']][fit['axis_dimension_indices']]
                np.testing.assert_allclose(fit['axis_dimensions'], expected_axes)
                again = estimate_column_geometry(later, color, 'thin_prism', fit['model'], plane, fit)
                np.testing.assert_array_equal(again['dims'], fit['dims'])
                self.assertEqual(again['model']['dimension_refinement_count'], 1)
                # Resolution uncertainty is a hard interval, not permission to
                # enlarge dimensions indefinitely or alter the measured height.
                narrow = dict(model, dims_upper_m=before + .00001)
                with self.assertRaises(ColumnGeometryConflict):
                    estimate_column_geometry(later, color, 'thin_prism', narrow, plane, prior)
                wrong = {'height': fixed['height'], 'dims': before - .005}
                with self.assertRaises(ColumnGeometryConflict):
                    estimate_column_geometry(later, color, 'thin_prism', wrong, plane)
                wrong_height = dict(model, height=model['height'] - .005)
                with self.assertRaises(ColumnGeometryConflict):
                    estimate_column_geometry(later, color, 'thin_prism', wrong_height, plane)

    def test_visible_three_dimensional_extent_rejects_contradictory_model(self):
        dimensions = np.array([.03, .04, .05])
        frame, center = cube_frame(np.eye(3), half_size=dimensions / 2)
        prior = {'center': center, 'normal': np.array([0., 0., 1.])}
        for axis in range(3):
            wrong = dimensions.copy()
            wrong[axis] -= .005
            with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, 'contradicts learned'):
                estimate_column_geometry(frame, 'red', 'prism',
                                         {'height': wrong[2], 'dims': wrong[:2]}, np.zeros(3), prior)

    def test_rotated_non_catalog_cuboid_keeps_learned_axis_lengths(self):
        dimensions = np.array([.017, .031, .053])
        model = {'height': dimensions[2], 'dims': dimensions[:2]}
        initial_frame, center = cube_frame(np.eye(3), half_size=dimensions / 2)
        initial = estimate_column_geometry(initial_frame, 'red', 'thin_prism', model, np.zeros(3))
        before = {'tool_position': center, 'tool_rotation': np.eye(3)}
        for angles in ([.08, -.06, 1.0], [0., 0., np.pi / 2], [.4, .3, -1.2]):
            rotation = Rotation.from_euler('xyz', angles).as_matrix()
            frame, _ = cube_frame(rotation, half_size=dimensions / 2)
            robot = {'tool_position': center + [.002, -.002, .001], 'tool_rotation': rotation}
            with self.subTest(angles=angles):
                # The model's axes can be reordered by a different visible face
                # cluster without changing the physical lengths attached to them.
                for permutation in (np.array([0, 1, 2]), np.array([1, 2, 0])):
                    prior = dict(initial, rotation=initial['rotation'][:, permutation],
                                 axis_dimensions=initial['axis_dimensions'][permutation])
                    fit = update_partial_column_geometry(frame, 'red', 'thin_prism', model, prior,
                                                         before, robot, np.zeros(3))
                    np.testing.assert_allclose(fit['center'], center, atol=.0003)
                    np.testing.assert_allclose(np.sort(fit['axis_dimensions']), np.sort(dimensions))
                    np.testing.assert_allclose(fit['model']['dims'], dimensions[:2])
                    signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
                    expected = center + (signs * dimensions / 2) @ rotation.T
                    distance = np.linalg.norm(fit['vertices'][:, None] - expected[None], axis=2)
                    self.assertLess(float(np.max(np.min(distance, axis=1))), .0003)

    def test_equal_cube_axes_keep_existing_fit_bitwise_unchanged(self):
        rotation = Rotation.from_euler('xyz', [.1, -.08, .2]).as_matrix()
        frame, center = cube_frame(rotation, half_size=np.repeat(.015, 3))
        model = {'height': .03, 'dims': np.repeat(.03, 2)}
        original = estimate_column_geometry(frame, 'red', 'cube', model, np.zeros(3))
        prior = dict(original, center=center + [.002, -.002, .001])
        updated = estimate_column_geometry(frame, 'red', 'cube', model, np.zeros(3), prior)
        for field in ('center', 'vertices', 'normal', 'axis_dimensions', 'top_vertices'):
            np.testing.assert_array_equal(updated[field], original[field])

    def test_rectangular_axis_ambiguity_without_orientation_prior_is_rejected(self):
        dimensions = np.array([.017, .031, .053])
        rotation = Rotation.from_euler('xyz', [.08, -.06, 1.0]).as_matrix()
        frame, _ = cube_frame(rotation, half_size=dimensions / 2)
        # The old world-axis size assignment cannot establish which learned
        # rectangle edge rotated into this view. It must not silently resize it.
        with self.assertRaisesRegex(ValueError, 'contradicts learned'):
            estimate_column_geometry(frame, 'red', 'thin_prism',
                                     {'height': dimensions[2], 'dims': dimensions[:2]}, np.zeros(3))

    def test_cube_shape_corrects_narrow_legacy_silhouette_dimensions(self):
        frame, center = cube_frame(np.eye(3), half_size=np.repeat(.015, 3))
        model = {'height': .03, 'dims': np.array([.025, .027])}
        fit = estimate_column_geometry(frame, 'red', 'cube', model, np.zeros(3))
        np.testing.assert_allclose(fit['dims'], [.03, .03])
        np.testing.assert_allclose(fit['center'], center, atol=.0003)


if __name__ == '__main__':
    unittest.main()
