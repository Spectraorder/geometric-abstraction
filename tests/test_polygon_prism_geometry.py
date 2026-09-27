"""Calibrated pixel fixtures for the isolated convex-prism session extension."""
from pathlib import Path
import ast
import sys
import unittest

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_polygon_prism_session import library
from build_geometry_session import library as legacy_library
from test_cube_geometry import cube_frame


def footprint(kind='hexagon'):
    if kind == 'hexagon':
        angle = np.arange(6)*np.pi/3
        return .022*np.column_stack((np.cos(angle), np.sin(angle)))
    return np.array([[-.025, -.018], [.025, -.018], [.014, .018], [-.014, .018]])


def prism_frame(polygon=None, height=.032, center=None, rotation=None, stamp=1., resolution_scale=1):
    """Intersect rays with polygon halfspaces; truth stays inside this fixture."""
    polygon = footprint() if polygon is None else np.asarray(polygon)
    center = np.array([.01, -.02, height/2]) if center is None else np.asarray(center)
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    frame, _ = cube_frame(np.eye(3))
    if resolution_scale != 1:
        rows, cols = frame['depth_m'].shape
        rows, cols = rows*resolution_scale, cols*resolution_scale
        frame['depth_m'] = np.empty((rows, cols))
        frame['rgb'] = np.empty((rows, cols, 3), dtype=np.uint8)
        frame['intrinsics'][:2, :2] *= resolution_scale
        frame['intrinsics'][:2, 2] = (frame['intrinsics'][:2, 2]+.5)*resolution_scale-.5
    camera = frame['camera_to_world'][:3, 3]
    v, u = np.indices(frame['depth_m'].shape)
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(frame['intrinsics']).T
    rays = rays @ frame['camera_to_world'][:3, :3].T
    origin, direction = (camera-center)@rotation, rays@rotation
    boundary = ConvexHull(polygon).equations
    normals = np.concatenate((np.column_stack((boundary[:, :2], np.zeros(len(boundary)))),
                              [[0., 0., 1.], [0., 0., -1.]]))
    offsets = np.r_[boundary[:, 2], -height/2, -height/2]
    denominator = direction@normals.T
    numerator = -(origin@normals.T+offsets)
    with np.errstate(divide='ignore', invalid='ignore'):
        crossings = numerator/denominator
    entry = np.max(np.where(denominator < -1e-12, crossings, -np.inf), axis=2)
    leave = np.min(np.where(denominator > 1e-12, crossings, np.inf), axis=2)
    parallel_outside = np.any((abs(denominator) <= 1e-12) & (numerator < 0), axis=2)
    table = -camera[2]/rays[..., 2]
    hit = (entry > 0) & (entry < leave) & (entry < table) & ~parallel_outside
    frame['depth_m'] = np.where(hit, entry, table)
    frame['valid_depth'] = np.isfinite(frame['depth_m']) & (frame['depth_m'] > 0)
    frame['rgb'][:] = [160, 120, 70]
    frame['rgb'][hit] = [210, 30, 30]
    frame['time'] = stamp
    return frame


def generated_footprints(seed=20260927):
    """Generate convex boundaries without selecting named shape templates.

    The small radial/angular perturbations keep features resolvable at the test
    resolution. The fitter receives only rendered pixels, not this generator or N.
    """
    random = np.random.default_rng(seed)
    for count in range(3, 13):
        for repeat in range(2):
            step = 2*np.pi/count
            angles = step*(np.arange(count)+random.uniform(-.12, .12, count))
            radii = random.uniform(.025, .033)*random.uniform(.97, 1.03, count)
            polygon = radii[:, None]*np.column_stack((np.cos(angles), np.sin(angles)))
            polygon = polygon@np.array([[random.uniform(.9, 1.1), random.uniform(-.12, .12)],
                                        [0., random.uniform(.9, 1.1)]])
            hull = ConvexHull(polygon)
            polygon = polygon[hull.vertices]
            yield polygon, random.uniform(.015, .055), random.uniform(-np.pi, np.pi)


class PrismPixelAPI:
    """Kinematic API contract fixture, not a physical grasp-success model."""
    def __init__(self, polygon):
        self.polygon = polygon
        self.center = np.array([.01, -.02, .016])
        self.position = np.array([.01, -.02, .12])
        self.now = 1.
        self.attached = False
        self.records, self.actions = [], []

    def task(self):
        return {'nodes': [{'color': 'red', 'shape': 'polygon_prism'}]}

    def camera(self, view='side'):
        return prism_frame(self.polygon, center=self.center, stamp=self.now)

    def proprioception(self):
        return {'tool_position': self.position.copy(), 'time': self.now}

    def gripper_geometry(self):
        return {'time': self.now, 'coordinate_frame': 'world', 'units': 'meters',
                'source': 'robot_model_and_joint_encoders', 'visual_spheres': []}

    def move_tool(self, point, seconds=1.5, **kwargs):
        point = np.asarray(point)
        if self.attached:
            self.center += point-self.position
        self.position = point.copy()
        self.now += seconds
        self.actions.append(('move', point.tolist(), seconds))

    def gripper(self, value, seconds):
        self.attached = value < 0
        if not self.attached:
            self.center[2] = .016
        self.now += seconds
        self.actions.append(('gripper', value, seconds))

    def wait(self, seconds):
        self.now += seconds
        self.actions.append(('wait', seconds))

    def record_geometry(self, record):
        self.records.append(record)


class PolygonPrismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = {}
        exec(library('symbolic'), cls.ns)

    def fit(self, frame, model=None):
        return self.ns['observe_polygon_prism'](frame, 'red', model, np.zeros(3))

    def test_non_catalog_caps_recover_vertices_and_area_centroid(self):
        for kind, count in (('hexagon', 6), ('trapezoid', 4)):
            polygon = footprint(kind)
            fit = self.fit(prism_frame(polygon))
            area, centroid, _ = self.ns['point_polygon_moments'](polygon)
            expected = np.array([.01, -.02, .016])+np.r_[centroid, 0.]
            with self.subTest(kind=kind):
                self.assertEqual(len(fit['top_polygon_m']), count)
                np.testing.assert_allclose(fit['center'], expected, atol=.001)
                self.assertAlmostEqual(fit['height'], .032, delta=.0005)
                truth = np.column_stack((polygon, np.repeat(.032, count)))+[.01, -.02, 0.]
                distances = np.linalg.norm(fit['top_polygon_m'][:, None]-truth[None], axis=2)
                self.assertLess(np.max(np.min(distances, axis=1)), .0025)
                self.assertLess(abs(fit['model']['cap_area_m2']-area), area*.12)

    def test_current_pixels_update_translation_and_tilt_without_pose_prior(self):
        polygon = footprint('trapezoid')
        model = self.fit(prism_frame(polygon))['model']
        rotation = Rotation.from_euler('xyz', [.10, -.06, .45]).as_matrix()
        center = np.array([-.025, .005, .075])
        moved = self.fit(prism_frame(polygon, center=center, rotation=rotation, stamp=2.), model)
        _, centroid, _ = self.ns['point_polygon_moments'](polygon)
        np.testing.assert_allclose(moved['center'], center+rotation@np.r_[centroid, 0.], atol=.001)
        np.testing.assert_allclose(moved['normal'], rotation[:, 2], atol=.001)
        self.assertEqual(moved['observation_time'], 2.)
        self.assertEqual(moved['height'], model['height'])

    def test_missing_depth_is_rejected_but_supported_occlusion_can_update(self):
        initial = prism_frame()
        model = self.fit(initial)['model']
        points, _ = self.ns['inventory_points'](initial)
        cap = (initial['rgb'][..., 1] == 30) & (abs(points[..., 2]-.032) < 1e-6)
        for kind in ('missing_depth', 'nearer_occluder'):
            frame = {key: value.copy() if isinstance(value, np.ndarray) else value
                     for key, value in initial.items()}
            cut = ndimage.binary_dilation(cap) & (points[..., 0] > .021)
            if kind == 'missing_depth':
                frame['valid_depth'][cut] = False
            else:
                frame['rgb'][cut] = [30, 40, 210]
                frame['depth_m'][cut] *= .94
            if kind == 'missing_depth':
                with self.assertRaisesRegex(ValueError, 'boundary|planar'):
                    self.fit(frame, model)
            else:
                # Initialization still needs a complete cap. A current partial
                # contour can register only an already measured shape.
                with self.assertRaisesRegex(ValueError, 'occluded'):
                    self.fit(frame)
                fitted = self.fit(frame, model)
                np.testing.assert_allclose(fitted['center'], self.fit(initial)['center'], atol=.001)
                self.assertIn('partial_edges', fitted['pose_update_mode'])
                self.assertGreaterEqual(fitted['registration_supported_edges'], 2)

    def test_visible_concavity_and_image_crop_are_not_completed_as_convex_shapes(self):
        horizontal = prism_frame(np.array([[-.03, -.025], [.03, -.025], [.03, -.005], [-.03, -.005]]))
        vertical = prism_frame(np.array([[-.03, -.025], [-.008, -.025], [-.008, .025], [-.03, .025]]))
        closer = vertical['depth_m'] < horizontal['depth_m']
        horizontal['depth_m'][closer] = vertical['depth_m'][closer]
        horizontal['rgb'][closer] = vertical['rgb'][closer]
        with self.assertRaisesRegex(ValueError, 'concave|straight|contour|footprint'):
            self.fit(horizontal)
        frame = prism_frame()
        occupied = np.where(frame['rgb'][..., 1] == 30)[1]
        edge = int(np.median(occupied))
        for field in ('rgb', 'depth_m', 'valid_depth'):
            frame[field] = frame[field][:, :edge]
        with self.assertRaisesRegex(ValueError, 'clipped'):
            self.fit(frame)

    def test_invalid_polygon_cannot_update_ledger(self):
        fit = self.fit(prism_frame())
        scene = self.ns['MetricScene']()
        scene.observe('red', fit)
        for field, value in (('top_polygon_m', fit['top_polygon_m'][::-1]),
                             ('axis_dimensions', fit['axis_dimensions']*2),
                             ('bottom', fit['bottom']-.01)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                scene.observe('red', dict(fit, observation_time=2., **{field: value}))
            self.assertEqual(scene.get('red')['observation_time'], 1.)

    def test_pixel_noise_and_yaw_do_not_require_a_shape_catalog(self):
        random = np.random.default_rng(42)
        for kind in ('hexagon', 'trapezoid'):
            polygon = footprint(kind)
            model = self.fit(prism_frame(polygon))['model']
            for yaw in np.linspace(0, 2*np.pi, 8, endpoint=False):
                rotation = Rotation.from_euler('z', yaw).as_matrix()
                frame = prism_frame(polygon, rotation=rotation)
                frame['depth_m'] += random.normal(0, .00004, frame['depth_m'].shape)
                with self.subTest(kind=kind, yaw=yaw):
                    fit = self.fit(frame, model)
                    _, centroid, _ = self.ns['point_polygon_moments'](polygon)
                    expected = np.array([.01, -.02, .016])+rotation@np.r_[centroid, 0.]
                    np.testing.assert_allclose(fit['center'], expected, atol=.0015)
                    self.assertEqual(len(fit['top_polygon_m']), len(polygon))
                    self.assertEqual(fit['edges_m'].shape, (3*len(polygon), 2, 3))

    def test_generated_boundaries_need_no_named_shape_or_vertex_count(self):
        observer = self.ns['MixedPnPObservation']([{'color': 'red', 'shape': 'polygon_prism'}])
        recovered_counts = set()
        for index, (polygon, height, yaw) in enumerate(generated_footprints()):
            rotation = Rotation.from_euler('z', yaw).as_matrix()
            center = np.array([.01, -.02, height/2])
            frame = prism_frame(polygon, height, center, rotation, resolution_scale=3)
            with self.subTest(index=index, vertices=len(polygon)):
                # The same public shape-family descriptor is used in every case.
                fit = observer.observe(frame, 'red', None, np.zeros(3))
                _, centroid, _ = self.ns['point_polygon_moments'](polygon)
                expected = center+rotation@np.r_[centroid, 0.]
                self.assertEqual(len(fit['top_polygon_m']), len(polygon))
                np.testing.assert_allclose(fit['center'], expected, atol=.001)
                self.assertAlmostEqual(fit['height'], height, delta=.0005)
                recovered_counts.add(len(fit['top_polygon_m']))
                scene = self.ns['MetricScene']()
                scene.observe('red', fit)
                face = self.ns['SymbolicScene'](scene).reference('red.top_face')
                self.assertEqual(len(face['vertices_m']), len(polygon))
                # Refitting an unseen contour at a new pose also uses one path.
                moved_center = center+[-.015, .015, .025]
                moved_rotation = Rotation.from_euler('z', yaw+.28).as_matrix()
                moved = observer.observe(prism_frame(polygon, height, moved_center, moved_rotation,
                                                     stamp=2., resolution_scale=3),
                                         'red', fit['model'], np.zeros(3))
                np.testing.assert_allclose(moved['center'], moved_center+moved_rotation@np.r_[centroid, 0.], atol=.001)
        self.assertEqual(recovered_counts, set(range(3, 13)))

    def test_wrong_height_or_shape_model_is_rejected(self):
        frame = prism_frame()
        model = self.fit(frame)['model']
        for changed in (dict(model, height=.009), dict(model, cap_area_m2=.01),
                        dict(model, kind='legacy')):
            with self.assertRaises(ValueError):
                self.fit(frame, changed)

    def test_initial_tilt_is_not_mistaken_for_table_supported_extrusion(self):
        rotation = Rotation.from_euler('x', .12).as_matrix()
        with self.assertRaisesRegex(ValueError, 'rest upright'):
            self.fit(prism_frame(center=[.01, -.02, .04], rotation=rotation))

    def test_polygon_references_alignment_and_staleness_keep_existing_contract(self):
        scene = self.ns['MetricScene']()
        fit = self.fit(prism_frame())
        scene.observe('red', fit)
        symbolic = self.ns['SymbolicScene'](scene)
        top, bottom = symbolic.reference('red.top_face'), symbolic.reference('red.bottom_face')
        self.assertEqual(top['type'], 'plane_patch')
        self.assertEqual(len(top['vertices_m']), 6)
        self.assertIn('inferred', bottom['semantics'])
        np.testing.assert_allclose(symbolic.reference('red.axis')['direction'], [0, 0, 1], atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'box corners'):
            symbolic.reference('red.corner_0')
        with self.assertRaisesRegex(ValueError, 'symmetry'):
            symbolic.reference('red.long_axis')
        goal = symbolic.align('red.bottom_face', [.02, .03, .05], [0, 0, -1])
        np.testing.assert_allclose(goal['rotation']@bottom['center_m']+goal['translation'], [.02, .03, .05])
        self.assertFalse(goal['execution_or_contact_certified'])
        scene.unavailable('red', 2., 'occluded')
        with self.assertRaisesRegex(ValueError, 'No current observation'):
            symbolic.reference('red.center')

    def test_support_uses_polygon_not_its_bounding_box(self):
        from test_mixed_scene import box
        fit = self.fit(prism_frame())
        scene = self.ns['MetricScene']()
        scene.observe('red', fit)
        # This small cube is inside the enclosing rectangle but outside the hexagon.
        scene.observe('blue', box(center=[.030, -.003, .033], dimensions=(.002, .002, .002)))
        symbolic = self.ns['SymbolicScene'](scene)
        result = symbolic.relation('support_candidate', 'blue', 'red')
        self.assertEqual(result['component_truth']['overlap'], 'false')
        self.assertEqual(symbolic.relation('contact', 'blue', 'red')['truth'], 'unknown')

    def test_known_shape_fixture_remains_identical(self):
        original = {}
        exec(legacy_library('symbolic'), original)
        with np.load(ROOT/'tests/fixtures/cuboid_correspondence_rgbd.npz') as data:
            frame = {key[8:]: data[key] for key in data.files if key.startswith('initial_')}
        objects = [{'color': 'green', 'shape': 'thin_prism', 'orientation': 'flat_x'}]
        packets = []
        for namespace in (original, self.ns):
            measured = namespace['MixedPnPObservation'](objects, correspondence=True).observe(
                frame, 'green', None, namespace['inventory_table'](frame))
            scene = namespace['MetricScene']()
            scene.observe('green', measured)
            symbolic = namespace['SymbolicScene'](scene)
            references = [symbolic.reference('green.'+role) for role in ('center', 'top_face', 'bottom_face')]
            checkpoint = namespace['GeometryStepState']('green').update(
                'inspect', measured, measured['observation_time'], references)
            packets.append({'compact': symbolic.packet(), 'detailed': symbolic.packet(details=True),
                            'checkpoint': checkpoint})
        self.assertEqual(packets[0], packets[1])

    def test_points_connectivity_and_existing_reference_call_agree(self):
        fit = self.fit(prism_frame())
        scene = self.ns['MetricScene']()
        scene.observe('red', fit)
        symbolic = self.ns['SymbolicScene'](scene)
        count = len(fit['top_polygon_m'])
        for index, point in enumerate(fit['vertices']):
            ref = symbolic.reference('red.vertex_'+str(index))
            np.testing.assert_array_equal(ref['position_m'], point)
            self.assertIn('current_observation_only', ref['id_scope'])
            self.assertEqual(ref['vertex_sources'], [fit['vertex_sources'][index]])
        for index, vertices in enumerate(fit['edge_vertex_indices']):
            ref = symbolic.reference('red.edge_'+str(index))
            self.assertEqual(ref['type'], 'line_segment')
            np.testing.assert_array_equal(ref['endpoints_m'], fit['vertices'][vertices])
        for index, vertices in enumerate(fit['face_vertex_indices']):
            ref = symbolic.reference('red.face_'+str(index))
            np.testing.assert_array_equal(ref['vertices_m'], fit['vertices'][vertices])
            np.testing.assert_allclose((ref['vertices_m']-ref['center_m'])@ref['normal'], 0., atol=1e-9)
        top = symbolic.reference('red.top_face')
        np.testing.assert_allclose(symbolic.reference('red.face_0')['center_m'], top['center_m'])
        self.assertIn('inferred', symbolic.reference('red.face_1')['semantics'])
        self.assertIn('inferred', symbolic.reference('red.vertex_0')['vertex_sources'][0])
        self.assertIn('fitted', symbolic.reference('red.vertex_'+str(count))['vertex_sources'][0])
        for name in ('vertex_999', 'edge_-1', 'face_foo'):
            with self.assertRaises(ValueError):
                symbolic.reference('red.'+name)
        bad = dict(fit, edge_vertex_indices=fit['edge_vertex_indices'][::-1])
        with self.assertRaisesRegex(ValueError, 'connectivity'):
            scene.observe('red', bad)

    def test_point_reference_layer_has_no_shape_label_requirement(self):
        # A tetrahedral point set exercises the representation independently of
        # the current prism perception backend. This is not a perception claim.
        fit = {'vertices': np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
               'edge_vertex_indices': [[0, 1]], 'face_vertex_indices': [[0, 1, 2]],
               'vertex_sources': ['fitted_from_current_depth']*4}
        reference = self.ns['vertex_geometry_reference']
        np.testing.assert_array_equal(reference(fit, 'vertex_3')['position_m'], [0., 0., 1.])
        np.testing.assert_array_equal(reference(fit, 'edge_0')['endpoints_m'], fit['vertices'][[0, 1]])
        np.testing.assert_allclose(reference(fit, 'face_0')['normal'], [0., 0., 1.])
        fit['face_vertex_indices'] = [[0, 1, 2, 3]]
        with self.assertRaisesRegex(ValueError, 'not coplanar'):
            reference(fit, 'face_0')
        fit['vertices'][2] = [2., 0., 0.]
        fit['face_vertex_indices'] = [[0, 1, 2]]
        with self.assertRaisesRegex(ValueError, 'do not define a plane'):
            reference(fit, 'face_0')

    def test_named_faces_and_detailed_packets_share_the_point_representation(self):
        fit = self.fit(prism_frame())
        scene = self.ns['MetricScene']()
        scene.observe('red', fit)
        symbolic = self.ns['SymbolicScene'](scene)
        for alias, indexed in (('top_face', 'face_0'), ('bottom_face', 'face_1')):
            named, generic = symbolic.reference('red.'+alias), symbolic.reference('red.'+indexed)
            for field in ('center_m', 'normal', 'vertices_m', 'vertex_indices'):
                np.testing.assert_array_equal(named[field], generic[field])
            self.assertEqual(named['vertex_sources'], generic['vertex_sources'])
            self.assertEqual(named['id_scope'], generic['id_scope'])
        compact = scene.packet()['objects']['red']
        self.assertEqual(compact['observability_condition'], fit['observability_condition'])
        for key in ('vertices', 'edge_vertex_indices', 'face_vertex_indices', 'vertex_sources'):
            self.assertNotIn(key, compact)
        detailed = scene.packet(details=True)['objects']['red']
        np.testing.assert_array_equal(detailed['vertices'], fit['vertices'])
        np.testing.assert_array_equal(detailed['edge_vertex_indices'], fit['edge_vertex_indices'])
        self.assertEqual(detailed['face_vertex_indices'], fit['face_vertex_indices'])
        self.assertEqual(detailed['vertex_sources'], fit['vertex_sources'])
        for invalid in (dict(fit, edge_vertex_indices=fit['edge_vertex_indices'].astype(float)),
                        dict(fit, vertex_id_scope='persistent_material_points')):
            with self.assertRaises(ValueError):
                scene.observe('red', invalid)

    def test_compact_checkpoint_keeps_evidence_without_implicit_sensing(self):
        api = PrismPixelAPI(footprint())
        session = self.ns['GEOMETRY'].session(api, api.task()['nodes'])
        fit = session.get('red')
        count = len(fit['top_polygon_m'])
        def unexpected(*args, **kwargs):
            raise AssertionError('Read-only geometry query unexpectedly requested a camera')
        api.camera = unexpected
        refs = ('red.vertex_0', 'red.vertex_'+str(count), 'red.top_face', 'red.bottom_face')
        for repeat in range(2):
            session.get('red')
            for name in refs:
                session.reference(name)
            record = session.checkpoint('red', 'inspect', reference_names=refs)
            self.assertEqual(record, api.records[-1]['record'])
            self.assertEqual(record['geometry']['observability_condition'], fit['observability_condition'])
            self.assertEqual(record['geometry']['vertex_id_scope'], fit['vertex_id_scope'])
            points = record['requested_points']
            self.assertEqual(points['vertex_0']['evidence_sources'], ['inferred_parallel_bottom_under_prism_model'])
            self.assertEqual(points['vertex_'+str(count)]['evidence_sources'], ['fitted_from_current_rgbd_contour'])
            self.assertEqual(points['bottom_face']['evidence_sources'], points['vertex_0']['evidence_sources'])
            self.assertIn('inferred', points['vertex_0']['semantics'])
            for point in points.values():
                self.assertEqual(point['id_scope'], fit['vertex_id_scope'])
                self.assertNotIn('vertices_m', point)
                self.assertNotIn('vertex_indices', point)
            self.assertNotIn('vertices', record['geometry'])
        # Motion invalidates cached geometry; compact records must not retain it.
        api.now += 1.
        unavailable = session.checkpoint('red', 'after_motion', reference_names=refs)
        self.assertEqual(unavailable['status'], 'unavailable')
        self.assertNotIn('requested_points', unavailable)
        self.assertNotIn('geometry', unavailable)

    def test_shared_actuator_code_is_unchanged_in_each_bundle(self):
        shared = ast.parse((ROOT/'policies/pick_place_rgbd_shared.py').read_text())
        expected = next(node for node in shared.body if isinstance(node, ast.ClassDef)
                        and node.name == 'RGBDPickPlace')
        for arm in ('basic', 'symbolic'):
            bundle = ast.parse(library(arm))
            actual = next(node for node in bundle.body if isinstance(node, ast.ClassDef)
                          and node.name == 'RGBDPickPlace')
            self.assertEqual(ast.dump(expected), ast.dump(actual))

    def test_both_arms_use_same_pick_place_commands_with_current_rgbd(self):
        basic = {}
        exec(library('basic'), basic)
        for kind in ('hexagon', 'trapezoid'):
            actions = []
            for namespace in (basic, self.ns):
                api = PrismPixelAPI(footprint(kind))
                session = namespace['GEOMETRY'].session(api, api.task()['nodes'])
                picked = session.pick('red')
                self.assertEqual(picked['status'], 'lift_visually_confirmed')
                placed = session.place([-.025, -.025, 0.])
                self.assertFalse(placed['held'])
                _, centroid, _ = self.ns['point_polygon_moments'](api.polygon)
                np.testing.assert_allclose(api.center[:2]+centroid, [-.025, -.025], atol=.001)
                session.checkpoint('red', 'released', action_status=placed['status'])
                if namespace is basic:
                    self.assertNotIn('observation', picked)
                    self.assertEqual(api.records[-1]['record']['status'], 'visual_input_only')
                actions.append(api.actions)
            self.assertEqual(actions[0], actions[1])


if __name__ == '__main__':
    unittest.main()
