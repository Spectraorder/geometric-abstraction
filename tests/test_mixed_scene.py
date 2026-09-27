"""State provenance and symbolic geometry must preserve physical shape semantics."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from aspire_so100.metric_scene import MetricScene
from aspire_so100.symbolic_scene import SymbolicScene
from aspire_so100.pnp_observation import MixedPnPObservation
from aspire_so100.instance_geometry import inventory_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_mixed_scene import build


def box(center=(0., 0., .015), rotation=None, dimensions=(.03, .03, .03), stamp=1., shape='cube', axes=(True, True, True)):
    center = np.asarray(center); rotation = np.eye(3) if rotation is None else rotation
    dimensions = np.asarray(dimensions); signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    vertices = center+(signs*dimensions/2)@rotation.T
    axis = int(np.argmax(abs(rotation[2]))); normal = rotation[:, axis]*np.sign(rotation[2, axis])
    return {'shape': shape, 'center': center, 'rotation': rotation, 'axis_dimensions': dimensions,
            'height': float(dimensions[2]), 'dims': dimensions[:2], 'vertices': vertices,
            'normal': normal, 'top': float(vertices[:, 2].max()), 'bottom': float(vertices[:, 2].min()),
            'extent_z': float(np.ptp(vertices[:, 2])/2), 'constrained_axes': np.array(axes),
            'inferred_orientation': False, 'coordinate_frame': 'world', 'units': 'meters',
            'observation_time': stamp, 'source': 'analytic_test_geometry'}


class MixedSceneTests(unittest.TestCase):
    def test_stale_and_missing_poses_do_not_become_current_memory(self):
        scene = MetricScene(); scene.observe('a', box()); scene.observe('b', box(stamp=2.))
        with self.assertRaisesRegex(ValueError, 'No current observation'):
            scene.get('a')
        packet = scene.packet(); self.assertEqual(packet['objects']['a']['status'], 'stale')
        self.assertNotIn('center', packet['objects']['a'])
        scene.unavailable('a', 2., 'occluded')
        self.assertEqual(scene.packet()['objects']['a']['status'], 'unavailable')
        self.assertEqual(scene.history('a')['last_observation']['observation_time'], 1.)
        with self.assertRaisesRegex(ValueError, 'nondecreasing'):
            scene.observe('a', box(stamp=1.))

    def test_targets_remain_fixed_and_returned_arrays_cannot_mutate_state(self):
        scene = MetricScene(); initial = box(); scene.observe('a', initial)
        goal = scene.target('a', [.1, .2, .3]); goal['center'][0] = 10
        initial['center'][0] = 5
        current = scene.get('a'); current['center'][0] = 3
        np.testing.assert_allclose(scene.get('a')['center'], [0., 0., .015])
        scene.observe('a', box(center=[.01, 0., .015], stamp=2.))
        packet = scene.packet()
        np.testing.assert_allclose(packet['desired_states']['a']['center'], [.1, .2, .3])
        np.testing.assert_allclose(packet['objects']['a']['change']['estimated_translation_m'], [.01, 0., 0.])
        np.testing.assert_allclose(scene.goal_error('a')['translation_to_goal_m'], [.09, .2, .285])

    def test_conditional_prediction_is_separate_and_reports_disagreement(self):
        transform = np.eye(4); transform[0, 3] = .02
        scene = MetricScene(); scene.observe('a', box())
        for kind in ('desired_state', 'conditional_prediction'):
            with self.assertRaisesRegex(ValueError, 'cannot be ingested'):
                scene.observe('a', dict(box(stamp=2.), kind=kind))
        with self.assertRaisesRegex(ValueError, 'disabled'):
            scene.predict('a', transform, 2., 'payload remains grasped')
        scene = MetricScene(predictions=True); scene.observe('a', box())
        scene.predict('a', transform, 2., 'payload remains grasped')
        np.testing.assert_allclose(scene.get('a')['center'], [0., 0., .015])
        scene.observe('a', box(center=[.006, 0., .015], stamp=2.))
        row = scene.packet()['objects']['a']['prediction_residual']
        np.testing.assert_allclose(row['translation_residual_m'], [-.014, 0., 0.])
        self.assertFalse(row['condition_verified'])
        self.assertEqual(scene.packet()['conditional_predictions']['a']['kind'], 'conditional_prediction')

    def test_rotated_cuboid_face_uses_current_axis_extent(self):
        rotation = Rotation.from_euler('y', 90, degrees=True).as_matrix()
        fit = box(rotation=rotation, dimensions=(.061, .027, .013), shape='thin_prism')
        scene = MetricScene(); scene.observe('beam', fit); symbolic = SymbolicScene(scene)
        face = symbolic.reference('beam.top_face')
        self.assertAlmostEqual(face['center_m'][2], fit['center'][2]+.061/2)
        np.testing.assert_allclose(face['normal'], [0., 0., 1.], atol=1e-12)
        np.testing.assert_allclose(face['vertices_m'][:, 2], fit['vertices'][:, 2].max())
        face['center_m'][0] = 100
        self.assertLess(symbolic.reference('beam.top_face')['center_m'][0], 1.)

    def test_cylinder_has_disks_and_axis_but_no_physical_corners(self):
        fit = box(dimensions=(.03, .03, .06), shape='cylinder'); fit['model'] = {'radius': .015}
        scene = MetricScene(); scene.observe('cylinder', fit); symbolic = SymbolicScene(scene)
        face = symbolic.reference('cylinder.top_face')
        self.assertEqual(face['type'], 'disk'); self.assertNotIn('vertices_m', face)
        self.assertEqual(symbolic.reference('cylinder.axis')['type'], 'unoriented_axis')
        with self.assertRaisesRegex(ValueError, 'physical corners'):
            symbolic.reference('cylinder.corner_0')
        scene.observe('cube', box())
        with self.assertRaisesRegex(ValueError, 'unique long_axis'):
            symbolic.reference('cube.long_axis')

    def test_relations_keep_missing_axes_and_contact_unknown(self):
        scene = MetricScene(); scene.observe('lower', box()); scene.observe('upper', box(center=[0., 0., .045]))
        symbolic = SymbolicScene(scene)
        self.assertEqual(symbolic.relation('support_candidate', 'upper', 'lower')['truth'], 'true')
        self.assertEqual(symbolic.relation('contact', 'upper', 'lower')['truth'], 'unknown')
        scene.observe('upper', box(center=[0., 0., .045], axes=(False, True, True)))
        self.assertEqual(symbolic.relation('aligned_xy', 'upper', 'lower')['truth'], 'unknown')
        scene.observe('upper', box(center=[0., 0., -.045], axes=(False, True, True)))
        self.assertEqual(symbolic.relation('support_candidate', 'upper', 'lower')['truth'], 'false')
        scene.unavailable('lower', 2., 'missing')
        self.assertEqual(symbolic.relation('above', 'upper', 'lower')['truth'], 'unknown')

    def test_face_alignment_generates_goal_without_mutating_observation(self):
        scene = MetricScene(); scene.observe('beam', box(dimensions=(.06, .03, .015), shape='thin_prism'))
        symbolic = SymbolicScene(scene); before = scene.get('beam')
        goal = symbolic.align('beam.bottom_face', [.1, -.2, .03], [0., 0., -1.])
        self.assertEqual(goal['kind'], 'desired_state'); self.assertTrue(goal['unconstrained_twist'])
        self.assertFalse(goal['execution_or_contact_certified'])
        desired_bottom = goal['rotation']@symbolic.reference('beam.bottom_face')['center_m']+goal['translation']
        np.testing.assert_allclose(desired_bottom, [.1, -.2, .03])
        np.testing.assert_array_equal(scene.get('beam')['center'], before['center'])
        self.assertEqual(scene.packet()['desired_states'], {})

    def test_exported_numeric_and_symbolic_share_exact_observations(self):
        numeric = {}; symbolic = {}
        exec(compile(build(False), '<numeric>', 'exec'), numeric)
        exec(compile(build(True), '<symbolic>', 'exec'), symbolic)
        self.assertNotIn('SymbolicScene', numeric)
        self.assertNotIn('point_axis_goal', numeric)
        with np.load(ROOT/'tests/fixtures/cuboid_correspondence_rgbd.npz') as data:
            frame = {k[8:]: data[k] for k in data.files if k.startswith('initial_')}
        objects = [{'color': 'green', 'shape': 'thin_prism', 'orientation': 'flat_x'}]
        outputs = []
        for library in (numeric, symbolic):
            observed = library['MixedPnPObservation'](objects, correspondence=True).observe(frame, 'green', None, inventory_table(frame))
            state = library['MetricScene'](); state.observe('green', observed); outputs.append(state.packet())
        self.assertEqual(outputs[0], outputs[1])
        self.assertIn('green', symbolic['SymbolicScene'](state).packet()['objects'])

    def test_session_checks_robot_time_even_without_a_new_camera_call(self):
        from test_rgbd_pnp_base import PixelsOnlyAPI
        class API(PixelsOnlyAPI):
            def __init__(self):
                super().__init__()
                self.now = float(self.frame['time'])
            def camera(self):
                return dict(self.frame, time=self.now)
            def proprioception(self):
                return {'tool_position': self.position.copy(), 'time': self.now}
            def gripper_geometry(self):
                return {'time': self.now, 'source': 'robot_model_and_joint_encoders',
                        'coordinate_frame': 'world', 'units': 'meters', 'visual_spheres': []}
        library = {}; exec(compile(build(True), '<session>', 'exec'), library)
        for kind in ('MixedSession', 'SymbolicSession'):
            api = API(); session = library[kind](api, [{'color': 'blue', 'shape': 'cube'}])
            self.assertEqual(session.state()['objects']['blue']['status'], 'available')
            api.now += 1.
            with self.assertRaisesRegex(ValueError, 'No current observation'):
                session.get('blue')
            self.assertEqual(session.state()['objects']['blue']['status'], 'stale')
            session.target('blue', [0., 0., .1])
            self.assertEqual(session.state()['desired_states']['blue']['recorded_time'], api.now)
            self.assertEqual(session.snapshot()['objects']['blue']['status'], 'available')
            self.assertEqual(api.actions, [])

    def test_invalid_landmarks_cannot_poison_serialized_state(self):
        scene = MetricScene(); scene.observe('a', box())
        for field, value in [('normal', [0., 0., np.nan]), ('vertices', np.full((8, 3), np.inf))]:
            with self.assertRaises(ValueError):
                scene.observe('a', dict(box(stamp=2.), **{field: value}))
        self.assertEqual(scene.packet()['time'], 1.)
        self.assertEqual(scene.packet()['objects']['a']['status'], 'available')
