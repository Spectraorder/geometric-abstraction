"""Real recorded RGB-D validates scoped sensing and baseline projection."""
import sys
from pathlib import Path
import unittest
import numpy as np
from test_shared_observer_experiment import API, OBJECTS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from build_geometry_session import surface_library as library
from run_mixed_scene_experiment import validate_planner


class RecordingAPI(API):
    def __init__(self):
        super().__init__()
        self.logged = []

    def record_geometry(self, packet):
        self.logged.append(packet)


class StepSessionTests(unittest.TestCase):
    def test_same_sensor_estimates_with_target_only_records_and_no_implicit_sensing(self):
        sources = [library(arm) for arm in ('basic', 'symbolic')]
        self.assertEqual(sources[0].replace('SURFACE_ONLY = True', 'SURFACE_ONLY = False'), sources[1])
        centers = []
        for source in sources:
            namespace = {}; exec(compile(source, '<step-library>', 'exec'), namespace)
            api = RecordingAPI(); session = namespace['GEOMETRY'].session(api, OBJECTS)
            color = OBJECTS[0]['color']
            def unexpected(*args, **kwargs):
                raise AssertionError('Checkpoint unexpectedly requested a camera')
            api.camera = unexpected
            packet = session.checkpoint(color, 'check')
            self.assertEqual(packet['object'], color)
            self.assertNotIn('objects', packet); self.assertNotIn('facts', packet)
            self.assertEqual(api.logged[-1]['record'], packet)
            geometry = packet['geometry']
            centers.append(geometry.get('center_m', geometry.get('center')))
            if namespace['SURFACE_ONLY']:
                self.assertNotIn('dimensions_m', geometry)
                self.assertNotIn('estimated_change', packet)
        np.testing.assert_allclose(*centers, atol=5e-7, rtol=0)

    def test_explicit_refresh_only_observes_focus_and_missing_fit_stays_unknown(self):
        namespace = {}; exec(compile(library('symbolic'), '<step-library>', 'exec'), namespace)
        api = RecordingAPI(); session = namespace['GEOMETRY'].session(api, OBJECTS)
        color = OBJECTS[0]['color']; original = session.observe; called = []
        def observed(name):
            called.append(name); return original(name)
        session.observe = observed
        session.checkpoint(color, 'check', refresh=True)
        self.assertEqual(called, [color])
        def unavailable(name):
            raise ValueError('No current observation')
        session.get = unavailable
        record = session.checkpoint(color, 'after_motion')
        self.assertEqual(record['status'], 'unavailable')
        self.assertNotIn('geometry', record)

    def test_checkpoint_goal_uses_the_existing_desired_state(self):
        namespace = {}; exec(compile(library('symbolic'), '<step-library>', 'exec'), namespace)
        api = RecordingAPI(); session = namespace['GEOMETRY'].session(api, OBJECTS)
        color = OBJECTS[0]['color']; before = session.get(color)['center'].copy()
        session.target(color, before + np.array([.01, 0., 0.]))
        record = session.checkpoint(color, 'check')
        self.assertEqual(record['remaining_translation_m'], [.01, 0., 0.])
        np.testing.assert_equal(session.get(color)['center'], before)
        record = session.checkpoint(color, 'check', goal_center=before)
        self.assertEqual(record['remaining_translation_m'], [0., 0., 0.])
        np.testing.assert_equal(session.goal_error(color)['translation_to_goal_m'], [0., 0., 0.])

    def test_place_intent_uses_extent_not_face_or_center_and_never_refreshes(self):
        for arm in ('basic', 'symbolic'):
            namespace = {}; exec(compile(library(arm), '<step-library>', 'exec'), namespace)
            api = RecordingAPI(); public = namespace['GEOMETRY'].session(api, OBJECTS)
            session = public.backend if arm == 'basic' else public
            color = OBJECTS[0]['color']; fit = session.get(color)
            actual = np.array([fit['center'][0], fit['center'][1], fit['bottom']])
            destination = actual + [.004, -.003, .002]
            session.target(color, fit['center'] + [.01, 0., 0.])
            session.pnp.held = {'color': color, 'confirmed': True}
            commands = []
            def place(point):
                commands.append(point.copy())
                session.pnp.held = None
                return {'status': 'released', 'held': False}
            session.pnp.place = place
            def unexpected(*args, **kwargs):
                raise AssertionError('Feedback unexpectedly requested a camera')
            api.camera = unexpected
            public.place(destination)
            np.testing.assert_equal(commands, [destination])
            destination[:] = 99
            record = public.checkpoint(color, 'place')
            if arm == 'basic':
                self.assertNotIn('placement_goal', record)
            else:
                self.assertEqual(record['placement_goal']['reference'], 'center_xy_min_z')
                self.assertEqual(record['placement_goal']['remaining_translation_m'], [.004, -.003, .002])
                self.assertEqual(record['remaining_translation_m'], [.01, 0., 0.])
                self.assertNotIn('placement_goal', public.checkpoint(OBJECTS[1]['color'], 'check'))
                def unavailable(name):
                    raise ValueError('Expired observation')
                session.get = unavailable
                record = public.checkpoint(color, 'wait')
                self.assertIsNone(record['placement_goal']['remaining_translation_m'])
                self.assertNotIn('geometry', record)

    def test_failed_place_retains_intent_and_new_pick_clears_it(self):
        namespace = {}; exec(compile(library('symbolic'), '<step-library>', 'exec'), namespace)
        session = namespace['GEOMETRY'].session(RecordingAPI(), OBJECTS)
        color = OBJECTS[0]['color']; point = np.array([.1, 0., .4])
        session.pnp.held = {'color': color, 'confirmed': True}
        def failed_place(point):
            raise RuntimeError('Injected loss of vision during placement')
        session.pnp.place = failed_place
        with self.assertRaises(RuntimeError):
            session.place(point)
        self.assertEqual(session.checkpoint(color, 'failed')['placement_goal']['point_m'], point.tolist())
        calls = []
        def pick(name, transit_z):
            calls.append((name, transit_z))
            return {'status': 'grasp_unconfirmed', 'held': False}
        session.pnp.held = None
        session.pnp.pick = pick
        result = session.pick_and_place(color, point)
        self.assertEqual(result['status'], 'grasp_unconfirmed')
        self.assertEqual(calls, [(color, float(point[2]+.085))])
        self.assertNotIn('placement_goal', session.checkpoint(color, 'pick'))

    def test_new_common_starter_passes_actual_actor_validator(self):
        path = Path(__file__).resolve().parents[1]/'policies/support_graph_step_starter.py'
        validate_planner(path.read_text())


if __name__ == '__main__':
    unittest.main()
