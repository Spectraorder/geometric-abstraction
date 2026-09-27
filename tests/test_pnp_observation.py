"""A shared callback can influence execution without changing its motion logic."""
import sys
from pathlib import Path
import unittest
import numpy as np
from test_rgbd_pnp_base import PixelsOnlyAPI, ROOT
from aspire_so100.pnp_observation import MixedPnPObservation

sys.path.insert(0, str(ROOT/'scripts'))
from build_mixed_pnp_observer import build

NS = {}; exec(compile(build(), '<mixed-bundle>', 'exec'), NS)
ObservedPnP = NS['ObservedRGBDPickPlace']


class SharedObserverTests(unittest.TestCase):
    def test_default_path_is_identical_and_motion_methods_are_inherited(self):
        basic = NS['RGBDPickPlace'](PixelsOnlyAPI()); changed = ObservedPnP(PixelsOnlyAPI())
        for name in ('pick', 'place', 'move', 'pick_and_place'):
            self.assertIs(getattr(ObservedPnP, name), getattr(NS['RGBDPickPlace'], name))
        for color in ('blue', 'red', 'green'):
            first, second = basic.observe(color), changed.observe(color)
            for key in ('center', 'normal', 'height', 'dims', 'bottom', 'extent_z'):
                np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(changed.api.actions, [])

    def test_current_observation_required_and_backends_cannot_mix_models(self):
        api = PixelsOnlyAPI(); pnp = ObservedPnP(api)
        observer = MixedPnPObservation([{'color': 'green', 'shape': 'cube'}])
        def stale(frame, color, model, plane):
            result = observer.observe(frame, color, model, plane)
            return dict(result, observation_time=result['observation_time']-1.)
        pnp.set_observer(stale)
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            pnp.observe('green')
        self.assertEqual(pnp.models, {})
        pnp.set_observer(observer.observe); measured = pnp.observe('green')
        self.assertEqual(measured['source'], 'mixed_rigid_rgbd_observation')
        with self.assertRaisesRegex(ValueError, 'before observing'):
            pnp.set_observer(None)
        self.assertEqual(api.actions, [])

    def test_invalid_units_fail_before_model_or_action_changes(self):
        api = PixelsOnlyAPI(); pnp = ObservedPnP(api)
        observer = MixedPnPObservation([{'color': 'red', 'shape': 'cube'}])
        def wrong_units(frame, color, model, plane):
            return dict(observer.observe(frame, color, model, plane), units='millimeters')
        pnp.set_observer(wrong_units)
        with self.assertRaisesRegex(ValueError, 'meters'):
            pnp.pick('red')
        self.assertEqual(api.actions, [])
        self.assertEqual(pnp.models, {})

    def test_actual_bridge_rgbd_export_matches_package_observer(self):
        with np.load(ROOT/'tests/fixtures/mixed_bridge_release_rgbd.npz') as data:
            frames = {label: {k[len(label)+1:]: data[k] for k in data.files if k.startswith(label+'_')}
                      for label in ('initial', 'held')}
        objects = [{'color': 'blue', 'shape': 'cube'}, {'color': 'yellow', 'shape': 'cube'},
                   {'color': 'green', 'shape': 'thin_prism', 'orientation': 'flat_x'},
                   {'color': 'red', 'shape': 'cylinder'}]
        api = PixelsOnlyAPI(); api.frame = frames['initial']; pnp = ObservedPnP(api)
        packaged = MixedPnPObservation(objects); bundled = NS['MixedPnPObservation'](objects)
        pnp.set_observer(bundled.observe); models = {}
        for obj in objects:
            color = obj['color']; expected = packaged.observe(api.frame, color, None, pnp.plane)
            actual = pnp.observe(color); models[color] = expected['model']
            np.testing.assert_allclose(actual['center'], expected['center'], atol=1e-12)
        api.frame = frames['held']; expected = packaged.observe(api.frame, 'green', models['green'], pnp.plane)
        actual = pnp.observe('green')
        np.testing.assert_allclose(actual['center'], expected['center'], atol=1e-12)
        np.testing.assert_array_equal(actual['constrained_axes'], expected['constrained_axes'])
        self.assertEqual(api.actions, [])

    def test_shared_release_gate_uses_injected_current_center(self):
        class FixedRobotAPI(PixelsOnlyAPI):
            def move_tool(self, position, seconds=1.5):
                self.actions.append(('move', position))
        api = FixedRobotAPI(); pnp = ObservedPnP(api)
        observer = MixedPnPObservation([{'color': 'red', 'shape': 'cube'}]); calls = []
        def estimate(frame, color, model, plane):
            calls.append(color)
            return observer.observe(frame, color, model, plane)
        pnp.set_observer(estimate)
        first = pnp.observe('red'); api.position = first['center']+np.array([0., 0., .05])
        pnp.held = {'color': 'red', 'confirmed': True, 'park': api.position.copy()}
        # Static recorded RGB-D cannot follow commanded corrections. The common
        # gate must retain the payload instead of releasing at an unobserved target.
        target = np.r_[first['center'][:2]+[.03, 0.], first['bottom']]
        result = pnp.place(target)
        self.assertEqual(result['status'], 'placement_unconfirmed')
        self.assertGreater(len(calls), 3)
        self.assertFalse(any(action[0] == 'gripper' for action in api.actions))

    def test_post_release_observation_failure_preserves_completed_release(self):
        for error_type in (ValueError, RuntimeError):
            with self.subTest(error=error_type.__name__):
                api = PixelsOnlyAPI(); pnp = ObservedPnP(api)
                fixed = {'center': np.array([.1, -.2, .418]), 'bottom': .403, 'extent_z': .015}
                pnp.held = {'color': 'red', 'confirmed': True, 'park': api.position.copy()}
                def observe(color, near=None):
                    if pnp.held is None:
                        raise error_type('Follow-up fit unavailable')
                    return fixed
                pnp.observe = observe
                result = pnp.place([.1, -.2, .4])
                self.assertEqual(result['status'], 'released')
                self.assertFalse(result['held'])
                self.assertIsNone(pnp.held)
                self.assertFalse(result['task_success_certified'])
                self.assertIn('Follow-up fit unavailable', result['observation_unavailable'])
                self.assertNotIn('observation', result)
                self.assertTrue(any(a[0] == 'gripper' for a in api.actions))

    def test_pre_release_observation_failure_still_propagates(self):
        api = PixelsOnlyAPI(); pnp = ObservedPnP(api)
        pnp.held = {'color': 'red', 'confirmed': True, 'park': api.position.copy()}
        def observe(color, near=None):
            raise ValueError('Current grasp geometry unavailable')
        pnp.observe = observe
        with self.assertRaisesRegex(ValueError, 'Current grasp geometry'):
            pnp.place([.1, -.2, .4])
        self.assertIsNotNone(pnp.held)
        self.assertEqual(api.actions, [])

    def test_view_fallback_requires_same_time_and_records_unavailable_side(self):
        initial = PixelsOnlyAPI().frame
        observer = MixedPnPObservation([{'color': 'blue', 'shape': 'cube'}])
        pnp = ObservedPnP(PixelsOnlyAPI())
        blank = dict(initial, valid_depth=np.zeros_like(initial['valid_depth']))
        estimate = observer.observe_views([('side', blank), ('wrist', initial)], 'blue', None, pnp.plane)
        self.assertEqual(estimate['view'], 'wrist')
        self.assertEqual(estimate['rejected_views'][0]['view'], 'side')
        stale = dict(initial, time=float(initial['time'])-1.)
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            observer.observe_views([('side', blank), ('wrist', stale)], 'blue', None, pnp.plane)
        with self.assertRaisesRegex(ValueError, 'No usable current view'):
            observer.observe_views([('side', blank), ('wrist', blank)], 'blue', None, pnp.plane)
