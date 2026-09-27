"""A task-free visual PnP mechanism and its explicit task adapter boundary."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pnp_base', ROOT/'policies/pick_place_rgbd_base.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sys.path.insert(0, str(ROOT/'scripts'))
from build_rgbd_pnp_policy import build
from run_vision_trial import validate_source


class PixelsOnlyAPI:
    def __init__(self):
        with np.load(ROOT/'tests/fixtures/ablation_cube_initial_rgbd.npz') as data:
            self.frame = dict(data)
        self.actions = []
        self.position = np.array([0., -.2, .6])
    def camera(self):
        return self.frame
    def task(self):
        raise AssertionError('The reusable mechanism must not query a task')
    def proprioception(self):
        return {'tool_position': self.position.copy()}
    def move_tool(self, position, seconds=1.5):
        self.actions.append(('move', position))
        self.position = np.asarray(position)
    def gripper(self, angle, seconds=.6):
        self.actions.append(('gripper', angle))
    def wait(self, seconds):
        pass


class RGBDPickPlaceTests(unittest.TestCase):
    def test_initialize_and_observe_without_goal_or_actions(self):
        api = PixelsOnlyAPI()
        pnp = base.RGBDPickPlace(api)
        for color in ('red', 'green', 'blue'):
            measured = pnp.observe(color)
            self.assertAlmostEqual(measured['height'], .03, delta=.003)
            self.assertTrue(np.isfinite(measured['center']).all())
            self.assertEqual(measured['source'], 'rgbd_visible_top_estimate')
            self.assertNotIn('vertices', measured)
        self.assertEqual(api.actions, [])

    def test_invalid_destination_does_not_start_a_grasp(self):
        api = PixelsOnlyAPI()
        pnp = base.RGBDPickPlace(api)
        for point in ([0., 0.], [0., 0., np.nan]):
            with self.assertRaises(ValueError):
                pnp.pick_and_place('red', point)
        self.assertEqual(api.actions, [])
        with self.assertRaises(RuntimeError):
            pnp.place([0., -.2, .4])

    def test_uncertain_placement_does_not_release_or_claim_success(self):
        api = PixelsOnlyAPI()
        pnp = base.RGBDPickPlace(api)
        fixed = {'center': np.array([.1, -.2, .43]), 'bottom': .415, 'extent_z': .015}
        pnp.observe = lambda color, near=None: fixed
        pnp.held = {'color': 'red', 'confirmed': True, 'park': api.position.copy()}
        result = pnp.place([-.1, -.2, .4])
        self.assertEqual(result['status'], 'placement_unconfirmed')
        self.assertTrue(result['held'])
        self.assertFalse(any(kind=='gripper' for kind, _ in api.actions))
        with self.assertRaises(RuntimeError):
            pnp.pick('green')

    def test_adapter_follows_public_order_instead_of_color_roles(self):
        calls = []
        class PnP:
            def __init__(self, api):
                pass
            def observe(self, color):
                return {'center': np.array([0., 0., .4]), 'top': .415}
            def pick_and_place(self, color, point):
                calls.append(color)
                return {'status': 'released'}
        class API:
            def task(self):
                return {'kind': 'single_column', 'levels': [{'color': c} for c in ('yellow','green','red','blue')]}
            def note(self, text):
                pass
        namespace = {'RGBDPickPlace': PnP}
        exec((ROOT/'policies/stack_from_rgbd_pnp_adapter.py').read_text(), namespace)
        namespace['run'](API())
        self.assertEqual(calls, ['green','red','blue'])

    def test_bundle_uses_only_the_sensor_policy_language(self):
        validate_source(build())


if __name__ == '__main__':
    unittest.main()
