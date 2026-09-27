"""The shared executor remains task-independent and changes IK using robot data only."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from test_rgbd_pnp_base import PixelsOnlyAPI, ROOT, build, validate_source

spec = importlib.util.spec_from_file_location('shared_pnp', ROOT/'policies/pick_place_rgbd_shared.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


class SharedPnPTests(unittest.TestCase):
    def test_source_is_task_independent_and_preserves_visible_top_interface(self):
        api = PixelsOnlyAPI()
        pnp = shared.RGBDPickPlace(api)
        measured = pnp.observe('green')
        self.assertEqual(measured['source'], 'rgbd_visible_top_estimate')
        self.assertNotIn('vertices', measured)
        self.assertEqual(api.actions, [])
        validate_source(build(base=ROOT/'policies/pick_place_rgbd_shared.py'))

    def test_robot_only_fallback_preserves_requested_position(self):
        class API(PixelsOnlyAPI):
            def move_tool(self, position, seconds=1.5, orientation=None):
                if orientation is None:
                    raise RuntimeError('Requested tool position is unreachable within 5 mm')
                self.actions.append(('move', np.asarray(position), np.asarray(orientation)))
            def plan_tool(self, position, orientation=None):
                return {'tool_rotation': np.eye(3), 'position_error': .002 if orientation is not None else .006}
        api=API();pnp=shared.RGBDPickPlace(api);target=[.03,-.23,.51];pnp.move(target)
        self.assertEqual(len(api.actions),1)
        np.testing.assert_array_equal(api.actions[0][1],target)
        np.testing.assert_array_equal(api.actions[0][2],np.eye(3))

    def test_fallback_rejects_an_unreachable_position(self):
        class API(PixelsOnlyAPI):
            def move_tool(self, position, seconds=1.5, orientation=None):
                raise RuntimeError('unreachable')
            def plan_tool(self, position, orientation=None):
                return {'tool_rotation': np.eye(3), 'position_error': .02}
        api=API();pnp=shared.RGBDPickPlace(api)
        with self.assertRaisesRegex(RuntimeError,'unreachable'):
            pnp.move([.8,.8,.8])
