"""A surface ablation must not change the shared estimator or actuation path."""
from pathlib import Path
import sys
import unittest
import numpy as np
from test_mixed_scene_contract import RecordedSceneAPI, OBJECTS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from build_mixed_scene import build
from run_vision_trial import validate_source
from observation_context_prototype import bind_session
from test_observation_context_prototype import initial_context

ROOT = Path(__file__).resolve().parents[1]


class SurfaceProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = build(False)+'\n'+(ROOT/'scripts/surface_projection_prototype.py').read_text()
        validate_source(source)
        cls.library = {}; exec(compile(source, '<projected-session>', 'exec'), cls.library)

    def test_actual_sensor_fits_match_shared_fields_and_freshness(self):
        factory = bind_session(self.library['MixedSession'], initial_context())
        raw_api, projected_api = RecordedSceneAPI(), RecordedSceneAPI()
        raw = factory(raw_api, OBJECTS)
        projected = self.library['SurfaceProjectionSession'](factory(projected_api, OBJECTS))
        for obj in OBJECTS:
            first, second = raw.observe(obj['color']), projected.observe(obj['color'])
            for key, value in second.items():
                np.testing.assert_equal(value, first[key])
            self.assertFalse({'rotation', 'vertices', 'axis_dimensions', 'model'} & set(second))
        self.assertEqual(projected.snapshot(), self.library['surface_packet'](raw.snapshot()))
        returned = projected.get('blue'); returned['center'][:] = 99
        self.assertLess(np.max(projected.get('blue')['center']), 1)
        projected_api.frame['time'] = float(projected_api.frame['time'])+1
        self.assertEqual(projected.state()['objects']['blue']['status'], 'stale')
        with self.assertRaises(ValueError): projected.get('blue')
        self.assertEqual(raw_api.actions, projected_api.actions)

    def test_action_delegation_preserves_arguments_and_filters_nested_fit(self):
        calls = []
        result = {'status': 'released', 'held': False, 'xy_error_m': .001,
                  'surface_gap_m': .003, 'task_success_certified': False,
                  'observation': {'center': np.zeros(3), 'vertices': np.ones((8,3)),
                                  'model': {'private_dimensions': [1,2,3]}}}
        class Backend:
            def pick(self, color, transit_z):
                calls.append(('pick', color, transit_z)); return result
            def place(self, point):
                calls.append(('place', point)); return result
            def pick_and_place(self, color, point):
                calls.append(('pick_and_place', color, point)); return result
        projected = self.library['SurfaceProjectionSession'](Backend())
        point = [1, 2, 3]
        for output in (projected.pick('red', .5), projected.place(point), projected.pick_and_place('blue', point)):
            self.assertEqual(set(output['observation']), {'center'})
            self.assertEqual(output['status'], 'released')
            self.assertFalse(output['task_success_certified'])
            output['observation']['center'][:] = 99
        np.testing.assert_array_equal(result['observation']['center'], np.zeros(3))
        self.assertEqual(calls, [('pick','red',.5),('place',point),('pick_and_place','blue',point)])

    def test_initial_context_is_compatible_with_the_worker_library_validator(self):
        validate_source((ROOT/'scripts/observation_context_prototype.py').read_text())


if __name__ == '__main__':
    unittest.main()
