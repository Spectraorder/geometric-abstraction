"""Initial priors cannot be changed by goal metadata or descriptor projection."""
from pathlib import Path
import sys
import unittest
import numpy as np
from test_mixed_scene_contract import RecordedSceneAPI, OBJECTS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from build_mixed_scene import build
from observation_context_prototype import ObservationContext, bind_session


def initial_context():
    return {'schema': 'initial_observation_context_v1',
            'source': 'public_initial_scene_description',
            'objects': [{'color': obj['color'], 'shape': obj['shape'],
                         'initial_orientation_assumption': obj.get('orientation', 'upright')}
                        for obj in OBJECTS]}


class InitialContextTests(unittest.TestCase):
    def test_goal_changes_projection_and_order_do_not_change_descriptors(self):
        context = ObservationContext(initial_context())
        stripped = [{'color': x['color'], 'shape': x['shape']} for x in OBJECTS]
        changed_goal = [dict(x, orientation='upright', grid=[99, 99], supports=['unrelated']) for x in reversed(OBJECTS)]
        expected = context.descriptors(OBJECTS)
        self.assertEqual(context.descriptors(stripped), expected)
        self.assertEqual(context.descriptors(changed_goal), expected)
        self.assertEqual(expected[2]['orientation'], 'flat_x')

    def test_context_is_copied_and_invalid_inventory_or_metric_priors_rejected(self):
        packet = initial_context()
        seen = []
        session = bind_session(lambda api, objects: seen.append(objects), packet)
        packet['objects'][2]['initial_orientation_assumption'] = 'upright'
        session(None, OBJECTS)
        self.assertEqual(seen[0][2]['orientation'], 'flat_x')
        context = ObservationContext(initial_context())
        returned = context.packet(); returned['objects'].clear()
        self.assertEqual(len(context.packet()['objects']), 4)
        for request in (OBJECTS[:-1], OBJECTS+[OBJECTS[0]], [dict(x, shape='cube') for x in OBJECTS]):
            with self.assertRaises(ValueError): context.descriptors(request)
        for field in ('center', 'dimensions', 'orientation'):
            malformed = initial_context(); malformed['objects'][0][field] = [0, 0, 0]
            with self.assertRaises(ValueError): ObservationContext(malformed)
        missing = initial_context(); del missing['objects'][2]['initial_orientation_assumption']
        with self.assertRaises(ValueError): ObservationContext(missing)

    def test_same_recorded_rgbd_yields_identical_fits_for_both_descriptor_forms_and_arms(self):
        by_arm = []
        stripped = [{'shape': obj['shape'], 'color': obj['color']} for obj in reversed(OBJECTS)]
        for symbolic in (False, True):
            namespace = {}; exec(compile(build(symbolic), '<session-test>', 'exec'), namespace)
            factory = bind_session(namespace['SymbolicSession' if symbolic else 'MixedSession'], initial_context())
            sessions = [factory(RecordedSceneAPI(), descriptor) for descriptor in (OBJECTS, stripped)]
            fits = [[session.observe(obj['color']) for obj in OBJECTS] for session in sessions]
            for first, second in zip(*fits):
                for field in ('center', 'rotation', 'axis_dimensions', 'constrained_axes', 'vertices'):
                    np.testing.assert_array_equal(first[field], second[field])
            by_arm.append(fits[0])
        for first, second in zip(*by_arm):
            for field in ('center', 'rotation', 'axis_dimensions', 'constrained_axes', 'vertices'):
                np.testing.assert_array_equal(first[field], second[field])


if __name__ == '__main__':
    unittest.main()
