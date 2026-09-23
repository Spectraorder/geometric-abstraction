"""Instance identity regressions using real RGB-D plus controlled associations."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from aspire_so100.instance_geometry import InstanceTracker, depth_components, inventory_instances
from test_vision_boundary import runner

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'policies/stack_inventory_cpu.py'
spec = importlib.util.spec_from_file_location('inventory_candidate', SOURCE)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def detection(x, shape='cube'):
    return dict(color='red', shape=shape, observed_center=np.array([x, 0., .03]),
                center=np.array([x, 0., .03]), mask=np.ones((3, 3), bool))


class InstanceGeometryTests(unittest.TestCase):
    def test_real_same_color_distractors_have_separate_shapes_and_ids(self):
        with np.load(ROOT / 'tests/fixtures/inventory_six_rgbd.npz') as archive:
            frame = dict(archive)
        tracker = InstanceTracker()
        detected = tracker.update(frame)
        self.assertEqual(len(detected), 6)
        self.assertEqual({t['shape'] for t in detected},
                         {'cube', 'thin_prism', 'prism', 'cylinder', 'triangular_prism'})
        target = tracker.select('red', 'thin_prism')
        base = tracker.select('blue', 'cube')
        self.assertNotEqual(target, base)
        red = [t for t in detected if t['color'] == 'red']
        self.assertEqual(len(red), 4)
        masked = tracker.isolated_frame(frame, target)
        self.assertEqual(np.count_nonzero(masked['valid_depth']), tracker.tracks[target]['pixel_count'])
        for other in red:
            if other['id'] != target:
                self.assertFalse(masked['valid_depth'][other['mask']].any())
        bundled, plane = policy.inventory_instances(frame)
        expected, _ = inventory_instances(frame)
        for a, b in zip(bundled, expected):
            self.assertEqual(a['shape'], b['shape'])
            np.testing.assert_allclose(a['center'], b['center'])

    def test_touching_image_regions_split_at_depth_discontinuity(self):
        v, u = np.indices((8, 16))
        points = np.stack((u*.001, v*.001, np.where(u < 8, .4, .46)), axis=-1)
        parts = depth_components(np.ones((8, 16), bool), points)
        self.assertEqual(sorted(np.count_nonzero(m) for m in parts), [64, 64])

    def test_robot_motion_prediction_preserves_held_identity_past_distractor(self):
        tracker = InstanceTracker()
        with patch('aspire_so100.instance_geometry.inventory_instances',
                   return_value=([detection(0, 'thin_prism'), detection(.1)], np.zeros(3))):
            tracker.update({})
        target = tracker.select('red', 'thin_prism')
        distractor = tracker.select('red', 'cube')
        # Order swaps and the held object's instantaneous shape is unknown.
        with patch('aspire_so100.instance_geometry.inventory_instances',
                   return_value=([detection(.1), detection(.2, 'unknown')], np.zeros(3))):
            tracker.update({}, {target: np.array([.2, 0., .03])})
        self.assertEqual(tracker.select('red', 'thin_prism'), target)
        self.assertEqual(tracker.select('red', 'cube'), distractor)
        self.assertAlmostEqual(tracker.tracks[target]['center'][0], .2)

    def test_missing_target_does_not_fall_back_to_same_color_distractor(self):
        tracker = InstanceTracker()
        with patch('aspire_so100.instance_geometry.inventory_instances',
                   return_value=([detection(0, 'thin_prism'), detection(.1)], np.zeros(3))):
            tracker.update({})
        target = tracker.select('red', 'thin_prism')
        with patch('aspire_so100.instance_geometry.inventory_instances',
                   return_value=([detection(.1)], np.zeros(3))):
            tracker.update({})
        with self.assertRaisesRegex(ValueError, 'occluded'):
            tracker.isolated_frame({}, target)
        with self.assertRaisesRegex(ValueError, 'unique'):
            tracker.select('red', 'thin_prism')

    def test_duplicate_role_is_rejected(self):
        tracker = InstanceTracker()
        with patch('aspire_so100.instance_geometry.inventory_instances',
                   return_value=([detection(0), detection(.1)], np.zeros(3))):
            tracker.update({})
        with self.assertRaisesRegex(ValueError, 'unique'):
            tracker.select('red', 'cube')

    def test_partial_surface_can_update_but_cannot_create_an_identity(self):
        tracker = InstanceTracker()
        partial = dict(detection(.005), initial_eligible=False)
        with patch('aspire_so100.instance_geometry.inventory_instances',
                   return_value=([partial], np.zeros(3))):
            self.assertEqual(tracker.update({}), [])
        with patch('aspire_so100.instance_geometry.inventory_instances',
                   return_value=([detection(0)], np.zeros(3))):
            tracker.update({})
        key = tracker.select('red', 'cube')
        with patch('aspire_so100.instance_geometry.inventory_instances',
                   return_value=([partial], np.zeros(3))):
            tracker.update({})
        self.assertEqual(tracker.select('red', 'cube'), key)

    def test_ambiguous_overlap_does_not_reassign_selected_ids(self):
        tracker = InstanceTracker()
        with patch('aspire_so100.instance_geometry.inventory_instances',
                   return_value=([detection(0), detection(.04, 'prism')], np.zeros(3))):
            tracker.update({})
        old_ids = list(tracker.tracks)
        with patch('aspire_so100.instance_geometry.inventory_instances',
                   return_value=([detection(.02, 'unknown')], np.zeros(3))):
            tracker.update({})
        self.assertTrue(all(not tracker.tracks[key]['visible'] for key in old_ids))

    def test_blackout_stops_before_any_robot_call(self):
        with np.load(ROOT / 'tests/fixtures/inventory_six_rgbd.npz') as archive:
            frame = dict(archive)
        frame['rgb'][:] = 0
        frame['valid_depth'][:] = False
        class API:
            def camera(self):
                return frame
            def note(self, text):
                self.message = text
        api = API()
        policy.run(api)
        self.assertEqual(api.message, 'Insufficient tabletop depth.')

    def test_bundle_is_reproducible_and_sensor_only(self):
        spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_inventory_policy.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        self.assertEqual(SOURCE.read_text(), builder.build_policy())
        runner.validate_source(SOURCE.read_text())
