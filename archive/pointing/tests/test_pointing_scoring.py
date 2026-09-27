"""Offline scoring distinguishes target identity, misses, and abstentions."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('offline_pointer_score', ROOT / 'scripts/score_pointing_replays.py')
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


class PointingScoringTests(unittest.TestCase):
    def test_target_pixel_required_without_nearest_object_tolerance(self):
        target = np.zeros((20, 30), bool)
        target[10:13, 15:18] = True
        self.assertEqual(scorer.point_outcome(target, {'status': 'point', 'point_px': [15.9, 10.1]}), 'target_hit')
        self.assertEqual(scorer.point_outcome(target, {'status': 'point', 'point_px': [14.9, 10.1]}), 'wrong_pixel')
        for point in ([30, 10], [-.1, 10], [15, 20]):
            self.assertEqual(scorer.point_outcome(target, {'status': 'point', 'point_px': point}), 'out_of_bounds')

    def test_absence_control_and_visible_false_abstention(self):
        target = np.zeros((20, 30), bool)
        abstain = {'status': 'abstain', 'point_px': None}
        self.assertEqual(scorer.point_outcome(target, abstain), 'correct_abstention')
        self.assertEqual(scorer.point_outcome(target, {'status': 'point', 'point_px': [15, 10]}), 'false_positive_absent')
        target[10, 15] = True
        self.assertEqual(scorer.point_outcome(target, abstain), 'false_abstention')

    def test_malformed_predictions_cannot_become_successes(self):
        target = np.ones((20, 30), bool)
        for point in ([float('nan'), 10], [True, 10], ['15', 10], [15], None):
            self.assertEqual(scorer.point_outcome(target, {'status': 'point', 'point_px': point}), 'invalid_prediction')
        self.assertEqual(scorer.point_outcome(target, {'status': 'abstain', 'point_px': [15, 10]}), 'invalid_prediction')


if __name__ == '__main__':
    unittest.main()
