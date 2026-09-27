"""Reject malformed subagent output and mismatched RGB-D validation frames."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('evaluate_pointer', ROOT / 'scripts/evaluate_pointing_agent.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class PointingAgentTests(unittest.TestCase):
    def test_abstention_does_not_read_depth_or_become_a_point(self):
        result, mask = evaluator.validate_prediction({}, {'status': 'abstain', 'point_px': None}, 'red')
        self.assertEqual(result, {'status': 'abstained'})
        self.assertIsNone(mask)

    def test_malformed_records_do_not_reach_geometry(self):
        for prediction in ({'status': 'abstain', 'point_px': [10, 20]},
                           {'status': 'point', 'point_px': None},
                           {'status': 'point', 'point_px': [True, 10]},
                           {'status': 'point', 'point_px': ['10', '20']},
                           {'status': 'unknown', 'point_px': [10, 20]}):
            with self.subTest(prediction=prediction):
                result, mask = evaluator.validate_prediction({}, prediction, 'red')
                self.assertEqual(result['status'], 'rejected')
                self.assertIsNone(mask)

    def test_duplicate_ids_are_not_silently_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            evaluator.indexed([{'task_id': 'A'}, {'task_id': 'A'}])

    def test_changed_rgb_cannot_validate_against_another_observation(self):
        source = ROOT / 'tests/fixtures/inventory_six_rgbd.npz'
        with np.load(source) as archive:
            rgb = archive['rgb'].copy()
        rgb[0, 0, 0] ^= 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'input.png'
            Image.fromarray(rgb).save(image)
            records = {
                'tasks': [{'task_id': 'A', 'image_path': str(image), 'image_width': 320,
                           'image_height': 240, 'target': 'red cube'}],
                'frames': [{'task_id': 'A', 'frame_path': str(source), 'color': 'red'}],
                'predictions': [{'task_id': 'A', 'status': 'abstain', 'point_px': None}],
            }
            args = ['evaluate', '--output', str(root / 'out')]
            for name, data in records.items():
                path = root / (name + '.json')
                path.write_text(json.dumps(data))
                args += ['--' + name, str(path)]
            with patch('sys.argv', args), self.assertRaisesRegex(ValueError, 'RGB input and camera frame differ'):
                evaluator.main()


if __name__ == '__main__':
    unittest.main()
