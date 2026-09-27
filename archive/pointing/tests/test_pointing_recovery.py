"""Pointing cannot bypass frame binding, geometry fitting, or missing pixels."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from aspire_so100.pointing_exchange import PointingExchange, checked_point_reply
from aspire_so100.pointing_recovery import PointingRecovery
from aspire_so100.column_tracking import ColumnTracker
from aspire_so100.beam_geometry import estimate_flat_beam_geometry

ROOT = Path(__file__).resolve().parents[1]


class PointingRecoveryTests(unittest.TestCase):
    def test_reply_requires_current_image_and_finite_bounded_coordinates(self):
        req = {'task_id': 'current', 'image_sha256': 'rgb123', 'image_width': 320, 'image_height': 240}
        good = dict(task_id='current', image_sha256='rgb123', status='point', point_px=[150, 120])
        self.assertEqual(checked_point_reply(good, req)['point_px'], [150, 120])
        for change in ({'task_id': 'old'}, {'image_sha256': 'other'}, {'point_px': [320, 120]},
                       {'point_px': [float('nan'), 120]}, {'point_px': [True, 120]},
                       {'status': 'abstain'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                checked_point_reply(dict(good, **change), req)
        self.assertEqual(checked_point_reply(dict(good, status='abstain', point_px=None), req)['status'], 'abstain')

    def test_timeout_abstains_without_synthetic_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            box = PointingExchange(Path(temp) / 'mailbox', timeout=.001, max_requests=1)
            frame = {'rgb': np.zeros((240,320,3), np.uint8), 'time': 1.}
            with self.assertRaises(TimeoutError):
                box.request(frame, 'red', 'cube', 'upright')
            self.assertEqual(box.records[0]['status'], 'timeout')
            self.assertNotIn('reply', box.records[0])
            with self.assertRaisesRegex(ValueError, 'budget'):
                box.request(frame, 'red', 'cube', 'upright')

    def test_recovered_track_does_not_persist_into_a_new_observation(self):
        with np.load(ROOT / 'tests/fixtures/inventory_six_rgbd.npz') as archive:
            frame = dict(archive)
        frame['time'] = 1.
        tracker = ColumnTracker();tracker.update(frame)
        target = next(t for t in tracker.tracks.values() if t['color']=='red')
        v,u = np.nonzero(target['mask']);point=[int(u[len(u)//2]),int(v[len(v)//2])]
        class API:
            def note(self, text): pass
            def point_target(self, *args): return {'status':'point','point_px':point}
        recovery = PointingRecovery(API(),tracker)
        node = {'color':'red','shape':target['shape'],'orientation':'upright'}
        self.assertTrue(recovery.reassociate(frame,node,target['id']))
        self.assertFalse(recovery.reassociate(frame,node,target['id']))
        recovery.apply_frame(dict(frame,time=2.))
        self.assertEqual(recovery.overrides,{})

    def test_blackout_cannot_be_recovered_by_a_model_point(self):
        with np.load(ROOT / 'tests/fixtures/inventory_six_rgbd.npz') as archive:
            frame = dict(archive)
        frame.update(time=1., rgb=np.zeros_like(frame['rgb']), valid_depth=np.zeros_like(frame['valid_depth']))
        class API:
            def note(self, text): pass
            def point_target(self, *args): return {'status':'point','point_px':[160,120]}
        tracker = ColumnTracker(plane=np.zeros(3))
        recovery = PointingRecovery(API(),tracker)
        self.assertFalse(recovery.reassociate(frame,{'color':'red','shape':'cube','orientation':'upright'}))
        self.assertEqual(tracker.tracks,{})

    def test_generated_variant_passes_numerical_source_validator(self):
        spec=importlib.util.spec_from_file_location('runner_for_pointing',ROOT/'scripts/run_vision_trial.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        runner.validate_source((ROOT/'policies/assemble_structure_pointing.py').read_text())


if __name__ == '__main__':
    unittest.main()
