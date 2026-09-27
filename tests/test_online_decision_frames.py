"""Verify same-time reuse, independent views, freshness and mutation isolation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from online_decision_frames import DecisionFrameReuse


class Sensor:
    def __init__(self):
        self.time = 1.
        self.captures = 0
        self.advance = False
    def proprioception(self):
        return {'time':self.time}
    def camera(self, view='side'):
        self.captures += 1
        if self.advance: self.time += 1.
        return {'time':self.time, 'view':view, 'rgb':[[self.captures]]}


class CachedSensor(DecisionFrameReuse, Sensor):
    pass


class FrameReuseTests(unittest.TestCase):
    def test_reuses_per_view_and_isolates_mutations(self):
        sensor = CachedSensor()
        with sensor.decision_frames():
            first = sensor.camera()
            first['rgb'][0][0] = 999
            self.assertEqual(sensor.camera()['rgb'], [[1]])
            sensor.camera('wrist')
            self.assertEqual(sensor.captures, 2)
            self.assertEqual(sensor.frame_reuse_hits, 1)
        sensor.camera()
        self.assertEqual(sensor.captures, 3)

    def test_time_change_invalidates_every_view(self):
        sensor = CachedSensor()
        with sensor.decision_frames():
            sensor.camera(); sensor.camera('wrist')
            sensor.time += .002
            sensor.camera(); sensor.camera('wrist')
            self.assertEqual(sensor.captures, 4)

    def test_exception_closes_scope_and_nested_scope_rejected(self):
        sensor = CachedSensor()
        with self.assertRaisesRegex(RuntimeError, 'nested'):
            with sensor.decision_frames():
                sensor.camera()
                with sensor.decision_frames(): pass
        sensor.camera()
        self.assertEqual(sensor.captures, 2)
        with sensor.decision_frames(): sensor.camera()
        self.assertEqual(sensor.captures, 3)

    def test_readout_that_advances_time_is_never_cached(self):
        sensor = CachedSensor()
        sensor.advance = True
        with sensor.decision_frames():
            sensor.camera(); sensor.camera()
            self.assertEqual(sensor.captures, 2)
            self.assertEqual(sensor.frame_reuse_hits, 0)


if __name__ == '__main__':
    unittest.main()
