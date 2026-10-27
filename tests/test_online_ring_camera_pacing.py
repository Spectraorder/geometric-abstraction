"""Regress cumulative sensing lag without changing clocks, observations or budgets."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import online_ring_camera_pacing as pacing
import online_ring_loop
import resume_online_extension_camera as resume


class ClockEnv:
    def __init__(self):
        self.now = self.sim = self.lag = 0.
        self.deadline = 200.
        self.reset_count = 1

    def advance_while_waiting(self):
        if self.now >= self.deadline:
            raise TimeoutError('Same episode deadline')
        self.lag = max(self.lag, self.now - self.sim)
        self.sim = self.now

    def wait(self, seconds):
        self.sim += seconds
        self.now += seconds


class Sensors:
    def __init__(self, env):
        self.env = env
        self.views = []

    def camera(self, view='side'):
        self.views.append(view)
        return dict(view=view, time=self.env.sim)


class CameraPacingTests(unittest.TestCase):
    def retries(self, sensor):
        for _ in range(3):
            for view in ('side', 'wrist'):
                sensor.camera(view)
                sensor.env.now += .6
            sensor.env.wait(.12)
        sensor.env.advance_while_waiting()

    def test_same_retry_work_advances_physics_without_reset_or_free_time(self):
        old, new = Sensors(ClockEnv()), Sensors(ClockEnv())
        original = Sensors.camera
        self.retries(old)
        with patch.object(online_ring_loop, 'OnlineRingSensors', Sensors):
            with pacing.installed():
                self.retries(new)
            self.assertIs(Sensors.camera, original)
        self.assertGreater(old.env.lag, 3.)
        self.assertLessEqual(new.env.lag, .601)
        self.assertAlmostEqual(old.env.now, new.env.now)
        self.assertEqual(new.env.deadline, 200.)
        self.assertEqual(new.env.reset_count, 1)
        self.assertEqual(new.views, old.views)

    def test_camera_uses_advanced_timestamp_and_propagates_timeout(self):
        sensor = Sensors(ClockEnv())
        sensor.env.now = 1.
        with patch.object(online_ring_loop, 'OnlineRingSensors', Sensors), pacing.installed():
            self.assertEqual(sensor.camera('wrist'), dict(view='wrist', time=1.))
            sensor.env.now = 201.
            with self.assertRaises(TimeoutError):
                sensor.camera()
        self.assertEqual(sensor.views, ['wrist'])

    def test_coordinator_launches_new_worker_and_stops_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root, evidence = Path(d) / 'resume', Path(d) / 'evidence'
            root.mkdir(); evidence.mkdir()
            (root / 'baseline_records.json').write_text('[]')
            slots = resume.study.schedule()
            reg = dict(evidence_root=str(evidence), completed_slots=slots[:160],
                       pending_slots=slots[160:], remaining=110)
            with patch.object(resume.gpu, 'check', return_value=reg), \
                 patch.object(resume.gpu, 'gpu_check', return_value={}), \
                 patch.object(resume.gpu, 'environment', return_value={}), \
                 patch.object(resume.gpu.subprocess, 'call', return_value=1) as launch:
                with self.assertRaisesRegex(ValueError, 'Worker failed'):
                    resume.execute(root)
                self.assertEqual(launch.call_count, 4)
                for call in launch.call_args_list:
                    command = call.args[0]
                    self.assertEqual(Path(command[2]).name, 'resume_online_extension_camera.py')
                    self.assertIn(int(command[-1]), range(160, 164))
                self.assertFalse(resume.read(root / 'STOP.json')['automatic_retry'])


if __name__ == '__main__':
    unittest.main()
