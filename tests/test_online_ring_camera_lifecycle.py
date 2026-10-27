"""Terminal capture must not re-enter physics or swallow live terminal signals."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import online_ring_camera_lifecycle as lifecycle
import online_ring_loop
import online_ring_scene


class Env:
    def __init__(self):
        self.steps = 0
        self.terminal = None
        self.snapshots = []

    def advance_while_waiting(self):
        if self.terminal is not None:
            raise self.terminal
        self.steps += 1

    def archive_terminal_physics(self, output):
        self.snapshots.append(self.steps)
        return {'steps': self.steps}


class Sensors:
    def __init__(self, env):
        self.env = env

    def camera(self, view='side'):
        return dict(view=view, steps=self.env.steps)


class LifecycleTests(unittest.TestCase):
    def test_success_and_timeout_final_images_freeze_at_archived_state(self):
        original_camera, original_archive = Sensors.camera, Env.archive_terminal_physics
        for terminal in (RuntimeError('success'), TimeoutError('deadline')):
            env = Env(); sensor = Sensors(env)
            with patch.object(online_ring_loop, 'OnlineRingSensors', Sensors), \
                 patch.object(online_ring_scene, 'OnlineRingEnv', Env), lifecycle.installed() as counts:
                self.assertEqual(sensor.camera()['steps'], 1)
                env.terminal = terminal
                state = env.archive_terminal_physics(Path('unused'))
                self.assertEqual(sensor.camera('side')['steps'], state['steps'])
                self.assertEqual(env.steps, 1)
                self.assertEqual(counts, dict(live_camera_attempts=1, terminal_camera_attempts=1,
                                              terminal_archives=1))
            self.assertIs(Sensors.camera, original_camera)
            self.assertIs(Env.archive_terminal_physics, original_archive)

    def test_live_deadline_is_not_suppressed_by_terminal_capture_guard(self):
        env = Env(); env.terminal = TimeoutError('deadline')
        with patch.object(online_ring_loop, 'OnlineRingSensors', Sensors), \
             patch.object(online_ring_scene, 'OnlineRingEnv', Env), lifecycle.installed():
            with self.assertRaisesRegex(TimeoutError, 'deadline'):
                Sensors(env).camera()
        self.assertEqual(env.steps, 0)

    def test_only_terminal_environment_stops_pacing(self):
        finished, live = Env(), Env()
        with patch.object(online_ring_loop, 'OnlineRingSensors', Sensors), \
             patch.object(online_ring_scene, 'OnlineRingEnv', Env), lifecycle.installed():
            finished.archive_terminal_physics(Path('unused'))
            Sensors(finished).camera()
            Sensors(live).camera()
        self.assertEqual(finished.steps, 0)
        self.assertEqual(live.steps, 1)


if __name__ == '__main__':
    unittest.main()
