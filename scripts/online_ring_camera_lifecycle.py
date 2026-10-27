"""Pace live Ring sensing, but freeze physics before terminal evidence capture.

The episode runner archives physics before taking its final image. That archive
entry is an explicit terminal boundary: final images must not step the world or
re-enter success/timeout monitoring. This opt-in wrapper leaves frozen sources
and the live sensing/deadline behavior unchanged.
"""
from contextlib import contextmanager
from functools import wraps
from unittest.mock import patch


@contextmanager
def installed():
    from online_ring_loop import OnlineRingSensors
    from online_ring_scene import OnlineRingEnv
    original_camera = OnlineRingSensors.camera
    original_archive = OnlineRingEnv.archive_terminal_physics
    counts = dict(live_camera_attempts=0, terminal_camera_attempts=0, terminal_archives=0)

    @wraps(original_archive)
    def archive(self, *args, **kwargs):
        self._ring_terminal_archiving = True
        counts['terminal_archives'] += 1
        return original_archive(self, *args, **kwargs)

    @wraps(original_camera)
    def camera(self, view='side'):
        if getattr(self.env, '_ring_terminal_archiving', False):
            counts['terminal_camera_attempts'] += 1
        else:
            counts['live_camera_attempts'] += 1
            self.env.advance_while_waiting()
        return original_camera(self, view)

    with patch.object(OnlineRingEnv, 'archive_terminal_physics', archive), \
         patch.object(OnlineRingSensors, 'camera', camera):
        yield counts
