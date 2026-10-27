"""Opt-in online pacing amendment; leave all frozen episode sources untouched.

Ring's local skill can retry unavailable observations without returning to the
agent loop. Advance elapsed physics before each new camera sample so successive
CPU fits do not accumulate an unserviced lag. Individual slow calls still count
against the original lag gate and the same wall deadline.
"""
from contextlib import contextmanager
from functools import wraps
from unittest.mock import patch


@contextmanager
def installed():
    from online_ring_loop import OnlineRingSensors
    original = OnlineRingSensors.camera

    @wraps(original)
    def camera(self, view='side'):
        self.env.advance_while_waiting()
        return original(self, view)

    with patch.object(OnlineRingSensors, 'camera', camera):
        yield
