"""Reuse sensor frames only within an explicit, synchronous decision readout."""
from contextlib import contextmanager
from copy import deepcopy
import math


class DecisionFrameReuse:
    """Mixin for the shared sensor API; does not cache object pose estimates.

    Use before the sensor API in the inheritance list. Scope reuse around decision
    evidence and feedback preparation, never around execution or model waits.
    """
    @contextmanager
    def decision_frames(self):
        if getattr(self, '_decision_frames', None) is not None:
            raise RuntimeError('Decision frame scopes cannot be nested')
        self._decision_frames = {}
        self._decision_frame_time = None
        self.frame_reuse_hits = 0
        try:
            yield self
        finally:
            self._decision_frames = None
            self._decision_frame_time = None

    def camera(self, view='side'):
        cache = getattr(self, '_decision_frames', None)
        if cache is None:
            return super().camera(view)
        stamp = float(self.proprioception()['time'])
        if not math.isfinite(stamp):
            raise ValueError('Cannot reuse frames without finite sensor time')
        if stamp != self._decision_frame_time:
            cache.clear()
            self._decision_frame_time = stamp
        if view in cache:
            self.frame_reuse_hits += 1
            return deepcopy(cache[view])
        frame = super().camera(view)
        # A readout that advances time cannot seed this same-time cache.
        if float(frame['time']) == stamp == float(self.proprioception()['time']):
            cache[view] = deepcopy(frame)
        return frame
