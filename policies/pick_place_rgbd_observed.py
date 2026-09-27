"""Optional observation callback for the unchanged shared PnP motion loop.

Bundle after pick_place_rgbd_shared.py. Every inherited grasp, motion, visual
check and release timing stays identical. The callback sees only a fresh RGB-D
frame, color, previously learned dimension model and measured table plane.
"""
import numpy as np


class ObservedRGBDPickPlace(RGBDPickPlace):
    observer = None

    def set_observer(self, observer):
        """Select a backend before learning models; never mix model schemas."""
        if self.models or self.held is not None:
            raise ValueError('Select the observer before observing or picking objects')
        if observer is not None and not callable(observer):
            raise ValueError('Observer must be callable or None')
        self.observer = observer

    def observe(self, color, near=None):
        if self.observer is None:
            return RGBDPickPlace.observe(self, color, near)
        if color not in ('red', 'yellow', 'blue', 'green'):
            raise ValueError('Expected a supported unique color')
        frame = self.api.camera()
        measured = self.observer(frame, color, self.models.get(color), self.plane.copy())
        if measured.get('coordinate_frame') != 'world' or measured.get('units') != 'meters':
            raise ValueError('Observer must return world geometry in meters')
        stamp = float(measured['observation_time'])
        if not np.isfinite(stamp) or abs(stamp-float(frame['time'])) > 1e-6:
            raise ValueError('Observer must use the current camera timestamp')
        center = np.asarray(measured['center'], float)
        normal = np.asarray(measured['normal'], float)
        dims = np.asarray(measured['dims'], float)
        scalars = np.asarray([measured[k] for k in ('height', 'top', 'bottom', 'extent_z')], float)
        if (center.shape != (3,) or normal.shape != (3,) or dims.shape != (2,)
                or not all(np.isfinite(v).all() for v in (center, normal, dims, scalars))
                or np.any(dims <= 0) or scalars[0] <= 0 or scalars[3] <= 0
                or abs(np.linalg.norm(normal)-1.) > 1e-5 or normal[2] <= 0
                or measured['bottom'] > measured['top']
                or abs(center[2]-measured['extent_z']-measured['bottom']) > 1e-6):
            raise ValueError('Invalid rigid observation for shared PnP')
        if near is not None and np.linalg.norm(center-np.asarray(near)) > .12:
            raise RuntimeError('Observed object is too far from the expected tool neighborhood')
        self.models[color] = measured['model']
        return measured
