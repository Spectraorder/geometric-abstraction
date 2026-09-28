"""Matched public sensor/action boundary for Base and Full ring policies."""
import base64
import io
import numpy as np
from ring_peg_development import DevelopmentSensors


class ComparisonSensors(DevelopmentSensors):
    def __init__(self, env, output, arm):
        if arm not in ('basic', 'symbolic'):
            raise ValueError('Unknown experimental arm')
        super().__init__(env, output)
        self.arm = arm

    def task(self):
        task = super().task()
        task['geometry_available'] = self.arm == 'symbolic'
        return task

    def observe(self, name):
        if self.arm != 'symbolic':
            raise ValueError('Supplied object geometry is unavailable in Base; use calibrated camera observations.')
        return super().observe(name)

    def dispatch(self, method, params):
        if method == 'camera':
            frame = self.camera(**params)
            public = {key: frame[key] for key in
                      ('rgb', 'depth_m', 'valid_depth', 'intrinsics', 'camera_to_world', 'time')}
            stream = io.BytesIO()
            np.savez_compressed(stream, **public)
            return {'rgbd_npz_base64': base64.b64encode(stream.getvalue()).decode('ascii')}
        return super().dispatch(method, params)
