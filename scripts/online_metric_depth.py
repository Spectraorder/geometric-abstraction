"""Shared point queries on the exact calibrated RGB-D frame shown to an agent.

This interface backprojects selected pixels. It does not segment objects, infer
hidden surfaces, choose supports, fit poses, or construct manipulation targets.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import numpy as np


class MetricDepthFrames:
    """Keep only the latest displayed frame per view, with explicit provenance."""
    def __init__(self, max_points=16):
        if type(max_points) is not int or max_points < 1:
            raise ValueError('max_points must be a positive integer')
        self.max_points = max_points
        self.frames = {}
        self.sequence = 0

    def register(self, view, frame):
        if view not in ('side', 'wrist'):
            raise ValueError('Unknown camera view')
        rgb = np.array(frame['rgb'], copy=True)
        depth = np.array(frame['depth_m'], dtype=float, copy=True)
        valid = np.array(frame['valid_depth'], copy=True)
        K = np.array(frame['intrinsics'], dtype=float, copy=True)
        T = np.array(frame['camera_to_world'], dtype=float, copy=True)
        stamp = float(frame['time'])
        if (depth.ndim != 2 or min(depth.shape) < 1 or rgb.shape != (*depth.shape, 3)
                or rgb.dtype != np.uint8 or valid.shape != depth.shape or valid.dtype != bool):
            raise ValueError('Mismatched RGB-D arrays or invalid mask type')
        if (not math.isfinite(stamp) or stamp < 0 or K.shape != (3, 3) or T.shape != (4, 4)
                or not np.isfinite(K).all() or not np.isfinite(T).all()):
            raise ValueError('Invalid calibration or timestamp')
        if (K[0,0] <= 0 or K[1,1] <= 0 or not np.allclose(K[2], [0,0,1])
                or abs(np.linalg.det(K)) < 1e-12 or not np.allclose(T[3], [0,0,0,1])
                or not np.allclose(T[:3,:3].T@T[:3,:3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(T[:3,:3]), 1., atol=1e-6)):
            raise ValueError('Calibration must describe pinhole rays and a rigid transform')
        self.sequence += 1
        identifier = f'{view}:{self.sequence}'
        descriptor = {'frame_id':identifier, 'view':view, 'observation_time':stamp,
                      'width':int(depth.shape[1]), 'height':int(depth.shape[0]),
                      'pixel_convention':'integer [u,v], origin at top left',
                      'depth_convention':'optical-axis meters', 'max_points_per_query':self.max_points}
        self.frames[view] = dict(descriptor=descriptor, depth=depth, valid=valid,
                                 inverse_K=np.linalg.inv(K), T=T)
        return deepcopy(descriptor)

    def sample(self, frame_id, pixels, *, current_time):
        if not isinstance(frame_id, str):
            raise ValueError('frame_id must identify a displayed image')
        if (not isinstance(pixels, list) or not 1 <= len(pixels) <= self.max_points
                or any(not isinstance(p, (list, tuple)) or len(p)!=2
                       or any(type(x) is not int for x in p) for p in pixels)):
            raise ValueError('pixels must contain 1 to max_points integer [u,v] pairs')
        now = float(current_time)
        if not math.isfinite(now):
            raise ValueError('A finite current sensor time is required')
        frame = next((f for f in self.frames.values() if f['descriptor']['frame_id']==frame_id), None)
        if frame is None:
            return {'status':'frame_unavailable', 'frame_id':frame_id,
                    'reason':'Only the latest displayed frame of each view is retained'}
        descriptor = frame['descriptor']
        stamp = descriptor['observation_time']
        if now < stamp:
            raise ValueError('A displayed observation cannot come from the future')
        samples = []
        for u,v in pixels:
            row = {'pixel':[u,v]}
            if not (0 <= u < descriptor['width'] and 0 <= v < descriptor['height']):
                row['status'] = 'out_of_frame'
            else:
                z = float(frame['depth'][v,u])
                if not frame['valid'][v,u] or not math.isfinite(z) or z <= 0:
                    row['status'] = 'invalid_depth'
                else:
                    with np.errstate(over='ignore', invalid='ignore'):
                        camera = z*(frame['inverse_K']@np.array([u,v,1.]))
                        world = frame['T'][:3,:3]@camera+frame['T'][:3,3]
                    if np.isfinite(world).all():
                        row.update(status='observed_surface_sample', depth_m=z, world_point_m=world.tolist())
                    else:
                        row['status'] = 'invalid_projection'
            samples.append(row)
        return dict(descriptor, status='sampled', current_time=now,
                    at_current_sensor_time=now==stamp, age_seconds=now-stamp, samples=samples,
                    semantics='Visible surface samples at observation_time, not object centers or current poses')


class MetricDepthSensorMixin:
    """Bind depth queries to existing image archives without additional captures."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_frames = MetricDepthFrames()
        self.depth_query_count = 0
        self.depth_query_seconds = 0.

    def image(self, name, view='side'):
        path, calibration = super().image(name, view)
        # The raw sensor archive is produced by the same image() call. Reading it
        # avoids a second capture whose time or rendered content could differ.
        with np.load(Path(path).with_suffix('.npz')) as frame:
            descriptor = self.metric_frames.register(view, frame)
        return path, dict(calibration, metric_depth_frame=descriptor)

    def sample_depth(self, frame_id, pixels):
        started = time.perf_counter()
        self.depth_query_count += 1
        try:
            result = self.metric_frames.sample(frame_id, pixels,
                                              current_time=self.proprioception()['time'])
        except (ValueError, TypeError) as error:
            result = {'status':'invalid_query', 'reason':str(error)}
        elapsed = time.perf_counter()-started
        self.depth_query_seconds += elapsed
        record = {'frame_id':frame_id, 'pixels':pixels, 'result':result, 'query_seconds':elapsed}
        try:
            encoded = json.dumps(record, allow_nan=False)
        except (TypeError, ValueError):
            encoded = json.dumps({'invalid_input_repr':repr((frame_id,pixels)),
                                  'result':result, 'query_seconds':elapsed}, allow_nan=False)
        with (self.output/'depth_queries.jsonl').open('a') as stream:
            stream.write(encoded+'\n')
        return result
