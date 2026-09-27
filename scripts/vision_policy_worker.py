"""Isolated policy client: only JSON RPC observations cross the process boundary."""
import base64
import json
import sys
import traceback
import time
import numpy as np


def unpack(value):
    if isinstance(value, dict) and '__array__' in value:
        return np.frombuffer(base64.b64decode(value['__array__']),dtype=value['dtype']).reshape(value['shape']).copy()
    if isinstance(value,dict):
        return {k:unpack(v) for k,v in value.items()}
    if isinstance(value,list):
        return [unpack(v) for v in value]
    return value


class API:
    def __init__(self):
        self.rpc_cpu_seconds = 0.0
        self.compute_intervals = []
        self.last_return_cpu = time.process_time()

    def call(self, method, **params):
        started = time.process_time()
        self.compute_intervals.append(started - self.last_return_cpu)
        try:
            return self.exchange(method, **params)
        finally:
            self.last_return_cpu = time.process_time()
            self.rpc_cpu_seconds += self.last_return_cpu - started

    def exchange(self, method, **params):
        print(json.dumps({'method':method,'params':params}),flush=True)
        line=sys.stdin.readline()
        if not line:
            raise RuntimeError('Simulator disconnected')
        result=json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return unpack(result['result'])
    def camera(self, view='side'):
        return self.call('camera', view=view)
    def gripper_geometry(self):
        return self.call('gripper_geometry')
    def record_geometry(self, record):
        return self.call('record_geometry', record=record)
    def task(self):
        return self.call('task')
    def point_target(self, color, shape, orientation, frame_time):
        return self.call('point_target', color=color, shape=shape,
                         orientation=orientation, frame_time=float(frame_time))
    def proprioception(self):
        return self.call('proprioception')
    def move_tool(self, position, seconds=1.5, orientation=None):
        params = {'position': np.asarray(position).tolist(), 'seconds': float(seconds)}
        if orientation is not None:
            params['orientation'] = np.asarray(orientation).tolist()
        return self.call('move_tool', **params)
    def plan_tool(self, position, orientation=None):
        params = {'position': np.asarray(position).tolist()}
        if orientation is not None:
            params['orientation'] = np.asarray(orientation).tolist()
        return self.call('plan_tool', **params)
    def move_joints(self, angles, seconds=1.5):
        return self.call('move_joints',angles=np.asarray(angles).tolist(),seconds=float(seconds))
    def gripper(self, angle, seconds=.6):
        return self.call('gripper',angle=float(angle),seconds=float(seconds))
    def wait(self, seconds):
        return self.call('wait',seconds=float(seconds))
    def note(self, text):
        return self.call('note',text=str(text)[:2000])


namespace={}
api = API()
started_cpu = time.process_time()


def profile():
    intervals = api.compute_intervals + [time.process_time() - api.last_return_cpu]
    return {'policy_compute_cpu_seconds': time.process_time() - started_cpu - api.rpc_cpu_seconds,
            'rpc_cpu_seconds': api.rpc_cpu_seconds,
            'compute_interval_count': len(intervals),
            'compute_interval_cpu_p50_ms': float(np.percentile(intervals, 50)*1000),
            'compute_interval_cpu_p95_ms': float(np.percentile(intervals, 95)*1000),
            'scope': 'worker compile/import/run; excludes RPC serialization, simulator, rendering and host IK'}


try:
    if len(sys.argv) == 3:
        geometry_namespace = {}
        exec(compile(open(sys.argv[2]).read(), sys.argv[2], 'exec'), geometry_namespace)
        namespace['geometry'] = geometry_namespace['GEOMETRY']
    exec(compile(open(sys.argv[1]).read(),sys.argv[1],'exec'),namespace)
    namespace['run'](api)
    print(json.dumps({'method':'done','params':{'cpu_profile': profile()}}),flush=True)
except BaseException as exc:
    print(json.dumps({'method':'failed','params':{'error':type(exc).__name__+': '+str(exc), 'cpu_profile': profile()}}),flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
