"""Isolated policy client: only JSON RPC observations cross the process boundary."""
import base64
import json
import sys
import traceback
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
    def call(self, method, **params):
        print(json.dumps({'method':method,'params':params}),flush=True)
        line=sys.stdin.readline()
        if not line:
            raise RuntimeError('Simulator disconnected')
        result=json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return unpack(result['result'])
    def camera(self):
        return self.call('camera')
    def proprioception(self):
        return self.call('proprioception')
    def move_tool(self, position, seconds=1.5):
        return self.call('move_tool',position=np.asarray(position).tolist(),seconds=float(seconds))
    def move_joints(self, angles, seconds=1.5):
        return self.call('move_joints',angles=np.asarray(angles).tolist(),seconds=float(seconds))
    def gripper(self, angle, seconds=.6):
        return self.call('gripper',angle=float(angle),seconds=float(seconds))
    def wait(self, seconds):
        return self.call('wait',seconds=float(seconds))
    def note(self, text):
        return self.call('note',text=str(text)[:2000])


namespace={}
try:
    exec(compile(open(sys.argv[1]).read(),sys.argv[1],'exec'),namespace)
    namespace['run'](API())
    print(json.dumps({'method':'done','params':{}}),flush=True)
except BaseException as exc:
    print(json.dumps({'method':'failed','params':{'error':type(exc).__name__+': '+str(exc)}}),flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
