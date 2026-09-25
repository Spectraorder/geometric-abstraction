"""Local fault-injection adapter; original runner, worker sandbox and evaluator stay intact."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / 'scripts/run_vision_trial.py'

class Faults:
    def __init__(self, kind):
        self.kind = kind
        self.events = []
        self.side_count = 0
        self.fired = False
        self.armed = False
        self.environment = None

    def log(self, env, event, **detail):
        self.events.append({'time': float(env.data.time), 'event': event, **detail})

    def camera(self, env, frame, view):
        if view != 'side': return
        self.side_count += 1
        limit = 2 if self.kind == 'occlusion' else 999 if self.kind == 'persistent_occlusion' else 0
        if self.side_count > limit: return
        # An image-space occluding rectangle located from RGB, never an object ID/pose.
        r, g, b = frame['rgb'].astype(float).transpose(2, 0, 1)
        mask = (b > 65) & (b > 1.55*r) & (b > 1.2*g)
        ys, xs = np.nonzero(mask)
        if not len(xs): raise RuntimeError('Fault setup: no blue pixels available')
        y0, y1 = max(0, int(ys.min())-4), min(len(r), int(ys.max())+5)
        x0, x1 = max(0, int(xs.min())-4), min(r.shape[1], int(xs.max())+5)
        frame['rgb'][y0:y1, x0:x1] = 55
        frame['depth_m'][y0:y1, x0:x1] = np.nan
        frame['valid_depth'][y0:y1, x0:x1] = False
        self.log(env, 'sensor_rectangle', side_camera=self.side_count, rectangle=[x0,y0,x1,y1], mechanism='synthetic image occlusion; not a physical occluder')

    def before(self, env, method, params):
        self.environment = env
        if method != 'gripper' or params['angle'] >= 0 or self.fired or self.armed: return
        if self.kind == 'miss':
            self.fired = True
            self.log(env, 'jaw_close_suppressed', requested=params['angle'], applied=.8)
            params['angle'] = .8
        if self.kind == 'slip':
            self.armed = True
            origin_z = float(env.data.site_xpos[env.site_id,2])
            original_step = env._step
            active_until = [None]
            def step():
                if not self.fired and float(env.data.site_xpos[env.site_id,2]) > origin_z+.035:
                    self.fired = True
                    active_until[0] = float(env.data.time)+.45
                    self.log(env, 'jaw_open_during_lift', robot_lift_m=.035, duration_s=.45, mechanism='actuator disturbance, not friction randomization')
                if active_until[0] is not None:
                    if env.data.time < active_until[0]: env.data.ctrl[5] = .8
                    else:
                        env.data.ctrl[5] = -.174
                        active_until[0] = None
                        self.log(env, 'jaw_command_restored')
                original_step()
            env._step = step

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fault', choices=['none','occlusion','persistent_occlusion','miss','slip'], required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=4001)
    args = p.parse_args()
    if args.output.exists(): p.error('Output must be new')
    fault = Faults(args.fault)
    source = RUNNER.read_text()
    hooks = [
        ("                        try:\n                            response=None", "                        try:\n                            params=dict(params)\n                            fault.before(env,method,params)\n                            response=None"),
        ("                                if a.sensor_blackout:", "                                fault.camera(env,response,view)\n                                if a.sensor_blackout:")]
    for old, new in hooks:
        if source.count(old) != 1: raise RuntimeError('Upstream runner changed; review adapter before running')
        source = source.replace(old, new)
    module = types.ModuleType('local_fault_runner')
    module.__file__ = str(RUNNER)
    module.__dict__['fault'] = fault
    sys.modules[module.__name__] = module
    exec(compile(source, str(RUNNER), 'exec'), module.__dict__)
    sys.argv = [str(RUNNER), '--candidate', str(args.candidate), '--output', str(args.output), '--seed', str(args.seed), '--task', 'tower', '--layout', 'random_xy']
    try: module.main()
    finally:
        if args.output.exists():
            (args.output/'fault_events.private.json').write_text(json.dumps({'fault':args.fault, 'events':fault.events, 'runner_sha256':hashlib.sha256(RUNNER.read_bytes()).hexdigest(), 'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'actor_receives_fault_metadata':False},indent=2)+'\n')

if __name__ == '__main__': main()
