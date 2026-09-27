"""Measure frozen CPU geometry processing on one recorded occlusion case.

This is an in-memory replay microbenchmark, not end-to-end policy latency or an
agent speedup experiment. The simulator and model are never called.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(library, output, repeats, warmup):
    if output.exists():
        raise ValueError('Refusing to overwrite timing evidence')
    if repeats < 10 or warmup < 1:
        raise ValueError('Require at least ten measured repetitions and one warmup')
    packet_path = ROOT/'tests/fixtures/partial_cuboid_failure_rgbd.json'
    frame_path = packet_path.with_suffix('.npz')
    packet = json.loads(packet_path.read_text())
    with np.load(frame_path) as data:
        frames = {view:{k[len(view)+1:]:data[k] for k in data.files if k.startswith(view+'_')}
                  for view in ('side','wrist')}
    namespace = {}; exec(compile(library.read_text(), str(library), 'exec'), namespace)

    class RecordedAPI:
        def __init__(self): self.calls = []; self.records = []
        def gripper_geometry(self):
            self.calls.append('robot'); return packet['robot_geometry']
        def camera(self, view):
            self.calls.append(view); return frames[view]
        def proprioception(self):
            self.calls.append('proprioception'); return {'time':packet['time']}
        def record_geometry(self, row): self.records.append(row)

    def fresh_session():
        session = namespace['GeometryStepSession'].__new__(namespace['GeometryStepSession'])
        session.api = RecordedAPI(); session.colors = ['red']
        session.ledger = namespace['MetricScene']()
        session.backend = namespace['MixedPnPObservation'](
            [dict(color='red', **packet['descriptor'])], correspondence=True)
        return session

    rows = []; expected = None
    load_start = os.getloadavg()
    for iteration in range(warmup + repeats):
        session = fresh_session()
        w0, c0 = time.perf_counter_ns(), time.process_time_ns()
        try:
            session.current_observation(frames['side'], 'red', packet['visual_model'], packet['table_plane'])
        except ValueError as error:
            if 'wide hidden position intervals' not in str(error):
                raise
        else:
            raise ValueError('Frozen fixture unexpectedly acquired a point pose')
        w1, c1 = time.perf_counter_ns(), time.process_time_ns()
        record = session.checkpoint('red', 'place', action_status='exception: visual pose unavailable')
        serialized = json.dumps(record, separators=(',', ':'))
        w2, c2 = time.perf_counter_ns(), time.process_time_ns()
        if record['status'] != 'unavailable' or 'geometry' in record or 'constraints' not in record:
            raise ValueError('Constraint-only contract changed')
        if expected is None: expected = serialized
        if serialized != expected:
            raise ValueError('Repeated identical input changed the abstraction')
        if session.api.calls != ['robot','wrist','proprioception','proprioception']:
            raise ValueError('Unexpected sensing calls: '+str(session.api.calls))
        if iteration >= warmup:
            rows.append({'observation_wall_ms':(w1-w0)/1e6, 'observation_cpu_ms':(c1-c0)/1e6,
                         'checkpoint_and_json_wall_ms':(w2-w1)/1e6,
                         'checkpoint_and_json_cpu_ms':(c2-c1)/1e6,
                         'total_wall_ms':(w2-w0)/1e6, 'total_cpu_ms':(c2-c0)/1e6})
    summary = {key:{'median':float(np.median([r[key] for r in rows])),
                   'p95':float(np.percentile([r[key] for r in rows],95)),
                   'min':min(r[key] for r in rows), 'max':max(r[key] for r in rows)}
               for key in rows[0]}
    report = {'scope':__doc__, 'repetitions':repeats, 'warmup':warmup,
        'fixture':{'seed':990205,'time':packet['time'],'object':packet['object'],
                   'views':{k:list(v['depth_m'].shape) for k,v in frames.items()}},
        'checkpoint_characters':len(expected), 'checkpoint':json.loads(expected),
        'summary':summary, 'samples':rows, 'model_calls':0, 'simulator_calls':0,
        'environment':{'platform':platform.platform(), 'python':platform.python_version(),
                       'numpy':np.__version__, 'logical_cpus':os.cpu_count(),
                       'cpu_affinity':sorted(os.sched_getaffinity(0)),
                       'thread_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
                       'load_average_start':load_start, 'load_average_end':os.getloadavg()},
        'source_sha256':{str(p):sha(p) for p in (Path(__file__),library,packet_path,frame_path)},
        'limitations':['One repeated recorded occlusion case, not a representative frame or task distribution.',
            'Includes robot masking, original failed-view fits, extra partial-constraint fit, checkpoint and JSON.',
            'Excludes module import, initial object-model fitting, sensor IO, simulation, model reasoning and actions.',
            'CPU timings include all process threads; wall timings can be affected by concurrent campaign activity.',
            'Single-thread environment requested; no isolated-core or hardware-real-time guarantee.',
            'No VLM timing comparison, control-frequency claim or policy success benefit follows.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'repetitions':repeats,'checkpoint_characters':len(expected),'summary':summary},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--warmup',type=int,default=3)
    args=parser.parse_args()
    measure(args.library.resolve(),args.output.resolve(),args.repeats,args.warmup)
