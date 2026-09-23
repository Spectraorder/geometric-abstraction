"""Offline perception audit and CPU benchmark on recorded initial RGB-D frames.

Only this evaluator reads private trace labels. The actor module takes the
camera frame alone. Timing excludes loading, label matching and GPU rendering.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from aspire_so100.instance_geometry import inventory_instances


def audit(directories, repeats=3):
    frames, confusion, failures = [], Counter(), []
    matched = correct = missed = extras = roles_correct = 0
    for directory in directories:
        with np.load(directory / 'camera_001.npz', allow_pickle=False) as archive:
            frame = dict(archive)
        frames.append(frame)
        detections, _ = inventory_instances(frame)
        trace = json.loads((directory / 'trace.json').read_text())
        truth = trace['primitives'][0]['before']['objects']
        names = list(truth)
        costs = np.full((len(names), len(detections)), 1e6)
        for i, name in enumerate(names):
            for j, detection in enumerate(detections):
                if name.split('_', 1)[0] == detection['color']:
                    costs[i, j] = np.linalg.norm(np.array(truth[name]['position'])[:2] - detection['center'][:2])
        rows, cols = linear_sum_assignment(costs)
        seen = set()
        right_roles = set()
        for i, j in zip(rows, cols):
            if costs[i, j] > .035:
                continue
            name = names[i]
            expected = name.split('_', 1)[1]
            actual = detections[j]['shape']
            matched += 1
            seen.add(j)
            correct += expected == actual
            confusion[(expected, actual)] += 1
            if expected != actual:
                failures.append(dict(directory=str(directory), object=name, predicted=actual))
            elif name in ('red_thin_prism', 'blue_cube'):
                color = name.split('_', 1)[0]
                if sum(d['shape'] == actual and d['color'] == color for d in detections) == 1:
                    right_roles.add(name)
        missed += len(names)-len(seen)
        extras += len(detections)-len(seen)
        roles_correct += len(right_roles) == 2
    # Warm up numerical imports/caches before timing; frames already in memory.
    for frame in frames[:3]:
        inventory_instances(frame)
    cpu, wall = [], []
    for _ in range(repeats):
        for frame in frames:
            c, w = time.process_time(), time.perf_counter()
            inventory_instances(frame)
            wall.append(time.perf_counter()-w)
            cpu.append(time.process_time()-c)
    return dict(frames=len(frames), matched_objects=matched, correctly_classified=correct,
                missed_objects=missed, extra_detections=extras, unique_correct_roles=roles_correct,
                confusion={a+' -> '+b: n for (a, b), n in sorted(confusion.items())},
                failures=failures, benchmark=dict(samples=len(cpu), machine=platform.machine(),
                    cpu_ms_p50=float(np.percentile(cpu, 50)*1000),
                    cpu_ms_p95=float(np.percentile(cpu, 95)*1000),
                    wall_ms_p50=float(np.percentile(wall, 50)*1000),
                    wall_ms_p95=float(np.percentile(wall, 95)*1000),
                    scope='one initial RGB-D frame: table, components and shapes; preloaded; excludes simulator, rendering, IO, tracking, servo and IK'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=Path, nargs='+', required=True, help='Folders containing seed_*/result.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directories = sorted(p.parent for root in args.trials for p in root.glob('seed_*/result.json'))
    if not directories:
        parser.error('No completed trials')
    report = audit(directories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
