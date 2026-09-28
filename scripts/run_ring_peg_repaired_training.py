"""A separately registered Full training pilot after ring interface repairs."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch
import run_ring_peg_training as original

ROOT = original.ROOT
ADAPTER_FILES = ('ring_peg_development.py', 'check_ring_peg_repair.py', 'run_ring_peg_repaired_training.py')
CONTRACT = original.CONTRACT + '''
This separately registered run uses calibrated 640x480 RGB-D for local perception.
Color association includes every declared scene color. Current partial-boundary
validation uses perspective camera-ray intersections with the cap plane.
plan_grasp checks the requested direction at the final fixed-pad IK solution;
its returned closing_alignment is a cosine and fixed_pad_error_m is in meters.
These are kinematic predictions, not contact or grasp-success measurements.
'''
_original_sources = original.sources


def sources():
    result = _original_sources()
    result.update({str(Path('scripts')/name): original.digest(ROOT/'scripts'/name) for name in ADAPTER_FILES})
    return result


def rollout(root, candidate, stage, seed):
    original.check(root)
    output = stage/'trials'/f'seed_{seed}'
    if output.exists():
        raise RuntimeError('Existing attempt retained; no automatic retry')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.log').open('x') as log:
        subprocess.run([sys.executable, 'scripts/check_ring_peg_repair.py', '--output', str(output),
                        '--candidate', str(candidate), '--seed', str(seed)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180,
                       env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='egl'))
    return original.read(output/'result.json')


@contextmanager
def installed():
    with patch.object(original, 'sources', sources), patch.object(original, 'CONTRACT', CONTRACT), \
            patch.object(original, 'rollout', rollout):
        yield


def prepare(root):
    # This reuses training layouts only. No held-out outcomes enter development.
    initial = ROOT/'outputs/ring_peg_dev/repair_initial_01'
    replay = ROOT/'outputs/ring_peg_dev/repair_policy_01'
    physical = ROOT/'outputs/ring_peg_dev/repair_physical_01'
    evidence = []
    for seed in original.SEEDS:
        result = initial/f'seed_{seed}/result.json'
        if not original.read(result)['initialization_complete']:
            raise RuntimeError('Repaired initialization has not passed all training layouts')
        evidence.append(result)
    for seed in (1, 2, 3):
        folder = replay/f'seed_{seed}'
        records = original.read(folder/'sensor_records.json')
        if not any(row.get('note') == 'Vision verified ring lift.' for row in records):
            raise RuntimeError('Repaired development lift has not been visually verified')
        evidence.extend((folder/'result.json', folder/'sensor_records.json', physical/f'seed_{seed}/result.json'))
    manifest = original.prepare(root)
    manifest.update(schema='ring_peg_single_policy_repaired_v1',
                    preceding_training='outputs/ring_peg_full_train_01',
                    sensor_resolution=[640, 480],
                    layout_scope='Reuse the five development/training layouts; no reserved evaluation.',
                    development_outcome='Initialization passed 5/5. Existing R1 verified lift on 3/3 development seeds but completed insertion on 0/3. The earlier development controller also completed insertion on 0/3 with this interface. Full end-to-end policy success remains unestablished.',
                    repair_evidence={str(p.relative_to(ROOT)): original.digest(p) for p in evidence})
    for path in evidence:
        target = root/'development_evidence'/path.relative_to(ROOT/'outputs/ring_peg_dev')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    original.write(root/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'train', 'check'), required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    try:
        with installed():
            if args.phase == 'prepare':
                print(prepare(root)['schema'])
            elif args.phase == 'check':
                print(original.check(root)['schema'])
            else:
                original.train(root)
    except BaseException as error:
        if root.exists():
            original.write(root/'STOP.json', {'error': type(error).__name__+': '+str(error), 'time': time.time()})
        raise
