"""Development-only one-line guard counterfactual on five existing training layouts.

No model calls, reserved layouts, changed libraries, or repairs to study artifacts.
This diagnoses a controller-contract mismatch, not an abstraction treatment effect.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import run_visual_input_experiment as variant

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def run(parent, output):
    manifest = read(parent/'manifest.json')
    variant.check(parent, manifest)
    if read(parent/'summary.json')['status'] != 'completed':
        raise ValueError('Parent study must be completed before this diagnostic')
    original = parent/'symbolic/round_003/bundled.py'
    before = original.read_text()
    old = 'if not np.isfinite(error) or error > .0025:'
    new = 'if not np.isfinite(error) or error > .005:'
    if before.count(old) != 1:
        raise ValueError('Expected exactly one workspace IK guard')
    after = before.replace(old, new, 1)
    variant.base.validate_source(after)
    output.mkdir(parents=True, exist_ok=False)
    candidate = output/'guard_5mm.py';candidate.write_text(after)
    shutil.copy2(original, output/'original_bundled.py')
    library = output/'symbolic_library.py'
    shutil.copy2(parent/'symbolic_library.py', library)
    shutil.copy2(Path(__file__), output/'driver_snapshot.py')
    protocol = {'scope': __doc__, 'parent': str(parent),
                'parent_manifest_sha256': digest(parent/'manifest.json'),
                'original_candidate_sha256': digest(original),
                'candidate_sha256': digest(candidate), 'library_sha256': digest(library),
                'driver_sha256': digest(Path(__file__)),
                'exact_replacement': {'before': old, 'after': new},
                'seeds': manifest['train_seeds'], 'model_calls': 0,
                'heldout_seeds_used': [], 'not_actor_feedback': True}
    write(output/'manifest.json', protocol)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='egl')
    rows = []
    for seed in manifest['train_seeds']:
        trial = output/'trials'/f'seed_{seed}'
        command = [sys.executable, str(ROOT/'scripts/run_vision_trial.py'),
                   '--candidate', str(candidate), '--output', str(trial), '--seed', str(seed),
                   '--task', 'inventory_structure', '--layout', 'sampled',
                   '--structure-recipe', manifest['recipe'],
                   '--structure-color-map', json.dumps(manifest['color_maps'][str(seed)]),
                   '--object-count', '4', '--wrist-depth', '--record-geometry', '--timeout', '120',
                   '--geometry-library', str(library)]
        with (output/f'seed_{seed}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=240)
        prior = parent/'symbolic/round_003/trials'/f'seed_{seed}'
        result, previous = read(trial/'result.json'), read(prior/'result.json')
        if result['candidate_sha256'] != digest(candidate) or previous['candidate_sha256'] != digest(original):
            raise ValueError('Candidate provenance changed')
        for key in ('seed','task','layout','scene','physics','sensor','wrist_depth','sensor_blackout','geometry_library_sha256'):
            if result[key] != previous[key]:
                raise ValueError('Counterfactual settings changed: '+key)
        if digest(trial/'initial.png') != digest(prior/'initial.png'):
            raise ValueError('Initial RGB differs')
        calls, previous_calls = read(trial/'api_calls.json'), read(prior/'api_calls.json')
        motions = lambda data: sum(c['method'] in ('move_tool','move_joints','gripper') for c in data)
        row = {'seed': seed, 'original_success': previous['success'], 'modified_success': result['success'],
               'original_motion_commands': motions(previous_calls), 'modified_motion_commands': motions(calls),
               'original_policy_error': previous['policy_error'], 'modified_policy_error': result['policy_error'],
               'notes': [c['params']['text'] for c in calls if c['method']=='note'],
               'initial_rgb_identical': True,
               'artifact_sha256': {str(p): digest(p) for folder in (prior, trial)
                                  for p in (folder/'result.json',folder/'api_calls.json',folder/'initial.png')}}
        rows.append(row)
        write(output/'progress.json', {'status':'running','completed':len(rows),'rows':rows})
        print({k:v for k,v in row.items() if k not in ('artifact_sha256','notes')}, flush=True)
    variant.check(parent, manifest)
    report = {'status':'completed', 'scope':__doc__, 'protocol_sha256':digest(output/'manifest.json'),
              'original_successes':sum(r['original_success'] for r in rows),
              'modified_successes':sum(r['modified_success'] for r in rows),
              'trials':len(rows), 'rows':rows,
              'limitations':['Post-training intervention on five inspected training layouts.',
                             'One changed numeric guard; other conservative checks remain.',
                             'No new held-out result and no abstraction-effect inference.',
                             'Original learning curves remain unchanged. No actor receives this repaired candidate.']}
    write(output/'summary.json', report);write(output/'progress.json', report)
    write(ROOT/'docs/evaluations/visual_guard_threshold_development.json',report)
    print('SUMMARY '+json.dumps({k:report[k] for k in ('original_successes','modified_successes','trials')}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args();run(args.parent.resolve(),args.output.resolve())
