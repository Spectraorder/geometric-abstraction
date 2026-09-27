"""Validate partial-constraint feedback on five development training trajectories.

No actor calls or held-out layouts. Require unchanged actions, sensing and physics;
only compact failed-fit feedback may differ. This is not an SR-effect experiment.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from build_partial_geometry_session import library as extend
from run_visual_guard_counterfactual import digest, read, write, ROOT


def ordinary(calls):
    result = []
    for original in calls:
        if original['method'] == 'record_geometry': continue
        call = json.loads(json.dumps(original))
        if call['method'] == 'done': call['params'].pop('cpu_profile', None)
        result.append(call)
    return result


def run(parent, output):
    parent_manifest = read(parent/'manifest.json')
    study = Path(parent_manifest['parent']); manifest = read(study/'manifest.json')
    if read(parent/'summary.json')['status'] != 'completed': raise ValueError('Require completed parent')
    output.mkdir(parents=True, exist_ok=False)
    candidate = output/'candidate.py'; shutil.copy2(parent/'guard_5mm.py', candidate)
    lib = output/'library.py'; lib.write_text(extend((parent/'symbolic_library.py').read_text()))
    source_paths = [Path(__file__), ROOT/'scripts/build_partial_geometry_session.py', ROOT/'policies/partial_geometry_state.py']
    for path in source_paths: shutil.copy2(path, output/path.name)
    protocol = {'scope': __doc__, 'parent': str(parent), 'seeds': parent_manifest['seeds'],
        'heldout_seeds_used': [], 'model_calls': 0, 'not_actor_feedback': True,
        'candidate_sha256': digest(candidate), 'library_sha256': digest(lib),
        'parent_manifest_sha256': digest(parent/'manifest.json'),
        'sources': {str(p): digest(p) for p in source_paths},
        'required_equality': ['motion and sensing API calls (excluding terminal CPU profile)',
            'camera archives', 'opaque rollout archive', 'original geometry records after removing new constraints'],
        'success_rate_claim': False}
    write(output/'manifest.json', protocol)
    rows = []
    for seed in protocol['seeds']:
        previous = parent/'trials'/f'seed_{seed}'; trial = output/'trials'/f'seed_{seed}'
        command = [sys.executable, str(ROOT/'scripts/run_vision_trial.py'), '--candidate', str(candidate),
            '--output', str(trial), '--seed', str(seed), '--task', 'inventory_structure', '--layout', 'sampled',
            '--structure-recipe', manifest['recipe'], '--structure-color-map', json.dumps(manifest['color_maps'][str(seed)]),
            '--object-count', '4', '--wrist-depth', '--record-geometry', '--timeout', '120', '--geometry-library', str(lib)]
        with (output/f'seed_{seed}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='egl'),
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
        a, b = read(trial/'result.json'), read(previous/'result.json')
        for key in ('seed','candidate_sha256','task','layout','scene','physics','sensor','wrist_depth','sensor_blackout','success','policy_error'):
            if a[key] != b[key]: raise ValueError('Prototype changed '+key)
        if a['geometry_library_sha256'] != digest(lib): raise ValueError('Incorrect library')
        if ordinary(read(trial/'api_calls.json')) != ordinary(read(previous/'api_calls.json')):
            raise ValueError('Action/sensor calls changed')
        names = sorted(p.name for p in trial.glob('camera_*.npz'))
        if names != sorted(p.name for p in previous.glob('camera_*.npz')): raise ValueError('Camera sequence changed')
        for name in [*names, 'initial.png', 'rollout.npz']:
            # Physics archives are compared as opaque bytes, never read as policy inputs.
            if digest(trial/name) != digest(previous/name): raise ValueError('Image/physics differs: '+name)
        records = read(trial/'geometry_records.json') if (trial/'geometry_records.json').exists() else []
        prior_records = read(previous/'geometry_records.json') if (previous/'geometry_records.json').exists() else []
        stripped = json.loads(json.dumps(records)); partial = []
        for row in stripped:
            if row.get('kind') == 'geometry_step' and 'constraints' in row['record']:
                partial.append(row['record'].pop('constraints'))
        if stripped != prior_records: raise ValueError('Original geometry changed')
        row = {'seed': seed, 'equivalence': 'passed', 'success_unchanged': a['success'],
            'constraint_packets': partial,
            'max_checkpoint_chars': max([len(json.dumps(r['record'],separators=(',', ':'))) for r in records if r.get('kind')=='geometry_step'] or [0]),
            'source_hashes': {str(trial/n):digest(trial/n) for n in ['result.json','api_calls.json','rollout.npz',*names]}}
        rows.append(row); write(output/'progress.json', {'status':'running','rows':rows})
        print({k:v for k,v in row.items() if k not in ('source_hashes','constraint_packets')}, 'partial_records',len(partial),flush=True)
    report = {'status':'completed', 'scope':__doc__, 'protocol_sha256':digest(output/'manifest.json'),
        'equivalent_trials':len(rows), 'new_constraint_packets':sum(len(r['constraint_packets']) for r in rows),
        'unchanged_successes':sum(r['success_unchanged'] for r in rows), 'rows':rows,
        'limitations':['New feedback only; no actor has used these packets and no SR gain is claimed.',
            'Only unique rectangular-prism correspondence is supported; missing or ambiguous fits remain unknown.',
            'Finite bounds depend on fitted orientation and visual dimensions, not calibrated confidence.']}
    write(output/'summary.json',report);write(output/'progress.json',report)
    write(ROOT/'docs/evaluations/partial_geometry_development.json',report)
    print({k:report[k] for k in ('status','equivalent_trials','new_constraint_packets','unchanged_successes')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.parent.resolve(),args.output.resolve())
