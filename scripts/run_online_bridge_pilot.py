"""Freeze and execute a small, sequential, paired online bridge-tower pilot."""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
import time

from build_online_bridge_geometry import online_library
from online_bridge_skills import SOURCES
from run_online_bridge_episode import run
from summarize_online_bridge_pilot import summarize, plot

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(2028101001, 2028101006))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def source_paths():
    paths = {p for name in ('scripts', 'policies', 'prototypes', 'src/aspire_so100')
             for p in (ROOT/name).rglob('*.py') if '__pycache__' not in p.parts}
    paths.update(ROOT/name for name, _ in SOURCES.values())
    paths.update(ROOT.glob('*.py'))
    paths.update(p for p in (ROOT/'assets').rglob('*') if p.is_file())
    paths.update(ROOT/name for name in ('pyproject.toml', 'uv.lock') if (ROOT/name).exists())
    return sorted(paths)


def package_versions():
    return {name: importlib.metadata.version(name)
            for name in ('numpy', 'scipy', 'mujoco', 'mjlab', 'pillow')}


def client_version():
    return subprocess.check_output(['codex', '--version'], text=True).strip()


def make_slots(seeds=SEEDS, repetitions=2):
    slots = []
    for repetition in range(repetitions):
        for index, seed in enumerate(seeds):
            arms = ('basic', 'symbolic') if (index+repetition) % 2 == 0 else ('symbolic', 'basic')
            for arm in arms:
                slots.append({'seed': seed, 'repetition': repetition, 'arm': arm,
                              'path': f'episodes/seed_{seed}/repeat_{repetition}/{arm}'})
    return slots


def prepare(root, development):
    if root.exists():
        raise FileExistsError('Registration output already exists')
    development = Path(development)
    results = [json.loads(p.read_text()) for p in development.rglob('result.json')
               if 'sources' not in p.parts]
    observed_seeds = {r['seed'] for r in results if 'seed' in r}
    if not set(range(2028100101, 2028100104)).issubset(observed_seeds):
        raise ValueError('Validate all three distinct development layouts before registration')
    if set(SEEDS) & observed_seeds:
        raise ValueError('Reserved layouts overlap development')
    online = [r for r in results if 'model_requests' in r]
    if not all(any(r.get('arm') == arm and r.get('success') for r in online)
               for arm in ('basic', 'symbolic')):
        raise ValueError('Each arm needs a successful continuous development episode')
    if not any(any(row.get('model') == 'gpt-6-luna' for row in r['model_requests'])
               for r in online):
        raise ValueError('No verified fast-model development decision')
    root.mkdir(parents=True)
    hashes = {}
    for path in source_paths():
        relative = path.relative_to(ROOT)
        target = root/'sources'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        hashes[str(relative)] = digest(path)
    geometry = online_library()
    (root/'geometry.py').write_text(geometry)
    protocol = {'name': 'online_bridge_tower_pilot', 'registered_at': time.time(),
        'seeds': list(SEEDS), 'development_seeds': list(range(2028100101, 2028100104)),
        'repetitions': 2, 'arms': ['basic', 'symbolic'], 'slots': make_slots(),
        'budget_s': 360., 'concurrency': 1,
        'episode_termination': 'Verified success or wall-clock timeout; infrastructure errors stop dispatch',
        'routing': {'initial_and_replanning': {'model': 'gpt-6-astra', 'effort': 'medium'},
                    'routine': {'model': 'gpt-6-luna', 'effort': 'low'},
                    'escalation': 'Explicit replan or two consecutive unconfirmed skill calls'},
        'success': 'Full private support-graph checks for five continuous simulated seconds, '
                   'with at most 3 mm object drift, within the wall-clock budget',
        'one_reset_per_episode': True, 'physics_during_model_wait': True,
        'paced_actions': True, 'max_timing_lag_for_realtime_claim_s': 2.,
        'shared_skills': SOURCES, 'source_sha256': hashes,
        'geometry_sha256': hashlib.sha256(geometry.encode()).hexdigest(),
        'package_versions': package_versions(),
        'model_client_version': client_version(),
        'development_results': {str(p.relative_to(development)): digest(p)
                                for p in development.rglob('result.json')},
        'selection': 'Five fresh fixed seeds; no reserved outcomes inspected or used for tuning',
        'failure_policy': 'Keep every attempt; stop dispatch on infrastructure errors; no automatic retries'}
    write(root/'protocol.json', protocol)
    (root/'protocol.sha256').write_text(digest(root/'protocol.json')+'\n')
    write(root/'progress.json', {'status': 'registered', 'completed': 0, 'total': 20})
    return protocol


def verify_sources(root):
    if digest(root/'protocol.json') != (root/'protocol.sha256').read_text().strip():
        raise ValueError('Registered protocol changed')
    protocol = json.loads((root/'protocol.json').read_text())
    if package_versions() != protocol['package_versions']:
        raise ValueError('Registered runtime package versions changed')
    if client_version() != protocol['model_client_version']:
        raise ValueError('Registered model client version changed')
    for name, expected in protocol['source_sha256'].items():
        if digest(ROOT/name) != expected or digest(root/'sources'/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    if digest(root/'geometry.py') != protocol['geometry_sha256']:
        raise ValueError('Archived geometry bundle changed')
    if hashlib.sha256(online_library().encode()).hexdigest() != protocol['geometry_sha256']:
        raise ValueError('Generated geometry differs from registration')
    return protocol


def execute(root):
    import fcntl
    root = root.resolve()
    protocol = verify_sources(root)
    if (root/'STOP.json').exists():
        raise RuntimeError('Inspect the existing STOP receipt; no automatic resume')
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        completed = 0
        try:
            for slot in protocol['slots']:
                verify_sources(root)
                directory = root/slot['path']
                if directory.exists():
                    raise FileExistsError('Preserve previous attempt: '+str(directory))
                write(root/'progress.json', {'status': 'running', 'completed': completed,
                      'total': len(protocol['slots']), 'active': slot, 'started_at': started})
                result = run(directory, slot['seed'], slot['arm'], protocol['budget_s'])
                verify_sources(root)
                completed += 1
                print(slot['seed'], slot['repetition'], slot['arm'], result['status'],
                      result['capped_completion_seconds'], flush=True)
                if result['status'] == 'infrastructure_error':
                    raise RuntimeError('Episode infrastructure failure: '+str(directory))
            summary = summarize(root)
            write(root/'summary.json', summary)
            plot(summary, root, protocol['budget_s'])
            write(root/'progress.json', {'status': 'completed', 'completed': completed,
                  'total': len(protocol['slots']), 'wall_seconds': time.time()-started})
        except Exception as error:
            write(root/'STOP.json', {'time': time.time(), 'error': type(error).__name__+': '+str(error),
                                    'completed': completed})
            write(root/'progress.json', {'status': 'stopped', 'completed': completed,
                                        'total': len(protocol['slots'])})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'run', 'check'), required=True)
    parser.add_argument('--development', type=Path, default=ROOT/'outputs/online_bridge_tower_dev')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.root, args.development)
    elif args.phase == 'run':
        execute(args.root)
    else:
        verify_sources(args.root)
        print('Registered sources match')


if __name__ == '__main__':
    main()
