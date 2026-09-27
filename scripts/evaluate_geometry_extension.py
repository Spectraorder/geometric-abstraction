"""Evaluate training-selected extended policies on a newly registered shared split."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import statistics
import time

import extend_geometry_training as extension
from summarize_mixed_scene_learning import interval, paired, require

base = extension.base
experiment = extension.experiment
read = extension.read


def fresh_seeds(seeds, inspected):
    require(len(seeds) == 20 and len(set(seeds)) == 20, 'Require 20 unique shared layouts')
    require(not set(seeds).intersection(inspected), 'Fresh evaluation overlaps inspected layouts')
    return seeds


def prepare(root, rounds, output, seeds, workers):
    require(not output.exists(), 'Evaluation directory already exists; do not overwrite evidence')
    require(workers in (1, 2, 4), 'Invalid worker count')
    audit = extension.audit(root, rounds)
    manifest = read(root/'manifest.json')
    inspected = set(manifest['train_seeds'] + manifest['heldout_seeds'])
    for pattern in ('**/*manifest*.json', '**/*amendment*.json'):
        for path in (experiment.ROOT/'outputs').glob(pattern):
            record = read(path)
            if isinstance(record, dict):
                for key in ('train_seeds', 'heldout_seeds', 'reserved_seeds', 'test_seeds'):
                    inspected.update(record.get(key, []))
    fresh_seeds(seeds, inspected)
    ext = extension.extension_dir(root, rounds)
    paths = [root/'manifest.json', ext/'amendment.json', ext/'training_summary.json',
             ext/'all_candidates_frozen.json', ext/'audit.json']
    for arm in manifest['arms']:
        paths.extend([ext/arm/'frozen.py', root/arm/'.actor_session/state.json'])
    registration = {'schema': 'fresh_extended_policy_evaluation_v1',
                    'registered_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'rounds': rounds, 'reserved_seeds': seeds, 'workers': workers,
                    'required_rollouts': 2 * len(seeds), 'selected': audit['selected'],
                    'runner_sha256': base.digest(Path(__file__)),
                    'preserved_files': {str(p.relative_to(root)): base.digest(p) for p in paths},
                    'scope': 'Only training-selected policies; no actor calls or test-based selection.',
                    'seed_rule': 'Consecutive new seeds fixed before any fresh rollout; no outcome screening.',
                    'limitations': ['One learned chain per arm and 20 paired layouts.',
                                    'Continuation followed inspection of the earlier reserved split.',
                                    'Fresh evaluation measures final selected policies, not revision learning curves.']}
    output.mkdir(parents=True)
    base.write(output/'registration.json', registration)
    return registration


def check(root, output):
    registration = read(output/'registration.json')
    require(base.digest(Path(__file__)) == registration['runner_sha256'], 'Evaluation runner changed')
    for name, digest in registration['preserved_files'].items():
        require(base.digest(root/name) == digest, 'Preserved evidence changed: ' + name)
    _, _, manifest = extension.check(root, registration['rounds'])
    manifest = deepcopy(manifest)
    fresh_seeds(registration['reserved_seeds'], manifest['train_seeds'] + manifest['heldout_seeds'])
    manifest['heldout_seeds'] = registration['reserved_seeds']
    manifest['color_maps'].update({str(seed): base.color_map(seed) for seed in manifest['heldout_seeds']})
    return registration, manifest


def audit(root, output):
    registration, manifest = check(root, output)
    rows = {}; hashes = {}; results = {}
    for arm in manifest['arms']:
        rows[arm] = []; results[arm] = []
        for seed in registration['reserved_seeds']:
            trial = output/arm/'heldout'/f'seed_{seed}'
            result = read(trial/'result.json')
            require(result['seed'] == seed and type(result['success']) is bool, 'Result identity mismatch')
            require(not result.get('invalid_candidate', False), 'Invalid evaluation candidate')
            require(result['candidate_sha256'] == registration['selected'][arm]['sha256'], 'Policy changed')
            require(result['geometry_library_sha256'] == manifest['library_sha256'][arm], 'Observer changed')
            require(result['task'] == manifest['task'] and result['layout'] == 'sampled' and
                    result['scene']['seed'] == seed and result['scene']['recipe'] == manifest['recipe'] and
                    result['scene']['color_map'] == manifest['color_maps'][str(seed)] and
                    result['scene']['object_count'] == manifest['object_count'] and
                    result['physics'] == manifest['physics'] and result['sensor'] == manifest['sensor'] and
                    result['wrist_depth'] and not result['sensor_blackout'], 'Runtime settings changed')
            sandbox = read(trial/'sandbox.json')
            require(sandbox['isolation'] == 'bwrap' and sandbox['network'] == 'disabled', 'Sandbox mismatch')
            rows[arm].append({key: result[key] for key in ('seed', 'success', 'policy_error')})
            results[arm].append(result)
            for name in ('result.json', 'sandbox.json', 'api_calls.json'):
                hashes[str((trial/name).relative_to(output))] = base.digest(trial/name)
    arms = manifest['arms']
    for first, second in zip(results[arms[0]], results[arms[1]]):
        require(first['scene'] == second['scene'], 'Paired scene mismatch')
    report = {'status': 'completed', 'valid_rollouts': sum(map(len, rows.values())), 'invalid_slots': 0,
              'registration_sha256': base.digest(output/'registration.json'), 'selected': registration['selected'],
              'model_calls': 0, 'model_tokens': 0, 'arms': {}, 'paired': paired(rows[arms[0]], rows[arms[1]]),
              'result_hashes': hashes, 'limitations': registration['limitations']}
    require(report['valid_rollouts'] == registration['required_rollouts'], 'Incomplete evaluation')
    for arm in arms:
        records = results[arm]; count = sum(r['success'] for r in records)
        report['arms'][arm] = {'revision': registration['selected'][arm]['revision'], 'successes': count,
                              'trials': len(records), 'success_rate': count/len(records),
                              'wilson_95_interval': interval(count, len(records)), 'results': rows[arm],
                              'mean_camera_calls': statistics.mean(r['camera_calls'] for r in records),
                              'mean_policy_compute_cpu_seconds': statistics.mean(r['cpu_profile']['policy_compute_cpu_seconds'] for r in records),
                              'median_policy_sim_seconds': statistics.median(r['policy_sim_seconds'] for r in records)}
    report['training_audit'] = extension.audit(root, registration['rounds'])['status']
    return report


def run(root, output):
    registration, manifest = check(root, output)
    require(not (output/'summary.json').exists(), 'Evaluation already completed')
    ext = extension.extension_dir(root, registration['rounds'])
    started = time.monotonic(); completed = 0
    base.write(output/'progress.json', {'status': 'evaluating', 'completed': 0, 'total': registration['required_rollouts']})
    with experiment.configured(), ThreadPoolExecutor(max_workers=registration['workers']) as pool:
        jobs = {}
        for index, seed in enumerate(registration['reserved_seeds']):
            arms = manifest['arms'] if index % 2 == 0 else list(reversed(manifest['arms']))
            for arm in arms:
                job = pool.submit(base.execute, root, arm, ext/arm/'frozen.py', output/arm/'heldout', [seed], manifest)
                jobs[job] = (arm, seed)
        for job in as_completed(jobs):
            job.result(); completed += 1
            base.write(output/'progress.json', {'status': 'evaluating', 'completed': completed,
                       'total': registration['required_rollouts'], 'elapsed_seconds': time.monotonic()-started})
    report = audit(root, output)
    report['elapsed_seconds'] = time.monotonic()-started
    base.write(output/'summary.json', report)
    base.write(output/'audit.json', report)
    base.write(output/'progress.json', {'status': 'completed', 'completed': completed, 'total': completed,
                                       'elapsed_seconds': report['elapsed_seconds']})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'evaluate', 'audit'), required=True)
    parser.add_argument('--seed-start', type=int, default=2026112701)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(); root = args.root.resolve(); output = args.output.resolve()
    require(output.is_relative_to(root), 'Keep evaluation within the study root')
    with (root/'training_coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'prepare':
            result = prepare(root, args.rounds, output, list(range(args.seed_start, args.seed_start+20)), args.workers)
        elif args.phase == 'evaluate':
            result = run(root, output)
        else:
            result = audit(root, output)
        print(json.dumps({k:v for k,v in result.items() if k not in ('preserved_files', 'result_hashes')}, indent=2))
