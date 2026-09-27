"""Evaluate training-selected interface-guided policies on their registered split."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import run_interface_contract_geometry_experiment as experiment
from summarize_mixed_scene_learning import interval, paired, require

base = experiment.base


def read(path):
    return json.loads(path.read_text())


@contextmanager
def configured():
    with experiment.installed(), experiment.neutral.configured():
        yield


def prepare(root, output, workers):
    require(not output.exists(), 'Evaluation directory already exists')
    require(workers in (1, 2, 4), 'Invalid worker count')
    training = experiment.audit(root, allow_partial=True)
    require(training['status'] == 'trained', 'Require completed training before evaluation')
    manifest = read(root/'manifest.json')
    trained = read(root/'training_summary.json')
    seeds = manifest['heldout_seeds']
    require(len(seeds) == len(set(seeds)) == 20, 'Require 20 shared registered layouts')
    require(not set(seeds).intersection(manifest['train_seeds']), 'Training/test overlap')
    require(not list(root.glob('*/heldout/seed_*')), 'Reserved evaluation already started')
    require(not list(root.glob('*/reserved_revisions/*/heldout/seed_*')), 'Reserved revisions already evaluated')
    paths = [root/name for name in ('manifest.json', 'training_summary.json', 'all_candidates_frozen.json', 'audit_training.json')]
    for arm in manifest['arms']:
        paths.extend([root/arm/'frozen.py', root/arm/'.actor_session/state.json'])
    registration = {
        'schema': 'selected_interface_policy_evaluation_v1',
        'registered_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'reserved_seeds': seeds, 'workers': workers,
        'required_rollouts': len(manifest['arms']) * len(seeds), 'selected': trained['frozen'],
        'runner_sha256': base.digest(Path(__file__)),
        'preserved_files': {str(path.relative_to(root)): base.digest(path) for path in paths},
        'scope': 'Training-selected policies only on the original reserved split; no actor calls or test-based selection.',
        'limitations': ['One learned chain per arm and 20 paired layouts.',
                        'Selected-policy evaluation does not complete the original full-revision protocol.',
                        'Original 30000-character training budget and invalid R5 records remain unchanged.']}
    output.mkdir(parents=True)
    base.write(output/'registration.json', registration)
    check(root, output)
    return registration


def check(root, output):
    registration = read(output/'registration.json')
    require(base.digest(Path(__file__)) == registration['runner_sha256'], 'Evaluation runner changed')
    for name, digest in registration['preserved_files'].items():
        require(base.digest(root/name) == digest, 'Preserved evidence changed: ' + name)
    manifest = read(root/'manifest.json')
    experiment.check(root, manifest)
    require(registration['reserved_seeds'] == manifest['heldout_seeds'], 'Reserved split changed')
    require(registration['selected'] == read(root/'training_summary.json')['frozen'] ==
            read(root/'all_candidates_frozen.json')['selected'], 'Training selection changed')
    for arm in manifest['arms']:
        require(base.digest(root/arm/'frozen.py') == registration['selected'][arm]['sha256'], 'Selected source changed')
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
            experiment.neutral.check_goal(trial, result, manifest)
            require((trial/'initial.png').stat().st_mtime >= (output/'registration.json').stat().st_mtime, 'Rollout predates registration')
            sandbox = read(trial/'sandbox.json')
            require(sandbox['isolation'] == 'bwrap' and sandbox['network'] == 'disabled', 'Sandbox mismatch')
            rows[arm].append({key: result[key] for key in ('seed', 'success', 'policy_error')})
            results[arm].append(result)
            for name in ('result.json', 'sandbox.json', 'api_calls.json', 'goal_contract.json', 'initial.png'):
                hashes[str((trial/name).relative_to(output))] = base.digest(trial/name)
    arms = manifest['arms']
    for first, second in zip(results[arms[0]], results[arms[1]]):
        require(first['scene'] == second['scene'], 'Paired scene mismatch')
        seed = first['seed']
        require(base.digest(output/arms[0]/'heldout'/f'seed_{seed}'/'initial.png') ==
                base.digest(output/arms[1]/'heldout'/f'seed_{seed}'/'initial.png'), 'Paired initial image mismatch')
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
    report['training_audit'] = experiment.audit(root, allow_partial=True)['status']
    return report


def run(root, output):
    registration, manifest = check(root, output)
    require(not (output/'summary.json').exists(), 'Evaluation already completed')
    started = time.monotonic()
    completed = 0
    def progress(status, **extra):
        record = {'status': status, 'pid': os.getpid(), 'completed': completed,
                  'total': registration['required_rollouts'], 'elapsed_seconds': time.monotonic()-started, **extra}
        base.write(output/'progress.json', record)
        print(json.dumps(record), flush=True)
    progress('evaluating')
    try:
        with configured(), ThreadPoolExecutor(max_workers=registration['workers']) as pool:
            jobs = {}
            for index, seed in enumerate(registration['reserved_seeds']):
                arms = manifest['arms'] if index % 2 == 0 else list(reversed(manifest['arms']))
                for arm in arms:
                    job = pool.submit(base.execute, root, arm, root/arm/'frozen.py', output/arm/'heldout', [seed], manifest)
                    jobs[job] = (arm, seed)
            for job in as_completed(jobs):
                job.result()
                completed += 1
                progress('evaluating', last_arm=jobs[job][0], last_seed=jobs[job][1])
        progress('auditing')
        report = audit(root, output)
        report['elapsed_seconds'] = time.monotonic()-started
        base.write(output/'summary.json', report)
        base.write(output/'audit.json', report)
        progress('completed')
        return report
    except BaseException as error:
        progress('failed', error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'evaluate', 'audit'), required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    require(output.is_relative_to(root) and output != root, 'Keep evaluation within the study root')
    with (root/'training_coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'prepare':
            result = prepare(root, output, args.workers)
        elif args.phase == 'evaluate':
            result = run(root, output)
        else:
            result = audit(root, output)
        print(json.dumps({key: result[key] for key in ('status', 'required_rollouts', 'valid_rollouts', 'paired') if key in result}, indent=2))
