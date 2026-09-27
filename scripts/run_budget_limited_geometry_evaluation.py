"""Evaluate a fixed prefix of reserved layouts in parallel after a budget reduction.

Keep the original manifest, frozen policies and all previously completed results.
The reduced-scope report never claims completion of the original larger protocol.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

import run_constraint_geometry_experiment as experiment
from summarize_mixed_scene_learning import paired, require


def read(path):
    return json.loads(path.read_text())


def timestamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def frozen_batches(root, manifest, trained):
    require(trained['manifest_sha256'] == experiment.base.digest(root / 'manifest.json'), 'Training manifest changed')
    require(read(root / 'all_candidates_frozen.json') == {'selected': trained['frozen'], 'revisions': trained['revision_frozen']}, 'Freeze gate changed')
    batches = []
    for arm in manifest['arms']:
        rows = trained['curves'][arm]
        selected = max(rows, key=lambda row: row['successes'])['revision']
        require(selected == trained['frozen'][arm]['revision'], 'Selection differs from training')
        require(experiment.base.digest(root / arm / 'frozen.py') == trained['frozen'][arm]['sha256'], 'Selected policy changed')
        entries = trained['revision_frozen'][arm]
        require([entry['revision'] for entry in entries] == list(range(manifest['rounds'] + 1)), 'Incomplete frozen revisions')
        for entry in entries:
            revision = entry['revision']
            candidate = root / arm / 'frozen_revisions' / f'revision_{revision:03d}.py'
            original = root / arm / ('revision_000' if revision == 0 else f'round_{revision:03d}') / 'bundled.py'
            require(experiment.base.digest(candidate) == entry['sha256'] == experiment.base.digest(original), 'Frozen source changed')
            if revision == selected:
                require(entry['sha256'] == trained['frozen'][arm]['sha256'], 'Selected revision source mismatch')
                directory = root / arm / 'heldout'
            else:
                directory = root / arm / 'reserved_revisions' / f'revision_{revision:03d}' / 'heldout'
            batches.append((arm, revision, candidate, directory, entry.get('validation_error')))
    return batches


def prepare(root, layouts, workers):
    manifest = read(root / 'manifest.json')
    experiment.check(root, manifest)
    require(1 <= layouts <= len(manifest['heldout_seeds']), 'Invalid layout limit')
    require(1 <= workers <= 8, 'Require between one and eight rollout workers')
    require((root / 'user_budget_change.json').exists(), 'Budget-change receipt is required')
    require(not (root / 'summary.json').exists(), 'Original campaign is already complete')
    trained = read(root / 'training_summary.json')
    batches = frozen_batches(root, manifest, trained)
    seeds = manifest['heldout_seeds'][:layouts]
    existing = {str(path.relative_to(root)): experiment.base.digest(path)
                for _, _, _, directory, _ in batches for path in directory.glob('seed_*/result.json')}
    missing = [(arm, revision, seed) for arm, revision, _, directory, _ in batches
               for seed in seeds if not (directory / f'seed_{seed}' / 'result.json').exists()]
    amendment = {'schema': 'reserved_budget_reduction_v1', 'registered_at_utc': timestamp(),
        'manifest_sha256': experiment.base.digest(root / 'manifest.json'),
        'training_sha256': experiment.base.digest(root / 'training_summary.json'),
        'freeze_sha256': experiment.base.digest(root / 'all_candidates_frozen.json'),
        'runner_sha256': experiment.base.digest(Path(__file__)),
        'original_reserved_layouts': len(manifest['heldout_seeds']), 'reserved_seeds': seeds,
        'selection_rule': 'First N seeds in the original registered order, independent of outcomes.',
        'workers': workers, 'required_slots': len(batches) * len(seeds),
        'additional_slots_at_registration': len(missing), 'existing_results': existing,
        'scope': 'User-directed budget reduction after evaluation began; retain all prior outcomes separately.'}
    path = root / 'evaluation_amendment.json'
    require(not path.exists(), 'An evaluation amendment is already registered')
    experiment.base.write(path, amendment)
    return amendment


def check_amendment(root):
    amendment = read(root / 'evaluation_amendment.json')
    manifest = read(root / 'manifest.json')
    experiment.check(root, manifest)
    for key, name in (('manifest_sha256', 'manifest.json'), ('training_sha256', 'training_summary.json'),
                      ('freeze_sha256', 'all_candidates_frozen.json')):
        require(amendment[key] == experiment.base.digest(root / name), 'Registered provenance changed: ' + name)
    require(amendment['runner_sha256'] == experiment.base.digest(Path(__file__)), 'Evaluation runner changed after registration')
    seeds = amendment['reserved_seeds']
    require(seeds and seeds == manifest['heldout_seeds'][:len(seeds)], 'Reserved subset is not the registered prefix')
    for relative, digest in amendment['existing_results'].items():
        require(experiment.base.digest(root / relative) == digest, 'Previously completed result changed')
    return amendment, manifest, read(root / 'training_summary.json')


def audit(root):
    amendment, manifest, trained = check_amendment(root)
    training = experiment.audit(root, allow_partial=True)
    require(training['exact_actor_contexts_verified'] == len(manifest['arms']) * manifest['rounds'], 'Incomplete actor history')
    batches = frozen_batches(root, manifest, trained)
    freeze_time = (root / 'all_candidates_frozen.json').stat().st_mtime
    initial, all_rows, hashes = {}, {}, {}
    valid, invalid = 0, 0
    for arm, revision, candidate, directory, _ in batches:
        rows = {}
        for path in sorted(directory.glob('seed_*/result.json')):
            result = read(path)
            seed = result['seed']
            require(seed in manifest['heldout_seeds'] and path.parent.name == f'seed_{seed}', 'Unexpected reserved layout')
            require(type(result['success']) is bool and result['candidate_sha256'] == experiment.base.digest(candidate), 'Result source or success mismatch')
            executed = not result.get('invalid_candidate', False)
            first = path.parent / 'initial.png' if executed else path
            require(first.stat().st_mtime >= freeze_time and path.stat().st_mtime >= freeze_time, 'Reserved execution predates global freeze')
            if executed:
                require(result['geometry_library_sha256'] == manifest['library_sha256'][arm], 'Observer changed')
                require(result['task'] == manifest['task'] and result['scene']['recipe'] == manifest['recipe'], 'Task changed')
                require(result['scene']['color_map'] == manifest['color_maps'][str(seed)] and result['scene']['object_count'] == 4, 'Scene changed')
                require(result['physics'] == manifest['physics'] and result['sensor'] == manifest['sensor'], 'Simulation settings changed')
                require(result['wrist_depth'] and not result['sensor_blackout'], 'Sensor settings changed')
                image_hash = experiment.base.digest(first)
                require(seed not in initial or initial[seed] == image_hash, 'Paired initial scenes differ')
                initial[seed] = image_hash
                valid += 1
            else:
                require(not result['success'] and not result['rollout_executed'], 'Invalid slot counted as success')
                invalid += 1
            rows[seed] = {'seed': seed, 'success': result['success'], 'policy_error': result['policy_error'], 'rollout_executed': executed}
            hashes[str(path.relative_to(root))] = experiment.base.digest(path)
        all_rows[arm, revision] = rows
    seeds = amendment['reserved_seeds']
    curves, selected_rows = {}, {}
    for arm in manifest['arms']:
        curves[arm] = []
        for revision in range(manifest['rounds'] + 1):
            rows = all_rows[arm, revision]
            require(all(seed in rows for seed in seeds), 'Reduced evaluation is incomplete')
            selected = [rows[seed] for seed in seeds]
            successes = sum(row['success'] for row in selected)
            curves[arm].append({'revision': revision, 'successes': successes, 'trials': len(seeds),
                                'sr': successes / len(seeds), 'results': selected})
        selected_rows[arm] = curves[arm][trained['frozen'][arm]['revision']]['results']
    return {'status': 'completed_reduced_scope', 'original_protocol_complete': False,
            'scope': amendment['scope'], 'reserved_seeds': seeds, 'workers': amendment['workers'],
            'manifest_sha256': amendment['manifest_sha256'],
            'amendment_sha256': experiment.base.digest(root / 'evaluation_amendment.json'),
            'required_reserved_slots': amendment['required_slots'],
            'actual_reserved_rollouts': valid, 'invalid_reserved_slots': invalid,
            'retained_extra_reserved_slots': valid + invalid - amendment['required_slots'],
            'training_selected_revisions': {arm: trained['frozen'][arm]['revision'] for arm in manifest['arms']},
            'curves': curves, 'training_selected_comparison': paired(selected_rows['basic'], selected_rows['symbolic']),
            'actors': training['actors'], 'result_hashes': hashes,
            'audit': {'original_training_and_actor_contexts': 'passed', 'frozen_sources': 'passed',
                      'all_existing_reserved_results': 'passed', 'paired_initial_scenes': 'passed'}}


def run(root):
    with (root / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        amendment, manifest, trained = check_amendment(root)
        batches = frozen_batches(root, manifest, trained)
        jobs = [(arm, revision, candidate, directory, error, seed)
                for arm, revision, candidate, directory, error in batches
                for seed in amendment['reserved_seeds']
                if not (directory / f'seed_{seed}' / 'result.json').exists()]
        # Interleave policy versions so no worker monopolizes a long batch.
        jobs.sort(key=lambda job: (job[-1], job[1], job[0]))
        def progress(phase, **extra):
            value = {'phase': phase, 'pid': os.getpid(), 'updated_at_utc': timestamp(),
                     'workers': amendment['workers'], 'remaining_slots': len(jobs), **extra}
            experiment.base.write(root / 'lifecycle.json', value)
            print(json.dumps(value), flush=True)
        def execute(job):
            arm, _, candidate, directory, error, seed = job
            return experiment.base.execute(root, arm, candidate, directory, [seed], manifest, error)
        progress('parallel_evaluate')
        try:
            with experiment.configured(), ThreadPoolExecutor(max_workers=amendment['workers']) as pool:
                pending = set()
                while jobs or pending:
                    while jobs and len(pending) < amendment['workers']:
                        pending.add(pool.submit(execute, jobs.pop(0)))
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        future.result()
                    progress('parallel_evaluate', in_flight=len(pending))
            progress('reduced_scope_audit')
            report = audit(root)
            experiment.base.write(root / 'audit_budget_limited.json', report)
            experiment.base.write(root / 'summary_budget_limited.json', {k: v for k, v in report.items() if k != 'result_hashes'})
            progress('completed_reduced_scope', actual_reserved_rollouts=report['actual_reserved_rollouts'],
                     retained_extra_reserved_slots=report['retained_extra_reserved_slots'])
        except BaseException as error:
            progress('interrupted', error_type=type(error).__name__, message=str(error))
            traceback.print_exc()
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'run', 'audit'), required=True)
    parser.add_argument('--layouts', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == 'prepare':
        value = prepare(root, args.layouts, args.workers)
        print(json.dumps({k: v for k, v in value.items() if k != 'existing_results'}, indent=2))
    elif args.phase == 'run':
        run(root)
    else:
        value = audit(root)
        experiment.base.write(root / 'audit_budget_limited.json', value)
        print(value['status'])
