"""Five paired ring training chains and post-freeze R0-R3 reserved evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import ring_peg_comparison_experiment as experiment
import geometry_actor_session as actor
import summarize_geometry_session_resources as resources
from verified_token_usage import turn_usage
from experiment_parallel import bounded_map, lease, stop, check_stop, audit_leases, write_json

ROOT = experiment.ROOT
read = experiment.legacy.read
digest = experiment.legacy.digest
ARMS = experiment.ARMS
TRAIN_SEEDS = list(range(2031101001, 2031101026))
TEST_SEEDS = list(range(2031109001, 2031109101))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def hardware():
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            memory[key] = int(value.strip().split()[0])*1024
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=10)
    return {'time': time.time(), 'cpus': os.cpu_count(), 'memory': memory,
            'gpu_free_mib': [int(x) for x in gpu.stdout.split()] if gpu.returncode == 0 else [],
            'load': list(os.getloadavg())}


def check(root):
    protocol = read(root/'protocol.json')
    require(digest(root/'protocol.json') == read(root/'registration.json')['protocol_sha256'], 'Protocol changed')
    for relative, sha in protocol['source_sha256'].items():
        require(digest(ROOT/relative) == sha == digest(root/'sources'/relative), 'Frozen source changed: '+relative)
    require(digest(root/'starter.py') == protocol['starter_sha256'], 'Starter changed')
    for arm in ARMS:
        require(digest(root/f'{arm}_contract.txt') == protocol['contracts'][arm], 'Contract changed')
    for replica in protocol['replicas']:
        require(digest(root/replica['name']/'manifest.json') == replica['manifest_sha256'], 'Manifest changed')
    return protocol


def prior_seeds():
    found = {1, 2, 3, *range(2030100101, 2030100106)}
    # Follow the known top-level SSD study links; avoid repeated source/trial trees.
    seen = set()
    for output in (ROOT/'outputs').iterdir():
        if not output.is_dir():
            continue
        target = output.resolve()
        if target in seen:
            continue
        seen.add(target)
        for directory, folders, files in os.walk(target):
            folders[:] = [f for f in folders if f not in ('sources', 'trials', 'rollouts', 'env',
                '.actor_session', 'scheduler', 'snapshot', 'scheduler_snapshot', 'development_evidence') and not f.startswith('seed_')]
            for name in set(files)&{'manifest.json', 'protocol.json', 'registration.json'}:
                data = read(Path(directory)/name)
                for key in ('train_seeds', 'heldout_seeds', 'reserved_seeds', 'test_seeds', 'seeds'):
                    found.update(x for x in data.get(key, []) if isinstance(x, int))
    return found


def prepare(root):
    require(not root.exists(), 'Use a new campaign directory')
    require(not set(TRAIN_SEEDS+TEST_SEEDS)&prior_seeds(), 'Seed overlap with previous registrations')
    require(shutil.disk_usage(root.parent).free > 20*1024**3, 'Insufficient disk reserve')
    proof = ROOT/'outputs/ring_peg_dev/comparison_preflight_01'
    for seed in (1, 2, 3):
        for arm in ARMS:
            result = read(proof/f'{arm}_{seed}/result.json')
            require(result['status'] == 'completed', 'RPC preflight failed')
        require(digest(proof/f'basic_{seed}/camera_000_side.png') ==
                digest(proof/f'symbolic_{seed}/camera_000_side.png'), 'Initial RGB differs across arms')
    motion_proof = ROOT/'outputs/ring_peg_dev/comparison_motion_01/result.json'
    require(read(motion_proof)['success'], 'Physical interface smoke check failed')
    root.mkdir(parents=True)
    source_hashes = experiment.sources()
    for relative in source_hashes:
        destination = root/'sources'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, destination)
    (root/'starter.py').write_bytes(experiment.STARTER_PATH.read_bytes())
    for arm in ARMS:
        (root/f'{arm}_contract.txt').write_text(experiment.contract(arm))
    replicas = []
    for index in range(5):
        name = f'replica_{index+1:02d}'
        directory = root/name
        directory.mkdir()
        manifest = {'schema': 'paired_ring_v1', 'arms': list(ARMS), 'rounds': 3,
                    'train_seeds': TRAIN_SEEDS[5*index:5*index+5], 'heldout_seeds': TEST_SEEDS,
                    'model': 'gpt-6-astra', 'actor_context_mode': 'persistent', 'model_timeout_s': 900,
                    'max_candidate_characters': 50000, 'task': 'ring_over_peg',
                    'training_slots': 40, 'model_calls': 6, 'test_slots': 800}
        write_json(directory/'manifest.json', manifest)
        replicas.append({'name': name, 'train_seeds': manifest['train_seeds'],
                         'manifest_sha256': digest(directory/'manifest.json')})
    protocol = {'schema': 'ring_comparison_study_v1', 'registered_at': time.time(),
                'replicas': replicas, 'train_seeds': TRAIN_SEEDS, 'heldout_seeds': TEST_SEEDS,
                'revisions': [0, 1, 2, 3], 'training_slots': 200, 'model_calls': 30, 'test_slots': 4000,
                'limits': {'model': 4, 'rollout': 24}, 'chain_processes': 4, 'rollouts_per_chain': 5,
                'evaluation_pilot_slots': 48, 'fallback_workers': 8, 'minimum_free_disk_bytes': 20*1024**3,
                'hardware': hardware(), 'source_sha256': source_hashes,
                'starter_sha256': digest(root/'starter.py'),
                'contracts': {arm: digest(root/f'{arm}_contract.txt') for arm in ARMS},
                'contrast': 'Matched task, qualitative guidance, raw RGB-D, CPU code budget and robot-only actuator. Full additionally receives supplied geometry. No executor or perception source in actor inputs.',
                'selection': 'Preserve every incumbent training success and add a success. Ties retain the incumbent.',
                'test_gate': 'All ten chains and actor/resource audits freeze before any reserved execution.',
                'failure_rule': 'Stop new dispatch on infrastructure errors; preserve attempts without retries or seed replacement.',
                'feedback_rule': 'Binary outcome, policy notes, at most four latest permitted records, two training images per revision. Base receives no supplied geometry.',
                'actor_scope': 'Fresh independent persistent context for each arm/replica; no prior trained candidates or reserved feedback.',
                'analysis': 'Per-chain raw and training-retained curves on 100 shared layouts; mean across five chains, resource curves. Shared layouts are not 500 independent layouts.',
                'runtime': {'resolution': [640, 480], 'rollout_seconds': 120, 'api_calls': 600,
                            'physics': 'Unchanged ring scene, noslip 0 and libccd', 'success_hold_seconds': 2},
                'preflight': {str(p.relative_to(ROOT)): digest(p) for p in proof.glob('*/result.json')},
                'physical_preflight': {str(motion_proof.relative_to(ROOT)): digest(motion_proof)}}
    write_json(root/'protocol.json', protocol)
    write_json(root/'registration.json', {'protocol_sha256': digest(root/'protocol.json')})
    write_json(root/'progress.json', {'status': 'registered', 'training_total': 200, 'test_total': 4000})
    check(root)


def stage_path(directory, arm, revision):
    return directory/arm/('revision_000' if revision == 0 else f'round_{revision:03d}')


def audit_trial(path, candidate, arm, seed):
    row = read(path/'result.json')
    require(row['arm'] == arm and row['seed'] == seed and row['candidate_sha256'] == digest(candidate), 'Result identity mismatch')
    if row['status'] == 'invalid_policy':
        require(not row['success'] and not row['rollout_executed'], 'Invalid source cannot succeed')
        return row
    require(row['status'] in ('completed', 'policy_error') and row['rollout_executed'], 'Incomplete trial')
    physical = row['private_evaluation']
    require(row['success'] == physical['success'] and physical['hold_s'] == 2., 'Physical evaluation mismatch')
    command = read(path/'sandbox.json')['command']
    require('--unshare-net' in command and 'bwrap' in command[0], 'Missing isolated worker')
    feedback = read(path/'actor_feedback.json')
    require(set(feedback) <= {'seed', 'success', 'status', 'error', 'observations'}, 'Private feedback fields')
    require(feedback['success'] == row['success'] and feedback['seed'] == seed, 'Feedback outcome mismatch')
    if arm == 'basic':
        require(all(set(record) <= {'note', 'time'} for record in feedback['observations']), 'Base geometry leak')
        records = read(path/'sensor_records.json')
        require(all(set(record) <= {'note', 'time'} for record in records), 'Base runtime geometry leak')
    return row


def rollout(root, directory, arm, revision, seed, reserved=False, workers=24):
    protocol = check(root)
    require(shutil.disk_usage(root).free > protocol['minimum_free_disk_bytes'], 'Insufficient free disk')
    stage = stage_path(directory, arm, revision)
    candidate = stage/'candidate.py'
    path = (root/'evaluation'/directory.name/arm/f'revision_{revision:03d}'/f'seed_{seed}'
            if reserved else stage/'trials'/f'seed_{seed}')
    require(not path.exists(), 'Existing trial retained; automatic retries are prohibited')
    path.parent.mkdir(parents=True, exist_ok=True)
    with lease(root, 'rollout', workers, str(path.relative_to(root))):
        with path.with_suffix('.log').open('x') as log:
            subprocess.run([sys.executable, 'scripts/run_ring_peg_comparison_trial.py', '--output', str(path),
                            '--candidate', str(candidate), '--arm', arm, '--seed', str(seed)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180,
                           env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='egl'))
        row = audit_trial(path, candidate, arm, seed)
    return row


def feedback_for(directory, arm, revision, seeds):
    result = []
    for seed in seeds:
        path = stage_path(directory, arm, revision)/'trials'/f'seed_{seed}'
        feedback = path/'actor_feedback.json'
        if feedback.exists():
            result.append(read(feedback))
        else:
            row = read(path/'result.json')
            result.append({k: row[k] for k in ('seed', 'success', 'status', 'error') if k in row})
    return result


def prompt_for(root, directory, arm, revision, retained, seeds):
    prefix = (root/f'{arm}_contract.txt').read_text() if revision == 1 else 'Continue this isolated training chain. Return complete policy code.\n'
    incumbent = stage_path(directory, arm, retained)/'candidate.py'
    return prefix+'\nTraining-only feedback:\n'+json.dumps(feedback_for(directory, arm, revision-1, seeds), separators=(',', ':'))+'\nRetained policy R'+str(retained)+':\n'+incumbent.read_text()


def chain(root, name, arm):
    protocol = check(root)
    directory = root/name
    manifest = read(directory/'manifest.json')
    require(name in [r['name'] for r in protocol['replicas']] and arm in ARMS, 'Unknown chain')
    target = directory/arm
    target.mkdir(exist_ok=False)
    rows, successes, retained = [], set(), 0
    for revision in range(4):
        check_stop(root)
        check(root)
        stage = stage_path(directory, arm, revision)
        write_json(target/'progress.json', {'status': 'proposing' if revision else 'training', 'revision': revision})
        if revision == 0:
            stage.mkdir()
            shutil.copyfile(root/'starter.py', stage/'candidate.py')
        else:
            prompt = prompt_for(root, directory, arm, revision, retained, manifest['train_seeds'])
            require(len(prompt) <= 100000, 'Prompt exceeds fixed budget')
            images = []
            for seed in manifest['train_seeds'][:2]:
                files = sorted((stage_path(directory, arm, revision-1)/'trials'/f'seed_{seed}').glob('camera_*_side.png'))
                if files:
                    images.append(files[-1])
            with lease(root, 'model', protocol['limits']['model'], f'{name}/{arm}/R{revision}'):
                actor.propose(stage, prompt, images, manifest)
                actor.audit_session(stage, manifest, read(stage/'invocation.json'), read(stage/'generation.json'))
        write_json(target/'progress.json', {'status': 'training', 'revision': revision})
        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(lambda seed: rollout(root, directory, arm, revision, seed), manifest['train_seeds']))
        current = {row['seed'] for row in results if row['success']}
        if current > successes:
            retained, successes = revision, current
        rows.append({'revision': revision, 'successes': len(current), 'total': 5,
                     'selected_revision': retained, 'retained_successes': len(successes),
                     'candidate_sha256': digest(stage/'candidate.py')})
        write_json(target/'training_curve.json', rows)
    shutil.copyfile(stage_path(directory, arm, retained)/'candidate.py', target/'frozen.py')
    write_json(target/'summary.json', {'status': 'completed', 'selected_revision': retained,
                                      'selected_sha256': digest(target/'frozen.py'), 'curves': rows})
    write_json(target/'progress.json', {'status': 'completed', 'selected_revision': retained})


def freeze_training(root):
    protocol = check(root)
    files, ids, images = {}, set(), {}
    for replica in protocol['replicas']:
        directory = root/replica['name']
        manifest = read(directory/'manifest.json')
        for arm in ARMS:
            target = directory/arm
            summary = read(target/'summary.json')
            require(summary['status'] == 'completed', 'Training unfinished')
            rows = summary['curves']
            successes, retained = set(), 0
            for revision in range(4):
                stage = stage_path(directory, arm, revision)
                if revision:
                    invocation, generation = read(stage/'invocation.json'), read(stage/'generation.json')
                    actor.audit_session(stage, manifest, invocation, generation)
                    require((stage/'prompt.txt').read_text() == prompt_for(root, directory, arm, revision, retained, manifest['train_seeds']), 'Actor prompt differs from registered feedback')
                    require(digest(stage/'prompt.txt') == invocation['prompt_sha256'], 'Prompt hash mismatch')
                    if revision == 1:
                        require(generation['session_id'] not in ids, 'Actor context shared across chains')
                        ids.add(generation['session_id'])
                    for item in invocation['images']:
                        require(digest(directory/item['name']) == item['sha256'], 'Actor training image changed')
                current = set()
                for seed in manifest['train_seeds']:
                    path = stage/'trials'/f'seed_{seed}'
                    row = audit_trial(path, stage/'candidate.py', arm, seed)
                    if row['success']:
                        current.add(seed)
                    if row['rollout_executed']:
                        sha = digest(path/'camera_000_side.png')
                        require(images.setdefault(seed, sha) == sha, 'Training initial images differ')
                if current > successes:
                    retained, successes = revision, current
                require(rows[revision]['selected_revision'] == retained and rows[revision]['successes'] == len(current), 'Retention mismatch')
                require(rows[revision]['candidate_sha256'] == digest(stage/'candidate.py'), 'Candidate changed after training')
            require(summary['selected_revision'] == retained and digest(target/'frozen.py') == digest(stage_path(directory, arm, retained)/'candidate.py'), 'Selected policy mismatch')
            for pattern in ('*/candidate.py', '*/prompt.txt', '*/generation.json', '*/session.jsonl', '*/trials/*/result.json', '*/trials/*/actor_feedback.json'):
                for path in target.glob(pattern):
                    files[str(path.relative_to(root))] = digest(path)
            files[str((target/'summary.json').relative_to(root))] = digest(target/'summary.json')
        with patch.object(resources, 'turn_usage', turn_usage):
            accounting = resources.summarize(directory, 3)
        write_json(directory/'resource_accounting.json', accounting)
        files[str((directory/'resource_accounting.json').relative_to(root))] = digest(directory/'resource_accounting.json')
    receipt = {'time': time.time(), 'files': files, 'actor_sessions': sorted(ids), 'protocol_sha256': digest(root/'protocol.json')}
    require(len(ids) == 10, 'Expected ten independent actor sessions')
    write_json(root/'all_candidates_frozen.json', receipt)
    return receipt


def check_gate(root):
    gate = read(root/'all_candidates_frozen.json')
    require(gate['protocol_sha256'] == digest(root/'protocol.json'), 'Freeze/protocol mismatch')
    for relative, sha in gate['files'].items():
        require(digest(root/relative) == sha, 'Training evidence changed: '+relative)
    return gate


def summarize(root):
    import numpy as np
    protocol, gate = check(root), check_gate(root)
    images, hashes, replicas = {}, {}, {}
    for replica in protocol['replicas']:
        directory = root/replica['name']
        replicas[directory.name] = {}
        accounting = read(directory/'resource_accounting.json')
        for arm in ARMS:
            rows, raw = read(directory/arm/'training_curve.json'), []
            for revision in range(4):
                successes = 0
                for seed in TEST_SEEDS:
                    path = root/'evaluation'/directory.name/arm/f'revision_{revision:03d}'/f'seed_{seed}'
                    result = audit_trial(path, stage_path(directory, arm, revision)/'candidate.py', arm, seed)
                    if result['rollout_executed']:
                        require(result['started_at'] >= gate['time'], 'Reserved execution before training freeze')
                        sha = digest(path/'camera_000_side.png')
                        require(images.setdefault(seed, sha) == sha, 'Reserved initial image mismatch')
                    hashes[str((path/'result.json').relative_to(root))] = digest(path/'result.json')
                    successes += int(result['success'])
                raw.append(successes/len(TEST_SEEDS))
            cumulative = [0]
            for turn in accounting['arms'][arm]['turns']:
                cumulative.append(cumulative[-1]+turn['usage']['total_tokens'])
            replicas[directory.name][arm] = {'raw': raw, 'retained': [raw[r['selected_revision']] for r in rows],
                                            'selected_revisions': [r['selected_revision'] for r in rows],
                                            'cumulative_tokens': cumulative}
    mean = {arm: {key: np.mean([r[arm][key] for r in replicas.values()], axis=0).tolist()
                  for key in ('raw', 'retained', 'cumulative_tokens')} for arm in ARMS}
    summary = {'status': 'completed', 'test_slots': len(hashes), 'layouts': 100, 'replicas': replicas,
               'mean_curves': mean, 'result_hashes': hashes, 'scheduler': audit_leases(root, protocol['limits']),
               'scope': 'Five paired independent training chains evaluated on 100 shared layouts; no test-based retention.'}
    write_json(root/'summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for arm, label in (('basic', 'Base'), ('symbolic', 'Full')):
        axes[0].plot(range(4), mean[arm]['raw'], '--', alpha=.45)
        axes[0].plot(range(4), mean[arm]['retained'], 'o-', label=label)
        axes[1].plot(np.asarray(mean[arm]['cumulative_tokens'])/1000, mean[arm]['retained'], 'o-', label=label)
    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.set_ylabel('Reserved success rate')
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_xlabel('Revision (solid: training-retained; dashed: raw)')
    axes[0].set_xticks(range(4))
    axes[1].set_xlabel('Mean cumulative model tokens (thousands)')
    for suffix in ('pdf', 'png'):
        fig.savefig(root/f'learning_curves.{suffix}', dpi=180)
    plt.close(fig)
    return summary


def pipeline(root):
    protocol = check(root)
    lock = (root/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (root/'lifecycle.json').exists(), 'Existing campaign attempt; inspect rather than restart')
    state = {'status': 'training', 'pid': os.getpid(), 'training_total': 200, 'test_total': 4000,
             'started_at': time.time(), 'rollout_workers': 24}
    finished = threading.Event()
    pilot_samples = []
    def heartbeat():
        while not finished.is_set():
            snapshot = dict(state)
            snapshot['time'] = time.time()
            snapshot['training_completed'] = len(list(root.glob('replica_*/*/*/trials/*/result.json')))
            snapshot['models_completed'] = len(list(root.glob('replica_*/*/round_*/generation.json')))
            snapshot['test_completed'] = len(list(root.glob('evaluation/*/*/*/*/result.json')))
            snapshot['chains'] = {}
            for path in root.glob('replica_*/*/progress.json'):
                snapshot['chains'][str(path.parent.relative_to(root))] = read(path)
            write_json(root/'progress.json', snapshot)
            try:
                sample = hardware()
                write_json(root/'resource_status.json', sample)
                with (root/'hardware.jsonl').open('a') as handle:
                    handle.write(json.dumps(sample)+'\n')
                if state['status'] == 'evaluating_pilot':
                    pilot_samples.append(sample)
            except Exception as error:
                write_json(root/'resource_status.json', {'error': str(error), 'time': time.time()})
            finished.wait(5)
    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    write_json(root/'lifecycle.json', state)
    try:
        def launch(job):
            name, arm = job
            with (root/name/f'{arm}_campaign.log').open('x') as log:
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(root),
                                '--phase', 'chain', '--replica', name, '--arm', arm], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True,
                               env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='egl'))
        jobs = [(r['name'], arm) for r in protocol['replicas'] for arm in ARMS]
        for _, _ in bounded_map(launch, jobs, 4, on_error=lambda e: stop(root, e)):
            pass
        freeze_training(root)
        state['status'] = 'evaluating_pilot'
        jobs = [(r['name'], arm, revision, seed) for seed in TEST_SEEDS
                for revision in (3, 2, 1, 0) for r in protocol['replicas'] for arm in ARMS]
        def evaluate(job):
            name, arm, revision, seed = job
            return rollout(root, root/name, arm, revision, seed, reserved=True, workers=state['rollout_workers'])
        for _, _ in bounded_map(evaluate, jobs[:48], 24, on_error=lambda e: stop(root, e)):
            pass
        measured = hardware()
        samples = list(pilot_samples)+[measured]
        if any(row['memory']['MemAvailable'] < 8*1024**3 or not row['gpu_free_mib'] or min(row['gpu_free_mib']) < 2000 for row in samples):
            state['rollout_workers'] = 8
        state['status'] = 'evaluating'
        write_json(root/'evaluation_pilot.json', {'completed': 48, 'hardware_samples': samples, 'workers_after_pilot': state['rollout_workers']})
        for _, _ in bounded_map(evaluate, jobs[48:], state['rollout_workers'], on_error=lambda e: stop(root, e)):
            pass
        summarize(root)
        state['status'] = 'completed'
    except BaseException as error:
        stop(root, error)
        state.update(status='failed', error=str(error))
        raise
    finally:
        finished.set()
        monitor.join(timeout=15)
        state['finished_at'] = time.time()
        state['elapsed_seconds'] = state['finished_at']-state['started_at']
        state['training_completed'] = len(list(root.glob('replica_*/*/*/trials/*/result.json')))
        state['test_completed'] = len(list(root.glob('evaluation/*/*/*/*/result.json')))
        state['models_completed'] = len(list(root.glob('replica_*/*/round_*/generation.json')))
        write_json(root/'progress.json', state)
        write_json(root/'lifecycle.json', state)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'run', 'chain', 'check', 'summarize'), required=True)
    parser.add_argument('--replica')
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == 'prepare':
        prepare(root)
    elif args.phase == 'check':
        print(check(root)['schema'])
    elif args.phase == 'chain':
        try:
            chain(root, args.replica, args.arm)
        except BaseException as error:
            stop(root, error)
            raise
    elif args.phase == 'summarize':
        summarize(root)
    else:
        pipeline(root)
