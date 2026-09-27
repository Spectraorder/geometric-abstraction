"""Post-training R0-R3 evaluation with frozen selection and up to 16 rollout workers."""
import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import threading
import time

import run_parallel_geometry_study as study
from experiment_parallel import audit_leases, bounded_map, check_stop, lease, stop, write_json
from geometry_iteration import retention
from summarize_mixed_scene_learning import paired, require

ROOT = study.ROOT
base = study.base
read = study.read


def hardware():
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            memory[key] = int(value.strip().split()[0])*1024
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    return {'time': time.time(), 'logical_cpus': os.cpu_count(), 'load_average': list(os.getloadavg()),
            'memory': memory, 'gpu_csv': gpu.stdout.strip(), 'gpu_query_returncode': gpu.returncode}


def prepare(root, output, data_root, workers):
    protocol = study.check(root)
    require(protocol['rounds'] == 3, 'This curve protocol covers R0 through R3')
    study.check_gate(root, protocol)
    require(read(root/'summary.json')['status'] == 'completed', 'Selected-policy study must be complete')
    require(workers in (4, 8, 16), 'Supported rollout concurrency is 4, 8 or 16')
    require(not output.exists() and not data_root.exists(), 'Use new amendment and data directories')
    records = []
    preserved = {name: base.digest(root/name) for name in ('protocol.json', 'all_candidates_frozen.json', 'summary.json')}
    for replica in protocol['replicas']:
        name = replica['directory']
        directory = root/name
        trained = read(directory/'training_summary.json')
        previous_audit = read(directory/'selected_evaluation/audit.json')
        preserved[str((directory/'selected_evaluation/audit.json').relative_to(root))] = base.digest(directory/'selected_evaluation/audit.json')
        for arm in ('basic', 'symbolic'):
            for entry in trained['revision_frozen'][arm]:
                revision = entry['revision']
                source = directory/arm/'frozen_revisions'/f'revision_{revision:03d}.py'
                require(base.digest(source) == entry['sha256'], 'Frozen revision changed')
                preserved[str(source.relative_to(root))] = entry['sha256']
                reuse = revision == trained['frozen'][arm]['revision']
                for seed in protocol['heldout_seeds']:
                    trial = (directory/'selected_evaluation'/arm/'heldout'/f'seed_{seed}' if reuse else
                             directory/'curve_rollouts'/arm/f'revision_{revision:03d}'/f'seed_{seed}')
                    if reuse:
                        for relative, digest in previous_audit['result_hashes'].items():
                            if relative.startswith(f'{arm}/heldout/seed_{seed}/'):
                                preserved[str((directory/'selected_evaluation'/relative).relative_to(root))] = digest
                    records.append({'replica': name, 'arm': arm, 'revision': revision, 'seed': seed,
                                    'candidate': str(source.relative_to(root)), 'candidate_sha256': entry['sha256'],
                                    'validation_error': entry.get('validation_error'), 'reused': reuse,
                                    'trial': str(trial.relative_to(root))})
    data_root.mkdir(parents=True)
    output.mkdir(parents=True)
    for replica in protocol['replicas']:
        destination = data_root/replica['directory']
        destination.mkdir()
        (root/replica['directory']/'curve_rollouts').symlink_to(destination, target_is_directory=True)
    registration = {'schema': 'posthoc_geometry_learning_curves_v1', 'registered_at': time.time(),
        'root': str(root), 'data_root': str(data_root), 'max_workers': workers, 'pilot_rollouts': 32,
        'minimum_free_disk_bytes': 15*1024**3, 'minimum_available_ram_bytes': 6*1024**3,
        'model_calls': 0, 'records': records, 'slots': len(records),
        'reused_slots': sum(r['reused'] for r in records), 'new_slots': sum(not r['reused'] for r in records),
        'preserved_files': preserved, 'runner_sha256': base.digest(Path(__file__)),
        'primary_threshold': .8, 'secondary_thresholds': [.6, .7],
        'curves': ['Raw proposal at each revision', 'Training-retained policy at each proposal budget'],
        'scope': 'Post-hoc extension after selected-policy results were inspected. No test feedback to actors or test-based policy selection.',
        'ramp_rule': 'Run up to 32 non-baseline pending rollouts at the requested concurrency; use at most eight afterward if sampled available RAM falls below 6 GiB or GPU memory exceeds 90 percent.',
        'hardware_at_registration': hardware()}
    write_json(output/'registration.json', registration)
    shutil.copyfile(Path(__file__), output/'runner_snapshot.py')
    check(root, output)
    return registration


def check(root, output):
    registration = read(output/'registration.json')
    require(base.digest(Path(__file__)) == registration['runner_sha256'] == base.digest(output/'runner_snapshot.py'), 'Curve runner changed')
    protocol = study.check(root)
    study.check_gate(root, protocol)
    for name, digest in registration['preserved_files'].items():
        require(base.digest(root/name) == digest, 'Original evidence changed: '+name)
    for replica in protocol['replicas']:
        require((root/replica['directory']/'curve_rollouts').resolve() == Path(registration['data_root'])/replica['directory'], 'Data location changed')
    return registration


@contextmanager
def telemetry(output, samples):
    done = threading.Event()
    def monitor():
        while not done.is_set():
            try:
                sample = hardware()
                samples.append(sample)
                with (output/'hardware.jsonl').open('a') as handle:
                    handle.write(json.dumps(sample)+'\n')
            except Exception as error:
                with (output/'hardware.jsonl').open('a') as handle:
                    handle.write(json.dumps({'time': time.time(), 'error': str(error)})+'\n')
            done.wait(2)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()


def pilot_workers(samples, requested):
    healthy = bool(samples)
    for sample in samples:
        healthy &= sample['memory']['MemAvailable'] >= 6*1024**3
        if sample['gpu_query_returncode'] != 0:
            healthy = False
        for line in sample['gpu_csv'].splitlines():
            _, total, used, _ = [part.strip() for part in line.split(',')]
            healthy &= float(used) <= .9*float(total)
    return requested if healthy else min(requested, 8)


def audit_trial(root, row, manifest, initial_hash):
    trial = root/row['trial']
    result = read(trial/'result.json')
    require(result['seed'] == row['seed'] and type(result['success']) is bool, 'Result identity mismatch')
    require(result['candidate_sha256'] == row['candidate_sha256'], 'Candidate mismatch')
    if row['validation_error']:
        require(result.get('invalid_candidate') and not result['success'] and not result['rollout_executed'], 'Invalid candidate accounting changed')
    else:
        require(not result.get('invalid_candidate'), 'Unexpected invalid candidate')
        require(result['geometry_library_sha256'] == manifest['library_sha256'][row['arm']], 'Observer mismatch')
        require(result['task'] == manifest['task'] and result['layout'] == 'sampled' and
                result['scene']['seed'] == row['seed'] and result['scene']['recipe'] == manifest['recipe'] and
                result['scene']['color_map'] == manifest['color_maps'][str(row['seed'])] and
                result['scene']['object_count'] == manifest['object_count'] and
                result['physics'] == manifest['physics'] and result['sensor'] == manifest['sensor'] and
                result['wrist_depth'] and not result['sensor_blackout'], 'Runtime settings mismatch')
        study.neutral.check_goal(trial, result, manifest)
        sandbox = read(trial/'sandbox.json')
        require(sandbox['isolation'] == 'bwrap' and sandbox['network'] == 'disabled', 'Sandbox mismatch')
        require(base.digest(trial/'initial.png') == initial_hash, 'Initial image differs across revisions or replicas')
    return {'seed': row['seed'], 'success': result['success'], 'policy_error': result.get('policy_error')}


def curves(training, raw, tokens, thresholds):
    decisions = retention(training)
    retained = [raw[d['selected_revision']] for d in decisions]
    return {'raw': raw, 'retained': retained, 'retained_revisions': [d['selected_revision'] for d in decisions],
            'cumulative_generation_tokens': tokens,
            'first_raw_threshold': {str(t): next((i for i,v in enumerate(raw) if v >= t), None) for t in thresholds},
            'first_retained_threshold': {str(t): next((i for i,v in enumerate(retained) if v >= t), None) for t in thresholds}}


def audit(root, output):
    registration = check(root, output)
    rows = {}
    hashes = {}
    first_replica = study.read(root/'protocol.json')['replicas'][0]['directory']
    for row in registration['records']:
        manifest = read(root/row['replica']/'manifest.json')
        initial = base.digest(root/first_replica/'selected_evaluation/basic/heldout'/f"seed_{row['seed']}"/'initial.png')
        result = audit_trial(root, row, manifest, initial)
        if not row['reused']:
            require((root/row['trial']/'result.json').stat().st_mtime >= registration['registered_at'], 'Rollout predates curve registration')
        key = (row['replica'], row['arm'], row['revision'])
        rows.setdefault(key, []).append(result)
        hashes[row['trial']+'/result.json'] = base.digest(root/row['trial']/'result.json')
    report = {'status': 'completed', 'scope': registration['scope'], 'slots': registration['slots'],
              'reused_slots': registration['reused_slots'], 'new_slots': registration['new_slots'],
              'replicas': {}, 'mean_curves': {}, 'result_hashes': hashes,
              'registration_sha256': base.digest(output/'registration.json'),
              'scheduler': audit_leases(output, {'rollout': registration['max_workers']})}
    replicas = study.read(root/'protocol.json')['replicas']
    thresholds = [registration['primary_threshold']] + registration['secondary_thresholds']
    for replica in replicas:
        name = replica['directory']
        training = read(root/name/'training_summary.json')
        resources = read(root/name/'resource_accounting.json')
        report['replicas'][name] = {}
        for arm in ('basic', 'symbolic'):
            raw = [statistics.mean(row['success'] for row in rows[name,arm,r]) for r in range(4)]
            tokens = [0]
            for turn in resources['arms'][arm]['turns']:
                tokens.append(tokens[-1]+turn['usage']['total_tokens'])
            report['replicas'][name][arm] = curves(training['curves'][arm], raw, tokens, thresholds)
        report['replicas'][name]['paired_by_raw_revision'] = [paired(rows[name,'basic',r], rows[name,'symbolic',r]) for r in range(4)]
    for arm in ('basic', 'symbolic'):
        report['mean_curves'][arm] = {kind: [statistics.mean(r[arm][kind][i] for r in report['replicas'].values()) for i in range(4)]
                                           for kind in ('raw','retained','cumulative_generation_tokens')}
    write_json(output/'summary.json', report)
    plot(output, report)
    return report


def plot(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    colors = {'basic': '#2463a5', 'symbolic': '#c75c24'}
    for axis, kind, token_axis in zip(axes, ('raw','retained','retained'), (False,False,True)):
        for arm, label in (('basic','Base'),('symbolic','Full geometry')):
            mean = report['mean_curves'][arm]
            for replica in report['replicas'].values():
                x = [v/1000 for v in replica[arm]['cumulative_generation_tokens']] if token_axis else range(4)
                axis.plot(x, [100*v for v in replica[arm][kind]], color=colors[arm], alpha=.18, linewidth=1)
            x = [v/1000 for v in mean['cumulative_generation_tokens']] if token_axis else range(4)
            axis.plot(x, [100*v for v in mean[kind]], '-o', label=label, color=colors[arm], linewidth=2.5)
        axis.axhline(80, color='gray', linestyle=':', linewidth=1)
        axis.set_ylim(0, 100)
        axis.set_xlabel('Cumulative generation tokens (thousands)' if token_axis else 'Proposal iteration')
        if not token_axis: axis.set_xticks(range(4))
        axis.set_ylabel('Test success rate (%)')
        axis.grid(alpha=.2)
        axis.legend()
    axes[0].set_title('Raw proposals')
    axes[1].set_title('Training-retained policies')
    axes[2].set_title('Retained policies vs token budget')
    fig.suptitle('Five paired training chains; 100 shared test layouts\nThin lines: individual chains. Thick lines: means. Post-hoc evaluation.')
    fig.savefig(output/'learning_curves.png', dpi=180)
    fig.savefig(output/'learning_curves.pdf')
    plt.close(fig)
    with (output/'learning_curves.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['replica','arm','iteration','raw_sr','retained_sr','retained_revision','cumulative_tokens'])
        for name, replica in report['replicas'].items():
            for arm in ('basic','symbolic'):
                value = replica[arm]
                for i in range(4):
                    writer.writerow([name,arm,i,value['raw'][i],value['retained'][i],value['retained_revisions'][i],value['cumulative_generation_tokens'][i]])


def run(root, output):
    with study.exclusive(root/'coordinator.lock'), study.exclusive(output/'coordinator.lock'):
        registration = check(root, output)
        check_stop(output)
        if (output/'summary.json').exists():
            return audit(root, output)
        manifests = {replica['directory']: read(root/replica['directory']/'manifest.json') for replica in study.read(root/'protocol.json')['replicas']}
        pending = [r for r in registration['records'] if not r['reused'] and not (root/r['trial']/'result.json').exists()]
        # Pilot action-taking policies rather than the cheap observation-only baseline.
        pending.sort(key=lambda r: (r['revision']==0, r['seed'], r['revision'], r['replica'], r['arm']))
        completed = registration['new_slots']-len(pending)
        started = time.time()
        samples = []
        def execute(row):
            trial = root/row['trial']
            with lease(output, 'rollout', registration['max_workers'], row['trial']):
                require(shutil.disk_usage(trial.parent if trial.parent.exists() else Path(registration['data_root'])).free >= registration['minimum_free_disk_bytes'], 'Data disk reserve reached')
                return base.execute(root/row['replica'], row['arm'], root/row['candidate'], trial.parent,
                                    [row['seed']], manifests[row['replica']], row['validation_error'])
        def batch(jobs, workers):
            nonlocal completed
            for _, _ in bounded_map(execute, jobs, workers, on_error=lambda error: stop(output,error)):
                completed += 1
                write_json(output/'progress.json', {'status':'evaluating','pid':os.getpid(),'new_completed':completed,
                    'new_total':registration['new_slots'],'reused':registration['reused_slots'],
                    'workers':workers,'elapsed_seconds':time.time()-started,'time':time.time()})
        write_json(output/'progress.json', {'status':'pilot','pid':os.getpid(),'workers':registration['max_workers'],
                   'new_completed':completed,'new_total':registration['new_slots'],'reused':registration['reused_slots']})
        try:
            with telemetry(output, samples), study.variant.configured():
                pilot_path = output/'pilot.json'
                if not pilot_path.exists():
                    size = min(registration['pilot_rollouts'],len(pending))
                    batch(pending[:size], registration['max_workers'])
                    workers = pilot_workers(samples, registration['max_workers'])
                    write_json(pilot_path, {'completed':size,'elapsed_seconds':time.time()-started,'workers_after_pilot':workers,
                               'minimum_available_ram_bytes':min(s['memory']['MemAvailable'] for s in samples) if samples else None,
                               'hardware_samples':len(samples)})
                    pending = pending[size:]
                else:
                    workers = read(pilot_path)['workers_after_pilot']
                batch(pending,workers)
            write_json(output/'progress.json', {'status':'auditing','new_completed':completed,'new_total':registration['new_slots']})
            report = audit(root,output)
            write_json(output/'progress.json', {'status':'completed','new_completed':completed,'new_total':registration['new_slots'],
                       'reused':registration['reused_slots'],'elapsed_seconds':time.time()-started})
            return report
        except BaseException as error:
            stop(output,error)
            write_json(output/'progress.json', {'status':'failed','error':str(error),'new_completed':completed})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--phase', choices=('prepare','run','audit'), required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if args.phase == 'prepare':
        require(args.data_root is not None, 'Explicit data disk path required')
        with study.exclusive(root/'coordinator.lock'):
            report = prepare(root,output,args.data_root.resolve(),args.workers)
        print({key:report[key] for key in ('slots','reused_slots','new_slots','max_workers')})
    elif args.phase == 'audit':
        with study.exclusive(root/'coordinator.lock'):
            print(audit(root,output)['mean_curves'])
    else:
        def interrupt(signum, frame):
            stop(output,RuntimeError(f'Curve supervisor received signal {signum}'))
            raise SystemExit(128+signum)
        signal.signal(signal.SIGINT,interrupt)
        signal.signal(signal.SIGTERM,interrupt)
        run(root,output)
