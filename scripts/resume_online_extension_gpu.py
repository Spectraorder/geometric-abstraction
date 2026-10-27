"""Explicit GPU repair amendment: preserve attempted slots, dispatch only unstarted ones.

Historical timing violations remain flagged. Future batches retain the original
two-second gate; neither episode code nor thresholds are changed by this wrapper.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time

import run_online_extension_study as study
from online_gpu_preflight import environment, gpu_check

ROOT, read, write, require, digest = study.ROOT, study.read, study.write, study.require, study.digest


def pending_slots(evidence, slots):
    completed, pending = [], []
    for slot in slots:
        path = study.episode_path(evidence, slot)
        if not path.exists():
            pending.append(slot)
            continue
        require((path / 'result.json').is_file(), 'Partial attempt must not be retried')
        result = read(path / 'result.json')
        require(result['status'] in ('success', 'timeout') and not result.get('archive_error'),
                'Unresolved infrastructure attempt')
        require(result['seed'] == slot['seed'] and result['arm'] == slot['arm'], 'Slot identity changed')
        completed.append(slot)
    require(completed + pending == slots, 'Attempts are not a contiguous schedule prefix')
    require(len(completed) % 4 == 0 and pending, 'Resume must start at a complete batch boundary')
    return completed, pending


def annotate_quality(rows, *, gpu_ok, epoch):
    return [dict(row, timing_valid=row['max_simulation_lag_seconds'] <= 2.,
                 gpu_dispatch_query_ok=gpu_ok, runtime_epoch=epoch) for row in rows]


def valid_pairs(rows):
    """Sensitivity subset only: retain both arms or neither; never replace a slot."""
    study.paired(rows)
    excluded = {(r['task'], r['seed']) for r in rows
                if r.get('timing_valid') is False or r.get('gpu_dispatch_query_ok') is False}
    return [r for r in rows if (r['task'], r['seed']) not in excluded]


def flagged_ring_records(evidence, slots, report):
    """Normalize the stopped pair without treating its failed timing gate as passed."""
    require(report['status'] == 'passed', 'Historical interface audit failed')
    expected = {(s['seed'], s['arm']) for s in slots}
    require({(r['seed'], r['arm']) for r in report['episodes']} == expected,
            'Historical ring audit does not match stopped slots')
    records = []
    for row in report['episodes']:
        episode = evidence / 'tasks/ring' / row['episode']
        result = read(episode / 'result.json')
        require(result['status'] in ('success', 'timeout') and not result.get('archive_error'),
                'Invalid historical result')
        require(result['budget_s'] == 200. and not result['scripted'] and
                result['timing']['reset_count'] == 1, 'Historical episode settings changed')
        require(row['max_simulation_lag_seconds'] == result['timing']['max_simulation_lag_seconds'] > 2.,
                'Expected preserved timing violation')
        failures = [read(p) for p in episode.glob('context_*/decision_*/attempt_failure.json')]
        incomplete = sum(f['wall_seconds'] for f in failures)
        records.append(dict(task='ring', study=evidence.name, cohort='new_45', seed=row['seed'],
            arm={'basic': 'Base', 'symbolic': 'Full'}[row['arm']], status=result['status'],
            success=int(row['success']), budget_seconds=200., capped_seconds=row['capped_seconds'],
            success_seconds=row['capped_seconds'] if row['success'] else None,
            completed_model_seconds=row['model_seconds'], incomplete_model_seconds=incomplete,
            all_model_seconds=row['model_seconds'] + incomplete,
            completed_requests=row['completed_requests'], incomplete_requests=row['incomplete_requests'],
            usage_complete=not failures, **row['usage'], images=row['images'],
            camera_acquisitions=result['camera_acquisitions'], depth_queries=result['depth_queries'],
            action_seconds=result['action_seconds'], feedback_seconds=result['feedback_seconds'],
            initial_state_sha256=row['initial_state_sha256'],
            max_simulation_lag_seconds=row['max_simulation_lag_seconds'],
            source=str(episode / 'result.json'), source_sha256=digest(episode / 'result.json')))
    study.paired(records)
    return records


def baseline(evidence, completed):
    """Complete the interrupted audit, preserving the original STOP and reports."""
    require(len(completed) == 48, 'This amendment is scoped to the known 48-attempt stop')
    records = []
    for offset in range(0, 44, 4):
        records.extend(read(evidence / 'audits' / f'batch_{offset:03d}' / 'records.json'))
    require(len(records) == 44, 'Prior batch audits are incomplete')
    records.extend(flagged_ring_records(evidence, completed[44:46],
        read(evidence / 'audits/batch_044/ring/interface_audit.json')))
    records.extend(study.audit_batch(evidence, completed[46:48], 'gpu_repair_remaining_baseline'))
    expected = {(s['task'], s['seed'], {'basic': 'Base', 'symbolic': 'Full'}[s['arm']]): s for s in completed}
    require(len(records) == len(expected) and
            {(r['task'], r['seed'], r['arm']) for r in records} == set(expected), 'Baseline mismatch')
    result = []
    for row in records:
        slot = expected[(row['task'], row['seed'], row['arm'])]
        hardware = read(evidence / f"hardware_{slot['index'] // 4 * 4:03d}.json")
        result.extend(annotate_quality([row], gpu_ok=hardware['gpu_query_returncode'] == 0,
                                      epoch='before_gpu_repair'))
    study.paired(result)
    return result


def prepare(root, evidence, manifest, preflight):
    protocol = study.check(evidence)
    require(read(evidence / 'lifecycle.json')['status'] == 'stopped', 'Original run is not stopped')
    require((evidence / 'STOP.json').is_file(), 'Missing original STOP receipt')
    completed, pending = pending_slots(evidence, protocol['slots'])
    report = read(preflight)
    require(report['status'] == 'passed' and report['concurrency'] == 4 and
            report['runtime_manifest_sha256'] == digest(manifest), 'GPU preflight did not pass')
    gpu = gpu_check(manifest)
    root.mkdir(parents=True, exist_ok=False)
    rows = baseline(evidence, completed)
    write(root / 'baseline_records.json', rows)
    shutil.copy2(manifest, root / 'runtime_manifest.json')
    shutil.copy2(preflight, root / 'preflight.json')
    frozen = {}
    for filename in ('scripts/resume_online_extension_gpu.py', 'scripts/online_gpu_preflight.py'):
        dest = root / 'sources' / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / filename, dest)
        frozen[str(ROOT / filename)] = digest(dest)
        frozen[str(dest)] = digest(dest)
    for path in (evidence / 'protocol.json', evidence / 'STOP.json', evidence / 'progress.json',
                 evidence / 'lifecycle.json', root / 'baseline_records.json',
                 root / 'runtime_manifest.json', root / 'preflight.json'):
        frozen[str(path)] = digest(path)
    for row in rows:
        require(digest(Path(row['source'])) == row['source_sha256'], 'Existing outcome changed')
        frozen[row['source']] = row['source_sha256']
    for path in (evidence / 'audits').rglob('*.json'):
        frozen[str(path)] = digest(path)
    registration = dict(name='online_extension_private_gpu_resume_v1', registered_at=time.time(),
        authorization='User explicitly requested repair; continue only unstarted slots',
        evidence_root=str(evidence), original_protocol_sha256=digest(evidence / 'protocol.json'),
        completed_slots=completed, pending_slots=pending, remaining=len(pending), total=270,
        concurrency=4, budget_s=200., timing_lag_limit_s=2., gpu_at_registration=gpu,
        frozen_sha256=frozen, historical_timing_violations=sum(not r['timing_valid'] for r in rows),
        historical_gpu_query_failures=sum(not r['gpu_dispatch_query_ok'] for r in rows),
        repair='Private userspace NVIDIA 580.173.02 libraries matching the already loaded module; '
               'no system installation, module unload, reboot or episode implementation change',
        evidence_policy='Preserve all 48 outcomes and their timing/GPU quality flags; no replay or '
                        'replacement. Keep all-slot descriptive summaries and a separately labeled '
                        'paired sensitivity subset excluding lag violations or failed dispatch GPU queries.',
        failure_policy='Original strict audits plus NVIDIA EGL/NVML checks before each batch; '
                       'stop on any new failure, preserve attempts, no automatic retry or resume')
    write(root / 'registration.json', registration)
    (root / 'registration.sha256').write_text(digest(root / 'registration.json') + '\n')
    write(root / 'progress.json', dict(status='registered', completed=48, remaining=222, total=270,
                                     timing_valid=46, timing_flagged=2, legacy_completed=30, pooled_total=300))
    return registration


def check(root):
    require(digest(root / 'registration.json') == (root / 'registration.sha256').read_text().strip(),
            'Resume registration changed')
    reg = read(root / 'registration.json')
    for name, sha in reg['frozen_sha256'].items():
        require(digest(Path(name)) == sha, 'Frozen resume evidence changed: ' + name)
    protocol = study.check(Path(reg['evidence_root']))
    require(reg['completed_slots'] + reg['pending_slots'] == protocol['slots'], 'Resume schedule changed')
    require(reg['concurrency'] == 4 and reg['budget_s'] == 200. and reg['timing_lag_limit_s'] == 2.,
            'Resume conditions changed')
    environment(root / 'runtime_manifest.json')
    return reg


def summarize(root, rows):
    reg = check(root)
    evidence = Path(reg['evidence_root'])
    require(len(rows) == 270, 'Incomplete extended cohort')
    for row in rows:
        require(digest(Path(row['source'])) == row['source_sha256'], 'Outcome changed')
    legacy = [dict(r, cohort='original_5') for r in read(evidence / 'legacy_episodes.json')]
    combined = legacy + rows
    quality = dict(timing_flagged=sum(r.get('timing_valid') is False for r in rows),
                   gpu_dispatch_query_failed=sum(r.get('gpu_dispatch_query_ok') is False for r in rows),
                   note=reg['evidence_policy'])
    summary = dict(status='completed_with_preserved_quality_flags', quality=quality,
        original_5=study.summaries(legacy), new_45=study.summaries(rows),
        pooled_50=study.summaries(combined),
        sensitivity_valid_pairs_new=study.summaries(valid_pairs(rows)),
        sensitivity_valid_pairs_pooled=study.summaries(valid_pairs(combined)), records=combined,
        infrastructure_overhead=read(evidence / 'infrastructure_attempts.json'))
    write(root / 'summary.json', summary)
    fields = list(dict.fromkeys(k for row in combined for k in row))
    with (root / 'episodes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(combined)
    return summary


def execute(root):
    reg = check(root)
    evidence = Path(reg['evidence_root'])
    with (root / 'coordinator.lock').open('a') as lock, (evidence / 'coordinator.lock').open('a') as prior_lock:
        for handle in (lock, prior_lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (root / 'STOP.json').exists(), 'Stopped continuation needs a new explicit amendment')
        require(all(not study.episode_path(evidence, s).exists() for s in reg['pending_slots']),
                'A pending slot already has an attempt; refusing automatic replay')
        (root / 'workers').mkdir(exist_ok=False)
        (root / 'records').mkdir(exist_ok=False)
        rows = read(root / 'baseline_records.json')
        started = time.time()
        write(root / 'lifecycle.json', dict(status='running', pid=os.getpid(), started_at=started))

        def progress(status, active=()):
            completed = sum((study.episode_path(evidence, s) / 'result.json').exists()
                            for s in reg['completed_slots'] + reg['pending_slots'])
            write(root / 'progress.json', dict(status=status, completed=completed, total=270,
                remaining=270 - completed, audited=len(rows), timing_valid=sum(r['timing_valid'] for r in rows),
                timing_flagged=sum(not r['timing_valid'] for r in rows),
                gpu_query_flagged=sum(not r['gpu_dispatch_query_ok'] for r in rows),
                legacy_completed=30, pooled_completed=30 + completed, pooled_total=300,
                active_slots=list(active), started_at=started, updated_at=time.time(),
                tasks={task: dict(completed=sum(r['task'] == task for r in rows), total=90,
                    successes={arm: sum(r['success'] for r in rows if r['task'] == task and r['arm'] == arm)
                               for arm in ('Base', 'Full')}) for task in study.STUDIES}))

        def launch(slot):
            env = environment(root / 'runtime_manifest.json')
            with (root / 'workers' / f"{slot['index']:03d}.log").open('x') as log:
                return subprocess.call([str(ROOT / '.venv/bin/python'), '-u', str(Path(__file__).resolve()),
                    '--root', str(root), '--phase', 'episode', '--slot', str(slot['index'])],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)

        try:
            for offset in range(0, len(reg['pending_slots']), 4):
                check(root)
                require(not (root / 'STOP.json').exists(), 'External STOP receipt found')
                batch = reg['pending_slots'][offset:offset + 4]
                first = batch[0]['index']
                write(root / f'hardware_{first:03d}.json', gpu_check(root / 'runtime_manifest.json'))
                progress('running', batch)
                with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                    codes = list(pool.map(launch, batch))
                require(not any(codes), 'Worker failed; no further batch dispatched')
                audited = study.audit_batch(evidence, batch, f'gpu_resume_batch_{first:03d}')
                audited = annotate_quality(audited, gpu_ok=True, epoch='private_gpu_runtime')
                write(root / 'records' / f'batch_{first:03d}.json', audited)
                rows.extend(audited)
                progress('running')
            # Repeat the strict independent audits over the continuation, not the flagged old pair.
            final_rows = study.audit_batch(evidence, reg['pending_slots'], 'gpu_resume_final')
            require(len(final_rows) == reg['remaining'], 'Incomplete final audit')
            summarize(root, read(root / 'baseline_records.json') +
                      annotate_quality(final_rows, gpu_ok=True, epoch='private_gpu_runtime'))
            progress('completed_with_preserved_quality_flags')
            write(root / 'lifecycle.json', dict(status='completed', pid=os.getpid(), finished_at=time.time()))
        except Exception as error:
            if not (root / 'STOP.json').exists():
                write(root / 'STOP.json', dict(error=str(error), time=time.time(), automatic_retry=False))
            progress('stopped')
            write(root / 'lifecycle.json', dict(status='stopped', pid=os.getpid(), error=str(error), time=time.time()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'check', 'run', 'episode'), required=True)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--slot', type=int, choices=range(270))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == 'prepare':
        prepare(root, args.evidence.resolve(), args.manifest.resolve(), args.preflight.resolve())
    elif args.phase == 'check':
        check(root)
    elif args.phase == 'run':
        execute(root)
    else:
        reg = check(root)
        require(args.slot in {s['index'] for s in reg['pending_slots']}, 'Slot is not authorized for continuation')
        # The environment must be present before importing MuJoCo/OpenGL.
        require(all(os.environ.get(k) == v for k, v in read(root / 'runtime_manifest.json')['environment'].items()),
                'Worker private GPU environment missing')
        study.worker(Path(reg['evidence_root']), args.slot)
