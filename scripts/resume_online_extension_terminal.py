"""Recover image-only archive failures and finish the remaining online slots.

Raw evidence is immutable. The two known terminal-image failures get a separate
derived audit view; independent prompt/resource/physics audits must pass before
their measured outcomes are included. No episode is replayed.
"""
import argparse
import os
from pathlib import Path
import shutil
import time
from types import FunctionType

import run_online_extension_study as study
import resume_online_extension_gpu as gpu
import resume_online_extension_camera as camera
import online_ring_camera_lifecycle as pacing
from online_gpu_preflight import gpu_check

ROOT, read, write, require, digest = study.ROOT, study.read, study.write, study.require, study.digest
RECOVERABLE_INDICES = {164, 165}
ARCHIVE_ERROR = 'Ring remained released over the peg for two seconds'


def validate_recoverable(path):
    """Accept only the exact post-success image failure, never a policy/physics error."""
    import numpy as np
    result = read(path / 'result.json')
    require(result['status'] == 'success' and result['success'] is True and
            result.get('archive_error') == ARCHIVE_ERROR and not result.get('error'),
            'Not the known post-success image failure')
    require(result['budget_s'] == 200. and result['time_to_success_s'] <= 200. and
            result['timing']['reset_count'] == 1 and
            result['timing']['max_simulation_lag_seconds'] <= 2., 'Episode conditions failed')
    require(result['private_final_status']['on_peg'] is True, 'Terminal task state unavailable')
    with np.load(path / 'private_terminal_state.npz') as state:
        require(abs(float(state['time']) - result['private_success_evidence']['end_sim_time']) < 1e-9,
                'Archived physics is not the success-time state')
    require(len(list(path.glob('camera_*.npz'))) == result['camera_acquisitions'],
            'Failure did not precede the final image capture')
    require((path / 'private_model.mjb').is_file() and (path / 'private_monitor.json').is_file(),
            'Required independent physical evidence is missing')
    return result


def recover_archives(root, evidence, slots):
    view = root / 'archive_recovery'
    (view / 'tasks/ring').mkdir(parents=True, exist_ok=False)
    (view / 'sources').symlink_to(evidence / 'sources', target_is_directory=True)
    (view / 'tasks/ring/sources').symlink_to(evidence / 'sources', target_is_directory=True)
    receipts = []
    for slot in slots:
        require(slot['index'] in RECOVERABLE_INDICES and slot['task'] == 'ring', 'Unexpected recovery slot')
        original = study.episode_path(evidence, slot)
        raw = validate_recoverable(original)
        target = study.episode_path(view, slot)
        target.mkdir(parents=True, exist_ok=False)
        for item in original.iterdir():
            if item.name != 'result.json':
                (target / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
        derived = dict(raw)
        del derived['archive_error']
        write(target / 'result.json', derived)
        receipts.append(dict(index=slot['index'], original=str(original / 'result.json'),
            original_sha256=digest(original / 'result.json'),
            derived=str(target / 'result.json'), derived_sha256=digest(target / 'result.json'),
            sole_derived_change='Remove the known terminal-image archive error for independent auditing',
            original_archive_error=raw['archive_error'], final_image_available=False,
            model_requests_added=0, physics_steps_added=0, outcome_and_resource_values_changed=False,
            physical_evidence_sha256={name: digest(original / name) for name in
                ('private_model.mjb', 'private_terminal_state.npz', 'private_monitor.json')}))
    # Includes terminal evaluator replay, the two-second hold trace, exact model
    # prompts/images, input exclusion, session counters, targets and pairing.
    records = study.audit_batch(view, slots, 'independent_archive_recovery')
    for row in records:
        receipt = next(r for r in receipts if r['derived'] == row['source'])
        row.update(audit_source=row['source'], audit_source_sha256=row['source_sha256'],
                   source=receipt['original'], source_sha256=receipt['original_sha256'],
                   original_archive_error=receipt['original_archive_error'],
                   archive_recovery_verified=True, terminal_image_available=False)
    write(view / 'recovery.json', dict(status='independently_verified', episodes=receipts,
        scope='Recover measured outcomes from saved physical/input evidence; final illustration image remains missing'))
    return gpu.annotate_quality(records, gpu_ok=True, epoch='ring_camera_pacing_v1')


def inventory(evidence, slots):
    completed, pending = [], []
    for slot in slots:
        path = study.episode_path(evidence, slot)
        if not path.exists():
            pending.append(slot)
            continue
        require((path / 'result.json').is_file(), 'Incomplete attempt cannot be repeated')
        result = read(path / 'result.json')
        require(result['seed'] == slot['seed'] and result['arm'] == slot['arm'], 'Slot identity changed')
        if result.get('archive_error'):
            require(slot['index'] in RECOVERABLE_INDICES, 'Unexpected archive failure')
            validate_recoverable(path)
        else:
            require(result['status'] in ('success', 'timeout'), 'Unexpected unfinished result')
        completed.append(slot)
    require(completed + pending == slots and len(completed) == 168 and len(pending) == 102,
            'Unexpected continuation boundary')
    return completed, pending


def prepare(root, previous, preflight):
    prior = gpu.check(previous)
    require(read(previous / 'lifecycle.json')['status'] == 'stopped', 'Previous coordinator is not stopped')
    pid = read(previous / 'lifecycle.json')['pid']
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('Previous process still exists; inspect before continuation')
    evidence = Path(prior['evidence_root'])
    protocol = study.check(evidence)
    completed, pending = inventory(evidence, protocol['slots'])
    report = read(preflight)
    require(report['status'] == 'passed' and report['model_requests'] == 0 and
            report['reserved_layouts_used'] is False and
            {r['mode'] for r in report['results']} == {'success', 'timeout'} and
            report['pacing_source_sha256'] == digest(ROOT / 'scripts/online_ring_camera_lifecycle.py') and
            report['runtime_manifest_sha256'] == digest(previous / 'runtime_manifest.json'),
            'Lifecycle preflight is missing or mismatched')
    health = gpu_check(previous / 'runtime_manifest.json')
    root.mkdir(parents=True, exist_ok=False)
    rows = read(previous / 'baseline_records.json')
    for p in sorted((previous / 'records').glob('batch_*.json')):
        rows.extend(read(p))
    require(len(rows) == 164, 'Unexpected prior audited count')
    rows.extend(recover_archives(root, evidence, protocol['slots'][164:166]))
    other = study.audit_batch(evidence, protocol['slots'][166:168], 'terminal_repair_baseline')
    rows.extend(gpu.annotate_quality(other, gpu_ok=True, epoch='ring_camera_pacing_v1'))
    require(len(rows) == 168 and sum(not r['timing_valid'] for r in rows) == 4,
            'Historical timing flags changed')
    study.paired(rows)
    write(root / 'baseline_records.json', rows)
    shutil.copy2(previous / 'runtime_manifest.json', root / 'runtime_manifest.json')
    shutil.copy2(preflight, root / 'preflight.json')
    shutil.copy2(previous.parent / 'continuation.json', root / 'previous_continuation.json')
    frozen = {}
    for name in ('resume_online_extension_terminal.py', 'online_ring_camera_lifecycle.py',
                 'check_online_ring_camera_lifecycle.py', 'resume_online_extension_camera.py',
                 'resume_online_extension_gpu.py', 'online_gpu_preflight.py'):
        src = ROOT / 'scripts' / name
        dest = root / 'sources/scripts' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        frozen[str(src)] = digest(src)
        frozen[str(dest)] = digest(dest)
    paths = [evidence / 'protocol.json', previous / 'registration.json', previous / 'STOP.json',
             previous / 'progress.json', previous / 'lifecycle.json', root / 'baseline_records.json',
             root / 'runtime_manifest.json', root / 'preflight.json', root / 'previous_continuation.json']
    paths += list((evidence / 'audits').rglob('*.json'))
    paths += list((previous / 'records').glob('*.json'))
    paths += [p for p in (root / 'archive_recovery').rglob('*.json') if not p.is_symlink()]
    for path in paths:
        frozen[str(path)] = digest(path)
    for row in rows:
        require(digest(Path(row['source'])) == row['source_sha256'], 'Original result changed')
        frozen[row['source']] = row['source_sha256']
    change = dict(prior['runtime_change'], name='ring_camera_pacing_v2',
        terminal_boundary='archive_terminal_physics disables further camera-driven physics; '
                          'live camera pacing is identical to v1',
        interpretation='Archive-only correction after slot 167; no change to live policy decisions. '
                       'Separate runtime epochs and preserve the two recovered image-only failures.')
    reg = dict(name='online_extension_terminal_archive_resume_v1', registered_at=time.time(),
        authorization='User requested continuous monitoring/debugging until all registered slots finish',
        evidence_root=str(evidence), previous_resume_root=str(previous),
        original_protocol_sha256=digest(evidence / 'protocol.json'), completed_slots=completed,
        pending_slots=pending, remaining=102, total=270, concurrency=4, budget_s=200.,
        timing_lag_limit_s=2., gpu_at_registration=health, frozen_sha256=frozen,
        runtime_change=change, historical_timing_violations=4, historical_gpu_query_failures=16,
        archive_recovered_episodes=2, evidence_policy=prior['evidence_policy'].replace('160 outcomes','168 outcomes') +
            ' Two post-success image failures are independently recovered from immutable physical/input '
            'evidence in a separate audit view; original errors stay available.',
        failure_policy='Strict audits and GPU gates; stop on new failure for diagnosis. '
                       'User authorizes debug/continuation of unstarted slots; no automatic replay or extra slots.')
    write(root / 'registration.json', reg)
    (root / 'registration.sha256').write_text(digest(root / 'registration.json') + '\n')
    write(root / 'progress.json', dict(status='registered', completed=168, total=270, remaining=102,
        audited=168, timing_valid=164, timing_flagged=4, gpu_query_flagged=16,
        archive_recovered=2, legacy_completed=30, pooled_completed=198, pooled_total=300))
    gpu.check(root)
    return reg


def annotate_new(rows, *, gpu_ok, epoch):
    return gpu.annotate_quality(rows, gpu_ok=gpu_ok, epoch='ring_camera_pacing_v2')


def summarize(root, rows):
    result = camera.summarize(root, rows)
    result['archive_recovery'] = read(root / 'archive_recovery/recovery.json')
    result['quality']['archive_recovered_episodes'] = 2
    subset = [r for r in gpu.valid_pairs(rows) if not r.get('archive_recovery_verified')]
    result['sensitivity_without_archive_recoveries_new'] = study.summaries(subset)
    write(root / 'summary.json', result)
    return result


def execute(root):
    scope = dict(gpu.execute.__globals__, __file__=__file__,
                 annotate_quality=annotate_new, summarize=summarize)
    return FunctionType(gpu.execute.__code__, scope, gpu.execute.__name__)(root)


def worker(root, index):
    reg = gpu.check(root)
    require(index in {s['index'] for s in reg['pending_slots']}, 'Unregistered continuation slot')
    require(all(os.environ.get(k) == v for k,v in read(root / 'runtime_manifest.json')['environment'].items()),
            'Private GPU environment missing')
    slot = next(s for s in reg['pending_slots'] if s['index'] == index)
    evidence = Path(reg['evidence_root'])
    output = study.episode_path(evidence, slot)
    require(not output.exists(), 'Existing attempt cannot be overwritten')
    counts = {}
    try:
        with pacing.installed() as counts:
            study.worker(evidence, index)
    finally:
        if output.exists():
            write(output / 'runtime_amendment.json', dict(
                registration=str(root / 'registration.json'), registration_sha256=digest(root / 'registration.json'),
                pacing_source_sha256=digest(ROOT / 'scripts/online_ring_camera_lifecycle.py'),
                name='ring_camera_pacing_v2', task=slot['task'], arm=slot['arm'],
                hook_applies=slot['task'] == 'ring', camera_lifecycle_counts=counts))
    gpu.check(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'check', 'run', 'episode'), required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--slot', type=int, choices=range(270))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == 'prepare':
        prepare(root, args.previous.resolve(), args.preflight.resolve())
    elif args.phase == 'check':
        gpu.check(root)
    elif args.phase == 'run':
        execute(root)
    else:
        worker(root, args.slot)
