"""Continue only 110 unstarted slots with symmetric Ring camera pacing.

Reuse the prior resume coordinator and its strict gates. The new entry point
installs a process-local camera hook only inside episode workers; all prior
outcomes, stopped registrations and frozen source files remain unchanged.
"""
import argparse
import os
from pathlib import Path
import shutil
import time
from types import FunctionType

import run_online_extension_study as study
import resume_online_extension_gpu as gpu
import online_ring_camera_pacing as pacing
from online_gpu_preflight import gpu_check

ROOT, read, write, require, digest = study.ROOT, study.read, study.write, study.require, study.digest


def baseline(previous, evidence, slots):
    rows = read(previous / 'baseline_records.json')
    for path in sorted((previous / 'records').glob('batch_*.json')):
        rows.extend(read(path))
    require(len(rows) == 156, 'Unexpected completed audit prefix')
    tower = study.audit_batch(evidence, slots[156:158], 'camera_repair_tower_baseline')
    ring = gpu.flagged_ring_records(evidence, slots[158:160],
                                   read(evidence / 'audits/gpu_resume_batch_156/ring/interface_audit.json'))
    rows.extend(gpu.annotate_quality(tower + ring, gpu_ok=True, epoch='private_gpu_runtime'))
    require(len(rows) == 160 and sum(not r['timing_valid'] for r in rows) == 4,
            'Expected 160 preserved outcomes and four historical lag violations')
    expected = {(s['task'], s['seed'], {'basic': 'Base', 'symbolic': 'Full'}[s['arm']]) for s in slots[:160]}
    require({(r['task'], r['seed'], r['arm']) for r in rows} == expected, 'Baseline slot identities differ')
    study.paired(rows)
    return rows


def prepare(root, previous, preflight):
    prior = gpu.check(previous)
    require(read(previous / 'lifecycle.json')['status'] == 'stopped' and
            (previous / 'STOP.json').exists(), 'Previous coordinator must be stopped')
    evidence = Path(prior['evidence_root'])
    protocol = study.check(evidence)
    completed, pending = gpu.pending_slots(evidence, protocol['slots'])
    require(len(completed) == 160 and len(pending) == 110, 'Unexpected continuation boundary')
    report = read(preflight)
    require(report['status'] == 'passed' and report['concurrency'] == 4 and
            report['model_requests'] == 0 and report['reserved_layouts_used'] is False and
            report['pacing_source_sha256'] == digest(ROOT / 'scripts/online_ring_camera_pacing.py') and
            report['runtime_manifest_sha256'] == digest(previous / 'runtime_manifest.json'),
            'Missing matching development preflight')
    health = gpu_check(previous / 'runtime_manifest.json')
    root.mkdir(parents=True, exist_ok=False)
    rows = baseline(previous, evidence, protocol['slots'])
    write(root / 'baseline_records.json', rows)
    shutil.copy2(previous / 'runtime_manifest.json', root / 'runtime_manifest.json')
    shutil.copy2(preflight, root / 'preflight.json')
    shutil.copy2(previous.parent / 'continuation.json', root / 'previous_continuation.json')
    frozen = {}
    scripts = ('resume_online_extension_camera.py', 'online_ring_camera_pacing.py',
               'check_online_ring_camera_pacing.py', 'resume_online_extension_gpu.py', 'online_gpu_preflight.py')
    for name in scripts:
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
    for path in paths:
        frozen[str(path)] = digest(path)
    for row in rows:
        require(digest(Path(row['source'])) == row['source_sha256'], 'Existing outcome changed')
        frozen[row['source']] = row['source_sha256']
    change = dict(name='ring_camera_pacing_v1', applies_to='Both arms, Ring only',
        hook='Before each OnlineRingSensors.camera call, advance elapsed physics using the existing wait loop',
        cause='Consecutive unavailable sensor fits accumulated lag during the shared grasp prefix',
        unchanged='Model, prompts, geometry fitting, policy prior, submitted actions, task evaluator, '
                  '200-second deadline and two-second lag gate',
        interpretation='Runtime timing amendment after 160 slots; report runtime cohorts separately. '
                       'New camera samples reflect physics advanced during prior CPU sensing work.')
    reg = dict(name='online_extension_camera_pacing_resume_v1', registered_at=time.time(),
        authorization='User explicitly requested continuation after the second timing stop',
        evidence_root=str(evidence), previous_resume_root=str(previous),
        original_protocol_sha256=digest(evidence / 'protocol.json'),
        completed_slots=completed, pending_slots=pending, remaining=110, total=270,
        concurrency=4, budget_s=200., timing_lag_limit_s=2., gpu_at_registration=health,
        frozen_sha256=frozen, runtime_change=change,
        historical_timing_violations=4, historical_gpu_query_failures=16,
        evidence_policy=prior['evidence_policy'].replace('48 outcomes', '160 outcomes'),
        failure_policy=prior['failure_policy'])
    write(root / 'registration.json', reg)
    (root / 'registration.sha256').write_text(digest(root / 'registration.json') + '\n')
    write(root / 'progress.json', dict(status='registered', completed=160, total=270, remaining=110,
        audited=160, timing_valid=156, timing_flagged=4, gpu_query_flagged=16,
        legacy_completed=30, pooled_completed=190, pooled_total=300))
    gpu.check(root)
    return reg


def annotate_new(rows, *, gpu_ok, epoch):
    return gpu.annotate_quality(rows, gpu_ok=gpu_ok, epoch='ring_camera_pacing_v1')


def summarize(root, rows):
    result = gpu.summarize(root, rows)
    result['runtime_change'] = read(root / 'registration.json')['runtime_change']
    result['by_runtime_epoch'] = {
        epoch: study.summaries([r for r in rows if r['runtime_epoch'] == epoch])
        for epoch in sorted({r['runtime_epoch'] for r in rows})}
    write(root / 'summary.json', result)
    return result


def execute(root):
    # Reuse the frozen coordinator without editing it. Its nested launcher now
    # targets this entry point, which installs the registered hook in workers.
    scope = dict(gpu.execute.__globals__, __file__=__file__,
                 annotate_quality=annotate_new, summarize=summarize)
    runner = FunctionType(gpu.execute.__code__, scope, gpu.execute.__name__)
    return runner(root)


def worker(root, index):
    reg = gpu.check(root)
    require(index in {s['index'] for s in reg['pending_slots']}, 'Slot is not authorized for continuation')
    require(all(os.environ.get(k) == v for k, v in read(root / 'runtime_manifest.json')['environment'].items()),
            'Worker private GPU environment missing')
    slot = next(s for s in reg['pending_slots'] if s['index'] == index)
    evidence = Path(reg['evidence_root'])
    output = study.episode_path(evidence, slot)
    require(not output.exists(), 'Existing attempt cannot be overwritten')
    try:
        with pacing.installed():
            study.worker(evidence, index)
    finally:
        if output.exists():
            write(output / 'runtime_amendment.json', dict(
                registration=str(root / 'registration.json'),
                registration_sha256=digest(root / 'registration.json'),
                pacing_source_sha256=digest(ROOT / 'scripts/online_ring_camera_pacing.py'),
                name='ring_camera_pacing_v1', task=slot['task'], arm=slot['arm'],
                hook_applies=slot['task'] == 'ring'))
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
