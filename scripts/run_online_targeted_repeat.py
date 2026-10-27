"""Repeat explicitly selected historical quality pairs in a separate study.

The original 300 outcomes are immutable. Selection is frozen before repeat
outcomes, both arms run once, and no better-of-two or automatic replacement is
computed. Episode implementations, prompts and skills are unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import run_online_extension_study as study
import online_ring_camera_lifecycle as pacing
from online_gpu_preflight import environment, gpu_check

ROOT, read, write, require, digest = study.ROOT, study.read, study.write, study.require, study.digest
PRIMARY = ROOT / 'outputs/online_extension_50_01/terminal_resume_01'
EVIDENCE = ROOT / 'outputs/online_extension_50_01/client_bundle_02'
SCOPES = ('quality', 'quality_and_archive', 'interrupted_usage')


def selected_pairs(records, scope):
    require(scope in SCOPES, 'Unknown selection scope')
    study.paired(records)
    if scope == 'interrupted_usage':
        keys = {(r['task'], r['seed']) for r in records if not r['usage_complete']}
    else:
        keys = {(r['task'], r['seed']) for r in records if r.get('timing_valid') is False
                or r.get('gpu_dispatch_query_ok') is False}
        if scope == 'quality_and_archive':
            keys |= {(r['task'], r['seed']) for r in records if r.get('archive_recovery_verified')}
    return [r for r in records if (r['task'], r['seed']) in keys]


def make_slots(records):
    keys = sorted({(r['task'], r['seed']) for r in records},
                  key=lambda k: (k[1] % 1000, list(study.STUDIES).index(k[0]), k[1]))
    slots = []
    for pair_index, (task, seed) in enumerate(keys):
        for arm in (study.ARMS if pair_index % 2 == 0 else study.ARMS[::-1]):
            original = next(r for r in records if r['task'] == task and r['seed'] == seed
                            and r['arm'] == {'basic': 'Base', 'symbolic': 'Full'}[arm])
            slots.append(dict(index=len(slots), task=task, seed=seed, arm=arm,
                path=f'episodes/{seed}/{arm}', original_source=original['source'],
                original_source_sha256=original['source_sha256'],
                initial_state_sha256=original['initial_state_sha256']))
    for offset in range(0, len(slots), 4):
        batch = slots[offset:offset + 4]
        for key in {(s['task'], s['seed']) for s in batch}:
            require({s['arm'] for s in batch if (s['task'], s['seed']) == key} == set(study.ARMS),
                    'Dispatch must preserve complete pairs')
    return slots


def prepare(root, scope, preflight):
    require(not root.exists(), 'Repeat root already exists')
    original = study.check(EVIDENCE)
    completion = read(PRIMARY / 'completion_audit.json')
    require(digest(PRIMARY / 'summary.json') == completion['summary_sha256'], 'Primary summary changed')
    records = read(PRIMARY / 'summary.json')['records']
    selected = selected_pairs(records, scope)
    slots = make_slots(selected)
    require(slots, 'Empty selected set')
    report = read(preflight / 'scenes/report.json')
    manifest = preflight / 'runtime_manifest.json'
    require(report['status'] == 'passed' and report['model_requests'] == 0
            and report['reserved_layouts_used'] is False and report['concurrency'] == 4
            and report['runtime_manifest_sha256'] == digest(manifest), 'GPU preflight failed')
    health = gpu_check(manifest)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'bin').mkdir()
    (root / 'bin/codex').symlink_to((EVIDENCE / 'bin/codex').resolve())
    (root / 'sources').symlink_to((EVIDENCE / 'sources').resolve(), target_is_directory=True)
    for task in study.STUDIES:
        folder = root / 'tasks' / task
        folder.mkdir(parents=True)
        (folder / 'sources').symlink_to(root / 'sources', target_is_directory=True)
        artifact = 'skill_prior.py' if task == 'ring' else 'geometry.py'
        shutil.copy2(EVIDENCE / 'tasks' / task / artifact, folder / artifact)
    shutil.copy2(manifest, root / 'runtime_manifest.json')
    shutil.copy2(preflight / 'scenes/report.json', root / 'preflight.json')
    write(root / 'original_selected_records.json', selected)
    files = [PRIMARY / 'summary.json', PRIMARY / 'completion_audit.json', root / 'runtime_manifest.json',
             root / 'preflight.json', root / 'original_selected_records.json']
    files.extend(ROOT / r['source'] for r in selected)
    files.extend(root / 'tasks' / t / ('skill_prior.py' if t == 'ring' else 'geometry.py') for t in study.STUDIES)
    for name in ('run_online_targeted_repeat.py', 'online_ring_camera_lifecycle.py', 'online_gpu_preflight.py'):
        path = ROOT / 'scripts' / name
        dest = root / 'runner_sources' / name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(path, dest)
        files.extend([path, dest])
    reg = dict(name='online_targeted_repeat_v1', registered_at=time.time(), selection_scope=scope,
        authorization='User requested token recovery and rerunning problematic episodes; '
                      'narrow scope repeats both arms of timing/GPU-flagged pairs.',
        slots=slots, episodes=len(slots), paired_layouts=len(slots) // 2, concurrency=4,
        budget_s=200., timing_lag_limit_s=2., repetitions_per_selected_arm=1,
        primary_summary=str(PRIMARY / 'summary.json'), prior_evidence_root=str(EVIDENCE.resolve()),
        original_client_version=original['model_client_version'], package_versions=original['package_versions'],
        gpu_at_registration=health, runtime='Current matching NVIDIA runtime; Ring camera pacing v2 in both arms',
        unchanged='Model/medium reasoning, prompts, geometry fitting, priors, actions, scenes, success checks and 200-second clock',
        analysis='Report original and repeated outcomes side by side for the selected pairs. '
                 'Do not overwrite the primary 50-pair report or select the better attempt. '
                 'These selected repeated layouts are not new independent layouts.',
        token_policy='Unchanged cancellation at success/deadline; unknown terminal usage may recur. '
                     'Recover only agreeing archived CLI/session counters in a separate accounting pass.',
        failure_policy='Stop on infrastructure, pairing or strict audit failures. No automatic second repeat, new seeds, or extra time.',
        frozen_sha256={str(p): digest(p) for p in files})
    write(root / 'registration.json', reg)
    (root / 'registration.sha256').write_text(digest(root / 'registration.json') + '\n')
    write(root / 'progress.json', dict(status='registered', completed=0, audited=0, total=len(slots)))
    check(root)
    return reg


def check(root):
    require(digest(root / 'registration.json') == (root / 'registration.sha256').read_text().strip(), 'Registration changed')
    reg = read(root / 'registration.json')
    study.check(Path(reg['prior_evidence_root']))
    for name, sha in reg['frozen_sha256'].items():
        require(digest(Path(name)) == sha, 'Frozen repeat evidence changed: ' + name)
    require(reg['slots'] == make_slots(selected_pairs(read(Path(reg['primary_summary']))['records'], reg['selection_scope'])),
            'Selected pairs changed')
    require(reg['concurrency'] == 4 and reg['budget_s'] == 200. and reg['timing_lag_limit_s'] == 2., 'Budget changed')
    environment(root / 'runtime_manifest.json')
    return reg


def worker(root, index):
    reg = check(root)
    slot = reg['slots'][index]
    output = study.episode_path(root, slot)
    require(not output.exists(), 'Existing attempt must not be overwritten')
    require(all(os.environ.get(k) == v for k, v in read(root / 'runtime_manifest.json')['environment'].items()),
            'Registered GPU environment missing')
    os.environ['PATH'] = str(root / 'bin') + os.pathsep + os.environ['PATH']
    try:
        with pacing.installed() as counts:
            if slot['task'] == 'tower':
                from run_online_bound_development import _run_episode
                result = _run_episode(output, slot['seed'], slot['arm'], 200.,
                                      kind='explicit_target_reserved', compact_text=True)
            elif slot['task'] == 'ring':
                from run_online_ring_episode import run
                result = run(output, slot['seed'], slot['arm'], root / 'tasks/ring/skill_prior.py',
                             budget=200., kind='reserved')
            else:
                from run_online_relocation_episode import run
                result = run(output, slot['seed'], slot['arm'], kind='relocation_reserved')
    finally:
        if output.exists():
            write(output / 'repeat_provenance.json', dict(registration=str(root / 'registration.json'),
                registration_sha256=digest(root / 'registration.json'), original_slot=slot,
                runtime='ring_camera_pacing_v2', applies_to=slot['task'] == 'ring'))
    require(result['status'] in ('success', 'timeout') and not result.get('archive_error'), 'Episode infrastructure failed')
    require(result['private_initial_state_sha256'] == slot['initial_state_sha256'], 'Initial state differs from original layout')
    check(root)
    print(json.dumps(dict(slot, status=result['status'], success=result['success'])), flush=True)


def summarize(root, rows):
    reg = check(root)
    old = read(root / 'original_selected_records.json')
    comparisons = []
    for row in rows:
        ref = next(r for r in old if (r['task'], r['seed'], r['arm']) == (row['task'], row['seed'], row['arm']))
        row.update(cohort='targeted_repeat', runtime_epoch='matching_system_gpu_ring_pacing_v2',
                   timing_valid=True, gpu_dispatch_query_ok=True,
                   original_source=ref['source'], original_source_sha256=ref['source_sha256'])
        comparisons.append(dict(task=row['task'], seed=row['seed'], arm=row['arm'],
            original_success=ref['success'], repeated_success=row['success'],
            original_capped_seconds=ref['capped_seconds'], repeated_capped_seconds=row['capped_seconds'],
            original_tokens=ref['total_tokens'], repeated_tokens=row['total_tokens'],
            original_usage_complete=ref['usage_complete'], repeated_usage_complete=row['usage_complete']))
    output = dict(status='completed', episodes=len(rows), paired_layouts=reg['paired_layouts'],
                  original_selected=study.summaries(old), repeated=study.summaries(rows),
                  comparisons=comparisons, records=rows, interpretation=reg['analysis'],
                  token_scope='Receipt-only here; unknown interruption usage remains excluded until an independent accounting audit.')
    write(root / 'summary.json', output)
    for name, data in [('episodes.csv', rows), ('original_vs_repeat.csv', comparisons)]:
        with (root / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, list(dict.fromkeys(k for r in data for k in r)), lineterminator='\n')
            writer.writeheader()
            writer.writerows(data)


def execute(root):
    reg = check(root)
    with (root / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study.ensure_unstarted(root, reg['slots'])
        (root / 'workers').mkdir(exist_ok=False)
        (root / 'records').mkdir(exist_ok=False)
        rows, started = [], time.time()
        write(root / 'lifecycle.json', dict(status='running', pid=os.getpid(), started_at=started))

        def progress(status, active=()):
            completed = sum((study.episode_path(root, s) / 'result.json').exists() for s in reg['slots'])
            write(root / 'progress.json', dict(status=status, completed=completed, audited=len(rows),
                total=reg['episodes'], active_slots=list(active), updated_at=time.time(), started_at=started,
                tasks={t: dict(audited=sum(r['task'] == t for r in rows),
                    total=sum(s['task'] == t for s in reg['slots']),
                    successes={a: sum(r['success'] for r in rows if r['task'] == t and r['arm'] == a)
                               for a in ('Base', 'Full')}) for t in study.STUDIES}))

        def launch(slot):
            with (root / 'workers' / f"{slot['index']:03d}.log").open('x') as log:
                return subprocess.call([str(ROOT / '.venv/bin/python'), '-u', str(Path(__file__).resolve()),
                    '--root', str(root), '--phase', 'episode', '--slot', str(slot['index'])],
                    cwd=ROOT, env=environment(root / 'runtime_manifest.json'), stdout=log, stderr=subprocess.STDOUT)

        try:
            for offset in range(0, len(reg['slots']), 4):
                check(root)
                require(not (root / 'STOP.json').exists(), 'External stop requested')
                batch = reg['slots'][offset:offset + 4]
                write(root / f'hardware_{offset:03d}.json', gpu_check(root / 'runtime_manifest.json'))
                progress('running', batch)
                with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                    codes = list(pool.map(launch, batch))
                require(not any(codes), 'Worker failed; no further dispatch')
                audited = study.audit_batch(root, batch, f'batch_{offset:03d}')
                rows.extend(audited)
                write(root / 'records' / f'batch_{offset:03d}.json', audited)
                progress('running')
            final = study.audit_batch(root, reg['slots'], 'final')
            require(len(final) == reg['episodes'], 'Incomplete audit')
            summarize(root, final)
            progress('completed')
            write(root / 'lifecycle.json', dict(status='completed', pid=os.getpid(), finished_at=time.time()))
        except Exception as error:
            write(root / 'STOP.json', dict(error=str(error), time=time.time(), automatic_retry=False))
            progress('stopped')
            write(root / 'lifecycle.json', dict(status='stopped', pid=os.getpid(), error=str(error), time=time.time()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'check', 'run', 'episode'), required=True)
    parser.add_argument('--scope', choices=SCOPES, default='quality')
    parser.add_argument('--preflight', type=Path, default=ROOT / 'outputs/online_repeat_preflight_01')
    parser.add_argument('--slot', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == 'prepare':
        prepare(root, args.scope, args.preflight.resolve())
    elif args.phase == 'check':
        check(root)
        print('Registered repeat sources and GPU runtime match')
    elif args.phase == 'run':
        execute(root)
    else:
        worker(root, args.slot)
