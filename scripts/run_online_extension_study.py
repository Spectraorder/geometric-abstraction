"""Extend three completed online studies to 50 pairs without rewriting the pilots.

Freeze 45 new pairs per task, run at most four episodes globally, and audit each
batch before dispatching the next. Episode implementations and prompts are reused
unchanged. The original five pairs are evidence, never execution slots.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from run_online_bridge_pilot import ROOT, digest, package_versions, source_paths, write

STUDIES = {
    'tower': 'online_compact_tower_200_01',
    'ring': 'online_ring_200_01',
    'relocation': 'online_relocation_200_01',
}
SEEDS = {task: tuple(range(2035101001 + i * 1000, 2035101046 + i * 1000))
         for i, task in enumerate(STUDIES)}
ARMS = ('basic', 'symbolic')
AUDIT_FILES = ('audit.json', 'interface_audit.json', 'compact_prompt_audit.json',
               'relocation_audit.json')
PRESENTATION_ONLY = {'scripts/render_agentic_geometry_method.py'}


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def schedule():
    slots = []
    for pair_index in range(45):
        for task in STUDIES:
            seed = SEEDS[task][pair_index]
            for arm in (ARMS if pair_index % 2 == 0 else ARMS[::-1]):
                slots.append(dict(index=len(slots), task=task, seed=seed, arm=arm,
                                  path=f'episodes/{seed}/{arm}'))
    return slots


def validate_schedule(slots):
    require(slots == schedule(), 'Schedule differs from the authorized 270 new slots')
    for offset in range(0, len(slots), 4):
        batch = slots[offset:offset + 4]
        for task, seed in {(s['task'], s['seed']) for s in batch}:
            pair = [s for s in batch if (s['task'], s['seed']) == (task, seed)]
            require(len(pair) == 2 and {s['arm'] for s in pair} == set(ARMS),
                    'Every dispatch batch must contain complete pairs')


def seed_overlap():
    """Check prior registrations/manifests, pruning large per-episode archives."""
    proposed = {s for seeds in SEEDS.values() for s in seeds}
    checked, conflicts = {}, []

    def numbers(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from numbers(child)
        elif isinstance(value, list):
            for child in value:
                yield from numbers(child)
        elif type(value) is int:
            yield value

    excluded = {'sources', 'episodes', 'rollouts', 'actor_home', 'actor_work',
                'training', 'evaluation', '__pycache__', 'node_modules'}
    for directory, dirs, files in os.walk(ROOT / 'outputs', followlinks=True):
        depth = len(Path(directory).relative_to(ROOT / 'outputs').parts)
        dirs[:] = [d for d in dirs if d not in excluded and not d.startswith('context_')]
        if depth >= 5:
            dirs[:] = []
        for name in ('protocol.json', 'registration.json', 'manifest.json'):
            if name not in files:
                continue
            path = Path(directory) / name
            values = set(numbers(read(path)))
            checked[str(path.relative_to(ROOT))] = digest(path)
            if overlap := proposed & values:
                conflicts.append(dict(path=str(path), seeds=sorted(overlap)))
    require(not conflicts, 'Proposed seeds already registered: ' + repr(conflicts))
    return dict(status='passed', metadata_sha256=checked,
                scope='Integer seeds in prior protocol/registration/manifest files, depth <= 5; '
                      'per-episode and source archives excluded')


def prepare(root):
    require(not root.exists(), 'Extension root already exists')
    from export_online_results import collect, SOURCES
    from evaluate_geometry_learning_curves import hardware
    legacy_rows, _, _, _, _, _ = collect()
    require(len(legacy_rows) == 30, 'Expected 30 completed pilot episodes')
    parity = {}
    for task, study in STUDIES.items():
        original = read(ROOT / 'outputs' / study / 'protocol.json')
        changed = [name for name, sha in original['source_sha256'].items()
                   if not (ROOT / name).exists() or digest(ROOT / name) != sha]
        require(set(changed) <= PRESENTATION_ONLY, 'Frozen runtime changed: ' + repr(changed))
        require(package_versions() == original['package_versions'], 'Simulation packages changed')
        parity[task] = dict(original_client=original['model_client_version'],
                            changed_presentation_files=changed, runtime_matches=True)
    overlap = seed_overlap()
    slots = schedule()
    validate_schedule(slots)
    client = Path(shutil.which('codex'))
    version = subprocess.check_output([str(client), '--version'], text=True).strip()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'bin').mkdir()
    shutil.copy2(client, root / 'bin/codex')
    sources = {}
    for path in source_paths():
        relative = path.relative_to(ROOT)
        target = root / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[str(relative)] = digest(path)
    artifacts = {}
    for task, study in STUDIES.items():
        directory = root / 'tasks' / task
        directory.mkdir(parents=True)
        (directory / 'sources').symlink_to('../../sources', target_is_directory=True)
        name = 'skill_prior.py' if task == 'ring' else 'geometry.py'
        shutil.copy2(ROOT / 'outputs' / study / name, directory / name)
        artifacts[str((directory / name).relative_to(root))] = digest(directory / name)
    write(root / 'legacy_episodes.json', legacy_rows)
    artifacts['legacy_episodes.json'] = digest(root / 'legacy_episodes.json')
    write(root / 'seed_overlap.json', overlap)
    artifacts['seed_overlap.json'] = digest(root / 'seed_overlap.json')
    protocol = dict(
        name='online_three_task_extension_50_v1', registered_at=time.time(),
        authorization='50 pairs/task including the existing five; exactly 270 new episodes',
        legacy_studies=STUDIES, legacy_pairs_per_task=5, new_pairs_per_task=45,
        seeds=SEEDS, slots=slots, new_episodes=270, pooled_episodes=300,
        repetitions_per_new_layout_and_arm=1, budget_s=200., concurrency=4,
        model='gpt-6-astra', reasoning='medium',
        runtime='Unmodified original episode functions, skills, prompts, feedback and evaluators',
        transport={'tower': 'Original default provider', 'ring': 'HTTP-only', 'relocation': 'HTTP-only'},
        resets_per_episode=1, physics_during_model_wait=True, intervention=None,
        clock='Original 200-second clock including initialization after reset, sensing, model waits, '
              'execution and settling; shared ring grasp/lift included',
        timing_lag_limit_s=2.,
        parallelism='Global four workers; complete Base/Full pairs; tasks interleaved; '
                    'pair order alternates; one OpenMP/BLAS thread per worker',
        failure_policy='Stop further dispatch on infrastructure, source, audit or timing failure; '
                       'preserve every attempt; no automatic retry, resume, replacement or budget growth',
        analysis='Report original 5, new 45 and pooled 50 pairs separately. All timeouts count as '
                 '200 seconds. Paired-layout bootstrap 95% intervals (10000 resamples, seed 1050) '
                 'for Full-minus-Base success and capped time; success/time curves and all cost records. '
                 'Unknown interrupted tokens remain unknown; no test-based selection.',
        interpretation='Post-hoc sample extension after pilot inspection. Tests more layouts, not '
                       'repeated stochastic runs on each layout or separate geometry/representation ablations. '
                       'Client version and service-date differences remain explicit cohort differences.',
        parity=parity, model_client_version=version, client_original_path=str(client),
        client_sha256=digest(root / 'bin/codex'), package_versions=package_versions(),
        source_sha256=sources, artifact_sha256=artifacts,
        legacy_evidence_sha256=dict(SOURCES), hardware_at_registration=hardware(),
    )
    write(root / 'protocol.json', protocol)
    (root / 'protocol.sha256').write_text(digest(root / 'protocol.json') + '\n')
    write(root / 'progress.json', dict(status='registered', completed=0, total=270,
                                      legacy_completed=30, pooled_total=300))
    return protocol


def check(root):
    require(digest(root / 'protocol.json') == (root / 'protocol.sha256').read_text().strip(),
            'Protocol hash changed')
    protocol = read(root / 'protocol.json')
    validate_schedule(protocol['slots'])
    require(protocol['concurrency'] == 4 and protocol['budget_s'] == 200., 'Budget changed')
    require(package_versions() == protocol['package_versions'], 'Package version changed')
    require(digest(root / 'bin/codex') == protocol['client_sha256'], 'Pinned client changed')
    for name, sha in protocol['source_sha256'].items():
        require(digest(ROOT / name) == sha == digest(root / 'sources' / name),
                'Source changed: ' + name)
    for name, sha in protocol['artifact_sha256'].items():
        require(digest(root / name) == sha, 'Frozen artifact changed: ' + name)
    for name, sha in protocol['legacy_evidence_sha256'].items():
        require(digest(ROOT / name) == sha, 'Original evidence changed: ' + name)
    return protocol


def episode_path(root, slot):
    return root / 'tasks' / slot['task'] / slot['path']


def worker(root, index):
    protocol = check(root)
    slot = protocol['slots'][index]
    output = episode_path(root, slot)
    require(not output.exists(), 'Existing attempt cannot be overwritten')
    os.environ['PATH'] = str(root / 'bin') + os.pathsep + os.environ['PATH']
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
    check(root)
    require(result['status'] in ('success', 'timeout') and not result.get('archive_error'),
            'Episode infrastructure failed; preserve artifacts and stop')
    print(json.dumps(dict(slot, status=result['status'], success=result['success'])), flush=True)


def audit_batch(root, slots, batch_id):
    from run_online_ring_checked_study import installed
    records = []
    archive = root / 'audits' / batch_id
    archive.mkdir(parents=True, exist_ok=False)
    for task in STUDIES:
        selected = [s for s in slots if s['task'] == task]
        if not selected:
            continue
        directory = root / 'tasks' / task
        with installed():
            if task == 'tower':
                from audit_online_bound_development import audit
                from audit_online_compact_prompts import audit_prompts
                report = audit(directory, [s['path'] for s in selected])
                audit_prompts(directory, selected)
            elif task == 'ring':
                from online_ring_evidence import audit
                report = audit(directory, [s['path'] for s in selected])
            else:
                from audit_online_relocation import audit
                report = audit(directory, selected)
        require(report['status'] == 'passed', 'Episode audit failed')
        for name in AUDIT_FILES:
            if (directory / name).exists():
                target = archive / task / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(directory / name, target)
        for row in report['episodes']:
            episode = directory / row['episode']
            result = read(episode / 'result.json')
            require(result['status'] in ('success', 'timeout') and not result.get('archive_error'),
                    'Invalid episode status')
            require(result['budget_s'] == 200. and not result['scripted'], 'Unexpected episode settings')
            require(row['max_simulation_lag_seconds'] <= 2., 'Real-time pacing lag exceeds two seconds')
            require(result['timing']['reset_count'] == 1, 'Unexpected reset')
            failures = [read(p) for p in episode.glob('context_*/decision_*/attempt_failure.json')]
            incomplete_seconds = sum(f['wall_seconds'] for f in failures)
            records.append(dict(
                task=task, study=root.name, cohort='new_45', seed=row['seed'],
                arm={'basic': 'Base', 'symbolic': 'Full'}[row['arm']],
                status=result['status'], success=int(row['success']), budget_seconds=200.,
                capped_seconds=row['capped_seconds'],
                success_seconds=row['capped_seconds'] if row['success'] else None,
                completed_model_seconds=row['model_seconds'], incomplete_model_seconds=incomplete_seconds,
                all_model_seconds=row['model_seconds'] + incomplete_seconds,
                completed_requests=row['completed_requests'], incomplete_requests=row['incomplete_requests'],
                usage_complete=not failures, **row['usage'], images=row['images'],
                camera_acquisitions=result['camera_acquisitions'], depth_queries=result['depth_queries'],
                action_seconds=result['action_seconds'], feedback_seconds=result['feedback_seconds'],
                initial_state_sha256=row['initial_state_sha256'],
                max_simulation_lag_seconds=row['max_simulation_lag_seconds'],
                source=str(episode / 'result.json'), source_sha256=digest(episode / 'result.json')))
    require(len(records) == len(slots), 'Audit record count mismatch')
    paired(records)
    write(archive / 'records.json', records)
    return records


def paired(rows):
    pairs = []
    for task, seed in sorted({(r['task'], r['seed']) for r in rows}):
        selected = [r for r in rows if (r['task'], r['seed']) == (task, seed)]
        require(len(selected) == 2 and {r['arm'] for r in selected} == {'Base', 'Full'},
                'Missing or duplicate pair')
        base, full = [next(r for r in selected if r['arm'] == arm) for arm in ('Base', 'Full')]
        require(base['initial_state_sha256'] == full['initial_state_sha256'], 'Initial states differ')
        pairs.append(dict(task=task, seed=seed, base_success=base['success'], full_success=full['success'],
                          full_minus_base_sr=full['success'] - base['success'],
                          full_minus_base_seconds=full['capped_seconds'] - base['capped_seconds']))
    return pairs


def summaries(rows):
    import numpy as np
    arms, differences = [], []
    for task in STUDIES:
        subset = [r for r in rows if r['task'] == task]
        pairs = paired(subset)
        for arm in ('Base', 'Full'):
            selected = [r for r in subset if r['arm'] == arm]
            require(selected, 'Missing task/arm')
            record = dict(task=task, arm=arm, episodes=len(selected),
                          successes=sum(r['success'] for r in selected),
                          usage_complete=all(r['usage_complete'] for r in selected))
            for key in ('success', 'capped_seconds', 'total_tokens', 'all_model_seconds',
                        'completed_requests', 'images', 'camera_acquisitions', 'action_seconds'):
                record['mean_' + key] = statistics.mean(r[key] for r in selected)
            arms.append(record)
        rng = np.random.default_rng(1050)
        indices = rng.integers(0, len(pairs), size=(10000, len(pairs)))
        for metric in ('full_minus_base_sr', 'full_minus_base_seconds'):
            values = np.asarray([p[metric] for p in pairs])
            low, high = np.quantile(values[indices].mean(axis=1), [.025, .975])
            differences.append(dict(task=task, pairs=len(pairs), metric=metric,
                                    mean=float(values.mean()), ci95=[float(low), float(high)]))
    return dict(arms=arms, paired_differences=differences)


def summarize(root):
    protocol = check(root)
    rows = audit_batch(root, protocol['slots'], 'final')
    legacy = [dict(r, cohort='original_5') for r in read(root / 'legacy_episodes.json')]
    require(len(rows) == 270 and len(legacy) == 30, 'Incomplete extension')
    combined = legacy + rows
    output = dict(status='completed', original_5=summaries(legacy), new_45=summaries(rows),
                  pooled_50=summaries(combined), records=combined,
                  interpretation=protocol['interpretation'])
    write(root / 'summary.json', output)
    fields = list(dict.fromkeys(k for r in combined for k in r))
    with (root / 'episodes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(combined)
    return output


def ensure_unstarted(root, slots):
    require(not (root / 'STOP.json').exists(), 'Stopped campaign requires an explicit amendment')
    require(not any(episode_path(root, s).exists() for s in slots),
            'Existing attempts cannot be resumed or overwritten automatically')


def execute(root):
    import fcntl
    from evaluate_geometry_learning_curves import hardware
    with (root / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = check(root)
        ensure_unstarted(root, protocol['slots'])
        started = time.time()
        records = []
        (root / 'workers').mkdir(exist_ok=False)
        write(root / 'lifecycle.json', dict(status='running', pid=os.getpid(), started_at=started))

        def launch(slot):
            environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='egl',
                               PATH=str(root / 'bin') + os.pathsep + os.environ['PATH'])
            with (root / 'workers' / f"{slot['index']:03d}.log").open('x') as log:
                return subprocess.call([str(ROOT / '.venv/bin/python'), '-u', str(Path(__file__)),
                                        '--root', str(root), '--phase', 'episode',
                                        '--slot', str(slot['index'])], cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)

        def progress(status, active=()):
            completed = sum((episode_path(root, s) / 'result.json').exists() for s in protocol['slots'])
            write(root / 'progress.json', dict(status=status, completed=completed, audited=len(records),
                total=270, legacy_completed=30, pooled_completed=30 + completed, pooled_total=300,
                active_slots=list(active), started_at=started, updated_at=time.time(),
                wall_seconds=time.time() - started,
                tasks={task: dict(completed=sum(r['task'] == task for r in records), total=90,
                    successes={arm: sum(r['success'] for r in records if r['task'] == task and r['arm'] == arm)
                               for arm in ('Base', 'Full')}) for task in STUDIES}))

        try:
            for offset in range(0, len(protocol['slots']), 4):
                check(root)
                require(not (root / 'STOP.json').exists(), 'External STOP receipt found')
                batch = protocol['slots'][offset:offset + 4]
                write(root / f'hardware_{offset:03d}.json', hardware())
                progress('running', batch)
                with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                    codes = list(pool.map(launch, batch))
                require(not any(codes), 'Worker failed; no further batch dispatched')
                records.extend(audit_batch(root, batch, f'batch_{offset:03d}'))
                progress('running')
            summarize(root)
            progress('completed')
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
    parser.add_argument('--phase', choices=('prepare', 'check', 'run', 'episode', 'summarize'), required=True)
    parser.add_argument('--slot', type=int, choices=range(270))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == 'prepare':
        prepare(root)
    elif args.phase == 'check':
        check(root)
        print('Frozen extension sources, client and pilot evidence match')
    elif args.phase == 'run':
        execute(root)
    elif args.phase == 'summarize':
        summarize(root)
    else:
        if args.slot is None:
            parser.error('--slot is required for episode')
        worker(root, args.slot)
