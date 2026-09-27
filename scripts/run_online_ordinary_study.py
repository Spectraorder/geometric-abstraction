"""Register and run a bounded paired comparison on ordinary bridge-tower scenes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import time

from run_online_bridge_pilot import (
    ROOT, client_version, digest, package_versions, source_paths, verify_sources, write,
)
from build_online_bridge_geometry import online_library
from run_online_reasoning_development import run
from audit_online_reasoning_development import audit

SEEDS = tuple(range(2028103001, 2028103006))
ARMS = ('basic', 'symbolic')


def slots():
    return [dict(seed=seed, arm=arm, path=f'episodes/{seed}/{arm}')
            for index, seed in enumerate(SEEDS)
            for arm in (ARMS if index % 2 == 0 else ARMS[::-1])]


def prepare(root):
    development = ROOT/'outputs/online_reasoning_dev/clearance_preflight_03'
    results = [development/str(seed)/'result.json' for seed in range(2028100101, 2028100104)]
    for path in results:
        result = json.loads(path.read_text())
        if not (result['success'] and result['scripted'] and result['pickup_clearance']
                and result['private_contact_evaluator'] == 'geometry_and_body_ancestry_v1'):
            raise ValueError('Missing ordinary-scene physical development evidence')
    root.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for source in source_paths():
        relative = source.relative_to(ROOT)
        target = root/'sources'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        hashes[str(relative)] = digest(source)
    geometry = online_library()
    (root/'geometry.py').write_text(geometry)
    protocol = dict(
        name='ordinary_online_bridge_tower', registered_at=time.time(),
        seeds=SEEDS, arms=ARMS, slots=slots(), repetitions=1, budget_s=180., concurrency=4,
        scenario='Ordinary unassembled five-object bridge plus two-cube tower',
        intervention=None, one_reset_per_episode=True, physics_during_model_wait=True,
        pickup_clearance=True, planner='astra',
        routing='Astra medium for all model decisions; queued ordinary progress runs locally',
        feedback='Base: RGB and qualitative feedback. Full: initial RGB, then geometric text; '
                 'either arm can request additional images. Same public goal and skills.',
        success='Complete private support graph, five-second stability, at most 3 mm drift; '
                'robot contact exclusion includes unnamed geoms by owning body ancestry',
        clock='Starts immediately after the sole reset, before workspace initialization; '
              'includes sensing, model requests, actions and settling, with 1:1 simulation pacing',
        max_timing_lag_for_realtime_claim_s=2.,
        parallelism='Balanced batches of two pairs, then one pair; one BLAS/OpenMP thread per worker',
        failure_policy='Retain all slots and failures; no replacement seeds, retries or budget extension; '
                       'infrastructure failure stops new batches',
        selection='Fixed fresh seeds without performance screening. Ordinary means no injected '
                  'disruption, not guaranteed easy or successful construction.',
        primary_metrics=['successes / 5', 'mean capped completion seconds including failures'],
        secondary_metrics=['paired time differences', 'completed request token totals',
                           'incomplete requests with unknown usage', 'attached images',
                           'first request latency', 'total completed request latency',
                           'camera acquisitions', 'action seconds', 'campaign wall seconds'],
        interpretation='Exploratory five-pair study; no assumed positive outcome. Shared clearance '
                       'improvements do not themselves establish a geometric-text advantage.',
        source_sha256=hashes, geometry_sha256=hashlib.sha256(geometry.encode()).hexdigest(),
        package_versions=package_versions(), model_client_version=client_version(),
        development_results={str(path.relative_to(ROOT)):digest(path) for path in results},
    )
    write(root/'protocol.json', protocol)
    (root/'protocol.sha256').write_text(digest(root/'protocol.json')+'\n')
    write(root/'progress.json', dict(status='registered', completed=0, total=10))


def summarize(root):
    protocol = verify_sources(root)
    audit(root, [slot['path'] for slot in protocol['slots']])
    records = []
    violations = []
    for slot in protocol['slots']:
        directory = root/slot['path']
        result = json.loads((directory/'result.json').read_text())
        if (result['seed'], result['arm']) != (slot['seed'], slot['arm']):
            raise ValueError('Mismatched slot identity')
        if (result['scripted'] or not result['pickup_clearance'] or result['planner'] != 'astra'
                or result['budget_s'] != protocol['budget_s'] or result['intervention'] is not None
                or result['private_contact_evaluator'] != 'geometry_and_body_ancestry_v1'):
            raise ValueError('Episode settings differ from registration')
        receipts = [json.loads(p.read_text()) for p in sorted(directory.glob('context_*/decision_*/receipt.json'))]
        feedback = [json.loads(line) for line in (directory/'feedback.jsonl').read_text().splitlines()] if (directory/'feedback.jsonl').exists() else []
        if slot['arm'] == 'basic' and any('geometry' in row['packet'] for row in feedback):
            raise ValueError('Geometry exposed to Base')
        if result['timing']['max_simulation_lag_seconds'] > protocol['max_timing_lag_for_realtime_claim_s']:
            violations.append(slot['path']+': excessive timing lag')
        records.append(dict(slot, **{key:result[key] for key in (
            'success', 'status', 'capped_completion_seconds', 'usage', 'incomplete_requests',
            'model_requests_completed', 'model_seconds', 'camera_acquisitions', 'action_seconds',
            'private_initial_state_sha256')},
            images=sum(len(row['images']) for row in feedback),
            first_request_seconds=receipts[0]['wall_seconds'] if receipts else None))
    pairs = []
    for seed in protocol['seeds']:
        base, full = [next(row for row in records if row['seed']==seed and row['arm']==arm) for arm in ARMS]
        if base['private_initial_state_sha256'] != full['private_initial_state_sha256']:
            raise ValueError('Paired initial physics differs')
        pairs.append(dict(seed=seed, base_success=base['success'], full_success=full['success'],
            full_minus_base_capped_seconds=full['capped_completion_seconds']-base['capped_completion_seconds']))
    arms = {}
    for arm in ARMS:
        selected = [row for row in records if row['arm']==arm]
        latencies = [row['first_request_seconds'] for row in selected if row['first_request_seconds'] is not None]
        arms[arm] = dict(successes=sum(row['success'] for row in selected), episodes=len(selected),
            mean_capped_seconds=mean(row['capped_completion_seconds'] for row in selected),
            usage={key:sum(row['usage'][key] for row in selected) for key in selected[0]['usage']},
            incomplete_requests=sum(row['incomplete_requests'] for row in selected),
            usage_complete=all(row['incomplete_requests']==0 for row in selected),
            completed_requests=sum(row['model_requests_completed'] for row in selected),
            images=sum(row['images'] for row in selected),
            model_seconds=sum(row['model_seconds'] for row in selected),
            mean_first_request_seconds=mean(latencies) if latencies else None)
    report = dict(arms=arms, pairs=pairs, records=records, timing_violations=violations,
                  interpretation='Five exploratory pairs. Incomplete request usage is unknown, not zero. '
                                 'Report all failures; do not claim a real-time advantage if pacing is invalid.')
    write(root/'summary.json', report)
    return report


def worker(root, index):
    protocol = verify_sources(root)
    slot = protocol['slots'][index]
    result = run(root/slot['path'], slot['seed'], slot['arm'], protocol['budget_s'],
                 clearance=True, planner='astra')
    verify_sources(root)
    if result['status'] == 'infrastructure_error':
        raise RuntimeError('Infrastructure failure; preserve the attempt and stop new batches')
    print(json.dumps(dict(slot, status=result['status'], success=result['success'])), flush=True)


def execute(root):
    import fcntl
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = verify_sources(root)
        if (root/'episodes').exists() or (root/'STOP.json').exists():
            raise FileExistsError('Previous attempts exist; no automatic resume or retry')
        start = time.time()
        completed = 0
        def launch(index):
            environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='egl')
            with (root/f'worker_{index:02d}.log').open('w') as log:
                return subprocess.call([str(ROOT/'.venv/bin/python'), str(Path(__file__).resolve()),
                    '--root', str(root), '--phase', 'episode', '--slot', str(index)],
                    cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            for offset in range(0, len(protocol['slots']), protocol['concurrency']):
                verify_sources(root)
                indices = list(range(offset, min(offset+protocol['concurrency'], len(protocol['slots']))))
                write(root/'progress.json', dict(status='running', completed=completed, total=10,
                                                 active_slots=indices, started_at=start))
                with ThreadPoolExecutor(max_workers=len(indices)) as pool:
                    codes = list(pool.map(launch, indices))
                completed = sum((root/slot['path']/'result.json').exists() for slot in protocol['slots'])
                if any(codes):
                    raise RuntimeError('Worker failed; no further batch dispatched')
            summarize(root)
            write(root/'progress.json', dict(status='completed', completed=completed, total=10,
                                             wall_seconds=time.time()-start))
        except Exception as error:
            write(root/'STOP.json', dict(error=str(error), completed=completed, time=time.time()))
            write(root/'progress.json', dict(status='stopped', completed=completed, total=10))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'check', 'run', 'episode', 'summarize'), required=True)
    parser.add_argument('--slot', type=int, choices=range(10))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == 'prepare': prepare(root)
    elif args.phase == 'check': verify_sources(root); print('Registered sources match')
    elif args.phase == 'run': execute(root)
    elif args.phase == 'summarize': summarize(root)
    else:
        if args.slot is None: parser.error('--slot is required for episode')
        worker(root, args.slot)
