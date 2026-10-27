"""Development-only physical grasp and delayed-sensing checks; no LLM requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import time

from online_gpu_preflight import environment, gpu_check
from run_online_extension_study import ROOT, digest, read, require, write


def ring_probe(output, seed, prior, *, corrected):
    import online_ring_camera_pacing
    from ring_peg_development import installed
    from online_ring_scene import OnlineRingEnv
    from online_ring_loop import OnlineRingSensors
    from online_ring_skills import RingSkills
    require(seed in (1, 2), 'Only existing development seeds allowed')
    output.mkdir(parents=True, exist_ok=False)
    env = None
    try:
        with installed(), online_ring_camera_pacing.installed() if corrected else nullcontext():
            env = OnlineRingEnv(output / 'env', scene_seed=seed)
            env.reset(seed)
            env.start_episode(120., realtime=True)
            api = OnlineRingSensors(env, output)
            skills = RingSkills(api, prior)
            outcome = skills.grasp_lift()
            env.advance_while_waiting()
            grasp_timing = env.timing()
            # Replay the shape of a failed-observation retry sequence. Each
            # synthetic CPU interval leaves time charged to the same deadline.
            for _ in range(3):
                for view in ('side', 'wrist'):
                    api.camera(view)
                    time.sleep(.6)
                env.wait(.12)
            env.advance_while_waiting()
            timing = env.timing()
            result = dict(status='passed' if timing['max_simulation_lag_seconds'] <= 2. else 'lag_detected',
                corrected=corrected, seed=seed, model_requests=0, grasp_outcome=outcome,
                grasp_timing=grasp_timing, timing=timing,
                scope='Development grasp prefix and synthetic delayed-sensing stress, not evaluation SR')
            write(output / 'result.json', result)
            if corrected:
                require(result['status'] == 'passed', 'Corrected pacing still exceeds original lag limit')
            return result
    finally:
        if env is not None:
            env.close()


def run(output, manifest, prior):
    output.mkdir(parents=True, exist_ok=False)
    health = gpu_check(manifest)
    env = environment(manifest)
    probes = [('ring', 1), ('ring', 2), ('tower', 2028100101), ('relocation', 1)]

    def launch(item):
        i, (task, seed) = item
        path = output / f'{i}_{task}'
        if task == 'ring':
            command = [sys.executable, str(Path(__file__).resolve()), '--phase', 'ring',
                       '--output', str(path), '--seed', str(seed), '--prior', str(prior), '--corrected']
        else:
            command = [sys.executable, str(ROOT / 'scripts/online_gpu_preflight.py'), '--phase', 'scene',
                       '--output', str(path), '--task', task, '--seed', str(seed)]
        with (output / f'{i}_{task}.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=150, check=True)
        return read(path / 'result.json')

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(launch, enumerate(probes)))
    report = dict(status='passed', registered_at=time.time(), concurrency=4, gpu=health,
                  runtime_manifest_sha256=digest(manifest), results=records, model_requests=0,
                  reserved_layouts_used=False, prior_sha256=digest(prior),
                  pacing_source_sha256=digest(ROOT / 'scripts/online_ring_camera_pacing.py'))
    write(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('ring', 'run'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--corrected', action='store_true')
    args = parser.parse_args()
    if args.phase == 'run':
        run(args.output, args.manifest, args.prior)
    else:
        print(json.dumps(ring_probe(args.output, args.seed, args.prior, corrected=args.corrected)))
