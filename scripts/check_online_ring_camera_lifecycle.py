"""Exercise actual Ring runner success/timeout cleanup without model requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import online_ring_camera_lifecycle as lifecycle
from online_gpu_preflight import environment, gpu_check
from run_online_extension_study import ROOT, digest, read, require, write


def exercise(output, prior, mode):
    """Use existing development scenes; no reserved layouts or agent requests."""
    import run_online_ring_episode as episode
    require(mode in ('success', 'timeout'), 'Unknown lifecycle check')
    if mode == 'success':
        # The existing sensor-based development controller performs the entire
        # task, reaching the real success exception and the actual finally path.
        with lifecycle.installed() as counts:
            result = episode.run(output, 1, 'symbolic', prior, scripted=True,
                                 budget=120., kind='archive_success_development')
        require(result['status'] == 'success', 'Development controller did not reach success cleanup')
    else:
        def expire(api, skills):
            api.camera('side')
            time.sleep(1.05)
            api.camera('side')
        with patch.object(episode, 'scripted_check', expire), lifecycle.installed() as counts:
            result = episode.run(output, 2, 'symbolic', prior, scripted=True,
                                 budget=1., kind='archive_timeout_development')
        require(result['status'] == 'timeout', 'Did not reach timeout cleanup')
    require(not result.get('archive_error'), 'Terminal archive still failed')
    require(result['model_requests_attempted'] == 0, 'Unexpected model request')
    require(counts['terminal_archives'] == 1 and counts['terminal_camera_attempts'] == 1,
            'Actual terminal camera boundary was not exercised')
    require((output / 'private_terminal_state.npz').exists(), 'Terminal physics missing')
    # Acquisitions are recorded before the final archive-only camera frame.
    require(len(list(output.glob('camera_*.npz'))) == result['camera_acquisitions'] + 1,
            'Final archive camera frame missing')
    report = dict(status='passed', mode=mode, model_requests=0, counts=counts,
                  terminal_status=result['status'], source=str(output / 'result.json'),
                  source_sha256=digest(output / 'result.json'))
    write(output / 'lifecycle_check.json', report)
    return report


def run(output, manifest, prior):
    output.mkdir(parents=True, exist_ok=False)
    health = gpu_check(manifest)
    env = environment(manifest)

    def launch(mode):
        path = output / mode
        command = [sys.executable, str(Path(__file__).resolve()), '--phase', mode,
                   '--output', str(path), '--prior', str(prior)]
        with (output / f'{mode}.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=150, check=True)
        return read(path / 'lifecycle_check.json')

    with ThreadPoolExecutor(max_workers=2) as pool:
        checks = list(pool.map(launch, ('success', 'timeout')))
    report = dict(status='passed', time=time.time(), results=checks, gpu=health,
                  runtime_manifest_sha256=digest(manifest), model_requests=0,
                  reserved_layouts_used=False, prior_sha256=digest(prior),
                  pacing_source_sha256=digest(ROOT / 'scripts/online_ring_camera_lifecycle.py'))
    write(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('run', 'success', 'timeout'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--prior', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'run':
        run(args.output, args.manifest, args.prior)
    else:
        exercise(args.output, args.prior, args.phase)
