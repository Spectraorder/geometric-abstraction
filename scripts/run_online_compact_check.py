"""Run one bounded development pair to compare raw and compact Full text."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from run_online_bridge_pilot import ROOT, client_version, digest, package_versions, source_paths, verify_sources, write
from build_online_bridge_geometry import online_library


def run(root):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for source in source_paths():
        name = source.relative_to(ROOT)
        destination = root / 'sources' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        hashes[str(name)] = digest(source)
    geometry = online_library()
    (root / 'geometry.py').write_text(geometry)
    protocol = dict(name='compact_text_live_development', seed=2028100101, arm='symbolic',
                    budget_s=200., concurrency=2, variants=['raw', 'compact'],
                    registered_at=time.time(), repetitions=1,
                    scope='One existing development layout, no reserved outcomes; no automatic retries.',
                    intervention=None, resets_per_episode=1,
                    routing='Persistent Astra medium; identical explicit-target skills and sensors',
                    contrast='Only reference metadata serialization and its one-time explanation differ',
                    measures=['success', 'capped completion time', 'input/cached/output tokens',
                              'incomplete usage', 'model seconds', 'decisions', 'images', 'sensor costs'],
                    source_sha256=hashes, geometry_sha256=hashlib.sha256(geometry.encode()).hexdigest(),
                    package_versions=package_versions(), model_client_version=client_version())
    write(root / 'protocol.json', protocol)
    (root / 'protocol.sha256').write_text(digest(root / 'protocol.json') + '\n')
    verify_sources(root)
    started = time.time()
    write(root / 'progress.json', dict(status='running', started_at=started, coordinator_pid=os.getpid(), total=2))

    def launch(variant):
        command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/run_online_bound_development.py'),
                   '--output', str(root / variant), '--seed', str(protocol['seed']),
                   '--arm', 'symbolic', '--budget', '200']
        if variant == 'compact':
            command.append('--compact-text')
        with (root / (variant + '.log')).open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='egl'))
            write(root / (variant + '_launch.json'), dict(command=command, pid=process.pid, started_at=time.time()))
            return process.wait()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            codes = list(pool.map(launch, protocol['variants']))
        verify_sources(root)
        if any(codes):
            raise RuntimeError('Worker failed; preserve all attempts without retry')
        results = {variant: json.loads((root / variant / 'result.json').read_text()) for variant in protocol['variants']}
        if len({r['private_initial_state_sha256'] for r in results.values()}) != 1:
            raise ValueError('Initial physical states differ')
        for variant, result in results.items():
            expected = 'reference_records' if variant == 'raw' else 'shared_reference_metadata_v1'
            if result['geometry_serialization'] != expected or result['budget_s'] != 200.:
                raise ValueError('Registered variant mismatch')
        write(root / 'progress.json', dict(status='completed', total=2, completed=2,
                                           wall_seconds=time.time() - started, audit_status='pending'))
        print(json.dumps({variant: {key: result[key] for key in ('status', 'capped_completion_seconds', 'usage')}
                          for variant, result in results.items()}, indent=2), flush=True)
    except Exception as error:
        write(root / 'STOP.json', dict(error=str(error), time=time.time()))
        write(root / 'progress.json', dict(status='stopped', total=2,
                                           completed=sum((root / v / 'result.json').exists() for v in protocol['variants'])))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
