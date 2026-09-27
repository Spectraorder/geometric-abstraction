"""Run paired fresh baseline/pointing trials; the session services RGB mailboxes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads((args.root / 'manifest.json').read_text())
    (args.root / 'logs').mkdir(exist_ok=True)
    (args.root / 'mailboxes').mkdir(exist_ok=True)
    environment = dict(os.environ, MUJOCO_GL='egl', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

    def pair(case):
        records = []
        for mode, candidate in [('baseline', 'policies/assemble_structure_cpu.py'),
                                ('pointing', 'policies/assemble_structure_pointing.py')]:
            destination = args.root / mode / case['case_id']
            command = [sys.executable, 'scripts/run_vision_trial.py', '--candidate', candidate,
                       '--output', str(destination), '--seed', str(case['seed']),
                       '--layout', 'sampled', '--task', 'inventory_structure',
                       '--structure-recipe', 'thin_beam_bridge', '--object-count', str(case['object_count']),
                       '--structure-color-map', json.dumps(case['color_map']), '--timeout', '180']
            if mode == 'pointing':
                command += ['--pointing-mailbox', str(args.root / 'mailboxes' / case['case_id']),
                            '--pointing-timeout', '360']
            start = time.monotonic()
            with (args.root / 'logs' / (mode + '_' + case['case_id'] + '.log')).open('w') as log:
                run = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            record = {'case_id': case['case_id'], 'mode': mode, 'command': command,
                      'exit_code': run.returncode, 'wall_seconds': time.monotonic() - start}
            result_path = destination / 'result.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                record.update(success=result['success'], policy_error=result['policy_error'],
                              pointing_requests=result.get('pointing', {}).get('request_count', 0))
            records.append(record)
            (args.root / 'logs' / (mode + '_' + case['case_id'] + '_run.json')).write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps(record), flush=True)
        return records

    all_records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(pair, case) for case in manifest['cases']]
        for future in as_completed(futures):
            all_records.extend(future.result())
    (args.root / 'runs.json').write_text(json.dumps(all_records, indent=2) + '\n')


if __name__ == '__main__':
    main()
