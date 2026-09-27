"""Run paired development episodes sequentially, preserving every attempt."""
import argparse
import json
from pathlib import Path
import time

from run_online_bridge_episode import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=range(2028100101, 2028100104), required=True)
    parser.add_argument('--budget', type=float, default=360.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {'kind': 'development_only', 'seed': args.seed, 'arms': ['basic', 'symbolic'],
            'budget_s': args.budget, 'concurrency': 1, 'started_at': time.time()}
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    results = {}
    for arm in plan['arms']:
        (args.output/'progress.json').write_text(json.dumps({'status': 'running', 'arm': arm,
                                                            'completed': list(results)}, indent=2)+'\n')
        result = run(args.output/arm, args.seed, arm, plan['budget_s'])
        results[arm] = {k: result[k] for k in ('status', 'success', 'capped_completion_seconds', 'decisions')}
        print(arm, results[arm], flush=True)
        if result['status'] == 'infrastructure_error':
            break
    failed = any(value['status'] == 'infrastructure_error' for value in results.values())
    (args.output/'progress.json').write_text(json.dumps({'status': 'stopped' if failed else 'finished', 'results': results}, indent=2)+'\n')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
