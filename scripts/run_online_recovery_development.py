"""Run one controlled recovery development episode without resetting its scene."""
import argparse
import json
from pathlib import Path
from online_recovery_fixture import RecoveryFixture, KINDS
from run_online_event_development import run

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=range(2028100101, 2028100104), default=2028100101)
    p.add_argument('--kind', choices=KINDS, required=True)
    p.add_argument('--shift', type=float, default=.025)
    p.add_argument('--arm', choices=('basic', 'symbolic'), default='symbolic')
    p.add_argument('--budget', type=float, default=240.)
    p.add_argument('--scripted', action='store_true')
    a = p.parse_args()
    result = run(a.output, a.seed, a.arm, a.budget, a.scripted,
                 preparation=RecoveryFixture(a.kind, a.shift))
    print(json.dumps(result, indent=2))
    if result['status'] == 'infrastructure_error':
        raise SystemExit(1)
