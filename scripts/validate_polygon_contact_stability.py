"""Compare no-action contact stability under explicit collision backends.

This development diagnostic uses the same physical fixtures and unchanged scoring
as validate_polygon_prism_physics. It makes no policy or model calls.
"""
import argparse
import hashlib
from pathlib import Path
import shutil

import numpy as np

from validate_polygon_prism_physics import (
    COLLISION_PIPELINES, PhysicsEnv, ROOT, fixture, mujoco, physics_settings, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026092701, 2026092703])
    parser.add_argument('--pipelines', nargs='+', choices=COLLISION_PIPELINES,
                        default=['scene', 'libccd'])
    parser.add_argument('--settle-seconds', type=float, default=36.)
    args = parser.parse_args()
    if not np.isfinite(args.settle_seconds) or not 0 <= args.settle_seconds <= 120:
        parser.error('settle-seconds must be finite and within 0..120')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.pipelines)) != len(args.pipelines):
        parser.error('Duplicate seeds or pipelines')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT/'scripts/validate_polygon_prism_physics.py', ROOT/'view_scene.py']
    write_json(args.output/'protocol.json', {
        'purpose': 'Paired no-action collision-backend control, not policy evaluation',
        'seeds': args.seeds, 'pipelines': args.pipelines, 'settle_seconds': args.settle_seconds,
        'initial_settle_seconds': .65, 'evaluation_settle_seconds': .8, 'evaluation_window_seconds': 2.,
        'mujoco_version': mujoco.__version__, 'policy_rollouts': 0, 'model_calls': 0,
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'scoring': 'Unmodified PhysicsEnv.evaluate_placement; only static stability is relevant here',
    })
    for path in sources[:2]:
        shutil.copy(path, args.output/path.name)
    rows = []
    for seed in args.seeds:
        case = fixture(seed)
        for pipeline in args.pipelines:
            output = args.output/(str(seed)+'_'+pipeline)
            output.mkdir()
            write_json(output/'private_fixture.json', case)
            env = PhysicsEnv(output, case, 640, 480, pipeline)
            try:
                mujoco.mj_saveModel(env.model, str(output/'model.mjb'))
                env._step_for(args.settle_seconds)
                evaluation = env.evaluate_placement()
                row = {'seed': seed, 'requested_pipeline': pipeline, 'physics': physics_settings(env.model),
                       'stable': evaluation['checks']['stable_for_2s'],
                       'table_supported': evaluation['checks']['table_supported'],
                       'released_from_robot': evaluation['checks']['released_from_robot'],
                       'stability': evaluation['stability']}
                write_json(output/'result.json', row)
                write_json(output/'private_states.json', env.private_samples[-evaluation['stability']['sample_count']:])
                rows.append(row)
                write_json(args.output/'summary.json', {'status': 'running', 'results': rows})
                print(seed, pipeline, row['stable'], row['stability'], flush=True)
            finally:
                env.close()
    write_json(args.output/'summary.json', {'status': 'complete', 'results': rows})


if __name__ == '__main__':
    main()
