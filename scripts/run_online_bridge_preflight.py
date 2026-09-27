"""Development-only fixed skill sequence; no model or reserved evaluation."""
import argparse
import json
from pathlib import Path
import time
import traceback

from online_bridge_scene import OnlineBridgeTowerEnv, EpisodeSuccess
from online_bridge_skills import SensorAPI, SharedSkills, plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2028100101)
    parser.add_argument('--realtime', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = OnlineBridgeTowerEnv(args.output, scene_seed=args.seed)
    result = {'kind': 'development_fixed_sequence', 'seed': args.seed,
              'realtime': args.realtime, 'success': False}
    try:
        env.reset(args.seed)
        env.start_episode(300, realtime=args.realtime)
        api = SensorAPI(env, args.output)
        api.image('initial')
        started = time.monotonic()
        skills = SharedSkills(api)
        result['initialization_seconds'] = time.monotonic()-started
        print('INITIALIZED', result['initialization_seconds'], flush=True)
        (args.output/'initial_geometry.json').write_text(json.dumps(skills.packet(), indent=2))
        (args.output/'provenance.json').write_text(json.dumps(skills.provenance, indent=2))
        for key in skills.order:
            kind = 'stack' if len(skills.nodes[key]['supports']) == 1 else 'bridge'
            outcome = skills.execute(kind, key)
            print(key, outcome, flush=True)
            api.image('after_'+key)
            if outcome['status'] != 'visually_consistent':
                break
        env.wait(5.5)
        result['success'] = env.success_time is not None
        result['private_status'] = env.structure_status(list(env.order))
    except EpisodeSuccess:
        result['success'] = True
        result['private_status'] = env.structure_status(list(env.order))
    except Exception as error:
        result['error'] = type(error).__name__+': '+str(error)
        (args.output/'error.txt').write_text(traceback.format_exc())
    finally:
        if env.episode_started is not None:
            result['timing'] = env.timing()
        env.close()
        (args.output/'result.json').write_text(json.dumps(plain(result), indent=2)+'\n')
        print(json.dumps(plain(result)), flush=True)


if __name__ == '__main__':
    main()
