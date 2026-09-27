"""Accelerated development diagnostic: clear after a physical failure, without reset."""
import argparse
import json
from pathlib import Path
import traceback

from online_bridge_scene import OnlineBridgeTowerEnv, EpisodeSuccess
from online_bridge_skills import SensorAPI, SharedSkills, plain
from online_bridge_recovery import clear_object


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seed = 2028100102
    env = OnlineBridgeTowerEnv(args.output, scene_seed=seed)
    record = {'kind': 'development_clearing_diagnostic', 'seed': seed, 'realtime': False}
    try:
        env.reset(seed)
        env.start_episode(180, realtime=False)
        api = SensorAPI(env, args.output)
        skills = SharedSkills(api)
        record['sequence'] = []
        for key in skills.order:
            kind = 'stack' if len(skills.nodes[key]['supports']) == 1 else 'bridge'
            record['sequence'].append(skills.execute(kind, key))
        env.wait(2.)
        api.image('before_clear')
        record['cleared_node'] = skills.order[-1]
        record['clear_result'] = clear_object(skills, skills.order[-1])
        api.image('after_clear')
        record['reset_count'] = env.reset_count
        record['geometry_after'] = skills.packet()
    except EpisodeSuccess:
        record['early_success'] = True
    except Exception as error:
        record['error'] = type(error).__name__+': '+str(error)
        (args.output/'error.txt').write_text(traceback.format_exc())
    finally:
        record['timing'] = env.timing()
        env.close()
        (args.output/'result.json').write_text(json.dumps(plain(record), indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k not in ('geometry_after','sequence')}, indent=2))


if __name__ == '__main__':
    main()
