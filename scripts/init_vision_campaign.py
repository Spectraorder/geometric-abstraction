"""Initialize an empty vision learning campaign; no policy or skill is loaded."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from PIL import Image
from aspire_so100.tower import TowerEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    protocol = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'actor': 'fresh model context each round; only current campaign history',
        'existing_skills': False, 'existing_policies': False,
        'training_seed': 81, 'debug_confirmation_seeds': [82, 83],
        'heldout_seeds': [91, 92],
        'sensors': 'RGB-D + camera calibration + robot proprioception',
        'privileged_objects_to_actor': False,
        'online_evaluator_to_policy': False,
        'reward': 'post-trial binary success only',
        'prior_infrastructure': 'Supplied generic robot-model IK with soft downward gripper preference, simulator, rendering and SDK; no task policy',
        'every_attempt_video': True,
    }
    with tempfile.TemporaryDirectory() as temporary:
        env = TowerEnv(Path(temporary))
        try:
            env.reset(81)
            Image.fromarray(env.get_overhead_rgbd()['rgb']).save(args.root/'initial.png')
        finally:
            env.close()
    (args.root/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (args.root/'campaign.json').write_text(json.dumps(protocol | {'status':'initialized','history':[]}, indent=2)+'\n')
    print(f'Initialized {args.root}; no model call or policy execution.')


if __name__ == '__main__':
    main()
