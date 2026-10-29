"""Run the hand-written ring-on-peg starter on a development seed and save every physics state.

Same path as scripts/check_ring_peg_repair.py (development sensors, no model calls),
plus per-step state recording for video. The trusted repo controller runs as a plain
isolated-mode subprocess because bwrap cannot start inside the current sandbox.
"""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import ring_peg_development as repair  # noqa: E402
import run_ring_peg_trial as trial  # noqa: E402
from aspire_so100.playback import RecordingMixin  # noqa: E402
from ring_peg_scene import RingPegEnv  # noqa: E402

recorded = []


class RecordedRingPegEnv(RecordingMixin, RingPegEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        recorded.append(self)

    def reset(self, seed=None):
        result = super().reset(seed)
        self.frames.clear()
        self.capture()
        return result


def trusted_worker_command(directory, candidate):
    return [sys.executable, '-I', str(directory / 'worker.py'), str(candidate)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--candidate', type=Path, default=ROOT / 'policies/controllers/ring_peg_starter.py')
    a = p.parse_args()
    with repair.installed(), \
            patch.object(trial, 'RingSensors', repair.DevelopmentSensors), \
            patch.object(trial, 'RingPegEnv', RecordedRingPegEnv), \
            patch.object(trial, 'policy_worker_command', trusted_worker_command):
        result = trial.run(a.output, a.candidate, a.seed)
    env = recorded[-1]
    np.savez_compressed(a.output / 'rollout.npz', states=np.asarray(env.frames), timestep=env.model.opt.timestep,
                        state_type=int(env.state_type))
    (a.output / 'scene_builder.json').write_text(json.dumps({'kind': 'ring', 'seed': a.seed}) + '\n')
    print(json.dumps({k: result.get(k) for k in ('seed', 'success', 'status', 'error')}))


if __name__ == '__main__':
    main()
