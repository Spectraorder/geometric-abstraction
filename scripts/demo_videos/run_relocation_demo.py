"""Scripted tower-relocation demo in the online relocation scene, recording every physics state.

Demo controller only: it reads simulator object poses (privileged truth) to pick and
place cubes, unlike the study agents, which only see sensors. Grasp heights, jaw
angles and approach clearances follow policies/controllers/structure_policy.py.
Sequence: upper -> staging spot, middle -> destination, upper -> middle.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

import numpy as np  # noqa: E402
from aspire_so100.playback import RecordingMixin  # noqa: E402
from online_relocation_scene import RelocationEnv  # noqa: E402
from run_vision_trial import move_tool, plan_tool  # noqa: E402

OPEN, CLOSED, RELEASE = .8, -.174, 1.4
STAGING_XY = np.array([-.005, -.355])


class RecordedRelocationEnv(RecordingMixin, RelocationEnv):
    pass


class Controller:
    def __init__(self, env):
        self.env = env
        self.park = self.tool()

    def tool(self):
        return self.env.data.site_xpos[self.env.site_id].copy()

    def center(self, name):
        return self.env._object_state(name)['position'].copy()

    def half(self, name):
        return float(self.env.model.geom(f'{name}/{name}_geom').size[2])

    def move(self, point, seconds=1.5):
        point = np.asarray(point, float)
        try:
            move_tool(self.env, point, seconds)
        except RuntimeError:
            # Same fallback as structure_policy: accept the achievable orientation at tall waypoints.
            first = plan_tool(self.env, point)
            move_tool(self.env, point, seconds, first['tool_rotation'])

    def pick(self, name):
        c, h = self.center(name), 2 * self.half(name)
        self.env.set_gripper(OPEN, .5)
        self.move(c + [0, 0, h / 2 + .075], 1.8)
        grasp = c + [0, 0, .6 * h]
        self.move(grasp + [0, 0, .025], 1.1)
        self.move(grasp, 1.2)
        self.env.set_gripper(CLOSED, 1.)
        self.env.wait(.35)
        self.move(grasp + [0, 0, .06], 1.7)

    def place(self, name, bottom_xyz):
        offset = self.center(name) - self.tool()
        h = 2 * self.half(name)
        command = np.asarray(bottom_xyz, float) + [0, 0, h / 2 + .003] - offset
        high = self.tool()
        high[2] = max(high[2], command[2] + .05)
        self.move(high, 1.2)
        high[:2] = command[:2]
        self.move(high, 2.)
        self.move(command + [0, 0, .02], 1.4)
        self.move(command, 1.)
        self.env.wait(.3)
        self.env.set_gripper(RELEASE, .35)
        self.env.wait(.3)
        retreat = self.tool()
        self.move(retreat + [0, 0, .025], 1.5)
        self.move(retreat + [0, 0, .085], 1.5)

    def top_of(self, name):
        return self.center(name) + [0, 0, self.half(name)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    env = RecordedRelocationEnv(a.output / 'env', scene_seed=a.seed)
    try:
        env.reset(a.seed)
        env.frames.clear()
        env.capture()
        roles = env.roles
        ctl = Controller(env)
        env.wait(.8)
        staging = np.r_[STAGING_XY, env.geometry_bounds(roles['destination'])[0][2]]
        ctl.pick(roles['upper'])
        ctl.place(roles['upper'], staging)
        ctl.pick(roles['middle'])
        ctl.place(roles['middle'], ctl.top_of(roles['destination']))
        ctl.pick(roles['upper'])
        ctl.place(roles['upper'], ctl.top_of(roles['middle']))
        ctl.move(ctl.park, 1.8)
        env.wait(5.5)
        status = env.structure_status(list(env.order))
        np.savez_compressed(a.output / 'rollout.npz', states=np.asarray(env.frames),
                            timestep=env.model.opt.timestep, state_type=int(env.state_type))
        (a.output / 'scene_builder.json').write_text(json.dumps({'kind': 'relocation', 'seed': a.seed}) + '\n')
        result = {'seed': a.seed, 'success': status['stable'], 'reasons': status['reasons'],
                  'sim_seconds': float(env.data.time), 'controller': 'scripted demo, simulator truth'}
        (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
    finally:
        env.close()


if __name__ == '__main__':
    main()
