"""Render a close view of actual saved structure states; never execute a policy."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aspire_so100.playback import recorded_env_for_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    width, height = 640, 480
    canvas = Image.new('RGB', (width*len(args.trials), height+80), '#101722')
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=15)
    records = []
    for index, trial in enumerate(args.trials):
        result = json.loads((trial/'result.json').read_text())
        if result['task'] != 'inventory_structure':
            raise ValueError('The gallery requires recorded structure trials')
        with np.load(trial/'rollout.npz', allow_pickle=False) as record:
            state = record['states'][-1].copy()
            state_type = int(record['state_type'])
        with TemporaryDirectory() as temporary:
            env = recorded_env_for_result(result, Path(temporary))
            renderer = None
            try:
                mujoco.mj_setState(env.model, env.data, state, state_type)
                mujoco.mj_forward(env.model, env.data)
                # Private geometry only frames an external display camera. It
                # is never returned to or used to execute the policy.
                bounds = [env.geometry_bounds(name) for name in env.order]
                low = np.min([b[0] for b in bounds], axis=0)
                high = np.max([b[1] for b in bounds], axis=0)
                camera = mujoco.MjvCamera()
                camera.lookat[:] = (low+high)/2
                camera.distance = max(.31, 2.4*float(np.linalg.norm(high-low)))
                camera.azimuth = 105
                camera.elevation = -27
                renderer = mujoco.Renderer(env.model, width=width, height=height)
                renderer.update_scene(env.data, camera=camera)
                canvas.paste(Image.fromarray(renderer.render()), (width*index, 80))
            finally:
                if renderer is not None:
                    renderer.close()
                env.close()
        status = 'PASS' if result['success'] else 'FAIL'
        scene = result['scene']
        draw.text((width*index+16, 12), scene['recipe'].replace('_', ' ').title()+'  |  '+status,
                  fill='#83ebbc' if result['success'] else '#ffb199', font=title)
        extras = scene['object_count']-len(scene['order'])
        draw.text((width*index+16, 46), f'Frozen seed {result["seed"]} | {extras} distractors | actual final state',
                  fill='white', font=small)
        records.append({'recipe': scene['recipe'], 'seed': result['seed'], 'goal': result['goal'],
                        'success': result['success'], 'distractors': extras,
                        'candidate_sha256': result['candidate_sha256']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    args.output.with_suffix('.json').write_text(json.dumps({
        'source': 'Final archived MuJoCo integration states, external display camera; no policy rerun.',
        'selection': 'Selected illustrative trials, not a substitute for the full declared evaluation.',
        'records': records}, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
