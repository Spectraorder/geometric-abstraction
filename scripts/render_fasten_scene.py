"""Render the Fasten Screws scene for geometry and camera inspection."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
from PIL import Image
from view_scene import build_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/fasten_screws_preview'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _, model, data = build_scene(scene_task='fasten_screws')
    for _ in range(round(1/model.opt.timestep)):
        mujoco.mj_step(model, data)
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 960
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    camera = mujoco.MjvCamera()
    with mujoco.Renderer(model, height=960, width=1280) as renderer:
        for name, lookat, distance, elevation in (
            ('scene', (0., -.22, .43), .85, -45),
            ('fasteners', (0., -.28, .415), .48, -55),
            ('nut_internal_thread', (.085, -.34, .408), .105, -50),
        ):
            camera.lookat[:] = lookat
            camera.distance = distance
            camera.azimuth = 90
            camera.elevation = elevation
            renderer.update_scene(data, camera=camera, scene_option=option)
            path = args.output/f'{name}.png'
            Image.fromarray(renderer.render()).save(path)
            print(path, flush=True)


if __name__ == '__main__':
    main()
