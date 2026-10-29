"""Render a saved rollout (every physics step) as a continuous high-resolution demo video.

Replays recorded integration states only; the policy is not executed again.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
sys.path.insert(0, str(Path(__file__).parent))
import demo_recipes  # noqa: E402,F401  registers demo-only recipes


def build_env(trial, temp):
    """Rebuild the exact scene a trial ran in."""
    builder = trial / 'scene_builder.json'
    if builder.exists():
        spec = json.loads(builder.read_text())
        if spec['kind'] == 'ring':
            import ring_peg_development
            from ring_peg_scene import RingPegEnv
            with ring_peg_development.installed():
                return RingPegEnv(Path(temp), scene_seed=spec['seed'])
        if spec['kind'] == 'relocation':
            from online_relocation_scene import RelocationEnv
            return RelocationEnv(Path(temp), scene_seed=spec['seed'])
        raise ValueError(f"Unknown scene builder {spec['kind']}")
    from aspire_so100.playback import recorded_env_for_result
    return recorded_env_for_result(json.loads((trial / 'result.json').read_text()), Path(temp))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--title', default='')
    p.add_argument('--speed', type=float, default=2.)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--lookat', type=float, nargs=3, default=(0., -.22, .45))
    p.add_argument('--distance', type=float, default=.66)
    p.add_argument('--azimuth', type=float, default=145.)
    p.add_argument('--elevation', type=float, default=-24.)
    p.add_argument('--end-hold', type=float, default=1.5)
    p.add_argument('--preview', action='store_true', help='Write first/middle/last PNG frames only')
    p.add_argument('--clean', action='store_true', help='No burned-in title or time labels')
    a = p.parse_args()

    record = np.load(a.trial / 'rollout.npz')
    states, dt, state_type = record['states'], float(record['timestep']), int(record['state_type'])
    with tempfile.TemporaryDirectory() as temp:
        env = build_env(a.trial, temp)
        env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, a.width)
        env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, a.height)
        env.model.vis.quality.offsamples = 8
        # The side-camera mast is a non-colliding visual geom; hide it so it does not block the demo view.
        for geom in range(env.model.ngeom):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ''
            if name.startswith('table/camera_m') or name.startswith('camera_m'):
                env.model.geom_rgba[geom, 3] = 0.
        camera = mujoco.MjvCamera()
        camera.lookat[:] = a.lookat
        camera.distance, camera.azimuth, camera.elevation = a.distance, a.azimuth, a.elevation
        renderer = mujoco.Renderer(env.model, height=a.height, width=a.width)
        font = ImageFont.load_default(size=26)
        small = ImageFont.load_default(size=20)

        indices = np.clip(np.rint(np.arange(0, len(states) * dt, a.speed / a.fps) / dt).astype(int), 0, len(states) - 1)
        if indices[-1] != len(states) - 1:
            indices = np.r_[indices, len(states) - 1]
        if a.preview:
            indices = indices[[0, len(indices) // 2, -1]]

        def frame_at(index):
            mujoco.mj_setState(env.model, env.data, states[index], state_type)
            mujoco.mj_forward(env.model, env.data)
            renderer.update_scene(env.data, camera=camera)
            image = Image.fromarray(renderer.render())
            if a.clean:
                return image
            draw = ImageDraw.Draw(image, 'RGBA')
            draw.rounded_rectangle((18, 18, 30 + draw.textlength(a.title, font=font), 60), 8, fill=(16, 23, 34, 170))
            draw.text((24, 24), a.title, font=font, fill='white')
            label = f'{a.speed:g}x speed  |  t = {env.data.time:5.1f} s'
            width = draw.textlength(label, font=small)
            draw.rounded_rectangle((a.width - 42 - width, 18, a.width - 18, 54), 8, fill=(16, 23, 34, 170))
            draw.text((a.width - 30 - width, 24), label, font=small, fill='white')
            return image

        try:
            if a.preview:
                for count, index in enumerate(indices):
                    frame_at(index).save(a.output.with_name(f'{a.output.stem}_preview{count}.png'))
                return
            with imageio.get_writer(a.output, fps=a.fps, codec='libx264', pixelformat='yuv420p', macro_block_size=1,
                                    ffmpeg_params=['-crf', '23', '-preset', 'slow', '-movflags', '+faststart']) as writer:
                for index in indices:
                    last = np.asarray(frame_at(index))
                    writer.append_data(last)
                for _ in range(int(round(a.end_hold * a.fps))):
                    writer.append_data(last)
            print(json.dumps({'output': str(a.output), 'frames': len(indices), 'seconds': len(indices) / a.fps,
                              'bytes': a.output.stat().st_size}))
        finally:
            renderer.close()
            env.close()


if __name__ == '__main__':
    main()
