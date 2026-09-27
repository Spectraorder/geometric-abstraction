"""Render the passive single-hoop insertion scene for visual inspection."""
import argparse
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
from PIL import Image
from view_scene import build_scene


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/hoop_pole_preview'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    _,m,d=build_scene(scene_task='hoop_pole')
    for _ in range(round(1/m.opt.timestep)):mujoco.mj_step(m,d)
    m.vis.global_.offwidth=1280;m.vis.global_.offheight=960
    option=mujoco.MjvOption();option.geomgroup[3:]=0
    with mujoco.Renderer(m,height=960,width=1280) as renderer:
        for name,distance,lookat in [('scene',.8,(0,-.23,.44)),('objects',.31,(0,-.275,.42))]:
            camera=mujoco.MjvCamera();camera.lookat[:]=lookat
            camera.distance=distance;camera.azimuth=100;camera.elevation=-50
            renderer.update_scene(d,camera=camera,scene_option=option)
            path=a.output/f'{name}.png';Image.fromarray(renderer.render()).save(path);print(path,flush=True)


if __name__=='__main__':main()
