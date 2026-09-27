"""Render actual saved cube stacks; display camera and labels are offline only."""
import argparse
import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from aspire_so100.playback import recorded_env_for_result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=json.loads(args.report.read_text())['cases']
    seeds=sorted({r['seed'] for r in rows})
    gallery=Image.new('RGB',(1200,510*len(seeds)),'#101722')
    font=ImageFont.load_default(size=18)
    for row in rows:
        trial=Path(row['trial']);policy=trial.parent.name
        if policy not in ('baseline','precision'):continue
        result=json.loads((trial/'result.json').read_text())
        with np.load(trial/'rollout.npz') as archive:
            state=archive['states'][-1];stype=int(archive['state_type'])
        with tempfile.TemporaryDirectory() as tmp:
            env=recorded_env_for_result(result,Path(tmp))
            renderer=None
            try:
                mujoco.mj_setState(env.model,env.data,state,stype);mujoco.mj_forward(env.model,env.data)
                centers=np.array([env.data.xpos[env.model.body(f'{n}/{n}').id] for n in env.order])
                env.model.vis.global_.offwidth=600;env.model.vis.global_.offheight=420
                camera=mujoco.MjvCamera();camera.lookat[:]=centers.mean(axis=0)
                camera.distance=.28;camera.azimuth=115;camera.elevation=-22
                renderer=mujoco.Renderer(env.model,height=420,width=600)
                renderer.update_scene(env.data,camera=camera)
                panel=Image.new('RGB',(600,510),'#101722');panel.paste(Image.fromarray(renderer.render()),(0,90))
                draw=ImageDraw.Draw(panel)
                draw.text((12,8),f"{policy.upper()} | seed {row['seed']} | NoSlip {result.get('physics', {}).get('noslip_iterations', 0)}",font=font,fill='white')
                draw.text((12,34),f"Max corner mismatch: {row['max_corner_m']*1000:.2f} mm",font=font,fill='#80e0b0')
                durable = row.get('durability', {}).get('stable', False)
                draw.text((12,60),'Actual final state; extra 60 s: '+('PASS' if durable else 'FAIL / not tested'),font=font,fill='white')
                gallery.paste(panel,(0 if policy=='baseline' else 600,510*seeds.index(row['seed'])))
            finally:
                if renderer is not None:renderer.close()
                env.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    gallery.save(args.output)

if __name__=='__main__':main()
