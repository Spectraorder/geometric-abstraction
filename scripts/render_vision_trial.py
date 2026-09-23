"""Render a saved trial including failures; does not execute the policy again."""
import argparse
import json
from pathlib import Path
import tempfile
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from aspire_so100.playback import recorded_env_for_result
from aspire_so100.rgbd import depth_preview


def main():
 p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--speed',type=float,default=2);p.add_argument('--fps',type=int,default=20);a=p.parse_args()
 result=json.loads((a.trial/'result.json').read_text());record=np.load(a.trial/'rollout.npz');states=record['states'];dt=float(record['timestep']);stype=int(record['state_type'])
 calls=json.loads((a.trial/'api_calls.json').read_text())
 wrist_used=any(c['method']=='camera' and c['params'].get('view')=='wrist' for c in calls)
 with tempfile.TemporaryDirectory() as temp:
  env=recorded_env_for_result(result,Path(temp));env.model.vis.global_.offwidth=720;env.model.vis.global_.offheight=540
  camera=mujoco.MjvCamera();camera.lookat[:]=[0,-.26,.44];camera.distance=1.15;camera.azimuth=115;camera.elevation=-32
  renderer=mujoco.Renderer(env.model,height=540,width=720)
  try:
   env.get_overhead_rgbd();sensor=env._overhead_rgbd
   indices=np.clip(np.rint(np.arange(0,len(states)*dt,a.speed/a.fps)/dt).astype(int),0,len(states)-1)
   font=ImageFont.load_default(size=17);small=ImageFont.load_default(size=13)
   with imageio.get_writer(a.trial/'demo.mp4',fps=a.fps,codec='libx264',quality=7,pixelformat='yuv420p',macro_block_size=1,ffmpeg_params=['-movflags','+faststart']) as writer:
    for count,index in enumerate(indices):
     mujoco.mj_setState(env.model,env.data,states[index],stype);mujoco.mj_forward(env.model,env.data)
     renderer.update_scene(env.data,camera=camera)
     frame=Image.new('RGB',(960,600),'#101722');frame.paste(Image.fromarray(renderer.render()),(0,60));draw=ImageDraw.Draw(frame)
     rgbd=sensor.capture(env.data);sensor.renderer.update_scene(env.data,camera='robot/wrist_camera');wrist=sensor.renderer.render().copy()
     for y,label,pixels in [(60,'RGB-D: SIDE RGB',rgbd['rgb']),(240,'RGB-D: DEPTH',depth_preview(rgbd['depth_m'])),(420,'WRIST: POLICY RGB' if wrist_used else 'WRIST (DISPLAY ONLY)',wrist)]:
      frame.paste(Image.fromarray(pixels).resize((240,180)),(720,y));draw.rectangle((720,y,960,y+20),fill='#172633');draw.text((724,y+2),label,font=small,fill='white')
     draw.text((12,8),f'RGB-D {result.get("task", "tower").upper()} POLICY | {a.trial.parent.name} | seed {result["seed"]}',font=font,fill='white')
     end=env.data.time>=states[-1][0]-1.
     status=('PASS' if result['success'] else 'FAIL') if end else 'RUNNING'
     draw.text((12,34),f'RGB-D + proprioception | {a.speed:g}x | t={env.data.time:.2f}s | {status}',font=font,fill='#6fe6b0')
     writer.append_data(np.asarray(frame))
     if count in (0,len(indices)//2,len(indices)-1):frame.save(a.trial/f'video_frame_{count:04d}.png')
   (a.trial/'video.json').write_text(json.dumps({'frames':len(indices),'fps':a.fps,'speed':a.speed,'duration_seconds':len(indices)/a.fps,'source':'saved actual physics states','success':result['success']},indent=2)+'\n')
  finally:renderer.close();env.close()

if __name__=='__main__':main()
