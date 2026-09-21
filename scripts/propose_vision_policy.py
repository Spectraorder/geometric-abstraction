"""Fresh model invocation with no user config, host skills, tools, or repository context."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import os
from datetime import datetime, timezone

CONTRACT='''You are the policy learner for a new SO-100 three-block manipulation experiment.
No prior policy or skill library is supplied. Use only this task contract, attached sensor images,
and previous attempts from THIS experiment when supplied below. Never use tools or inspect files.
Return JSON with hypothesis and code; code defines run(api). Only numpy, math, and scipy imports.
No file/network/process/introspection. Do not access private attributes. All diagnostics via api.note.
TASK: blue block stays on table; pick red and place on blue; then pick green and place on red.
All are free rigid bodies. Objects must be grasped, physically transported, released, and robot withdrawn.
After placing red, leave it untouched for at least 3 seconds before picking green. After green, withdraw
and wait at least 6 seconds. A separate evaluator observes support, alignment, release, and drift.
The policy cannot query evaluator or read any object ground truth. You receive only success/failure AFTER a trial.

SENSOR API:
api.camera() -> rgb uint8[240,320,3], depth_m float32[240,320], valid_depth bool,
intrinsics 3x3, camera_to_world 4x4, time. Ideal side-mounted calibrated RGB-D.
Depth is optical-axis z in meters, NOT range. For pixel (u,v):
world=R @ (inv(K) @ [u,v,1] * depth[v,u]) + t. Camera axes right,down,forward.
Use RGB segmentation and point clouds to estimate object positions/heights; NO preknown object coordinates or sizes.
The table and objects are visible in the attached initial image. Object colors are blue, red, green.
api.proprioception() -> joint_angles length5, tool_position world XYZ meters (robot forward kinematics), time.

ACTUATION API (robot SDK, no object perception or hidden grasp heuristics):
api.move_tool([x,y,z], seconds=1.5): position with soft preference for downward gripper,
box-constrained robot-model IK; may tilt near reach limits. It moves joints along a smooth interpolation.
Returns no object state or grasp result. Site is an approximate gripper center, not a guaranteed object center.
api.move_joints(five_angles, seconds=1.5) optional raw control; joint limits radians
[-1.92,1.92],[-3.32,.174],[-.174,3.14],[-1.66,1.66],[-2.79,2.79].
api.gripper(angle, seconds=.6): jaw angle in [-.174,1.75], larger is more open; .8 is open.
api.wait(seconds): advances physics, max10s per call. api.note(text): logs your perception/decision (no state oracle).
All motion durations 0.1..10s. All locations MUST come from current camera measurements, not constants keyed
by seed or object names. Known camera calibration and measured robot joint/tool pose are allowed.
No api.get_observation(), no api.check_structure(), no privileged object state, no forced attachment.
Pixel and pointcloud operations may use numpy and scipy.ndimage. Do not print; use api.note for diagnostics.
There is a 120s wall-time, 600 RPC call limit per trial. Robot base is fixed; workspace actions should be conservative.
Develop a physically plausible policy and adjust it using subsequent rendered outcomes and your logged estimates.
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--round-dir',type=Path,required=True);p.add_argument('--image',type=Path,action='append',required=True);p.add_argument('--previous',type=Path);p.add_argument('--model',default='gpt-6-astra');p.add_argument('--timeout',type=int,default=300);a=p.parse_args()
 a.round_dir.mkdir(parents=True,exist_ok=False)
 prompt=CONTRACT
 if a.previous:
  prompt+='\nPrevious attempt from this experiment:\n'+(a.previous/'candidate.py').read_text()
  prompt+='\nTrial feedback:\n'+(a.previous/'trial_81/actor_feedback.json').read_text()
 prompt+='\nDesign the next policy. Return complete code, not a patch.'
 (a.round_dir/'prompt.txt').write_text(prompt)
 schema={'type':'object','properties':{'hypothesis':{'type':'string'},'code':{'type':'string'}},'required':['hypothesis','code'],'additionalProperties':False}
 (a.round_dir/'schema.json').write_text(json.dumps(schema))
 with tempfile.TemporaryDirectory(prefix='fresh-vision-actor-') as directory:
  work=Path(directory).resolve();(work/'schema.json').write_text(json.dumps(schema));images=[]
  fresh_home=work/'codex_home';fresh_home.mkdir(mode=0o700)
  auth=Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex')))/'auth.json'
  if auth.exists():
   shutil.copyfile(auth,fresh_home/'auth.json');(fresh_home/'auth.json').chmod(0o600)
  actor_env=dict(os.environ,CODEX_HOME=str(fresh_home))
  for i,src in enumerate(a.image):
   dest=work/f'sensor_{i}.png';shutil.copyfile(src,dest);images.append(dest)
  command=['codex','exec','--ignore-user-config','--ephemeral','--sandbox','read-only','--skip-git-repo-check','--cd',str(work),
           '--model',a.model,'-c','project_doc_max_bytes=0','--enable','skip_host_skill_discovery',
           '--disable','shell_tool','--disable','unified_exec','--disable','apps','--disable','browser_use','--disable','computer_use','--disable','skill_search',
           '--output-schema',str(work/'schema.json'),'--output-last-message',str(work/'response.json'),'--json']
  for image in images:command+=['--image',str(image)]
  command+=['-']
  meta={'started_at':datetime.now(timezone.utc).isoformat(),'model':a.model,'fresh_context':True,'user_config_loaded':False,'host_skill_discovery':False,'old_policy_paths_in_prompt':False,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'images':[str(x) for x in a.image],'command':command}
  (a.round_dir/'invocation.json').write_text(json.dumps(meta,indent=2)+'\n')
  with (a.round_dir/'model.jsonl').open('w') as stdout,(a.round_dir/'model.stderr').open('w') as stderr:
   subprocess.run(command,input=prompt,text=True,stdout=stdout,stderr=stderr,timeout=a.timeout,check=True,env=actor_env)
  response=json.loads((work/'response.json').read_text())
  (a.round_dir/'response.json').write_text(json.dumps(response,indent=2)+'\n')
  (a.round_dir/'candidate.py').write_text(response['code'])
  print(response['hypothesis'])

if __name__=='__main__':main()
