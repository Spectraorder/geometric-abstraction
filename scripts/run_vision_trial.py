"""Sensor-only RPC trial. Policy process cannot read simulator memory or prior code."""
import argparse
import ast
import base64
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import tempfile
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from PIL import Image
from aspire_so100.playback import RecordedTowerEnv
from aspire_so100.tower import ORDER
from aspire_so100.rgbd import depth_preview


def pack(v):
    if isinstance(v,np.ndarray):
        a=np.ascontiguousarray(v)
        return {'__array__':base64.b64encode(a.tobytes()).decode(),'dtype':str(a.dtype),'shape':a.shape}
    if isinstance(v,dict):return {k:pack(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [pack(x) for x in v]
    if isinstance(v,np.generic):return v.item()
    return v


class VisionTrialEnv(RecordedTowerEnv):
    def __init__(self,output):
        self.intermediate_stable_seconds=0.
        self.intermediate_verified=False
        self.last_monitor=-1.
        self.monitor_start_positions=None
        super().__init__(output)
    def _step(self):
        super()._step()
        # Private evaluator monitor. No result or timing flag is exposed to actor.
        if not self.intermediate_verified and self.data.time-self.last_monitor>=.05:
            now=float(self.data.time)
            stable=self.structure_status(list(ORDER[:2]))['stable']
            positions={n:self._object_state(n)['position'].copy() for n in ORDER[:2]}
            if stable and self.monitor_start_positions is None:
                self.monitor_start_positions=positions
            if stable and any(np.linalg.norm(positions[n]-self.monitor_start_positions[n])>.003 for n in ORDER[:2]):
                stable=False
            if not stable:
                self.monitor_start_positions=None
            self.intermediate_stable_seconds=self.intermediate_stable_seconds+(now-self.last_monitor) if stable else 0.
            self.last_monitor=now
            if self.intermediate_stable_seconds>=2.:
                self.intermediate_verified=True
                self.checks.append({'stable':True,'reasons':[],'blocks':list(ORDER[:2]),'duration_s':self.intermediate_stable_seconds,'time':now,'source':'private continuous monitor'})


def move_tool(env,position,seconds):
    """Robot SDK: robot-model-only numerical IK, no object state inputs."""
    target=np.asarray(position,float)
    if target.shape!=(3,) or not np.isfinite(target).all() or np.linalg.norm(target)>2:
        raise ValueError('position must be a finite world XYZ in meters')
    scratch=mujoco.MjData(env.model)
    q=env.data.qpos[env.arm_qpos_addresses].copy()
    limits=env.model.jnt_range[[j.id for j in env.arm_joints]]
    orientation=np.array([[0,0,1],[1,0,0],[0,1,0.]])
    def residual(angles):
        scratch.qpos[env.arm_qpos_addresses]=angles
        mujoco.mj_kinematics(env.model,scratch)
        R=scratch.site_xmat[env.site_id].reshape(3,3)
        return np.r_[scratch.site_xpos[env.site_id]-target,(.025*(R-orientation)).ravel()]
    result=least_squares(residual,np.clip(q,limits[:,0]+1e-8,limits[:,1]-1e-8),bounds=(limits[:,0],limits[:,1]),max_nfev=150,xtol=1e-10,ftol=1e-10,gtol=1e-10)
    if np.linalg.norm(residual(result.x)[:3])>.005:
        raise RuntimeError('Requested tool position is unreachable within 5 mm')
    env.move_to_joint_positions(result.x,seconds)


def sandbox_profile(directory):
    return ('(version 1)\n(allow default)\n(deny network*)\n'
            '(deny file-read-data (require-all (subpath '+json.dumps(str(Path.home()))+') '
            '(require-not (subpath '+json.dumps(str(Path(sys.prefix).resolve()))+'))))\n'
            '(deny file-write* (require-not (subpath '+json.dumps(directory)+')))\n')


def validate_source(source):
    tree=ast.parse(source)
    forbidden={'open','exec','eval','compile','getattr','setattr','delattr','globals','locals','vars',
               '__import__','input','breakpoint','print','load','save','savez','savez_compressed','fromfile',
               'tofile','memmap','ctypeslib','ctypes','sys','os','subprocess'}
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or '']
            if any(x.split('.')[0] not in {'numpy','scipy','math'} for x in names):raise ValueError('Import outside numerical libraries')
        if isinstance(n,ast.Attribute) and (n.attr.startswith('_') or n.attr in forbidden):raise ValueError('Forbidden attribute')
        if isinstance(n,ast.Name) and n.id in forbidden:raise ValueError('Forbidden name')
    compile(tree,'<vision policy>','exec')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=81);p.add_argument('--timeout',type=float,default=120)
    p.add_argument('--sensor-blackout',action='store_true',help='Diagnostic: remove all policy RGB-D pixels; evaluation remains unchanged')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    validate_source(a.candidate.read_text())
    env=VisionTrialEnv(a.output);calls=[];error=None;images=[]
    try:
        env.reset(a.seed);env.frames.clear();env.capture()
        start=float(env.data.time)
        initial=env.get_overhead_rgbd();Image.fromarray(initial['rgb']).save(a.output/'initial.png')
        with tempfile.TemporaryDirectory(prefix='vision-policy-') as tmp:
            tmp=str(Path(tmp).resolve());shutil.copy(a.candidate,Path(tmp)/'candidate.py')
            shutil.copy(Path(__file__).with_name('vision_policy_worker.py'),Path(tmp)/'worker.py')
            profile=Path(tmp)/'policy.sb';profile.write_text(sandbox_profile(tmp))
            (a.output/'sandbox.sb').write_text(profile.read_text())
            cmd=['/usr/bin/sandbox-exec','-f',str(profile),sys.executable,'-I',str(Path(tmp)/'worker.py'),str(Path(tmp)/'candidate.py')]
            clean={'PATH':'/usr/bin:/bin','HOME':tmp,'TMPDIR':tmp,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
            with (a.output/'worker.stderr').open('w') as stderr:
                proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,cwd=tmp,env=clean,bufsize=1)
                selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
                deadline=time.monotonic()+a.timeout
                try:
                    while True:
                        if time.monotonic()>deadline:raise TimeoutError('Policy wall time limit exceeded')
                        if not selector.select(timeout=1):continue
                        line=proc.stdout.readline()
                        if not line:raise RuntimeError('Policy process exited without completion; see worker.stderr')
                        request=json.loads(line);method=request['method'];params=request['params']
                        calls.append({'time':float(env.data.time),'method':method,'params':params})
                        if method=='done':break
                        if method=='failed':raise RuntimeError(params['error'])
                        if len(calls)>600:raise RuntimeError('API call limit exceeded')
                        try:
                            response=None
                            seconds=float(params.get('seconds',1.5))
                            if not np.isfinite(seconds) or not 0<=seconds<=10:raise ValueError('seconds must be 0..10')
                            if method=='camera':
                                response=env.get_overhead_rgbd()
                                if a.sensor_blackout:
                                    response['rgb'][:]=0
                                    response['depth_m'][:]=np.nan
                                    response['valid_depth'][:]=False
                                i=len(images);filename=f'camera_{i:03d}.npz'
                                np.savez_compressed(a.output/filename,**response)
                                Image.fromarray(response['rgb']).save(a.output/f'camera_{i:03d}.png')
                                images.append(filename)
                            elif method=='proprioception':
                                response={'joint_angles':env.data.qpos[env.arm_qpos_addresses].copy(),'tool_position':env.data.site_xpos[env.site_id].copy(),'time':float(env.data.time)}
                            elif method=='move_tool':move_tool(env,params['position'],max(.1,seconds))
                            elif method=='move_joints':
                                q=np.asarray(params['angles'],float)
                                if q.shape!=(5,) or not np.isfinite(q).all():raise ValueError('five finite joint angles required')
                                limits=env.model.jnt_range[[j.id for j in env.arm_joints]]
                                if np.any(q<limits[:,0]) or np.any(q>limits[:,1]):raise ValueError('Joint limits exceeded')
                                env.move_to_joint_positions(q,max(.1,seconds))
                            elif method=='gripper':
                                angle=float(params['angle'])
                                if not np.isfinite(angle) or not -.174<=angle<=1.75:raise ValueError('Jaw angle outside hardware limits')
                                env.set_gripper(angle,max(.1,seconds))
                            elif method=='wait':env.wait(seconds)
                            elif method=='note':pass
                            else:raise ValueError('API method not permitted')
                            proc.stdin.write(json.dumps({'result':pack(response)})+'\n');proc.stdin.flush()
                        except Exception as exc:
                            proc.stdin.write(json.dumps({'error':type(exc).__name__+': '+str(exc)})+'\n');proc.stdin.flush()
                except Exception as exc:
                    error=type(exc).__name__+': '+str(exc);env.failure=error
                finally:
                    proc.terminate()
                    try:proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    selector.close()
        policy_seconds=float(env.data.time)-start
        env.check_structure(list(ORDER),duration=5.)
        metrics=env.evaluate()
        env.write_trace(metrics,error)
        frame=env.get_overhead_rgbd();Image.fromarray(frame['rgb']).save(a.output/'outcome.png')
        env.wait(1.)
        np.savez_compressed(a.output/'rollout.npz',states=env.frames,timestep=env.model.opt.timestep,state_type=int(env.state_type))
        (a.output/'api_calls.json').write_text(json.dumps(calls,indent=2)+'\n')
        result={'seed':a.seed,'success':metrics.success,'policy_error':error,'policy_sim_seconds':policy_seconds,
                'candidate_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),'camera_calls':len(images),
                'sensor_blackout':a.sensor_blackout,
                'metrics':asdict(metrics),'observation_contract':'RGB-D + calibration + robot proprioception ONLY; evaluator results never returned to policy'}
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        # Only this reduced result and rendered sensor frames go back to the actor.
        (a.output/'actor_feedback.json').write_text(json.dumps({'success':metrics.success,'policy_error':error,'policy_sim_seconds':policy_seconds,'notes':[c['params']['text'] for c in calls if c['method']=='note']},indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:env.close()

if __name__=='__main__':main()
