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

if sys.platform == 'linux' and not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import numpy as np
from scipy.optimize import least_squares
from PIL import Image
from aspire_so100.playback import RecordedTowerEnv, RecordedPrismEnv, RecordedInventoryPrismEnv, RecordedInventoryTowerEnv, RecordedInventoryStructureEnv
from aspire_so100.tower_tasks import RECIPES, TowerRecipe, tower_recipe
from aspire_so100.structure_tasks import STRUCTURE_RECIPES, structure_recipe
from aspire_so100.tower import ORDER
from aspire_so100.rgbd import depth_preview
from aspire_so100.playback import RecordedFastenScrewsEnv, RecordedHoopPoleEnv


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


def plan_tool(env,position,orientation=None):
    """Robot-only IK preview. No object poses, contacts or task results."""
    cpu_started = time.process_time()
    target=np.asarray(position,float)
    if target.shape!=(3,) or not np.isfinite(target).all() or np.linalg.norm(target)>2:
        raise ValueError('position must be a finite world XYZ in meters')
    scratch=mujoco.MjData(env.model)
    q=env.data.qpos[env.arm_qpos_addresses].copy()
    limits=env.model.jnt_range[[j.id for j in env.arm_joints]]
    if orientation is None:
        orientation=np.array([[0,0,1],[1,0,0],[0,1,0.]])
    else:
        orientation=np.asarray(orientation,float)
        if (orientation.shape!=(3,3) or not np.isfinite(orientation).all()
                or not np.allclose(orientation.T @ orientation,np.eye(3),atol=1e-5)
                or np.linalg.det(orientation)<.999):
            raise ValueError('orientation must be a finite proper 3x3 rotation matrix')
    def residual(angles):
        scratch.qpos[env.arm_qpos_addresses]=angles
        mujoco.mj_kinematics(env.model,scratch)
        R=scratch.site_xmat[env.site_id].reshape(3,3)
        return np.r_[scratch.site_xpos[env.site_id]-target,(.025*(R-orientation)).ravel()]
    result=least_squares(residual,np.clip(q,limits[:,0]+1e-8,limits[:,1]-1e-8),bounds=(limits[:,0],limits[:,1]),max_nfev=150,xtol=1e-10,ftol=1e-10,gtol=1e-10)
    error=float(np.linalg.norm(residual(result.x)[:3]))
    env.robot_ik_cpu_seconds = getattr(env, 'robot_ik_cpu_seconds', 0.) + time.process_time() - cpu_started
    return {'joint_angles':result.x.copy(),'tool_position':scratch.site_xpos[env.site_id].copy(),
            'tool_rotation':scratch.site_xmat[env.site_id].reshape(3,3).copy(),'position_error':error}


def move_tool(env,position,seconds,orientation=None):
    """Robot SDK: robot-model-only numerical IK, no object state inputs."""
    plan=plan_tool(env,position,orientation)
    if plan['position_error']>.005:
        raise RuntimeError('Requested tool position is unreachable within 5 mm')
    env.move_to_joint_positions(plan['joint_angles'],seconds)


def sandbox_profile(directory):
    return ('(version 1)\n(allow default)\n(deny network*)\n'
            '(deny file-read-data (require-all (subpath '+json.dumps(str(Path.home()))+') '
            '(require-not (subpath '+json.dumps(str(Path(sys.prefix).resolve()))+'))))\n'
            '(deny file-write* (require-not (subpath '+json.dumps(directory)+')))\n')


def policy_worker_command(directory, candidate):
    """Run the policy with no network and only its scratch directory writable."""
    worker = str(directory / 'worker.py')
    candidate = str(candidate)
    if sys.platform == 'darwin':
        profile = directory / 'policy.sb'
        profile.write_text(sandbox_profile(str(directory)))
        return ['/usr/bin/sandbox-exec', '-f', str(profile), sys.executable,
                '-I', worker, candidate]
    if sys.platform == 'linux':
        if not shutil.which('bwrap'):
            raise RuntimeError('bwrap is required to run a vision policy on Linux')
        home = Path.home()
        checkout = Path(__file__).resolve().parents[1]
        hidden_roots = sorted({home,checkout},key=lambda p:(len(p.parts),str(p)))
        command = ['bwrap', '--die-with-parent', '--unshare-net', '--new-session', '--ro-bind', '/', '/']
        for root in hidden_roots:
            command += ['--tmpfs',str(root)]
        # uv commonly installs both the venv and its base interpreter under
        # HOME. Restore only their runtime directories after masking HOME;
        # the surrounding checkout and other home files stay hidden.
        runtime_paths = {Path(prefix) for prefix in (sys.prefix, sys.base_prefix)}
        runtime_paths.update(path.resolve() for path in tuple(runtime_paths))
        for path in sorted(runtime_paths, key=lambda p: (len(p.parts), str(p))):
            if any(path.is_relative_to(root) for root in hidden_roots):
                command += ['--ro-bind', str(path), str(path)]
        return command + ['--proc', '/proc', '--dev', '/dev',
                '--bind', str(directory), str(directory), '--chdir', str(directory),
                sys.executable, '-I', worker, candidate]
    raise RuntimeError(f'No vision policy sandbox for {sys.platform}')


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
    p.add_argument('--hoop-variant',choices=('standard','tight'),default='standard')
    p.add_argument('--seed',type=int,default=81);p.add_argument('--timeout',type=float,default=120)
    p.add_argument('--layout', choices=('nominal','random_xy','sampled'), default='nominal')
    p.add_argument('--task', choices=('tower','prism','inventory_prism','inventory_tower','inventory_structure','fasten_screws','hoop_pole'), default='tower')
    p.add_argument('--structure-recipe', choices=tuple(STRUCTURE_RECIPES))
    p.add_argument('--structure-color-map', type=json.loads, help='JSON color bijection for structure goals')
    p.add_argument('--tower-recipe', choices=(*RECIPES, 'random'))
    p.add_argument('--tower-order', nargs='+', help='Explicit inventory object names, bottom to top')
    p.add_argument('--tower-levels', type=int, help='2..4 levels, for --tower-recipe random')
    p.add_argument('--object-count', type=int, help='Inventory sample size, default 6; inventory tasks only')
    p.add_argument('--sensor-blackout',action='store_true',help='Diagnostic: remove all policy RGB-D pixels; evaluation remains unchanged')
    from aspire_so100.fasten_screws import CONTROLLERS
    p.add_argument('--fasten-controller', choices=CONTROLLERS, default='legacy')
    p.add_argument('--fasten-stage', choices=('pickup', 'fasten'), default='fasten')
    p.add_argument('--grasp-quality',choices=('legacy','grasp_quality_v1'),default='legacy')
    p.add_argument('--fasten-pairs',type=int,choices=(1,3),default=3)
    a=p.parse_args()
    fasten = a.task == 'fasten_screws'
    hoop = a.task == 'hoop_pole'
    if a.hoop_variant != 'standard' and not hoop:
        p.error('--hoop-variant requires --task hoop_pole')
    insertion = fasten or hoop
    if not fasten and (a.fasten_controller != 'legacy' or a.fasten_stage != 'fasten' or a.grasp_quality != 'legacy' or a.fasten_pairs != 3):
        p.error('Fastening controller/stage options require --task fasten_screws')
    inventory = a.task in ('inventory_prism', 'inventory_tower', 'inventory_structure') or insertion
    if insertion and a.object_count is not None:
        p.error('Insertion tasks use fixed pair counts; omit --object-count')
    if inventory != (a.layout == 'sampled'):
        p.error('Inventory tasks require --layout sampled; historical tasks use nominal/random_xy')
    if a.object_count is not None and (not inventory or not 2 <= a.object_count <= 20):
        p.error('--object-count must be 2..20 and requires an inventory task')
    if a.task != 'inventory_tower' and (a.tower_recipe is not None or a.tower_levels is not None or a.tower_order is not None):
        p.error('Tower recipe options require --task inventory_tower')
    if a.task == 'inventory_tower':
        try:
            if a.tower_order is not None:
                if a.tower_recipe is not None or a.tower_levels is not None:
                    raise ValueError('tower-order cannot be combined with recipe or levels')
                recipe = TowerRecipe('custom', tuple(a.tower_order))
            else:
                recipe = tower_recipe(a.tower_recipe or 'cube_triple', seed=a.seed, levels=a.tower_levels)
            if (a.object_count or 6) < len(recipe.order):
                raise ValueError('object-count must include every recipe level')
        except ValueError as error:
            p.error(str(error))
    if a.task != 'inventory_structure' and (a.structure_recipe is not None or a.structure_color_map is not None):
        p.error('Structure options require --task inventory_structure')
    if a.task == 'inventory_structure':
        try:
            recipe = structure_recipe(a.structure_recipe or 'cube_bridge', color_map=a.structure_color_map)
            if (a.object_count or 6) < len(recipe.order):
                raise ValueError('object-count must include every structure node')
        except (ValueError, TypeError) as error:
            p.error(str(error))
    a.output.mkdir(parents=True,exist_ok=False)
    validate_source(a.candidate.read_text())
    env=(RecordedHoopPoleEnv(a.output,scene_seed=a.seed,hoop_variant=a.hoop_variant) if hoop else
         RecordedFastenScrewsEnv(a.output,scene_seed=a.seed,controller=a.fasten_controller,
                               learning_stage=a.fasten_stage,grasp_quality=a.grasp_quality,pair_count=a.fasten_pairs) if fasten else
         RecordedInventoryStructureEnv(a.output, scene_seed=a.seed, object_count=a.object_count or 6,
                                      recipe=recipe.name, color_map=a.structure_color_map) if a.task == 'inventory_structure' else
         RecordedInventoryTowerEnv(a.output, scene_seed=a.seed, object_count=a.object_count or 6,
                                  recipe=recipe.name, recipe_order=recipe.order) if a.task == 'inventory_tower' else
         RecordedInventoryPrismEnv(a.output, scene_seed=a.seed, object_count=a.object_count or 6)
         if a.task=='inventory_prism' else (RecordedPrismEnv if a.task=='prism' else VisionTrialEnv)(a.output))
    calls=[];error=None;images=[];cpu_profile=None
    try:
        env.reset(a.seed, layout=a.layout);env.frames.clear();env.capture()
        start=float(env.data.time)
        initial=env.get_overhead_rgbd();Image.fromarray(initial['rgb']).save(a.output/'initial.png')
        with tempfile.TemporaryDirectory(prefix='vision-policy-') as tmp:
            directory=Path(tmp).resolve()
            shutil.copy(a.candidate,directory/'candidate.py')
            shutil.copy(Path(__file__).with_name('vision_policy_worker.py'),directory/'worker.py')
            cmd=policy_worker_command(directory,directory/'candidate.py')
            if sys.platform == 'darwin':
                (a.output/'sandbox.sb').write_text((directory/'policy.sb').read_text())
            else:
                (a.output/'sandbox.json').write_text(json.dumps({'isolation':'bwrap',
                    'network':'disabled','writable_paths':[str(directory)]},indent=2)+'\n')
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
                        if method in ('done', 'failed'):
                            cpu_profile = params.get('cpu_profile')
                        if method=='done':break
                        if method=='failed':raise RuntimeError(params['error'])
                        call_limit = 900 if hoop else 1800 if fasten else 300*len(env.order) if a.task in ('inventory_tower', 'inventory_structure') else 600
                        if len(calls)>call_limit:raise RuntimeError('API call limit exceeded')
                        try:
                            response=None
                            seconds=float(params.get('seconds',1.5))
                            if not np.isfinite(seconds) or not 0<=seconds<=10:raise ValueError('seconds must be 0..10')
                            if method=='camera':
                                view=params.get('view', 'side')
                                if view not in ('side', 'wrist'):
                                    raise ValueError('camera view must be side or wrist')
                                response=env.get_overhead_rgbd() if view=='side' else env.get_wrist_rgb()
                                if a.sensor_blackout:
                                    response['rgb'][:]=0
                                    if view=='side':
                                        response['depth_m'][:]=np.nan
                                        response['valid_depth'][:]=False
                                i=len(images);filename=f'camera_{i:03d}.npz'
                                np.savez_compressed(a.output/filename,**response)
                                Image.fromarray(response['rgb']).save(a.output/f'camera_{i:03d}.png')
                                images.append(filename)
                            elif method=='proprioception':
                                response={'joint_angles':env.data.qpos[env.arm_qpos_addresses].copy(),'tool_position':env.data.site_xpos[env.site_id].copy(),'tool_rotation':env.data.site_xmat[env.site_id].reshape(3,3).copy(),'time':float(env.data.time)}
                                calls[-1]['observation']={key: value.tolist() if isinstance(value,np.ndarray) else value for key,value in response.items()}
                            elif method=='task':
                                if a.task not in ('inventory_tower', 'inventory_structure') and not insertion:
                                    raise ValueError('Symbolic task descriptions require an inventory construction task')
                                response=env.public_goal() if insertion else env.recipe.public_goal()
                            elif method=='move_tool':move_tool(env,params['position'],max(.1,seconds),params.get('orientation'))
                            elif method=='plan_tool':response=plan_tool(env,params['position'],params.get('orientation'))
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
                            calls[-1]['completed']=True
                            proc.stdin.write(json.dumps({'result':pack(response)})+'\n');proc.stdin.flush()
                        except Exception as exc:
                            calls[-1]['error']=type(exc).__name__+': '+str(exc)
                            proc.stdin.write(json.dumps({'error':type(exc).__name__+': '+str(exc)})+'\n');proc.stdin.flush()
                except Exception as exc:
                    if isinstance(exc, RuntimeError) and str(exc).startswith('Policy process exited without completion'):
                        detail=(a.output/'worker.stderr').read_text()[-3000:]
                        raise RuntimeError(f'Vision worker failed before completion: {detail}') from exc
                    error=type(exc).__name__+': '+str(exc);env.failure=error
                finally:
                    proc.terminate()
                    try:proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    selector.close()
        policy_seconds=float(env.data.time)-start
        env.check_structure(list(env.order if inventory or a.task=='prism' else ORDER),duration=5.)
        metrics=env.evaluate()
        env.write_trace(metrics,error)
        frame=env.get_overhead_rgbd();Image.fromarray(frame['rgb']).save(a.output/'outcome.png')
        env.wait(1.)
        np.savez_compressed(a.output/'rollout.npz',states=env.frames,timestep=env.model.opt.timestep,state_type=int(env.state_type))
        (a.output/'api_calls.json').write_text(json.dumps(calls,indent=2)+'\n')
        result={'seed':a.seed,'layout':a.layout,'task':a.task,'success':metrics.success,'policy_error':error,'policy_sim_seconds':policy_seconds,
                'candidate_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),'camera_calls':len(images),
                'sensor_blackout':a.sensor_blackout,
                'cpu_profile':cpu_profile,'robot_ik_cpu_seconds':getattr(env, 'robot_ik_cpu_seconds', 0.),
                'metrics':asdict(metrics),'observation_contract':'Side RGB-D, wrist RGB, calibration + robot proprioception ONLY; evaluator results never returned to policy'}
        if inventory:
            result['scene'] = env.scene_metadata
        if insertion:
            result['goal'] = env.public_goal()
            result['physics_model'] = ('ordinary rigid-body contact and gravity; fixed pole; no attachments' if hoop else
                                     'solver-resolved helical surrogate with entry stop; fixed bolts; not resolved thread-face contact')
        if a.task in ('inventory_tower', 'inventory_structure'):
            result['goal'] = env.recipe.public_goal()
            result['sensor'] = {'depth_sampling': 'pixel_center', 'offscreen_samples': 0}
            result['api_call_limit'] = 300*len(env.order)
            result['observation_contract'] += '; public symbolic construction goal via task()'
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        # Only this reduced result and rendered sensor frames go back to the actor.
        feedback={'success':metrics.success,'policy_error':error,'policy_sim_seconds':policy_seconds,
                  'notes':[c['params']['text'] for c in calls if c['method']=='note']}
        if insertion:
            feedback['post_trial_metrics']=asdict(metrics)
            feedback['rpc_errors']=[c for c in calls if 'error' in c][-5:]
        (a.output/'actor_feedback.json').write_text(json.dumps(feedback,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:env.close()

if __name__=='__main__':main()
