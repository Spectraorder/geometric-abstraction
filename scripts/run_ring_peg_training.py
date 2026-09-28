"""One Full policy chain: five new training layouts and three model proposals."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import geometry_actor_session as actor
import summarize_geometry_session_resources as resources
from verified_token_usage import turn_usage
from run_ring_peg_trial import validate_policy

ROOT=Path(__file__).resolve().parents[1]
SEEDS=list(range(2030100101,2030100106))
STARTER="def run(api):\n    api.observe('ring')\n    api.observe('peg')\n"
CONTRACT='''Generate one executable CPU policy for placing a movable perforated ring over a fixed vertical peg.
Return JSON with hypothesis and complete Python code defining run(api). Only numpy,
scipy and math imports. No files, tools, repository, external skills, simulator
access, introspection, classes, global/nonlocal, attribute mutation or dynamic code.
Only imports, functions and docstrings at module scope; keep constants in functions.
Maximum source length 50000 characters. No model calls occur during a rollout.
Each rollout resets to its registered initial scene. You choose all observations,
actions, grasp location, approach, closing direction, wrist roll and release targets.
No task-solving starter or executor/perception source is provided. R0 only observes.

Public geometry assumptions: uniquely colored rigid ring, initially flat on a table,
with convex polygonal outer and inner boundaries and constant thickness. The
polygon is measured, not specified by vertex count. The cylindrical peg is fixed
in the world; its visually measured initial location can be reused under that
explicit assumption. The movable ring requires fresh observations after motion.
No metric object dimensions or current coordinates are supplied by the task goal.
All positions below are world meters, rotations radians, durations seconds.

API:
api.task(): goal, object identities and colors, declared shape assumptions only.
api.observe('ring'): current geometry from calibrated RGB-D, or RuntimeError if
  evidence is unavailable. Returns center_m, height_m, minimum_z_m,
  outer_boundary_m, walls (list of outer_mid_m and outward_normal), time,
  table_plane=[a,b,c] (table z=a*x+b*y+c), view, boundary_resolution_m,
  hole={center_m,axis,inscribed_radius_m,inner_boundary_m,evidence}.
  hole.center_m lies in the top cap plane, center_m is offset by half the initialized
  thickness along the normal. Walls' outer_mid_m is at mid-thickness.
  Partial updates register the initialized shape to current visible edges. Hidden
  points remain model-inferred; pixels have finite resolution, not exact geometry.
api.observe('peg'): top_center_m, axis, radius_m, time, table_plane and evidence.
  Observation may fail when the robot/ring occludes the peg.
api.robot(): fixed_pad_m, moving_pad_m, closing_axis (fixed toward moving pad),
  jaw_vertices_m (full collision geometry of both fingers), tool_position, roll,
  jaw, time. These are robot model/encoders, never contact or object measurements.
api.plan_grasp(fixed_pad_m, closing_direction, jaw): robot-only IK search for a roll
  aligning the fixed-to-moving pad axis with closing_direction. Returns roll,
  fixed_pad_m, moving_pad_m, jaw_vertices_m, tool_position, site_target_m,
  joint_angles and position_error_m. Does not move, choose an object wall, or check
  object collisions. The fixed_pad_m argument targets the CENTER of the fixed pad,
  not the tool marker, object center, or moving pad. Pad boxes have finite thickness.
api.plan_pad(fixed_pad_m, roll, jaw): same kinematic preview at a supplied roll.
api.move_pad(point, roll, seconds=1.5): move the fixed pad center to point at roll,
  using the current measured jaw angle. Returns fresh robot geometry. No automatic
  grasp, collision avoidance, lift, target correction or release is performed.
api.gripper(angle, seconds=.6): change the jaw angle and return robot geometry.
  Range [-.174,1.75]; larger opens, -.174 is the closed command. Choose the opening
  from robot kinematic previews and observed space; an opening may hit nearby walls.
api.wait(seconds): 0..10 seconds per call. api.note(text): bounded diagnostic log.
Roll in [-2.79,2.79], motion duration .1..10. IK can fail; not every target is reachable.
The robot is not rigidly attached to the ring: the object may tilt or slip after
closing. A jaw command does not certify a grasp. Verify progress from current vision.
Neither old object coordinates nor a commanded gripper displacement is a measurement.

Policy wall budget is 120 seconds, maximum 600 API calls. Private evaluation after
execution checks insertion, robot release and stability for two simulated seconds.
Only its binary outcome enters training; no truth coordinates, private contact
reasons or oracle traces enter the actor context. Handle natural failures within
an attempt; no policy reset operation exists. Prior images/geometry are historical.
Use concise diagnostics. Choose one falsifiable improvement per revision. Return
complete code. Training retention must preserve all incumbent successes and add
at least one; ties retain the incumbent. Failed proposals still consume a round.
'''


def read(path):return json.loads(path.read_text())
def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    paths=[ROOT/'scripts'/name for name in ('ring_peg_scene.py','ring_peg_geometry.py','ring_peg_robot.py',
        'ring_peg_runtime.py','ring_peg_policy_worker.py','run_ring_peg_trial.py','run_ring_peg_training.py',
        'geometry_actor_session.py','run_geometry_ablation.py','run_vision_trial.py',
        'verified_token_usage.py','summarize_geometry_session_resources.py','online_bridge_scene.py')]
    paths+=list((ROOT/'src/aspire_so100').glob('*.py'))+list((ROOT/'assets').rglob('*.xml'))
    paths+=[ROOT/'prototypes/polygon_prism/geometry.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def check(root):
    manifest=read(root/'manifest.json')
    for relative,sha in manifest['source_sha256'].items():
        if digest(ROOT/relative)!=sha or digest(root/'sources'/relative)!=sha:raise RuntimeError('Frozen source changed: '+relative)
    return manifest


def prepare(root):
    if root.exists():raise ValueError('Use a new output directory')
    proof=ROOT/'outputs/ring_peg_dev/sensor_physical_05'
    checks=[read(proof/f'seed_{seed}/result.json') for seed in (1,2,3)]
    if len(checks)!=3 or sum(row['success'] for row in checks)<2:
        raise ValueError('Require all three recorded development attempts and at least two sensor-driven successes')
    root.mkdir(parents=True)
    manifest={'schema':'ring_peg_single_policy_v1','registered_at':time.time(),'arms':['symbolic'],
        'actor_context_mode':'persistent','model':'gpt-6-astra','model_timeout_s':900,
        'max_candidate_characters':50000,'rounds':3,'train_seeds':SEEDS,'heldout_seeds':[],
        'training_slots':20,'model_calls':3,'rollout_workers':5,'source_sha256':sources(),
        'observation_contract':'Sensor-derived ring/peg geometry and public robot kinematics only.',
        'selection':'Preserve incumbent success set and strictly improve; ties retain incumbent.',
        'scope':'One Full training chain; no Base comparison or reserved success-rate estimate.',
        'development_outcome': 'Sensor-driven script succeeded on two of three development layouts; the failed grasp on seed 3 is retained.',
        'development_evidence':{str((proof/f'seed_{seed}/result.json').relative_to(ROOT)):digest(proof/f'seed_{seed}/result.json') for seed in (1,2,3)},
        'concurrency_rule':'Wait until the contract-only stacking training finishes before additional model requests; this study then uses one model call and five rollouts at a time.'}
    for relative in manifest['source_sha256']:
        p=root/'sources'/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,p)
    (root/'starter.py').write_text(STARTER);(root/'actor_contract.txt').write_text(CONTRACT)
    write(root/'manifest.json',manifest);write(root/'progress.json',{'status':'registered','training_slots':20,'model_calls':3})
    return manifest


def rollout(root,candidate,stage,seed):
    check(root);output=stage/'trials'/f'seed_{seed}'
    if output.exists():raise RuntimeError('Existing attempt retained; no automatic retry')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.log').open('x') as log:
        subprocess.run([sys.executable,'scripts/run_ring_peg_trial.py','--output',str(output),'--candidate',str(candidate),'--seed',str(seed)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180,
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl'))
    return read(output/'result.json')


def train(root):
    manifest=check(root)
    # Do not add a fifth concurrent model request to the existing registered study.
    other=ROOT/'outputs/geometry_cube_stack_contract_01'
    while other.exists() and not (other/'all_candidates_frozen.json').exists():
        lifecycle=read(other/'lifecycle.json') if (other/'lifecycle.json').exists() else {}
        if lifecycle.get('status')=='failed':raise RuntimeError('Other campaign stopped; inspect resource state before launch')
        write(root/'progress.json',{'status':'waiting_for_stacking_training','time':time.time()});time.sleep(10)
    rows=[];selected=0;successes=set();previous_feedback=None
    for revision in range(4):
        check(root)
        stage=root/'symbolic'/('revision_000' if revision==0 else f'round_{revision:03d}')
        if revision==0:
            stage.mkdir(parents=True);candidate=stage/'candidate.py';candidate.write_text(STARTER)
        else:
            prefix=CONTRACT if revision==1 else 'Continue the same isolated policy-training session. Return complete code.\n'
            incumbent=root/'symbolic'/('revision_000' if selected==0 else f'round_{selected:03d}')/'candidate.py'
            prompt=prefix+'\nTraining-only feedback:\n'+json.dumps(previous_feedback,separators=(',',':'))
            prompt+='\nRetained policy R'+str(selected)+':\n'+incumbent.read_text()
            if len(prompt)>100000:raise RuntimeError('Actor prompt exceeds registered budget')
            images=[]
            prev=root/'symbolic'/('revision_000' if revision==1 else f'round_{revision-1:03d}')
            for seed in SEEDS[:2]:
                paths=sorted((prev/'trials'/f'seed_{seed}').glob('camera_*_side.png'))
                if paths:images.append(paths[-1])
            write(root/'progress.json',{'status':'proposing','revision':revision,'completed_rollouts':len(rows)*5})
            actor.propose(stage,prompt,images,manifest)
            candidate=stage/'candidate.py'
            actor.audit_session(stage,manifest,read(stage/'invocation.json'),read(stage/'generation.json'))
        write(root/'progress.json',{'status':'training','revision':revision,'completed_rollouts':len(rows)*5})
        with ThreadPoolExecutor(max_workers=5) as pool:
            results=list(pool.map(lambda seed:rollout(root,candidate,stage,seed),SEEDS))
        current={row['seed'] for row in results if row['success']}
        if current>successes:successes=current;selected=revision
        rows.append({'revision':revision,'successes':len(current),'total':5,'selected_revision':selected,'retained_successes':len(successes)})
        previous_feedback=[]
        for seed in SEEDS:
            path=stage/'trials'/f'seed_{seed}'/'actor_feedback.json'
            if path.exists():previous_feedback.append(read(path))
            else:
                result=read(path.with_name('result.json'));previous_feedback.append({k:result[k] for k in ('seed','success','status','error') if k in result})
        write(root/'training_curve.json',rows)
    selected_path=root/'symbolic'/('revision_000' if selected==0 else f'round_{selected:03d}')/'candidate.py'
    shutil.copyfile(selected_path,root/'symbolic/frozen.py')
    with patch.object(resources,'turn_usage',turn_usage):accounting=resources.summarize(root,3)
    write(root/'resource_accounting.json',accounting)
    check(root)
    write(root/'summary.json',{'status':'completed','curves':rows,'selected_revision':selected,
        'selected_sha256':digest(root/'symbolic/frozen.py'),'scope':manifest['scope']})
    write(root/'progress.json',{'status':'completed','training_slots':20,'model_calls':3,'selected_revision':selected})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=('prepare','train','check'),required=True);a=p.parse_args();root=a.output.resolve()
    try:
        if a.phase=='prepare':print(prepare(root)['schema'])
        elif a.phase=='check':print(check(root)['schema'])
        else:train(root)
    except BaseException as error:
        if root.exists():write(root/'STOP.json',{'error':type(error).__name__+': '+str(error),'time':time.time()})
        raise
