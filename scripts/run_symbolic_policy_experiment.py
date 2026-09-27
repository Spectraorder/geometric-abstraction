"""Matched policy evolution with frozen execution and sensor-derived text feedback.

This pilot measures the supplied feedback representation, not runtime memory or
controller improvements. Never expose private simulator traces to the actors.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
from geometric_feedback import make_feedback, actor_projection
from run_geometry_ablation import ROOT, digest, write, provenance, propose, trial, feedback
from run_vision_trial import validate_source

ARMS = ('raw', 'symbolic')
PLANNER_LIMIT = 14000
FORBIDDEN_ATTRIBUTES = {'api', 'move_tool', 'move_joints', 'gripper', 'move',
    'models', 'held', 'plane', 'grasp_fraction', 'release_clearance', 'position_tolerance',
    'capture', 'masks', 'locate'}
CONTRACT = '''Improve a CPU high-level planner for stacking three cubes. Return JSON
with hypothesis and code defining run(api), plus optional numerical helper functions.
The frozen RGBDPickPlace implementation below is ALREADY supplied at runtime. Return
ONLY the planner, do not repeat, replace, subclass, mutate or tune the primitive.
No tools, files, external knowledge of this repository, or other policies. Only numpy,
scipy and math imports. No private attributes, introspection or dynamic execution.
Maximum planner source: 14000 characters. Invalid candidates consume a revision.

Public task: api.task()['levels'] is bottom-to-top color order; blue, red, green.
Build an upright, released, stable column, supported on the table. The goal does not
specify a fixed world position. Object dimensions and all targets must come from
current sensor measurements, never fixed object coordinates or a seed lookup.
The evaluator's support/contact/stability checks are private. Only final success,
execution errors and your own notes are returned after each trial. No oracle online.

Frozen executor: instantiate exactly RGBDPickPlace(api), with no other arguments.
Use observe(color), table_z(xy), tool(), pick(color, transit_z=None), place(surface_xyz),
pick_and_place(color,surface_xyz). See exact source below for returned fields/statuses.
Read-only measurements returned by observe must not be mutated; copy arrays/dicts first.
Use only these public primitive methods; no access to internal attributes.
Direct robot actuation is prohibited: all grasp/motion/release uses this same executor.
You may choose object order, measurement-derived targets and high-level recovery.
No attribute assignment or class definitions. Do not redefine RGBDPickPlace or ndimage.

Other available api methods: task(), camera('side' or 'wrist'), proprioception(),
plan_tool(position,orientation=None), gripper_geometry(), wait(seconds), note(text).
Camera returns rgb uint8, depth_m, valid_depth, intrinsics, camera_to_world,time.
Depth is optical-axis Z in meters; camera axes right/down/forward. Both cameras have
RGB-D; world=R @ (inv(K) @ [u,v,1]*depth[v,u])+t. Proprioception returns
robot-only tool_position/tool_rotation/joint_angles/time. plan_tool returns robot-only
IK tool_position/tool_rotation/position_error; it performs no motion/collision checks.
No simulator object state, dimensions or contacts. Policy limit 120 sec / 900 RPCs.

You receive initial and final side frames from two training trials, each as RGB and
a linear grayscale depth image: black=0 m, white=2 m; magenta pixels invalid.
Identical frame selection and calibration are used in both experiment conditions.
Full raw numerical RGB-D remains available online via camera. Build sensor-driven
logic that generalizes to new layouts. Runtime geometry-library access is NOT supplied.
'''


def validate_planner(source):
    validate_source(source)
    if len(source) > PLANNER_LIMIT:
        raise ValueError('Planner source exceeds frozen budget')
    tree = ast.parse(source)
    if not any(isinstance(n, ast.FunctionDef) and n.name == 'run' for n in tree.body):
        raise ValueError('Planner must define run(api)')
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)) and not (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            raise ValueError('Only functions, numerical imports and docstrings allowed at module level')
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.Global)):
            raise ValueError('Executor replacement or namespace mutation prohibited')
        if isinstance(node, ast.Attribute) and (node.attr in FORBIDDEN_ATTRIBUTES or isinstance(node.ctx, (ast.Store, ast.Del))):
            raise ValueError('Forbidden mutable/internal/actuation attribute: '+node.attr)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in {'RGBDPickPlace', 'ndimage'}:
            raise ValueError('Frozen executor name cannot be rebound')
        if isinstance(node, (ast.FunctionDef, ast.arg)) and getattr(node,'name',getattr(node,'arg',None)) in {'RGBDPickPlace','ndimage'}:
            raise ValueError('Frozen executor name cannot be shadowed')
        if isinstance(node, ast.alias) and (node.asname or node.name) in {'RGBDPickPlace','ndimage'}:
            raise ValueError('Frozen executor import cannot be shadowed')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'RGBDPickPlace':
            if len(node.args) != 1 or node.keywords or not isinstance(node.args[0],ast.Name) or node.args[0].id != 'api':
                raise ValueError('Executor construction must be RGBDPickPlace(api)')
    return tree


def prepare(root, rounds=3, train=None, heldout=None):
    train = train or list(range(52001,52006)); heldout = heldout or list(range(53001,53011))
    if rounds < 1 or len(set(train+heldout)) != len(train+heldout):
        raise ValueError('Require positive rounds and distinct train/reserved seeds')
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(ROOT/'policies/pick_place_rgbd_shared.py',root/'executor.py')
    shutil.copyfile(ROOT/'policies/stack_from_rgbd_pnp_adapter.py',root/'starter.py')
    validate_planner((root/'starter.py').read_text())
    (root/'base_policy.py').write_text((root/'executor.py').read_text()+'\n'+(root/'starter.py').read_text())
    manifest = {'schema':'symbolic_feedback_pilot_v1','rounds':rounds,'arms':list(ARMS),
        'train_seeds':train,'heldout_seeds':heldout,'model':'gpt-6-astra','model_timeout_s':900,
        'max_candidate_characters':PLANNER_LIMIT,'max_prompt_characters':110000,
        'task':'inventory_tower','layout':'sampled','wrist_depth':True,
        'physics':{'noslip_iterations':0},'sensor':{'depth_sampling':'pixel_center','offscreen_samples':0},
        'executor_sha256':digest(root/'executor.py'),'starter_sha256':digest(root/'starter.py'),
        'base_policy_sha256':digest(root/'base_policy.py'),'source_sha256':provenance(),
        'experiment_sha256':{name:digest(ROOT/'scripts'/name) for name in
            ['run_symbolic_policy_experiment.py','geometric_feedback.py']},
        'treatment':'additional instantaneous symbolic/metric feedback from the same RGB-D frames',
        'runtime_library':None,'temporal_memory':False,'conditional_predictions':False,
        'selection':'highest training success; earliest revision breaks ties; freeze both before reserved evaluation',
        'budget':'three revisions by default, equal rollout slots; tokens measured, not equalized',
        'limitations':['Pilot with one stochastic generation chain per condition.',
            'Tests supplied feedback abstraction, not runtime library or full state-memory benefit.',
            'Raw actor sees depth visualization; symbolic actor additionally gets sensor-derived metric text.',
            'Shared execution was developed using different seeds; prior manual planner is not supplied.']}
    write(root/'manifest.json',manifest)
    return manifest


def assert_frozen(root, manifest):
    if provenance() != manifest['source_sha256']:
        raise RuntimeError('Simulation/perception source changed after freeze')
    for name, expected in manifest['experiment_sha256'].items():
        if digest(ROOT/'scripts'/name) != expected:
            raise RuntimeError('Experiment source changed after freeze: '+name)
    for name,key in [('executor.py','executor_sha256'),('starter.py','starter_sha256'),('base_policy.py','base_policy_sha256')]:
        if digest(root/name) != manifest[key]:
            raise RuntimeError('Frozen source changed: '+name)


def sensor_context(directory):
    path=directory/'symbolic_feedback.json'
    if not path.exists():
        packet=make_feedback(directory) if (directory/'api_calls.json').exists() else {'observations':[]}
        write(path,packet)
    packet=json.loads(path.read_text()); images=[]; metadata=[]
    for item in packet['observations']:
        rgb=Path(item['image']); depth=rgb.with_name(rgb.stem+'_depth.png')
        if not depth.exists():
            with np.load(item['sensor_file']) as sensor:
                d=sensor['depth_m']; valid=sensor['valid_depth'] & np.isfinite(d)
                gray=np.rint(np.clip(np.nan_to_num(d)/2,0,1)*255).astype(np.uint8)
                pixels=np.repeat(gray[...,None],3,axis=-1);pixels[~valid]=[255,0,255]
                Image.fromarray(pixels).save(depth)
        images.extend([rgb,depth]);metadata.append({'time':item['time'],'calibration':item['calibration']})
    return packet,images,metadata


def context(root,arm,index,manifest):
    prompt=CONTRACT+'\nFrozen executor source:\n'+(root/'executor.py').read_text()
    prompt+='\nInitial planner:\n'+(root/'starter.py').read_text()
    for previous in range(index):
        directory=root/'bootstrap' if previous==0 else root/arm/f'round_{previous:03d}'/'trials'
        prompt+=f'\nRevision {previous} feedback:\n'+json.dumps([feedback(directory/f'seed_{s}') for s in manifest['train_seeds']])
        if previous>0:
            response=json.loads((directory.parent/'response.json').read_text())
            prompt+='\nOwn hypothesis: '+response['hypothesis'][:2000]
    if index>1:
        prompt+='\nLatest planner:\n'+(root/arm/f'round_{index-1:03d}'/'candidate.py').read_text()
    latest=root/'bootstrap' if index==1 else root/arm/f'round_{index-1:03d}'/'trials'
    images=[]
    for layout,seed in enumerate(manifest['train_seeds'][:2]):
        packet,frames,metadata=sensor_context(latest/f'seed_{seed}')
        images.extend(frames)
        prompt+=f'\nTraining layout {layout+1}: attached RGB then depth per frame; first and last recorded side observations. Metadata:\n'+json.dumps(metadata)
        if arm=='symbolic':
            prompt+='\nDerived symbolic/metric geometry (estimates, not contact or task-success certificates):\n'+json.dumps(actor_projection(packet))
    prompt+='\nReturn your next planner. No human patches or retries will be inserted.'
    if len(prompt)>manifest['max_prompt_characters']:
        raise RuntimeError('Prompt budget exceeded')
    return prompt,images


def execute(root,candidate,directory,seeds,manifest,error=None):
    results=[]
    runtime=dict(manifest,max_candidate_characters=PLANNER_LIMIT+len((root/'executor.py').read_text())+100)
    for seed in seeds:
        assert_frozen(root,manifest)
        out=directory/f'seed_{seed}'
        if error:
            out.mkdir(parents=True,exist_ok=True)
            value={'success':False,'policy_error':error,'notes':[],'rollout_executed':False,'failure_stage':'candidate_validation','seed':seed}
            write(out/'actor_feedback.json',value);write(out/'result.json',value)
        else:
            value=trial(candidate,out,seed,runtime)
        results.append({key:value.get(key) for key in ['seed','success','policy_error','rollout_executed','failure_stage','camera_calls','cpu_profile']})
        print(f'{directory.relative_to(root)} seed={seed} success={value["success"]} error={value["policy_error"]}',flush=True)
    return results


def campaign(root,manifest):
    assert_frozen(root,manifest)
    curves={arm:[] for arm in ARMS}
    write(root/'progress.json',{'status':'bootstrap'})
    baseline=execute(root,root/'base_policy.py',root/'bootstrap',manifest['train_seeds'],manifest)
    entry={'revision':0,'successes':sum(r['success'] for r in baseline),'trials':len(baseline),'results':baseline}
    for arm in ARMS: curves[arm].append(entry.copy())
    for index in range(1,manifest['rounds']+1):
        assert_frozen(root,manifest)
        write(root/'progress.json',{'status':'proposing','revision':index,'curves':curves})
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending=[]
            for arm in ARMS:
                directory=root/arm/f'round_{index:03d}'
                if not (directory/'generation.json').exists():
                    prompt,images=context(root,arm,index,manifest)
                    pending.append(pool.submit(propose,directory,prompt,images,manifest))
            for future in pending: future.result()
        order=ARMS if index%2 else tuple(reversed(ARMS))
        for arm in order:
            directory=root/arm/f'round_{index:03d}'
            source=(directory/'candidate.py').read_text();generation=json.loads((directory/'generation.json').read_text())
            if digest(directory/'candidate.py') != generation['candidate_sha256']:
                raise RuntimeError('Actor candidate changed after submission')
            error=None
            try: validate_planner(source)
            except (SyntaxError,ValueError) as exc: error=f'{type(exc).__name__}: {exc}'
            bundle=(root/'executor.py').read_text()+'\n'+source
            target=directory/'bundled.py'
            if target.exists() and target.read_text()!=bundle: raise RuntimeError('Candidate bundle changed')
            target.write_text(bundle)
            write(root/'progress.json',{'status':'executing','arm':arm,'revision':index,'curves':curves})
            results=execute(root,target,directory/'trials',manifest['train_seeds'],manifest,error)
            curves[arm].append({'revision':index,'successes':sum(r['success'] for r in results),
                'trials':len(results),'results':results,'generation':generation,'validation_error':error})
            write(root/'progress.json',{'status':'training','curves':curves})
    frozen={}
    for arm in ARMS:
        best=max(curves[arm],key=lambda e:e['successes'])
        source=root/'base_policy.py' if best['revision']==0 else root/arm/f'round_{best["revision"]:03d}'/'bundled.py'
        target=root/arm/'frozen.py';shutil.copyfile(source,target)
        frozen[arm]={'revision':best['revision'],'sha256':digest(target)}
    write(root/'frozen.json',frozen)
    write(root/'progress.json',{'status':'heldout','curves':curves,'frozen':frozen})
    heldout={}
    for arm in ARMS:
        heldout[arm]=execute(root,root/arm/'frozen.py',root/arm/'heldout',manifest['heldout_seeds'],manifest)
    summary={'status':'completed','manifest_sha256':digest(root/'manifest.json'),'curves':curves,
        'frozen':frozen,'heldout':heldout,'limitations':manifest['limitations']}
    write(root/'summary.json',summary);write(root/'progress.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    manifest=json.loads((root/'manifest.json').read_text()) if args.resume else prepare(root)
    campaign(root,manifest)
