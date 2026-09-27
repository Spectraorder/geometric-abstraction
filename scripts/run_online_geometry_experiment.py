"""Matched policy evolution with frozen execution and online symbolic/metric geometry.

This pilot measures an action-free online state interface plus derived feedback,
not temporal memory or controller improvements. Never expose private simulator traces to the actors.
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
from build_online_geometry import build as build_online_geometry

ARMS = ('raw', 'symbolic')
PLANNER_LIMIT = 30000
FORBIDDEN_ATTRIBUTES = {'api', 'move_tool', 'move_joints', 'gripper', 'move',
    'models', 'held', 'plane', 'grasp_fraction', 'release_clearance', 'position_tolerance',
    'capture', 'masks', 'locate', 'belief', 'dimension_models', 'identities',
    'table_coefficients', 'memory_enabled', 'errors', 'latest_time', 'predict',
    'current_fits','rejected_views','refresh_instantaneous','robot_exclusions'}
CONTRACT = '''Improve a CPU high-level planner for stacking three cubes. Return JSON
with hypothesis and code defining run(api), plus optional numerical helper functions.
The frozen RGBDPickPlace implementation below is ALREADY supplied at runtime. Return
ONLY the planner, do not repeat, replace, subclass, mutate or tune the primitive.
No tools, files, external knowledge of this repository, or other policies. Only numpy,
scipy and math imports. No private attributes, introspection or dynamic execution.
Maximum planner source: 30000 characters. Aim below 14000 characters; concise numerical
helpers are preferred. Invalid candidates consume a revision without a free rewrite.

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
No attribute assignment or class definitions. Do not redefine RGBDPickPlace, ndimage, or geometry.

Other available api methods: task(), camera('side' or 'wrist'), proprioception(),
plan_tool(position,orientation=None), gripper_geometry(), wait(seconds), note(text),
record_geometry(JSON-compatible-dict) for logging your sensor-derived state.
Camera returns rgb uint8, depth_m, valid_depth, intrinsics, camera_to_world,time.
Depth is optical-axis Z in meters; camera axes right/down/forward. Both cameras have
RGB-D; world=R @ (inv(K) @ [u,v,1]*depth[v,u])+t. Proprioception returns
robot-only tool_position/tool_rotation/joint_angles/time. plan_tool returns robot-only
IK tool_position/tool_rotation/position_error; it performs no motion/collision checks.
No simulator object state, dimensions or contacts. Host wall timeout is 120 seconds
and the RPC cap is 900. Proprioception time is SIMULATED time, not wall time; there
is no 120-second simulation-duration limit. Rollout duration is not an objective.

You receive initial and final side frames from two training trials, preferring one
failure and one success when both exist, each as RGB and
a linear grayscale depth image: black=0 m, white=2 m; magenta pixels invalid.
Identical frame selection and calibration are used in both experiment conditions.
Full raw numerical RGB-D remains available online via camera. Build sensor-driven
logic that generalizes to new layouts. Additional treatment access, if supplied,
is described below; otherwise implement any desired numerical perception yourself.
'''


RAW_CONTRACT = """
No supplied online geometry object/library is available. You may implement your
own RGB-D geometry, symbolic state, memory, predicates or planning in the planner.
The entire frozen PnP executor and shared sensors are identical in both groups.
"""
ONLINE_CONTRACT = """
ADDITIONAL ONLINE OBSERVATION ABSTRACTION (CPU, no actions or robot API access):
A read-only object `geometry` is already in globals. Do not import it, redefine it,
or reproduce its implementation. It only consumes frames you explicitly supply.
  scene = geometry.scene(api.camera('side'), colors)
Initial frame must show the upright cubes ON THE TABLE: rigid dimensions and the
static table plane are learned from it and retained. All object poses still come
from depth/color fits. It has no evaluator, contact, object truth, target-site
selection, grasping, retry, stability checking or motion execution.

scene.refresh(frame, view='side') -> JSON-compatible symbolic/metric packet.
After motion, explicitly refresh. Frames at a NEW timestamp discard older poses;
frames at the SAME timestamp can fuse complementary visible directions (e.g. side
and wrist). There is NO temporal pose memory or conditional prediction in this
interface. Missing objects remain unavailable; partial axes remain uncertain. Same-time
views undergo geometric consistency checks: incompatible weaker fits are rejected
and equally constrained contradictory fits cause abstention. The packet records
rejected_views with residuals; new weak views cannot silently overwrite strong fits.
scene.get(color) -> dict (world meters): center[3], rotation[3,3], height, normal[3],
  top, bottom, extent_z, dims[2], axis_dimensions[3], vertices[8,3],
  position_fully_observed, observed_translation_rank (0..3),
  observed_translation_projector[3,3], inferred_orientation, covariance_m2[3,3],
  source, age_s, last_observed. Returned arrays are independent copies.
  A usable dimension model does NOT imply a fully observed current pose. Do not
  use a partial center as a fully measured placement center. Hidden corners are
  inferred from the rigid model; covariance is heuristic, not calibrated certainty.
scene.reference('blue.top_face') -> plane_patch with center_m[3], normal[3],
  vertices_m[4,3], observation provenance. Also bottom_face, center, corner_0..7.
  Uniform cube corners are geometric labels, not identifiable material markings.
scene.relation(predicate, upper_color, lower_color) -> truth ('true','false',
  'unknown'), residuals (in millimeters), evidence/observability flags.
  Predicates: above, aligned_xy, support_candidate, contact. Support is only a
  geometric candidate. Contact remains unknown without a contact sensor.
scene.table_height(xy) -> measured table world Z in meters.
scene.target(color, desired_center, rotation=None) records a desired pose,
  separately from observation; packet gives target_delta_mm. It performs no action.
scene.packet(details=False) -> finite JSON dict, compact object/relationship state;
  details=True also includes inferred geometric landmarks. Text packet units mm.
scene.text() -> compact readable state. Numeric get/reference fields use meters.
Get/reference/relation may raise ValueError or KeyError for missing geometry.
Do not access internal fields or change attributes. Prediction is disabled.

OBSERVATION-ONLY EXAMPLE:
  scene = geometry.scene(api.camera('side'), colors)
  scene.refresh(api.camera('wrist'), 'wrist')
  api.record_geometry(scene.packet())
  observation = scene.get(colors[0])
  api.note(scene.text())
Record scene.packet() at meaningful planner stages so perception usage is auditable.
Reason about targets using the returned estimates; all actuation must still use
RGBDPickPlace with its unmodified defaults and implementation.
"""


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
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in {'RGBDPickPlace', 'ndimage', 'geometry'}:
            raise ValueError('Frozen executor name cannot be rebound')
        if isinstance(node, (ast.FunctionDef, ast.arg)) and getattr(node,'name',getattr(node,'arg',None)) in {'RGBDPickPlace','ndimage','geometry'}:
            raise ValueError('Frozen executor name cannot be shadowed')
        if isinstance(node, ast.alias) and (node.asname or node.name) in {'RGBDPickPlace','ndimage','geometry'}:
            raise ValueError('Frozen executor import cannot be shadowed')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'RGBDPickPlace':
            if len(node.args) != 1 or node.keywords or not isinstance(node.args[0],ast.Name) or node.args[0].id != 'api':
                raise ValueError('Executor construction must be RGBDPickPlace(api)')
    return tree


def prepare(root, rounds=3, train=None, heldout=None):
    train = list(range(71001,71006)) if train is None else list(train)
    heldout = list(range(72001,72031)) if heldout is None else list(heldout)
    if rounds < 1 or not train or not heldout or len(set(train+heldout)) != len(train+heldout):
        raise ValueError('Require positive rounds and distinct train/reserved seeds')
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(ROOT/'policies/pick_place_rgbd_shared.py',root/'executor.py')
    shutil.copyfile(ROOT/'policies/stack_from_rgbd_pnp_adapter.py',root/'starter.py')
    validate_planner((root/'starter.py').read_text())
    (root/'base_policy.py').write_text((root/'executor.py').read_text()+'\n'+(root/'starter.py').read_text())
    (root/'geometry.py').write_text(build_online_geometry())
    manifest = {'schema':'online_symbolic_geometry_pilot_v2','rounds':rounds,'arms':list(ARMS),
        'train_seeds':train,'heldout_seeds':heldout,'model':'gpt-6-astra','model_timeout_s':900,
        'max_candidate_characters':PLANNER_LIMIT,'max_prompt_characters':110000,
        'task':'inventory_tower','layout':'sampled','wrist_depth':True,'record_geometry':True,
        'physics':{'noslip_iterations':0},'sensor':{'depth_sampling':'pixel_center','offscreen_samples':0},
        'executor_sha256':digest(root/'executor.py'),'starter_sha256':digest(root/'starter.py'),
        'base_policy_sha256':digest(root/'base_policy.py'),'source_sha256':provenance(),
        'geometry_sha256':digest(root/'geometry.py'),
        'experiment_sha256':{name:digest(ROOT/'scripts'/name) for name in
            ['run_online_geometry_experiment.py','build_online_geometry.py','geometric_feedback.py']},
        'treatment':'online action-free instantaneous symbolic/metric state plus matched sensor-derived feedback',
        'runtime_library':'geometry.py','fusion':'same_time_consistency_gated_v2','temporal_memory':False,'conditional_predictions':False,
        'selection':'highest training success; earliest revision breaks ties; freeze both before reserved evaluation',
        'budget':'three revisions by default, equal rollout slots; tokens measured, not equalized',
        'training_threshold':.8,'primary_outcomes':['training_curve','iterations_to_threshold','selected_policy_reserved_success'],
        'limitations':['Pilot with one stochastic generation chain per condition.',
            'Tests online state plus derived feedback; temporal memory and prediction are disabled.',
            'Raw actor sees matched RGB-D feedback and full runtime RGB-D; treatment additionally has supplied numerical scene geometry and symbolic text.',
            'Shared execution was developed using different seeds; prior manual planner is not supplied.']}
    manifest['source_sha256']['view_scene.py']=digest(ROOT/'view_scene.py')
    write(root/'manifest.json',manifest)
    snapshots=dict(manifest['source_sha256'])
    snapshots.update({'scripts/'+name:sha for name,sha in manifest['experiment_sha256'].items()})
    for relative,sha in snapshots.items():
        target=root/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
        if digest(target)!=sha:raise RuntimeError('Source changed while preparing snapshot')
    return manifest


def assert_frozen(root, manifest):
    current_sources=provenance()
    current_sources['view_scene.py']=digest(ROOT/'view_scene.py')
    if current_sources != manifest['source_sha256']:
        raise RuntimeError('Simulation/perception source changed after freeze')
    for name, expected in manifest['experiment_sha256'].items():
        if digest(ROOT/'scripts'/name) != expected:
            raise RuntimeError('Experiment source changed after freeze: '+name)
    for name,key in [('executor.py','executor_sha256'),('starter.py','starter_sha256'),('base_policy.py','base_policy_sha256'),('geometry.py','geometry_sha256')]:
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
    prompt=CONTRACT+(ONLINE_CONTRACT if arm=='symbolic' else RAW_CONTRACT)+'\nFrozen executor source:\n'+(root/'executor.py').read_text()
    prompt+='\nInitial planner:\n'+(root/'starter.py').read_text()
    for previous in range(index):
        directory=root/'bootstrap' if previous==0 else root/arm/f'round_{previous:03d}'/'trials'
        prompt+=f'\nRevision {previous} feedback:\n'+json.dumps([feedback(directory/f'seed_{s}') for s in manifest['train_seeds']])
        if previous>0:
            response=json.loads((directory.parent/'response.json').read_text())
            prompt+='\nOwn hypothesis: '+response['hypothesis'][:2000]
    if index>1:
        prompt+='\nLatest planner:\n'+(root/arm/f'round_{index-1:03d}'/'candidate.py').read_text()
    records=[]
    for seed in manifest['train_seeds']:
        for previous in range(index-1,-1,-1):
            directory=root/'bootstrap' if previous==0 else root/arm/f'round_{previous:03d}'/'trials'
            trial_dir=directory/f'seed_{seed}'
            if (trial_dir/'api_calls.json').exists():
                records.append((seed,previous,trial_dir,feedback(trial_dir)['success']))
                break
    selected=[]
    for success in [False,True]:
        candidates=[row for row in records if row[3]==success]
        if candidates:selected.append(candidates[0])
    for row in records:
        if len(selected)>=2:break
        if row not in selected:selected.append(row)
    images=[]
    for layout,(seed,previous,trial_dir,success) in enumerate(selected):
        packet,frames,metadata=sensor_context(trial_dir)
        images.extend(frames)
        prompt+=f'\nTraining image example {layout+1}, revision {previous}, final success={success}: attached RGB then depth per frame; first and last recorded side observations. Metadata:\n'+json.dumps(metadata)
        if arm=='symbolic':
            prompt+='\nDerived symbolic/metric geometry (estimates, not contact or task-success certificates):\n'+json.dumps(actor_projection(packet))
    prompt+='\nReturn your next planner. No human patches or retries will be inserted.'
    if len(prompt)>manifest['max_prompt_characters']:
        raise RuntimeError('Prompt budget exceeded')
    return prompt,images


def execute(root,candidate,directory,seeds,manifest,error=None,library=None):
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
            value=trial(candidate,out,seed,runtime,library)
        results.append({key:value.get(key) for key in ['seed','success','policy_error','rollout_executed','failure_stage','camera_calls','cpu_profile']})
        print(f'{directory.relative_to(root)} seed={seed} success={value["success"]} error={value["policy_error"]}',flush=True)
    return results


def campaign(root,manifest,phase='all'):
    assert_frozen(root,manifest)
    if phase=='evaluate':
        trained=json.loads((root/'training_summary.json').read_text())
        if trained['manifest_sha256']!=digest(root/'manifest.json'):
            raise RuntimeError('Training manifest changed before reserved evaluation')
        return evaluate(root,manifest,trained['curves'],trained['frozen'])
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
            results=execute(root,target,directory/'trials',manifest['train_seeds'],manifest,error,
                root/'geometry.py' if arm=='symbolic' else None)
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
    trained={'status':'trained','manifest_sha256':digest(root/'manifest.json'),'curves':curves,'frozen':frozen}
    write(root/'training_summary.json',trained)
    write(root/'progress.json',trained)
    if phase=='train':return
    return evaluate(root,manifest,curves,frozen)


def evaluate(root,manifest,curves,frozen):
    assert_frozen(root,manifest)
    for arm in ARMS:
        if digest(root/arm/'frozen.py')!=frozen[arm]['sha256']:
            raise RuntimeError('Selected candidate changed before reserved evaluation')
    write(root/'progress.json',{'status':'heldout','curves':curves,'frozen':frozen})
    heldout={}
    for arm in ARMS:
        heldout[arm]=execute(root,root/arm/'frozen.py',root/arm/'heldout',manifest['heldout_seeds'],manifest,
            library=root/'geometry.py' if arm=='symbolic' else None)
    summary={'status':'completed','manifest_sha256':digest(root/'manifest.json'),'curves':curves,
        'frozen':frozen,'heldout':heldout,'limitations':manifest['limitations']}
    write(root/'summary.json',summary);write(root/'progress.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--phase',choices=['all','train','evaluate'],default='all')
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--train-seeds',type=int,nargs='+')
    parser.add_argument('--heldout-seeds',type=int,nargs='+')
    args=parser.parse_args();root=args.output.resolve()
    if args.phase=='evaluate' and not args.resume:parser.error('Evaluation requires --resume of frozen training')
    manifest=json.loads((root/'manifest.json').read_text()) if args.resume else prepare(root,args.rounds,args.train_seeds,args.heldout_seeds)
    campaign(root,manifest,args.phase)
