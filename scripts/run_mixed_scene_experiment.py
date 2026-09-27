"""Frozen three-arm agent policy evolution for a mixed-shape support graph.

Primary contrasts: basic surface PnP versus supplied numeric geometry/state;
then the same numeric estimator plus symbolic references/relations/goal operators.
No development planner or failure audit enters actor contexts.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image
from build_mixed_scene import build as build_scene
from run_geometry_ablation import propose, feedback
from run_vision_trial import validate_source

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('basic', 'numeric', 'symbolic')
LIMIT = 30000
FORBIDDEN = {'session', 'api', 'pnp', 'backend', 'ledger', 'records', 'models', 'held', 'plane', 'colors',
             'move_tool', 'move_joints', 'gripper', 'move', 'call', 'exchange', 'point_target', 'capture', 'masks', 'locate', 'set_observer',
             'current_observation', 'predictions', 'predict', 'predictions_enabled', 'latest_time',
             'grasp_fraction', 'release_clearance', 'position_tolerance', 'metric', 'objects', 'observer'}
COMMON = '''Improve a CPU high-level planner for a low-rise bridge support graph.
Return JSON with hypothesis and code defining run(api), plus optional numerical
helpers. Only numpy, scipy and math imports. No tools, repository, skills, files,
introspection, classes, attribute assignment, dynamic execution, or direct actuation.
Maximum planner source 30000 characters; aim below 14000. Invalid submissions consume
a revision and its training slots, with no human correction or free rewrite.

The public goal is api.task(): kind=support_graph, axes=world_xy,
length_unit={observed_cube: node_key}, nodes in a valid construction order.
Each node has key, color, shape, grid=[x,y] in measured reference-cube side units,
supports=[node_keys], and desired orientation. The required physical bridge has two
cube foundations with a gap, a flat thin-prism beam spanning both, and a cylinder
above the beam. Use the public graph, not fixed color roles. Color assignments are
permuted between scenes; colors are unique. Sources start on the table, with the
beam's broad face flat and long axis along world X. Catalog metric sizes and object
truth are unavailable. World sites, distances and object sizes must come from sensors.
The goal specifies relative locations, not a fixed world worksite.

Instantiate exactly Session(api, objects), where objects is the public list of
shape/color descriptors from the task. It learns initial dimensions on the table.
Do not redefine, subclass, tune or mutate Session or its internals. All actions use:
observe(color), table_z(xy), tool(), pick(color,transit_z=None), place(surface_xyz),
pick_and_place(color,surface_xyz). You choose object order, sensor-derived targets,
workspace and high-level recovery. Read returned geometry as immutable; copy first.
The common PnP motion loop source below is frozen. The observation backend varies
as specified for your condition; numerical and symbolic conditions share a backend.
A release status is not a task-success certificate. Unknown visual evidence is not
necessarily a physical failure. A closed gripper can remain held after an unconfirmed
placement; only the returned status/held fields establish the executor's own state.

All conditions also have snapshot() -> refresh all objects and return state(),
state() -> JSON-compatible current estimates/availability with meter units, and
get(color) -> last estimate ONLY if its timestamp is current. After motion, refresh
explicitly. Stale or unavailable geometry must not be used as a fresh measurement.
Runtime RGB-D is always available through api.camera('side' or 'wrist'), with rgb,
depth_m (optical-axis Z), valid_depth, intrinsics, camera_to_world and simulated time.
Deprojection: world=R @ (inv(K) @ [u,v,1]*depth[v,u])+t. No segmentation or object truth.
api.proprioception() returns robot-only tool_position, tool_rotation, joint_angles,time.
api.plan_tool(position,orientation=None) returns robot-only IK tool_position,
tool_rotation, joint_angles and position_error, with no collision test or motion.
api.gripper_geometry() returns robot-model/encoder pad geometry, not object contacts.
api.wait(seconds), note(text), record_geometry(JSON-compatible dict) are available.
No direct move_tool/move_joints/gripper calls. Session internal attributes are forbidden.
Record {'kind':'planner_state','label':stage,'state':session.snapshot()} at meaningful
stages so supplied state use is auditable. It changes no task or motion commands.

Host trial timeout is 120 seconds, total API cap 1200; simulated time is NOT wall time.
Rollout duration is not an objective. GPU simulation is allowed; policies run on CPU.
Feedback includes final binary success, execution errors and your own notes. Private
poses, contacts, stability traces and evaluator reasons are never included. Initial
and final SIDE RGB/depth for two training trials are attached, preferring one failure
and one success; same selection rule in every condition. Depth images: black=0 m,
white=2 m, magenta=invalid. Full numerical sensors remain available online.
'''
BASIC = '''Your supplied observer is the historical RGB-D visible-top estimator.
observe returns center, normal, height, dims[2], top, bottom, extent_z, model and source.
Hidden extents are inferred. No supplied rigid vertices, symbolic predicates, goal
operators, temporal prediction or persistent geometric pose estimate is available.
You may implement additional perception or numerical reasoning yourself within the
same planner budget. State packets contain the same basic surface measurements.
'''
NUMERIC = '''Your supplied CPU observer fits rigid cubes, boxes and cylinders from
RGB-D with measured dimensions. It tries side then same-time wrist depth when fitting
is unavailable, excludes known robot marker geometry, and can resolve box length-axis
correspondence from current surfaces. There is no temporal object-pose prior.
observe/get additionally return shape, rotation[3,3], axis_dimensions[3], vertices,
constrained_axes[3], inferred_orientation, position_fully_observed, observation_time,
and source/view. Some fits include finite_center_bounds_m and finite_center_width_m
in the fitted basis (center @ rotation), conditional on the measured model. Unobserved
coordinates/vertices remain model estimates. Cylinders use sampled rims, not corners.
A model's initial height is not necessarily the current upward face extent after rotation.
state() serializes these same values and changes between estimates. history(color)
returns historical observations only; they never become current automatically.
target(color,desired_center,rotation=None) stores a separate desired pose, with no action.
goal_error(color) reports current estimated translation to that fixed target.
Temporal predictions and temporal pose fusion are disabled. No contact is certified.
'''
SYMBOLIC = '''Additional symbolic operators over the SAME numerical observations:
reference('id.center') -> point(position_m); top_face/bottom_face -> plane_patch
(center_m,normal,vertices_m) or cylinder disk(center_m,normal,radius_m,rim_samples_m).
long_axis/short_axis require unique box lengths; cylinder axis is unoriented.
corner_0..7 are inferred box corners, not tracked material points. Shapes retain
symmetry semantics. References require CURRENT evidence and use world meters.
relation(predicate,first,second,tolerance_m=.004) -> truth true/false/unknown plus
numeric residuals. Predicates: above, aligned_xy, support_candidate, contact.
Support_candidate reports surface gap and projected overlap, not certified physical
support or stability. A missing component remains unknown; a known false component
can reject a conjunction. Contact is always unknown without a contact sensor.
Do not wait for contact=true or require every estimate to be fully observed to act.
align(source_face_reference,target_point,target_normal) -> a desired rigid transform,
desired_center_m, desired_rotation and unconstrained_twist. Target normal is the
wanted OUTWARD normal of the source face. This does not choose supports, order,
workspace, grasp or actions, and does not change observations or targets automatically.
One point and one axis leave twist free. To record a goal explicitly, call target.
state() includes these derived references/facts in addition to identical numeric state.
text() gives a compact textual summary. No stronger controller or sensing is supplied.
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def sources():
    paths = list((ROOT/'src/aspire_so100').glob('*.py'))
    paths += [ROOT/'view_scene.py']
    paths += [ROOT/'policies'/name for name in ('pick_place_rgbd_shared.py', 'pick_place_rgbd_observed.py',
              'mixed_scene_session.py', 'basic_scene_session.py', 'support_graph_starter.py')]
    paths += [ROOT/'scripts'/name for name in ('run_mixed_scene_experiment.py', 'build_mixed_scene.py',
              'build_mixed_pnp_observer.py', 'run_geometry_ablation.py', 'run_vision_trial.py', 'vision_policy_worker.py')]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def library(arm):
    if arm == 'basic':
        tree = ast.parse((ROOT/'src/aspire_so100/metric_scene.py').read_text())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'metric_copy')
        source = ('import numpy as np\n'+ast.unparse(helper)+'\n'+(ROOT/'policies/pick_place_rgbd_shared.py').read_text()
                  +'\n'+(ROOT/'policies/basic_scene_session.py').read_text())
        kind = 'BasicSession'
    else:
        source = build_scene(arm == 'symbolic')
        kind = 'SymbolicSession' if arm == 'symbolic' else 'MixedSession'
    return source+'\nclass SessionFactory:\n    __slots__ = ()\n    session = staticmethod('+kind+')\nGEOMETRY = SessionFactory()\n'


def validate_planner(source):
    validate_source(source)
    if len(source) > LIMIT:
        raise ValueError('Planner exceeds source budget')
    tree = ast.parse(source)
    if not any(isinstance(n, ast.FunctionDef) and n.name == 'run' for n in tree.body):
        raise ValueError('Require run(api)')
    # Only the documented constructor exists in the actor's public namespace.
    # Backend implementation names live in a separate module; numerical helpers
    # cannot overwrite them. The injected factory's .session is not a public API.
    reserved = {'Session'}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)) and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            raise ValueError('Only functions, imports and docstrings at module level')
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.Global, ast.Nonlocal)):
            raise ValueError('No class or namespace mutation')
        if isinstance(node, ast.Attribute) and (node.attr in FORBIDDEN or isinstance(node.ctx, (ast.Store, ast.Del))):
            raise ValueError('Forbidden internal/mutable/actuation attribute: '+node.attr)
        if isinstance(node, ast.Name) and node.id in reserved and node.id != 'Session':
            raise ValueError('Use only the public Session constructor')
        name = node.id if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) else node.name if isinstance(node, ast.FunctionDef) else node.arg if isinstance(node, ast.arg) else (node.asname or node.name) if isinstance(node, ast.alias) else None
        if name in reserved:
            raise ValueError('Frozen name cannot be rebound or shadowed')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Session':
            if len(node.args) != 2 or node.keywords or not isinstance(node.args[0], ast.Name) or node.args[0].id != 'api':
                raise ValueError('Session construction must be Session(api, objects)')
    return tree


def color_map(seed):
    # A declared deterministic cyclic permutation; recognition remains unique-color.
    colors = ['blue', 'yellow', 'green', 'red']; offset = int(seed)%4
    return {color: colors[(index+offset)%4] for index, color in enumerate(colors)}


def prepare(root, rounds=3, train=None, heldout=None, arms=None):
    arms = tuple(ARMS if arms is None else arms)
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("Require distinct supported experiment arms")
    train = list(range(76001, 76006)) if train is None else train
    heldout = list(range(77001, 77021)) if heldout is None else heldout
    if rounds < 1 or not train or not heldout or len(set(train+heldout)) != len(train+heldout):
        raise ValueError('Distinct nonempty training/reserved splits required')
    validate_planner((ROOT/'policies/support_graph_starter.py').read_text())
    root.mkdir(parents=True, exist_ok=False)
    frozen = {}
    for arm in arms:
        source = library(arm); validate_source(source)
        (root/(arm+'_library.py')).write_text(source); frozen[arm] = digest(root/(arm+'_library.py'))
    shutil.copyfile(ROOT/'policies/support_graph_starter.py', root/'starter.py')
    shutil.copyfile(ROOT/'policies/pick_place_rgbd_shared.py', root/'executor.py')
    manifest = {'schema': 'mixed_scene_learning_pilot_v1', 'registered_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'minimum_free_disk_bytes': 3 * 1024**3, 'arms': list(arms), 'rounds': rounds,
        'train_seeds': train, 'heldout_seeds': heldout, 'color_maps': {str(s): color_map(s) for s in train+heldout},
        'task': 'inventory_structure', 'recipe': 'thin_beam_bridge', 'object_count': 4,
        'model': 'gpt-6-astra', 'model_timeout_s': 900, 'max_candidate_characters': LIMIT,
        'max_prompt_characters': 120000, 'library_sha256': frozen, 'starter_sha256': digest(root/'starter.py'),
        'executor_sha256': digest(root/'executor.py'), 'source_sha256': sources(),
        'training_threshold': .8, 'selection': 'highest training success, earliest revision breaks ties',
        'all_arms_frozen_before_reserved': True, 'all_revisions_evaluated_after_freeze': True,
        'primary_metrics': ['reserved success at each fixed revision', 'first reserved revision reaching 80 percent',
                            'mean reserved success over revisions 0 through 3', 'training-selected candidate reserved success'],
        'reasoning_effort': 'high', 'physics': {'noslip_iterations': 0},
        'sensor': {'depth_sampling': 'pixel_center', 'offscreen_samples': 0}, 'wrist_depth': True,
        'temporal_pose_fusion': False, 'conditional_prediction': False,
        'contrasts': ['basic versus numeric: supplied perception/state package', 'numeric versus symbolic: references, relations, goal operators and their text feedback'],
        'feedback': 'own outcomes/notes, same example-selection rule; up to initial/final own state packets from those examples; no private truth',
        'limitations': ['Pilot: one independent model-generation chain per arm, requiring replication.',
            'Equal revisions and rollout slots; token usage recorded, not equalized.',
            'Basic and numeric differ in observation implementation; only numeric versus symbolic isolates the added reasoning interface.',
            'No development planner, inspected failure audit or reserved feedback is supplied to actors.',
            'Earlier negative cube comparisons and development results remain separate evidence.']}
    write(root/'manifest.json', manifest)
    for relative, sha in manifest['source_sha256'].items():
        target = root/'source_snapshot'/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
        if digest(target) != sha:
            raise RuntimeError('Source changed during freeze')
    return manifest


def assert_frozen(root, manifest):
    if sources() != manifest['source_sha256']:
        raise RuntimeError('Implementation changed after freeze')
    for arm, sha in manifest['library_sha256'].items():
        if digest(root/(arm+'_library.py')) != sha:
            raise RuntimeError('Library changed after freeze')
    if digest(root/'starter.py') != manifest['starter_sha256'] or digest(root/'executor.py') != manifest['executor_sha256']:
        raise RuntimeError('Starter or executor changed')


def bundle(root, arm, planner, target):
    source = 'Session = geometry.session\n'+planner
    if target.exists() and target.read_text() != source:
        raise RuntimeError('Existing candidate bundle differs')
    target.parent.mkdir(parents=True, exist_ok=True); target.write_text(source)
    return target


def execute(root, arm, candidate, directory, seeds, manifest, error=None):
    rows = []
    for seed in seeds:
        assert_frozen(root, manifest); out = directory/f'seed_{seed}'
        if not (out/'result.json').exists():
            if out.exists():
                raise RuntimeError('Incomplete trial retained: '+str(out))
            if error:
                out.mkdir(parents=True)
                value = {'seed': seed, 'success': False, 'policy_error': error, 'invalid_candidate': True,
                         'rollout_executed': False, 'candidate_sha256': digest(candidate), 'notes': []}
                write(out/'result.json', value); write(out/'actor_feedback.json', value)
            else:
                if shutil.disk_usage(root).free < manifest['minimum_free_disk_bytes']:
                    raise RuntimeError('Insufficient free disk; completed trials preserved for resumption')
                out.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(ROOT/'scripts/run_vision_trial.py'), '--candidate', str(candidate),
                    '--output', str(out), '--seed', str(seed), '--task', 'inventory_structure', '--layout', 'sampled',
                    '--structure-recipe', manifest['recipe'], '--structure-color-map', json.dumps(manifest['color_maps'][str(seed)]),
                    '--object-count', '4', '--wrist-depth', '--record-geometry', '--timeout', '120',
                    '--geometry-library', str(root/(arm+'_library.py'))]
                env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='egl')
                with out.with_suffix('.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
        result = json.loads((out/'result.json').read_text())
        if result['seed'] != seed or result['candidate_sha256'] != digest(candidate):
            raise RuntimeError('Trial source/seed mismatch')
        if not result.get('invalid_candidate'):
            if (result['geometry_library_sha256'] != manifest['library_sha256'][arm]
                    or result['task'] != manifest['task'] or result['scene']['recipe'] != manifest['recipe']
                    or result['scene']['color_map'] != manifest['color_maps'][str(seed)] or result['scene']['object_count'] != 4
                    or result['physics'] != manifest['physics'] or result['sensor'] != manifest['sensor']
                    or result['sensor_blackout'] or not result['wrist_depth']):
                raise RuntimeError('Trial settings differ from protocol')
        row = {key: result.get(key) for key in ('seed', 'success', 'policy_error', 'camera_calls', 'cpu_profile')}
        row['rollout_executed'] = not result.get('invalid_candidate', False); rows.append(row)
        print(str(directory.relative_to(root))+' seed='+str(seed)+' success='+str(row['success'])+' error='+str(row['policy_error']), flush=True)
    return rows


def sensor_context(trial):
    calls = [call for call in json.loads((trial/'api_calls.json').read_text()) if call['method'] == 'camera']
    files = sorted(trial.glob('camera_*.npz'))
    if len(calls) != len(files):
        raise ValueError('Camera metadata mismatch')
    side = [(p, c) for p, c in zip(files, calls) if c['params']['view'] == 'side']
    if not side:
        return [], []
    chosen = [side[0]] if len(side) == 1 else [side[0], side[-1]]
    images = []; metadata = []
    for path, call in chosen:
        rgb = path.with_suffix('.png'); depth_path = path.with_name(path.stem+'_actor_depth.png')
        with np.load(path) as sensor:
            depth = sensor['depth_m']; valid = sensor['valid_depth'] & np.isfinite(depth)
            pixels = np.repeat(np.rint(np.clip(np.nan_to_num(depth)/2, 0, 1)*255).astype(np.uint8)[..., None], 3, axis=-1)
            pixels[~valid] = [255, 0, 255]
            metadata.append({'time': float(sensor['time']), 'intrinsics': sensor['intrinsics'].tolist(),
                             'camera_to_world': sensor['camera_to_world'].tolist()})
        if not depth_path.exists():
            Image.fromarray(pixels).save(depth_path)
        images.extend([rgb, depth_path])
    return images, metadata


def context(root, arm, revision, manifest):
    prompt = COMMON+({'basic': BASIC, 'numeric': NUMERIC, 'symbolic': NUMERIC+SYMBOLIC}[arm])
    prompt += '\nFrozen common motion loop (the supplied Session delegates to this; do not reproduce it):\n'+(root/'executor.py').read_text()
    prompt += '\nIdentical starter planner:\n'+(root/'starter.py').read_text()
    for previous in range(revision):
        directory = root/arm/('revision_000' if previous == 0 else f'round_{previous:03d}')
        prompt += '\nOwn revision '+str(previous)+' feedback:\n'+json.dumps([feedback(directory/'trials'/f'seed_{s}') for s in manifest['train_seeds']])
        if previous:
            response = json.loads((directory/'response.json').read_text()); prompt += '\nOwn hypothesis: '+response['hypothesis'][:2000]
    if revision > 1:
        prompt += '\nLatest exact planner:\n'+(root/arm/f'round_{revision-1:03d}'/'candidate.py').read_text()
    examples = []
    for seed in manifest['train_seeds']:
        for previous in range(revision-1, -1, -1):
            directory = root/arm/('revision_000' if previous == 0 else f'round_{previous:03d}')/'trials'/f'seed_{seed}'
            if (directory/'api_calls.json').exists():
                examples.append((seed, previous, directory, feedback(directory)['success'])); break
    chosen = []
    for success in (False, True):
        eligible = [row for row in examples if row[3] == success]
        if eligible: chosen.append(eligible[0])
    for row in examples:
        if len(chosen) >= 2: break
        if row not in chosen: chosen.append(row)
    images = []
    for _, previous, directory, success in chosen:
        attached, metadata = sensor_context(directory); images.extend(attached)
        prompt += '\nTraining example from revision '+str(previous)+', success='+str(success)+'. First/last side RGB then depth, calibration:\n'+json.dumps(metadata)
        path = directory/'geometry_records.json'
        if path.exists():
            states = [r for r in json.loads(path.read_text()) if r.get('kind') == 'planner_state']
            if states:
                selected = [states[0]] if len(states) == 1 else [states[0], states[-1]]
                # The actor gets only its own supplied sensor-derived packet.
                prompt += '\nOwn initial/final state packets (estimates; not contact or success certificates):\n'+json.dumps(selected)
    prompt += '\nReturn the next planner. No manual policy fixes will be inserted.'
    if len(prompt) > manifest['max_prompt_characters']:
        raise RuntimeError('Prompt budget exceeded')
    return prompt, images


def campaign(root, manifest, phase='all'):
    arms = tuple(manifest['arms'])
    assert_frozen(root, manifest)
    if phase == 'all' and (root/'training_summary.json').exists():
        phase = 'evaluate'
    if phase != 'evaluate':
        curves = {arm: [] for arm in arms}
        for arm in arms:
            write(root/'progress.json', {'status': 'bootstrap', 'arm': arm, 'curves': curves})
            directory = root/arm/'revision_000'; target = bundle(root, arm, (root/'starter.py').read_text(), directory/'bundled.py')
            rows = execute(root, arm, target, directory/'trials', manifest['train_seeds'], manifest)
            curves[arm].append({'revision': 0, 'successes': sum(r['success'] for r in rows), 'trials': len(rows), 'results': rows})
        for revision in range(1, manifest['rounds']+1):
            assert_frozen(root, manifest)
            write(root/'progress.json', {'status': 'proposing', 'revision': revision, 'curves': curves})
            # Actor subprocesses are isolated experimental subjects; they have no tools.
            with ThreadPoolExecutor(max_workers=len(arms)) as pool:
                pending = []
                for arm in arms:
                    directory = root/arm/f'round_{revision:03d}'
                    if not (directory/'generation.json').exists():
                        prompt, images = context(root, arm, revision, manifest)
                        pending.append(pool.submit(propose, directory, prompt, images, manifest))
                for future in pending: future.result()
            for arm in (arms if revision%2 else tuple(reversed(arms))):
                directory = root/arm/f'round_{revision:03d}'; source = (directory/'candidate.py').read_text()
                generation = json.loads((directory/'generation.json').read_text())
                if digest(directory/'candidate.py') != generation['candidate_sha256']:
                    raise RuntimeError('Actor submission changed')
                error = None
                try: validate_planner(source)
                except (ValueError, SyntaxError) as exc: error = type(exc).__name__+': '+str(exc)
                target = bundle(root, arm, source, directory/'bundled.py')
                write(root/'progress.json', {'status': 'executing', 'arm': arm, 'revision': revision, 'curves': curves})
                rows = execute(root, arm, target, directory/'trials', manifest['train_seeds'], manifest, error)
                curves[arm].append({'revision': revision, 'successes': sum(r['success'] for r in rows),
                    'trials': len(rows), 'results': rows, 'generation': generation, 'validation_error': error})
                write(root/'progress.json', {'status': 'training', 'curves': curves})
        frozen = {}; revision_frozen = {}
        for arm in arms:
            best = max(curves[arm], key=lambda row: row['successes']); revision = best['revision']
            source = root/arm/('revision_000' if revision == 0 else f'round_{revision:03d}')/'bundled.py'
            target = root/arm/'frozen.py'; shutil.copyfile(source, target)
            frozen[arm] = {'revision': revision, 'sha256': digest(target)}
            revision_frozen[arm] = []
            for row in curves[arm]:
                index = row['revision']
                original = root/arm/('revision_000' if index == 0 else f'round_{index:03d}')/'bundled.py'
                archived = root/arm/'frozen_revisions'/f'revision_{index:03d}.py'
                archived.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(original, archived)
                revision_frozen[arm].append({'revision': index, 'sha256': digest(archived),
                                            'validation_error': row.get('validation_error')})
        trained = {'status': 'trained', 'frozen_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'manifest_sha256': digest(root/'manifest.json'), 'curves': curves, 'frozen': frozen, 'revision_frozen': revision_frozen}
        write(root/'training_summary.json', trained)
        write(root/'all_candidates_frozen.json', {'selected': frozen, 'revisions': revision_frozen})
        if phase == 'train':
            write(root/'progress.json', trained); return
    else:
        trained = json.loads((root/'training_summary.json').read_text()); curves = trained['curves']; frozen = trained['frozen']; revision_frozen = trained['revision_frozen']
        if trained['manifest_sha256'] != digest(root/'manifest.json'):
            raise RuntimeError('Training manifest changed')
    if json.loads((root/'all_candidates_frozen.json').read_text()) != {'selected': frozen, 'revisions': revision_frozen}:
        raise RuntimeError('Global freeze gate differs from training record')
    heldout = {}; heldout_curves = {arm: [] for arm in arms}
    for arm in arms:
        for entry in revision_frozen[arm]:
            if digest(root/arm/'frozen_revisions'/f"revision_{entry['revision']:03d}.py") != entry['sha256']:
                raise RuntimeError('Frozen revision changed')
        if digest(root/arm/'frozen.py') != frozen[arm]['sha256']:
            raise RuntimeError('Selected candidate changed')
    for arm in arms:
        write(root/'progress.json', {'status': 'heldout', 'arm': arm, 'curves': curves, 'frozen': frozen})
        heldout[arm] = execute(root, arm, root/arm/'frozen.py', root/arm/'heldout', manifest['heldout_seeds'], manifest)
    for revision in range(manifest['rounds']+1):
        for arm in arms:
            entry = revision_frozen[arm][revision]
            if revision == frozen[arm]['revision']:
                rows = heldout[arm]; trial_path = arm+'/heldout'
            else:
                directory = root/arm/'reserved_revisions'/f'revision_{revision:03d}'/'heldout'
                trial_path = str(directory.relative_to(root))
                write(root/'progress.json', {'status': 'heldout_curves', 'arm': arm, 'revision': revision,
                                             'curves': curves, 'heldout_curves': heldout_curves, 'frozen': frozen})
                rows = execute(root, arm, root/arm/'frozen_revisions'/f'revision_{revision:03d}.py',
                               directory, manifest['heldout_seeds'], manifest, entry['validation_error'])
            heldout_curves[arm].append({'revision': revision, 'successes': sum(r['success'] for r in rows),
                                       'trials': len(rows), 'trial_path': trial_path, 'results': rows})
    summary = {'status': 'completed', 'manifest_sha256': digest(root/'manifest.json'), 'curves': curves,
               'frozen': frozen, 'revision_frozen': revision_frozen, 'heldout': heldout, 'heldout_curves': heldout_curves, 'limitations': manifest['limitations']}
    write(root/'summary.json', summary); write(root/'progress.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--phase', choices=('all', 'train', 'evaluate'), default='all')
    parser.add_argument('--arms', nargs='+', choices=ARMS)
    parser.add_argument('--rounds', type=int, default=3); parser.add_argument('--train-seeds', type=int, nargs='+')
    parser.add_argument('--heldout-seeds', type=int, nargs='+'); args = parser.parse_args(); root = args.output.resolve()
    if args.phase == 'evaluate' and not args.resume: parser.error('Evaluation requires frozen training')
    manifest = json.loads((root/'manifest.json').read_text()) if args.resume else prepare(root, args.rounds, args.train_seeds, args.heldout_seeds, args.arms)
    campaign(root, manifest, args.phase)
