"""Controlled three-cube policy-generation pilot with a frozen geometry treatment."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('raw', 'geometry')
SCHEMA = {'type': 'object', 'properties': {'hypothesis': {'type': 'string'},
          'code': {'type': 'string'}}, 'required': ['hypothesis', 'code'], 'additionalProperties': False}
CONTRACT = '''You are writing a CPU policy for a new three-cube stacking experiment.
Use only this contract, attached sensor images, and your own campaign history below.
No tools, file access, network, prior repository, other policies, or external skills.
Return JSON with hypothesis and complete Python code defining run(api), not a patch.
Only numpy, scipy, math imports. No introspection, private attributes, file/process APIs,
printing, or dynamic code execution. Use api.note(text) for diagnostics.

TASK: stack three initially upright cubes, blue on the table, red on blue, green on red.
Initial positions vary. No distractors. Pick, carry, release and withdraw physically.
A private evaluator checks the full column: support contacts, uprightness, support margin,
release and low motion over a five-second final stability check. It never returns truth
online. Final success/failure is supplied only after a rollout. Intermediate grasp success
is NOT an API return value. No object sizes or poses are supplied: measure from sensors.
All object-dependent targets must derive from current observations, not seed/coordinates.

SHARED API (meters, seconds, radians):
api.camera('side') -> dict: rgb uint8[240,320,3], depth_m[240,320], valid_depth[240,320],
  intrinsics[3,3], camera_to_world[4,4], time. Depth is optical-axis z, not range.
  world = R @ (inv(K) @ [u,v,1] * depth[v,u]) + t; camera axes right/down/forward.
api.camera('wrist') -> rgb, intrinsics, camera_to_world, time; NO wrist depth.
api.task() -> {'kind':'single_column','levels':[{'color':..., 'shape':'cube'}, ...]}
  in bottom-to-top order. No object state/dimensions/contacts.
api.proprioception() -> joint_angles[5], tool_position[3], tool_rotation[3,3], time.
  This is robot forward kinematics; tool_position is an approximate gripper center,
  not a guaranteed fingertip or payload center. No gripper/object state oracle.
api.plan_tool(position, orientation=None) -> joint_angles, tool_position, tool_rotation,
  position_error. Robot-only IK preview; no movement, no collision/object check.
api.move_tool(position, seconds=1.5, orientation=None) -> None. Smooth joint motion.
  Five joints cannot independently satisfy every six-dimensional pose. Orientation is
  a soft preference: a position-reachable approach can have orientation residual.
  Log actual robot poses and distinguish approach tolerances from placement tolerances.
  If omitted,
  preferred tool rotation is [[0,0,1],[1,0,0],[0,1,0]]. Error >5 mm raises RuntimeError.
api.move_joints(five_angles, seconds=1.5) -> None. Joint limits:
  [-1.92,1.92],[-3.32,.174],[-.174,3.14],[-1.66,1.66],[-2.79,2.79].
api.gripper(angle, seconds=.6) -> None. Range [-.174,1.75], larger opens, .8 is open.
api.wait(seconds) -> None. api.note(text) -> None, max 2000 characters per note.
Motion durations 0.1..10 seconds; wait 0..10 per call. Trial wall limit 120 seconds,
900 total RPC calls. Simulation/rendering may use GPU; your policy uses CPU only.
No pointing/VLM calls at runtime. Handle occlusions and imperfect IK with sensor feedback.
You may create your own reusable perception and control functions within this campaign.

STARTER (identical in both groups):
def run(api):
    pass
'''
GEOMETRY_CONTRACT = '''
ADDITIONAL FROZEN OBSERVATION LIBRARY: a read-only object named geometry is already
in your Python globals; DO NOT import it or reproduce its implementation. It has NO
robot API access and performs NO motions, grasping, planning, retry, or stacking.
geometry.table(frame) -> plane[3], table z = a*x+b*y+c fitted from RGB-D.
geometry.cube(frame, color, model=None, table_plane=None, prior=None) -> dict:
  center[3], normal[3] (upward face normal), rotation[3,3], top (top-center world z),
  bottom (minimum corner z), extent_z (half vertical extent), height, dims[2],
  vertices[8,3], top_vertices[4,3], axis_dimensions[3], model (learned dimensions),
  model_valid (bool), geometry_reliable (bool), observed_face_count, constrained_axes[3],
  position_fully_observed (bool), position_source,
  inferred_orientation, inferred_axial_position, surface_rms_m.
  Initially model=None learns dimensions from an unobstructed ON-TABLE view.
  Retain that returned model for all later calls, including airborne/stacked objects.
  prior, when supplied, is a previous geometry dict with an estimated current center,
  normal and rotation; hidden translations may retain it. It is NOT a fresh observation.
  model_valid means a usable stored dimension measurement exists; geometry_reliable
  means the CURRENT full pose is supported by the fit. These are different. Retain a
  returned model even when geometry_reliable=False. Partial poses contain estimates
  along unobserved axes; do not treat these as observed positions for precise placement.
  With fewer than two visible faces, orientation is partly inferred; a center prior
  only updates translation, not orientation. Missing/unusable depth raises ValueError.
  Hidden corners are inferred. No collision or grasp success is certified.

OBSERVATION-ONLY EXAMPLE (no actions):
  frame = api.camera()
  plane = geometry.table(frame)
  observed = geometry.cube(frame, 'red', table_plane=plane)
  model = observed['model']  # Retain even when the current pose is only partial.
  # After another observation, reuse this dimension model; do not relearn in midair.
  later = geometry.cube(api.camera(), 'red', model=model, table_plane=plane)
  api.note(str({'faces': later['observed_face_count'],
                'position_observed': later['position_fully_observed']}))
geometry.gripper(frame, tool_position, support_z=None) -> dict:
  visible_points, tip_points, lowest_visible_z, fingertip_drop_m,
  observed_opening_m (or None), jaw_centers (or None), visible_clearance_m (or None),
  visible_only=True, pixel_count. Describes visible neutral-colored gripper surfaces,
  NOT a full collision model. May raise ValueError if too little depth is visible.
geometry.margin(point, top_vertices, normal) -> signed point-to-edge support margin.
  This is a geometric calculation, not evaluator/contact/stability information.
All inputs are your sensor frames or your estimates. No simulator state is accessible.
Do not modify the library. You may wrap it and implement your own controller.
'''
RAW_CONTRACT = '''
No prebuilt geometry library is supplied. You may implement any perception, geometry,
tracking, or controller using the shared observations and numerical libraries.
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def build_geometry_library(include_targets=False, include_state=False):
    """Export numerical observation functions, excluding all task controllers."""
    bodies = []
    selection = {
        'cube_geometry': {'rgbd_points', 'support_margin', 'estimate_cube_geometry'},
        'instance_geometry': {'inventory_color_mask', 'inventory_points', 'inventory_table'},
        'ablation_geometry': None,
        'gripper_geometry': None,
    }
    for module, keep in selection.items():
        tree = ast.parse((ROOT/'src/aspire_so100'/f'{module}.py').read_text())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('aspire_so100'):
                continue
            if module == 'instance_geometry' and isinstance(node, ast.ImportFrom):
                continue
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if keep is not None and node.name not in keep:
                    continue
                if node.name in {'select_column_levels', 'update_partial_column_geometry'}:
                    continue
            bodies.append(node)
    bodies.extend(ast.parse('''
color_mask = inventory_color_mask
class GeometryAPI:
    __slots__ = ()
    table = staticmethod(inventory_table)
    gripper = staticmethod(estimate_gripper_geometry)
    margin = staticmethod(support_margin)
    cube = staticmethod(observe_cube)
GEOMETRY = GeometryAPI()
''').body)
    if include_targets or include_state:
        tree = ast.parse((ROOT/'src/aspire_so100/rigid_state.py').read_text())
        bodies.extend(tree.body)
        bodies.extend(ast.parse('''
class StateGeometryAPI(GeometryAPI):
    __slots__ = ()
    targets = staticmethod(stack_targets)
    error = staticmethod(state_error)
    relation = staticmethod(support_relation)
GEOMETRY = StateGeometryAPI()
''').body)
    if include_state:
        tree = ast.parse((ROOT/'src/aspire_so100/geometric_state.py').read_text())
        bodies.extend(node for node in tree.body if not isinstance(node, ast.ImportFrom)
                      or not (node.module or '').startswith('aspire_so100'))
        tree = ast.parse((ROOT/'src/aspire_so100/symbolic_geometry.py').read_text())
        bodies.extend(node for node in tree.body if not isinstance(node, ast.ImportFrom)
                      or not (node.module or '').startswith('aspire_so100'))
        bodies.extend(ast.parse('\nclass MemoryGeometryAPI(StateGeometryAPI):\n    __slots__ = ()\n    state = staticmethod(GeometricState)\n    predicate = staticmethod(geometric_predicate)\n    symbols = staticmethod(symbolic_packet)\n    resolve = staticmethod(resolve_reference)\nGEOMETRY = MemoryGeometryAPI()\n').body)
    source = '# Frozen action-free observation library.\n' + ast.unparse(ast.Module(body=bodies, type_ignores=[])) + '\n'
    from run_vision_trial import validate_source
    validate_source(source)
    return source


def provenance():
    paths = list((ROOT/'src/aspire_so100').glob('*.py'))
    paths += [ROOT/'scripts'/name for name in
              ('run_geometry_ablation.py', 'summarize_geometry_ablation.py',
               'run_vision_trial.py', 'vision_policy_worker.py')]
    paths += list((ROOT/'assets').rglob('*.xml'))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def prepare(root, rounds, seeds, heldout, model, base_policy=None):
    if rounds < 1 or not seeds or not heldout or len(set(seeds + heldout)) != len(seeds + heldout):
        raise ValueError('Positive rounds and nonempty, distinct, disjoint seed lists required')
    base_source = None
    if base_policy is not None:
        from run_vision_trial import validate_source
        base_source = Path(base_policy).read_text()
        validate_source(base_source)
        tree = ast.parse(base_source)
        blocked = {'get_observation', 'check_structure', 'structure_status', 'evaluate'}
        if any(isinstance(n, ast.Attribute) and n.attr in blocked for n in ast.walk(tree)):
            raise ValueError('Base policy must use only the sensor SDK, not privileged legacy APIs')
        if any(isinstance(n, ast.Name) and n.id == 'geometry' for n in ast.walk(tree)):
            raise ValueError('Shared base cannot depend on the treatment library')
        if len(base_source) > 48000 or not any(isinstance(n, ast.FunctionDef) and n.name=='run' for n in tree.body):
            raise ValueError('Base requires run(api) and must fit the shared source budget')
    library = build_geometry_library(include_targets=base_source is not None)
    manifest = {'version': 3, 'purpose': 'pilot; one campaign per arm, not confirmatory evidence',
        'arms': list(ARMS), 'rounds': rounds, 'train_seeds': seeds, 'heldout_seeds': heldout,
        'model': model, 'model_timeout_s': 900, 'reasoning_effort': 'high',
        'max_prompt_characters': 100000, 'max_candidate_characters': 48000,
        'target_candidate_characters': 18000, 'generation_concurrency': 2,
        'geometry_interface': 'cube_observation_v1',
        'task': 'inventory_tower', 'layout': 'sampled', 'recipe': 'cube_triple', 'object_count': 3,
        'physics': {'noslip_iterations': 0},
        'sensor': {'depth_sampling': 'pixel_center', 'offscreen_samples': 0},
        'geometry_sha256': hashlib.sha256(library.encode()).hexdigest(),
        'source_sha256': provenance(),
        'budget': 'fixed revisions and rollout slots; tokens measured, not equalized',
        'feedback': 'own latest source + all own reduced feedback; latest two sensor outcomes',
        'selection': 'highest training success count, earliest revision on ties',
        'initial_policy': 'empty run(api), no prior controller', 'heldout_feedback_to_actor': False}
    if base_source is not None:
        manifest.update(version=4, geometry_interface='cube_current_target_error_v1',
            initial_policy='shared frozen RGB-D base, byte-identical in both arms',
            base_policy_sha256=hashlib.sha256(base_source.encode()).hexdigest(),
            base_policy_source=str(Path(base_policy).resolve()),
            wrist_depth=True, baseline_revision=0,
            baseline_accounting='one shared execution per training seed; included in both curves; no model call',
            selection='highest training success count including base revision zero, earliest revision on ties')
    root.mkdir(parents=True, exist_ok=True)
    if base_source is not None:
        frozen_base = root/'base_policy.py'
        if frozen_base.exists() and frozen_base.read_text() != base_source:
            raise ValueError('Frozen base policy changed')
    path = root/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Manifest mismatch: use a new campaign directory')
    if (root/'geometry.py').exists() and (root/'geometry.py').read_text() != library:
        raise ValueError('Frozen geometry library changed')
    write(path, manifest)
    if base_source is not None:
        (root/'base_policy.py').write_text(base_source)
    (root/'geometry.py').write_text(library)
    for arm in ARMS:
        (root/f'contract_{arm}.txt').write_text(actor_contract(arm, manifest))
    snapshot = root/'source_snapshot'
    for relative, sha in manifest['source_sha256'].items():
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != sha:
            raise ValueError('Frozen source snapshot changed')
        if not target.exists():
            shutil.copyfile(ROOT/relative, target)
    return manifest


def assert_frozen(manifest):
    if provenance() != manifest['source_sha256']:
        raise RuntimeError('Experiment implementation changed after freezing')


def trial(candidate, out, seed, manifest, library=None):
    """Keep raw evaluator artifacts host-side; actor feedback is projected separately."""
    from run_vision_trial import validate_source
    expected_geometry = None if library is None else digest(library)
    if library is not None and expected_geometry != manifest['geometry_sha256']:
        raise RuntimeError('Geometry library does not match the frozen treatment')
    expected = {'candidate_sha256': digest(candidate), 'geometry_library_sha256': expected_geometry,
                'seed': seed, 'task': manifest['task'], 'layout': manifest['layout']}
    result_path = out/'result.json'
    if not result_path.exists():
        if out.exists():
            raise RuntimeError(f'Incomplete trial retained for inspection: {out}')
        try:
            source = candidate.read_text()
            if len(source) > manifest['max_candidate_characters']:
                raise ValueError(f'Candidate has {len(source)} characters; maximum is {manifest["max_candidate_characters"]}')
            validate_source(source)
        except (ValueError, SyntaxError) as error:
            out.mkdir(parents=True)
            result = dict(expected, success=False, invalid_candidate=True, rollout_executed=False,
                          failure_stage='candidate_validation',
                          policy_error=f'{type(error).__name__}: {error}')
            write(result_path, result)
            write(out/'actor_feedback.json', {'success': False, 'policy_error': result['policy_error'], 'notes': []})
            return result
        command = [sys.executable, str(ROOT/'scripts/run_vision_trial.py'),
                   '--candidate', str(candidate), '--output', str(out), '--seed', str(seed),
                   '--task', manifest['task'], '--layout', manifest['layout'],
                   '--object-count', '3', '--tower-recipe', 'cube_triple']
        if manifest.get('wrist_depth'):
            command += ['--wrist-depth']
        if manifest.get('record_geometry'):
            command += ['--record-geometry']
        if library is not None:
            command += ['--geometry-library', str(library)]
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        if sys.platform == 'linux' and not os.environ.get('DISPLAY'):
            env.setdefault('MUJOCO_GL', 'egl')
        out.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with out.with_suffix('.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=240)
        write(out/'trial_timing.json', {'host_trial_wall_seconds': time.monotonic()-started,
              'scope': 'subprocess startup, policy, simulation, rendering, private evaluation and artifact IO'})
    result = json.loads(result_path.read_text())
    if any(result.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f'Trial provenance mismatch: {out}')
    if not result.get('invalid_candidate'):
        if (result['physics'] != manifest['physics'] or result['sensor'] != manifest['sensor']
                or result.get('sensor_blackout') or result['scene']['object_count'] != 3
                or result['scene']['order'] != ['blue_cube', 'red_cube', 'green_cube']
                or result.get('pointing') or bool(result.get('wrist_depth')) != bool(manifest.get('wrist_depth'))):
            raise RuntimeError(f'Trial settings mismatch: {out}')
    timing = json.loads((out/'trial_timing.json').read_text()) if (out/'trial_timing.json').exists() else {}
    return dict(result, **timing, rollout_executed=not result.get('invalid_candidate', False),
                failure_stage='candidate_validation' if result.get('invalid_candidate') else
                ('policy_execution' if result.get('policy_error') else ('task_evaluation' if not result['success'] else None)))


def feedback(out):
    value = json.loads((out/'actor_feedback.json').read_text())
    return {'success': value['success'], 'policy_error': value['policy_error'],
            'notes': [str(x)[:500] for x in value.get('notes', [])[-10:]]}


def actor_contract(arm, manifest):
    if arm not in ARMS:
        raise ValueError('Unknown experiment arm')
    budget = (f'\nSOURCE BUDGET: hard maximum {manifest["max_candidate_characters"]} characters '
              f'in the code string, including whitespace and comments. Aim for at most '
              f'{manifest["target_candidate_characters"]} characters using concise reusable functions. '
              'An oversized or invalid candidate consumes a revision and all its training slots '
              'as candidate-validation failures; it is not executed in the simulator. '
              'There is no free rewrite. Keep diagnostics concise.\n')
    contract = CONTRACT
    extra = ''
    if manifest.get('base_policy_sha256'):
        contract = contract.replace('No tools, file access, network, prior repository, other policies, or external skills.',
                                    'No tools, file access, network, prior repository or external skills. The shared base policy is explicitly supplied below.')
        contract = contract.replace("api.camera('wrist') -> rgb, intrinsics, camera_to_world, time; NO wrist depth.",
                                    "api.camera('wrist') -> same RGB-D/calibration fields as side; ideal simulated wrist depth.")
        contract = contract.replace('STARTER (identical in both groups):\ndef run(api):\n    pass',
                                    'STARTER: the shared frozen RGB-D base supplied below. Improve that policy rather than restarting from an empty function.')
        extra = """
SHARED ROBOT GEOMETRY: api.gripper_geometry() returns eight finger pad boxes
(center, rotation, half_size_m, vertices), joint angles and time, from the robot
model and joint encoders only. Both arms can use it; no object/contact data.
The supplied base was written for an older scene. Diagnose it on these three
cubes in the current scene from your own observations; dimensions must still be measured.
No manual repair is inserted into either arm. Both may modify the complete base.
"""
        if arm == 'geometry':
            extra += """
ADDITIONAL STRUCTURED GOAL FUNCTIONS (numerical only, no actions):
geometry.targets(observations, order, table_plane, anchor=None) constructs an
upright single-column target using observed cube dimensions and a visual base
anchor. anchor is None (use observations[order[0]]) or a complete observation
DICT with center[3], rotation[3,3], position_fully_observed and inferred_orientation.
Do not pass an XYZ array as anchor. Example:
  base_anchor = dict(observations[order[0]])
  targets = geometry.targets(observations, order, table_plane, anchor=base_anchor)
observations maps your color identities to cube observations; order is
bottom-to-top. Returns center, rotation, vertices, height, support and provenance
per object. After relocating the base, explicitly update the anchor once; do not
silently move the goal to match each new observation. Targets are not observations.
geometry.error(observed, target) -> translation_to_goal_m, center_error_m,
xy_error_m, rotation_error_rad (cube symmetry), vertex_rms_m,
pose_fully_observed. Partial poses remain uncertain despite a small error.
geometry.relation(upper, lower) -> lowest_vertex_gap_m,
projected_center_margin_m, both_poses_fully_observed, contact_or_stability_certified=False.
These functions expose geometric estimates, not contact/success certificates.
"""
    return contract + budget + (GEOMETRY_CONTRACT if arm == 'geometry' else RAW_CONTRACT) + extra


def context_for(root, arm, index, manifest):
    prompt = actor_contract(arm, manifest)
    images = [root/'bootstrap'/f'seed_{seed}'/'initial.png' for seed in manifest['train_seeds'][:2]]
    prompt += '\nAttached initial sensor images show two training layouts, before robot motion.\n'
    if manifest.get('base_policy_sha256'):
        if digest(root/'base_policy.py') != manifest['base_policy_sha256']:
            raise RuntimeError('Frozen base policy was modified')
        if index == 1:
            prompt += '\nShared initial base policy (identical source in both arms):\n' + (root/'base_policy.py').read_text()
        prompt += '\nShared revision-zero training feedback:\n' + json.dumps([
            feedback(root/'bootstrap'/f'seed_{seed}') for seed in manifest['train_seeds']])
        if index == 1:
            for seed in manifest['train_seeds'][:2]:
                outcome = root/'bootstrap'/f'seed_{seed}'/'outcome.png'
                if outcome.exists():
                    images.append(outcome)
                    prompt += '\nNext image: shared base-policy outcome on training layout '+str(manifest['train_seeds'].index(seed)+1)+'.\n'

    for previous in range(1, index):
        directory = root/arm/f'round_{previous:03d}'
        response = json.loads((directory/'response.json').read_text())
        prompt += f'\nOwn revision {previous} hypothesis: {response["hypothesis"][:1200]}\n'
        prompt += json.dumps([feedback(directory/'trials'/f'seed_{s}') for s in manifest['train_seeds']])
    if index > 1:
        latest = root/arm/f'round_{index-1:03d}'
        prompt += '\nYour latest complete policy:\n' + (latest/'candidate.py').read_text()
        for seed in manifest['train_seeds'][:2]:
            path = latest/'trials'/f'seed_{seed}'/'outcome.png'
            if path.exists():
                images.append(path)
                prompt += '\nNext attached image: latest outcome for training layout ' + str(manifest['train_seeds'].index(seed)+1) + '.\n'
    prompt += '\nWrite the next complete candidate. No human repairs or old policies will be inserted.\n'
    if len(prompt) > manifest['max_prompt_characters']:
        raise RuntimeError('Context budget exceeded; do not silently truncate')
    return prompt, images


def audit_events(events):
    """Fail closed on tools, unexpected item types, or absent completion."""
    allowed = {'agent_message', 'reasoning'}
    for event in events:
        item = event.get('item')
        if item is not None and item.get('type') not in allowed:
            raise RuntimeError('Actor isolation violation: ' + str(item.get('type')))
    completed = [e for e in events if e.get('type') == 'turn.completed']
    if len(completed) != 1:
        raise RuntimeError('Expected exactly one completed model turn')
    return completed[0].get('usage', {})


def propose(directory, prompt, images, manifest):
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'prompt.txt').write_text(prompt)
    write(directory/'schema.json', SCHEMA)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='ablation-actor-') as tmp:
        work = Path(tmp)
        fresh = work/'codex_home'
        fresh.mkdir(mode=0o700)
        auth = Path(os.environ.get('CODEX_HOME', str(Path.home()/'.codex')))/'auth.json'
        if auth.exists():
            shutil.copyfile(auth, fresh/'auth.json')
            (fresh/'auth.json').chmod(0o600)
        write(work/'schema.json', SCHEMA)
        command = ['codex', 'exec', '--ignore-user-config', '--ephemeral', '--sandbox', 'read-only',
            '--skip-git-repo-check', '--cd', str(work), '--model', manifest['model'],
            '-c', 'project_doc_max_bytes=0', '-c', 'model_reasoning_effort="high"',
            '-c', 'suppress_unstable_features_warning=true',
            '--enable', 'skip_host_skill_discovery', '--disable', 'shell_tool',
            '--disable', 'unified_exec', '--disable', 'apps', '--disable', 'browser_use',
            '--disable', 'computer_use', '--disable', 'skill_search', '--disable', 'multi_agent',
            '--output-schema', str(work/'schema.json'), '--output-last-message', str(work/'response.json'), '--json']
        for i, source in enumerate(images):
            target = work/f'sensor_{i}.png'
            shutil.copyfile(source, target)
            command += ['--image', str(target)]
        command += ['-']
        metadata = {'started_at': datetime.now(timezone.utc).isoformat(), 'command': command,
                    'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                    'images': [{'sha256': digest(p), 'name': str(p.relative_to(directory.parents[1]))} for p in images],
                    'fresh_context': True, 'tools_disabled': True, 'user_config_loaded': False,
                    'skills_loaded': False, 'repository_context': False}
        write(directory/'invocation.json', metadata)
        with (directory/'model.jsonl').open('w') as log, (directory/'model.stderr').open('w') as err:
            subprocess.run(command, input=prompt, text=True, stdout=log, stderr=err, check=True,
                           timeout=manifest['model_timeout_s'], env=dict(os.environ, CODEX_HOME=str(fresh)))
        events = [json.loads(line) for line in (directory/'model.jsonl').read_text().splitlines() if line.startswith('{')]
        usage = audit_events(events)
        response = json.loads((work/'response.json').read_text())
        if set(response) != {'code', 'hypothesis'} or not all(isinstance(x, str) for x in response.values()):
            raise ValueError('Invalid model response schema')
        write(directory/'response.json', response)
        source_budget_exceeded = len(response['code']) > manifest['max_candidate_characters']
        # Preserve the exact generated source even when validation rejects it.
        (directory/'candidate.py').write_text(response['code'])
        write(directory/'generation.json', {'usage': usage, 'wall_seconds': time.monotonic()-started,
                                            'audit': 'passed', 'source_budget_exceeded': source_budget_exceeded,
                                            'candidate_sha256': digest(directory/'candidate.py')})


def campaign(root, manifest):
    assert_frozen(manifest)
    write(root/'progress.json', {'status': 'bootstrap'})
    observe = root/'observe.py'
    observe.write_text('def run(api):\n    api.camera()\n')
    baseline_results = []
    if manifest.get('base_policy_sha256') and digest(root/'base_policy.py') != manifest['base_policy_sha256']:
        raise RuntimeError('Frozen base policy was modified')
    for seed in manifest['train_seeds']:
        candidate = root/'base_policy.py' if manifest.get('base_policy_sha256') else observe
        result = trial(candidate, root/'bootstrap'/f'seed_{seed}', seed, manifest)
        baseline_results.append({'seed': seed, 'success': result['success'], 'policy_error': result['policy_error'],
            'rollout_executed': result['rollout_executed'], 'failure_stage': result['failure_stage'],
            'camera_calls': result.get('camera_calls'), 'cpu_profile': result.get('cpu_profile'),
            'host_trial_wall_seconds': result.get('host_trial_wall_seconds')})
    curves = {arm: [] for arm in ARMS}
    if manifest.get('base_policy_sha256'):
        entry = {'revision': 0, 'successes': sum(r['success'] for r in baseline_results),
            'trials': len(baseline_results), 'cumulative_rollout_slots': len(baseline_results),
            'executed_rollouts': sum(r['rollout_executed'] for r in baseline_results),
            'candidate_validation_failures': 0, 'results': baseline_results,
            'generation': {'usage': {}, 'wall_seconds': 0., 'audit': 'shared_frozen_base_no_generation',
                           'candidate_sha256': manifest['base_policy_sha256']}}
        write(root/'baseline.json', entry)
        for arm in ARMS:
            curves[arm].append(dict(entry))
    for index in range(1, manifest['rounds']+1):
        # Generate independently in parallel, then serialize physics rollouts.
        assert_frozen(manifest)
        order = ARMS if index % 2 else tuple(reversed(ARMS))
        pending = {}
        with ThreadPoolExecutor(max_workers=manifest['generation_concurrency']) as pool:
            for arm in order:
                directory = root/arm/f'round_{index:03d}'
                if not (directory/'generation.json').exists():
                    prompt, images = context_for(root, arm, index, manifest)
                    print(f'Proposing {arm} revision {index}', flush=True)
                    pending[pool.submit(propose, directory, prompt, images, manifest)] = arm
            write(root/'progress.json', {'status': 'proposing', 'pending_arms': list(pending.values()),
                                        'revision': index, 'curves': curves})
            for future in as_completed(pending):
                future.result()
                write(root/'progress.json', {'status': 'proposing',
                    'pending_arms': [arm for f, arm in pending.items() if not f.done()],
                    'revision': index, 'curves': curves})
        for arm in order:
            assert_frozen(manifest)
            directory = root/arm/f'round_{index:03d}'
            generation = json.loads((directory/'generation.json').read_text())
            if generation['candidate_sha256'] != digest(directory/'candidate.py'):
                raise RuntimeError('Generated candidate was edited after model submission')
            write(root/'progress.json', {'status': 'executing', 'arm': arm, 'revision': index, 'curves': curves})
            successes = 0
            results = []
            for seed in manifest['train_seeds']:
                result = trial(directory/'candidate.py', directory/'trials'/f'seed_{seed}', seed, manifest,
                               root/'geometry.py' if arm == 'geometry' else None)
                successes += int(result['success'])
                results.append({'seed': seed, 'success': result['success'], 'policy_error': result['policy_error'],
                                'cpu_profile': result.get('cpu_profile'),
                                'camera_calls': result.get('camera_calls'),
                                'host_trial_wall_seconds': result.get('host_trial_wall_seconds'),
                                'rollout_executed': result['rollout_executed'], 'failure_stage': result['failure_stage']})
                print(f'{arm} revision {index}, seed {seed}: success={result["success"]}, error={result["policy_error"]}', flush=True)
            entry = {'revision': index, 'successes': successes, 'trials': len(results),
                     'cumulative_rollout_slots': (index+int(bool(manifest.get('base_policy_sha256'))))*len(results),
                     'executed_rollouts': sum(r['rollout_executed'] for r in results),
                     'candidate_validation_failures': sum(r['failure_stage'] == 'candidate_validation' for r in results),
                     'generation': generation, 'results': results}
            curves[arm].append(entry)
            write(root/'progress.json', {'status': 'training', 'curves': curves})
    frozen = {}
    # Freeze BOTH arms before running ANY held-out trial. Selection never uses private metrics.
    for arm in ARMS:
        best = max(curves[arm], key=lambda r: r['successes'])
        source = root/'base_policy.py' if best['revision']==0 else root/arm/f'round_{best["revision"]:03d}'/'candidate.py'
        target = root/arm/'frozen.py'
        shutil.copyfile(source, target)
        frozen[arm] = {'revision': best['revision'], 'candidate_sha256': digest(target)}
    write(root/'frozen.json', frozen)
    write(root/'progress.json', {'status': 'heldout', 'curves': curves, 'frozen': frozen})
    heldout = {}
    for arm in ARMS:
        heldout[arm] = []
        for seed in manifest['heldout_seeds']:
            assert_frozen(manifest)
            result = trial(root/arm/'frozen.py', root/arm/'heldout'/f'seed_{seed}', seed, manifest,
                           root/'geometry.py' if arm == 'geometry' else None)
            heldout[arm].append({'seed': seed, 'success': result['success'], 'policy_error': result['policy_error'],
                                 'cpu_profile': result.get('cpu_profile'),
                                'camera_calls': result.get('camera_calls'),
                                'host_trial_wall_seconds': result.get('host_trial_wall_seconds'),
                                'rollout_executed': result['rollout_executed'], 'failure_stage': result['failure_stage']})
            print(f'HELDOUT {arm} seed {seed}: success={result["success"]}', flush=True)
    summary = {'status': 'completed', 'manifest_sha256': digest(root/'manifest.json'),
               'curves': curves, 'frozen': frozen, 'heldout': heldout,
               'limitations': ['Single campaign per arm; no statistical efficacy claim.',
                              'Fixed revision/rollout budget; tokens and elapsed time are measured, not matched.',
                              'CPU intervals exclude RPC/rendering/IK; no realtime or VLM-latency claim.',
                              'Ideal side RGB-D, unique colors, three cubes, fixed order, no distractors.']}
    write(root/'summary.json', summary)
    write(root/'progress.json', {'status': 'completed', 'curves': curves,
         'heldout': {arm: {'successes': sum(r['success'] for r in results), 'trials': len(results)}
                     for arm, results in heldout.items()}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--train-seeds', type=int, nargs='+', default=[3301, 3302, 3303])
    parser.add_argument('--heldout-seeds', type=int, nargs='+', default=[3401, 3402, 3403, 3404, 3405])
    parser.add_argument('--model', default='gpt-6-astra')
    parser.add_argument('--base-policy', type=Path, help='Shared sensor-only starting policy; enables wrist depth and structured goals equally configured across arms')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    manifest = prepare(root, args.rounds, args.train_seeds, args.heldout_seeds, args.model, args.base_policy)
    if not args.prepare_only:
        try:
            campaign(root, manifest)
            from summarize_geometry_ablation import export
            export(root, root/'report.json', root/'learning_curves.png')
        except Exception as error:
            write(root/'infrastructure_error.json', {'error_type': type(error).__name__,
                  'message': str(error), 'time': datetime.now(timezone.utc).isoformat()})
            state_path = root/'progress.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            write(state_path, dict(state, status='stopped_on_error', error_type=type(error).__name__))
            raise


if __name__ == '__main__':
    main()
