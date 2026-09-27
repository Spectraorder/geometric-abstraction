"""Compare geometry interfaces without a supplied task-solving starter or recipe advice."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import run_constraint_geometry_experiment as parent
from summarize_mixed_scene_learning import require
from run_unseeded_vision_trial import CONTRACT, declarative_goal
from aspire_so100.structure_tasks import structure_recipe

ROOT = parent.ROOT
base = parent.base
PARENT_SOURCES = parent.sources
PARENT_EXECUTE = base.execute
VERSION = 'unseeded_geometry_actor_sessions_v1'
STARTER = ROOT/'policies/observation_only_starter.py'
DRIVER = ROOT/'scripts/run_unseeded_vision_trial.py'
EXTRA = ('policies/observation_only_starter.py', 'scripts/run_unseeded_vision_trial.py',
         'scripts/run_unseeded_geometry_experiment.py')
COMMON = '''Generate a CPU policy for the declarative manipulation goal returned by api.task().
Return JSON with hypothesis and code defining run(api), plus optional numerical
helpers. Only numpy, scipy and math imports. No tools, repository, skills, files,
introspection, classes, attribute assignment, dynamic execution or direct actuation.
Maximum source length is 30000 characters. Invalid submissions consume a revision
and its training slots, with no human correction or free rewrite.

The shared R0 baseline only captures images; it contains no perception algorithm,
target calculation, object ordering or task solution. At R1 create the first
complete policy from the API contract and goal. Later improve the retained policy
using your own training feedback. No previously successful policy is supplied.

api.task() returns a declarative goal: kind=support_graph, axes=world_xy,
length_unit={observed_cube: node_key}, and nodes. Each node has key, color, shape,
grid=[x,y] in measured reference-cube side units, supports=[node_keys], and desired
orientation. Empty supports specify the table. The grid specifies desired relative
XY locations; supports specify required final support relationships. Orientation
uses upright or flat_x in world axes. These fields describe the final state, not
an action program. Nodes are sorted by color labels, NOT construction order. Keys
are color identifiers, without task-role names. No fixed world worksite, metric
object dimensions or current object poses are supplied in the goal. Derive a plan
from its constraints and current permitted observations.

Instantiate Session(api, objects), passing the complete public task inventory.
Both arms share a frozen CPU RGB-D pick/place actuator. Common methods: time(),
tool(), pick(color,transit_z=None), place(surface_xyz), pick_and_place(color,surface_xyz),
and checkpoint(color,step,action_status=None). Place takes object center XY and
support-surface Z in world meters. Do not redefine, subclass, mutate or access
session internals. No direct move_tool, move_joints or gripper calls. Choose your
own observations, action sequence, targets, workspace and bounded recovery.

Runtime RGB-D comes from api.camera('side' or 'wrist'): rgb, depth_m (optical-axis Z),
valid_depth, intrinsics, camera_to_world and time. Deprojection is
world=R @ (inv(K) @ [u,v,1]*depth[v,u])+t. Segmentation and object truth are unavailable.
api.proprioception() returns robot-only tool_position, tool_rotation, joint_angles,time.
api.plan_tool(position,orientation=None) returns robot-only IK tool_position,
tool_rotation, joint_angles and position_error, without collision checking or motion.
api.gripper_geometry() reports robot-model pad geometry, not object contact.
api.wait(seconds), note(text), record_geometry(JSON-compatible dict) are available.

Perception and grasp primitives retain shared domain assumptions: unique colors,
known shape labels, table-supported initial sources, upright cubes/prisms/cylinders,
and flat_x thin prisms. These are assumptions, not visual measurements. No
metric catalog dimensions, ground-truth object poses or evaluator state are allowed.
The executor source describes general motion primitives, not a task controller.

After action calls, record checkpoint(color,step,action_status=returned_status).
Base checkpoints expose action metadata only; Full adds current visual geometry.
Geometry may be missing, stale or ambiguous. Refresh evidence after motion; never
turn a stored target into a measurement. Released and held are executor statuses,
not proof of stable placement or task success. Keep all observation, association
and release checks unchanged. No global snapshot/state/text/history methods.

Use one falsifiable hypothesis per revision, or an explicit initial plan at R1.
Record bounded policy_probe diagnostics with hypothesis, triggered, reason and
sensor-derived evidence. Label unavailable evidence unknown. You may change
perception, planning and recovery; do not hardcode seeds or inspected outcomes.
Acceptance requires preserving incumbent training successes and repairing at
least one failure; ties retain the incumbent. If R0 remains selected, construct
a complete policy instead of treating its empty action sequence as a constraint.
Feedback contains own binary outcomes, notes, bounded event-centered checkpoints
and issued-command comparisons. Custom reports are unverified policy statements.
The host never inserts a task-specific failure diagnosis or solution suggestion.
Images follow the registered schedule; old images are historical evidence.
Host timeout is 120 seconds and total API cap 1200. Policies run on CPU; simulation
may use GPU rendering. Rollout duration is not an objective.
'''
BASIC = '''Base receives raw RGB-D, calibration, public goal labels and robot-only
proprioception/CAD. No supplied object geometry or perception helper is available.
Implement any needed image processing and geometry within your policy budget.
Session has no observe, get, table_z, reference, relation, align, target or goal_error
methods. Pick/place returns only status and held. Checkpoint returns identity,
time, step and action status; refresh=True is unavailable. Use api.camera() directly.
'''


def sources():
    result = PARENT_SOURCES()
    result.update({name: base.digest(ROOT/name) for name in EXTRA})
    return result


@contextmanager
def installed():
    with patch.multiple(parent, VERSION=VERSION, COMMON=COMMON, sources=sources), \
         patch.object(parent.parent, 'BASIC', BASIC):
        yield


@contextmanager
def configured():
    with installed(), parent.configured(), patch.object(base, 'execute', execute):
        yield


def prepare(root, train, reserved, rounds=3):
    with installed():
        manifest = parent.prepare(root, train, reserved, rounds, 'persistent', 'on_failure', 4)
    parent.parent.validate_planner(STARTER.read_text())
    shutil.copyfile(STARTER, root/'starter.py')
    manifest.update(starter_sha256=base.digest(root/'starter.py'), initial_policy='observation_only',
                    public_goal_contract=CONTRACT, trial_driver_sha256=base.digest(DRIVER),
                    supplied_task_planner=False, supplied_recipe_advice=False,
                    primary_metrics=['training-selected reserved SR', 'reserved SR of first generated policy R1',
                                     'reserved SR at matched proposal budgets R1 through R3'],
                    baseline_interpretation='R0 collects images without acting; R1 is the first model-generated policy, not a revision of a supplied solution.')
    manifest['limitations'] += [
        'Task-solving starter and recipe-specific prompt advice are removed; task goals and manipulation primitives remain.',
        'Goal nodes are color-sorted and role keys are relabeled; task-specific final-state constraints remain required input.',
        'Shared observer/executor retains shape, color and initial-orientation assumptions; this is not assumption-free learning.',
        'One task family cannot establish cross-task generalization.',
        'R0 is a non-acting control, not a measure of model zero-shot ability. R1 receives R0 images/outcomes.']
    base.write(root/'manifest.json', manifest)
    return manifest


def check(root, manifest):
    with installed():
        parent.check(root, manifest)
    require(manifest.get('initial_policy') == 'observation_only' and
            manifest.get('public_goal_contract') == CONTRACT, 'Incorrect neutral-start contract')
    require((root/'starter.py').read_bytes() == STARTER.read_bytes(), 'Unexpected supplied starter')
    require(manifest['trial_driver_sha256'] == base.digest(DRIVER), 'Trial driver changed')


def execute(root, arm, candidate, directory, seeds, manifest, error=None):
    rows = []
    for seed in seeds:
        base.assert_frozen(root, manifest)
        out = directory/f'seed_{seed}'
        if not error and not (out/'result.json').exists():
            require(not out.exists(), 'Incomplete trial retained: ' + str(out))
            require(shutil.disk_usage(root).free >= manifest['minimum_free_disk_bytes'], 'Insufficient disk')
            out.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(DRIVER), '--candidate', str(candidate), '--output', str(out),
                       '--seed', str(seed), '--task', manifest['task'], '--layout', 'sampled',
                       '--structure-recipe', manifest['recipe'], '--structure-color-map',
                       json.dumps(manifest['color_maps'][str(seed)]), '--object-count', '4', '--wrist-depth',
                       '--record-geometry', '--timeout', '120', '--geometry-library', str(root/(arm+'_library.py'))]
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='egl')
            with out.with_suffix('.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=240)
        rows.extend(PARENT_EXECUTE(root, arm, candidate, directory, [seed], manifest, error))
        result = json.loads((out/'result.json').read_text())
        if not result.get('invalid_candidate'):
            check_goal(out, result, manifest)
    return rows


def check_goal(trial, result, manifest):
    receipt = json.loads((trial/'goal_contract.json').read_text())
    expected = declarative_goal(structure_recipe(manifest['recipe'], color_map=manifest['color_maps'][str(result['seed'])]))
    require(receipt == {'contract': CONTRACT, 'driver_sha256': manifest['trial_driver_sha256'], 'goal': expected},
            'Public goal receipt differs')
    require(result['goal'] == expected and result['scene']['goal'] == expected, 'Public goal changed')


def bootstrap(root, manifest):
    check(root, manifest)
    require(not list(root.glob('*/round_*/invocation.json')), 'Bootstrap must precede actor calls')
    with configured():
        for arm in manifest['arms']:
            output = root/arm/'revision_000'
            candidate = base.bundle(root, arm, (root/'starter.py').read_text(), output/'bundled.py')
            execute(root, arm, candidate, output/'trials', manifest['train_seeds'], manifest)
    parent.parent.audit_starters(root, manifest)
    base.write(root/'progress.json', {'status': 'bootstrapped'})


def audit(root, allow_partial=False):
    manifest = json.loads((root/'manifest.json').read_text()); check(root, manifest)
    with installed():
        report = parent.audit(root, allow_partial)
    for relative in report['result_hashes']:
        path = root/relative; result = json.loads(path.read_text())
        if not result.get('invalid_candidate'):
            check_goal(path.parent, result, manifest)
    report['public_goal_contract_audit'] = 'passed'
    report['supplied_task_planner'] = False
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--train-seeds', type=int, nargs='+')
    parser.add_argument('--reserved-seeds', type=int, nargs='+')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--phase', choices=('prepare', 'bootstrap', 'train', 'evaluate', 'audit'), default='prepare')
    args = parser.parse_args(); root = args.output.resolve()
    if not args.resume and (not args.train_seeds or not args.reserved_seeds):
        parser.error('Explicit training and reserved seeds required for a new registration')
    manifest = json.loads((root/'manifest.json').read_text()) if args.resume else prepare(root, args.train_seeds, args.reserved_seeds, args.rounds)
    with (root/'training_coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check(root, manifest)
        if args.phase == 'bootstrap':
            bootstrap(root, manifest)
        elif args.phase in ('train', 'evaluate'):
            if not (root/'starter_parity.json').exists():
                bootstrap(root, manifest)
            with configured():
                base.campaign(root, manifest, args.phase)
        if args.phase == 'audit':
            report = audit(root, allow_partial=True); base.write(root/'audit.json', report)
            print({key: report[key] for key in ('status', 'valid_rollouts', 'invalid_slots', 'exact_actor_contexts_verified', 'public_goal_contract_audit')})
