"""Shared learned grasp skill and explicit pad actions, with no automatic placement."""
import ast
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from ring_peg_runtime import plain
from run_ring_peg_comparison_trial import validate_policy

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_training_prior():
    """Choose among training-retained policies; never read reserved outcomes."""
    study = ROOT/'outputs/ring_peg_comparison_01'
    choices = []
    for path in sorted(study.glob('replica_*/symbolic/summary.json')):
        summary = json.loads(path.read_text())
        revision = summary['selected_revision']
        score = next(row['successes'] for row in summary['curves'] if row['revision'] == revision)
        source = path.parent/('revision_000' if revision == 0 else f'round_{revision:03d}')/'candidate.py'
        if digest(source) != summary['selected_sha256']:
            raise ValueError('Training-selected source changed')
        choices.append((score, path.parent.parent.name, revision, source, path))
    if not choices:
        raise ValueError('No completed training priors')
    score, replica, revision, source, summary = sorted(choices, key=lambda x:(-x[0], x[1], x[2]))[0]
    if score < 1:
        raise ValueError('No successful training-retained prior')
    return source, dict(selection='Highest retained training success; replica/revision order breaks ties',
                        replica=replica, revision=revision, training_successes=score, training_layouts=5,
                        source=str(source.relative_to(ROOT)), sha256=digest(source),
                        training_summary=str(summary.relative_to(ROOT)), training_summary_sha256=digest(summary),
                        test_outcomes_used=False,
                        scope='Only the grasp/lift prefix is reused; placement and recovery code are not executed')


class LiftBoundary(Exception):
    pass


class SkillAPI:
    """Trusted local sensor adapter, shared by both arms; not exposed to the model."""
    def __init__(self, sensors):
        self.sensors = sensors

    def task(self):
        return dict(self.sensors.task(), geometry_available=True)

    def __getattr__(self, name):
        if name in ('camera', 'observe'):
            return getattr(self.sensors, name)
        if name == 'robot':
            return self.sensors.robot.current
        if name in ('plan_pad', 'plan_grasp'):
            def preview(*args, **kwargs):
                self.sensors.env.advance_while_waiting()
                return getattr(self.sensors.robot, name)(*args, **kwargs)
            return preview
        if name == 'move_pad':
            return lambda point, roll, seconds: self.sensors.robot.move_pad(point, roll, seconds)
        if name == 'gripper':
            return lambda angle, seconds=.6: self.sensors.dispatch('gripper', dict(angle=angle, seconds=seconds))
        if name == 'wait':
            return lambda seconds: self.sensors.dispatch('wait', dict(seconds=seconds))
        if name == 'note':
            return lambda text: None
        raise AttributeError(name)


class RingSkills:
    def __init__(self, sensors, prior):
        self.sensors = sensors
        self.api = SkillAPI(sensors)
        source = Path(prior).read_text()
        validate_policy(source)
        # Only function/import declarations are loaded. The learned placement
        # function is replaced by a mandatory return-to-agent boundary.
        namespace = {}
        exec(compile(ast.parse(source), str(prior), 'exec'), namespace)
        self.learned = namespace
        self.action_seconds = 0.
        self.last_grasp_opening = None
        self.grasp_attempts = 0

    def grasp_lift(self):
        self.grasp_attempts += 1
        def boundary(api, ring, peg, command, roll, opening):
            self.last_grasp_opening = float(opening)
            raise LiftBoundary()
        self.learned['place'] = boundary
        try:
            self.learned['run'](self.api)
        except LiftBoundary:
            return {'status':'lift_observed', 'holding':'visually_supported_at_skill_end'}
        return {'status':'grasp_incomplete', 'holding':'unknown'}

    def execute(self, decision):
        start = time.monotonic()
        try:
            action = decision['action']
            if action == 'grasp_lift':
                return self.grasp_lift()
            if action == 'move_pad':
                point = decision['point_m']
                # Exact actor target, without inferring an object destination.
                self.sensors.robot.move_pad(point, decision['roll'], decision['seconds'])
                return {'status':'motion_completed', 'submitted_point_m':point}
            if action == 'release':
                self.sensors.dispatch('gripper', dict(angle=decision['jaw'], seconds=.7))
                current = self.sensors.robot.current()
                point = np.asarray(current['fixed_pad_m'])+np.array([0., 0., .06])
                self.sensors.robot.move_pad(point.tolist(), float(current['roll']), 1.4)
                self.sensors.env.wait(2.5)
                return {'status':'release_and_retreat_completed'}
            if action == 'wait':
                self.sensors.env.wait(decision['seconds'])
                return {'status':'wait_completed'}
            raise ValueError('Unknown shared skill')
        except (ValueError, RuntimeError) as error:
            # Sensor or kinematic errors are ordinary episode feedback. Private
            # geometry and internal numeric diagnostics never enter Base prompts.
            return {'status':'action_unavailable', 'reason':type(error).__name__}
        finally:
            self.action_seconds += time.monotonic()-start
