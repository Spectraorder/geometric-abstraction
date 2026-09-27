"""Verify caller targets survive execution, cancellation and failed attempts."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from online_bound_target_skills import BoundTargetSkills, drive_bound, surface_point
from online_event_skills import EventSkills


def completed(value):
    if False: yield
    return value


class BindingTests(unittest.TestCase):
    def setup_skills(self, output, held=False):
        skills = BoundTargetSkills.__new__(BoundTargetSkills)
        skills.nodes = {'red':{'key':'red','color':'red','supports':['blue']}}
        skills.scale = .03
        skills.last_result = None
        state = {'held':{'color':'red','lift_confirmed':True} if held else None}
        skills.holding = lambda: state['held']
        self.targets = []
        def pick(*args, **kwargs):
            state['held'] = {'color':'red','lift_confirmed':True}
            return completed({'held':True,'status':'lift_visually_confirmed'})
        def place(point):
            self.targets.append(point)
            state['held'] = None
            return completed({'held':False,'status':'released'})
        skills.session = SimpleNamespace(time=lambda:1., pick_steps=Mock(side_effect=pick),
                                         place_steps=Mock(side_effect=place))
        skills.api = SimpleNamespace(output=Path(output),wait=Mock())
        skills.bridge = SimpleNamespace(
            current_target=Mock(side_effect=AssertionError('Destination must not be inferred')),
            settled_check=Mock(return_value=(False,None,{})),
            observe=Mock(return_value={'height':.03}), fit_geometry=lambda value:value,
            orientation_evidence=Mock(return_value=True))
        return skills

    def operation(self, point):
        return dict(action='execute',skill='stack',node='red',surface_point_world_m=point)

    def test_deliberately_displaced_target_is_not_replaced(self):
        with TemporaryDirectory() as output:
            skills = self.setup_skills(output)
            target = [.17,-.23,.44]
            result = drive_bound(skills,self.operation(target))
            self.assertEqual(self.targets,[target])
            self.assertEqual(result['surface_point_world_m'],target)
            self.assertFalse(result['task_success_certified'])
            skills.bridge.current_target.assert_not_called()
            self.assertIsNone(skills._bound_target)

    def test_resume_held_object_uses_same_submitted_point(self):
        with TemporaryDirectory() as output:
            skills = self.setup_skills(output,held=True)
            target = [.11,-.19,.45]
            drive_bound(skills,self.operation(target))
            skills.session.pick_steps.assert_not_called()
            self.assertEqual(self.targets,[target])

    def test_cancellation_after_pick_drops_binding_and_keeps_holding(self):
        with TemporaryDirectory() as output:
            skills = self.setup_skills(output)
            result = drive_bound(skills,self.operation([.1,-.2,.4]),lambda event:True)
            self.assertEqual(result['status'],'support_changed')
            self.assertEqual(skills.holding()['color'],'red')
            self.assertEqual(self.targets,[])
            self.assertIsNone(skills._bound_target)

    def test_source_list_mutation_cannot_retarget_a_binding(self):
        with TemporaryDirectory() as output:
            skills = self.setup_skills(output)
            point = [.1,-.2,.4]
            with skills.bind_target('stack','red',point):
                point[0] = .9
                self.assertEqual(skills._bound_target[1],(.1,-.2,.4))

    def test_unbound_call_retains_legacy_behavior(self):
        with TemporaryDirectory() as output:
            skills = self.setup_skills(output)
            with patch.object(EventSkills,'_attempt',return_value=completed({'status':'legacy'})) as old:
                result = skills.run_steps('stack','red')
                with self.assertRaises(StopIteration) as done:next(result)
                self.assertEqual(done.exception.value['status'],'legacy')
                old.assert_called_once()

    def test_invalid_coordinates_rejected_before_actions(self):
        for value in ([1,2], [1,2,float('nan')], [1,2,float('inf')], ['1','2','3'], [True,False,True], [True,2,3]):
            with self.assertRaises(ValueError):surface_point(value)
        with TemporaryDirectory() as output:
            skills = self.setup_skills(output)
            with self.assertRaises(ValueError):drive_bound(skills,self.operation([1,2]))
            skills.session.pick_steps.assert_not_called()
            skills.session.place_steps.assert_not_called()

    def test_error_does_not_leak_target_to_next_call(self):
        with TemporaryDirectory() as output:
            skills = self.setup_skills(output,held=True)
            skills.session.place_steps.side_effect = ValueError('Placement unavailable')
            result = drive_bound(skills,self.operation([.1,-.2,.4]))
            self.assertEqual(result['status'],'recoverable_error')
            self.assertEqual(result['surface_point_world_m'],[.1,-.2,.4])
            self.assertFalse(result['task_success_certified'])
            self.assertIsNone(skills._bound_target)


if __name__ == '__main__':unittest.main()
