"""Online isolation, failure continuation and model-switch accounting contracts."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from online_bridge_actor import audit_decision_events, validate_response
from online_bridge_skills import SharedSkills
from run_online_bridge_episode import feedback
import run_online_bridge_episode as episode


class OnlineRuntimeTests(unittest.TestCase):
    def test_episode_continues_past_thirty_decisions_until_wall_timeout(self):
        clock = [0.]
        def now():
            clock[0] += .00001
            return clock[0]
        class Environment:
            def __init__(self, output, scene_seed):
                self.episode_started = None
                self.success_time = None
                self.success_evidence = None
                self.order = ()
                self.reset_count = 0
                self.data = SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1), time=0.)
            def reset(self, seed): self.reset_count += 1
            def start_episode(self, budget, realtime):
                self.episode_started = now()
                self.deadline = self.episode_started+budget
            def advance_while_waiting(self):
                if now() >= self.deadline: raise TimeoutError()
                self.data.time += .00001
            def timing(self): return {'reset_count': self.reset_count}
            def structure_status(self, names): return {'stable': False}
            def close(self): pass
        class API:
            def __init__(self, env, output):
                self.env, self.output = env, output
                self.camera_count = self.camera_seconds = self.action_seconds = 0
            def image(self, name, view): return self.output/(name+'.png'), {'time': 0.}
            def task(self): return {'nodes': []}
            def wait(self, seconds): pass
        class Skills:
            def __init__(self, api):
                self.provenance = {}
                self.last_result = None
                self.session = SimpleNamespace(time=lambda: api.env.data.time)
            def holding(self): return None
            def packet(self): return {}
        class Actor:
            def __init__(self, output): self.receipts = []
            def request(self, *args, **kwargs):
                return {'action': 'wait', 'skill': 'none', 'node': '', 'plan': [], 'reason': 'settle'}
        with TemporaryDirectory() as folder, patch.multiple(episode,
            OnlineBridgeTowerEnv=Environment, SensorAPI=API, SharedSkills=Skills, OnlineActor=Actor,
            time=SimpleNamespace(monotonic=now)):
            result = episode.run(Path(folder)/'episode', 1, 'symbolic', .01)
        self.assertEqual(result['status'], 'timeout')
        self.assertGreater(result['decisions'], 30)
        self.assertEqual(result['timing']['reset_count'], 1)

    def test_intended_model_switch_warning_is_not_a_tool_or_general_error_exception(self):
        warning = ('This session was recorded with model `gpt-6-astra` but is resuming with '
                   '`gpt-6-luna`. Consider switching back to `gpt-6-astra` as it may affect Codex performance.')
        events = [{'type': 'item.completed', 'item': {'type': 'error', 'message': warning}},
                  {'type': 'turn.completed', 'usage': {'input_tokens': 12}}]
        self.assertEqual(audit_decision_events(events, 'gpt-6-luna', True), {'input_tokens': 12})
        with self.assertRaises(RuntimeError):
            audit_decision_events(events, 'gpt-6-luna', False)
        for item in ({'type': 'command_execution'}, {'type': 'error', 'message': 'other error'}):
            with self.assertRaises(RuntimeError):
                audit_decision_events([{'type': 'item.completed', 'item': item}, events[-1]], 'gpt-6-luna', True)

    def test_base_projection_omits_private_geometry_and_numeric_error_details(self):
        api = SimpleNamespace(image=Mock(return_value=(Path('frame.png'), {'time': 3.})))
        skills = SimpleNamespace(session=SimpleNamespace(time=lambda: 3.), holding=lambda: None,
            last_result={'status': 'recoverable_error', 'error': 'metric detail 0.123',
                         'skill': 'bridge', 'node': 'red'}, packet=Mock(return_value={'metric': .123}))
        basic, images = feedback(api, skills, 'basic', 1)
        self.assertNotIn('geometry', basic)
        self.assertNotIn('error', basic['last_action'])
        skills.packet.assert_not_called()
        self.assertEqual(len(images), 1)
        full, images = feedback(api, skills, 'symbolic', 1)
        self.assertEqual(full['geometry'], {'metric': .123})
        self.assertEqual(images, [])

    def test_recoverable_failure_retains_session_and_next_call_can_succeed(self):
        with TemporaryDirectory() as folder:
            skills = SharedSkills.__new__(SharedSkills)
            skills.api = SimpleNamespace(output=Path(folder))
            session = SimpleNamespace(pnp=SimpleNamespace(held=None))
            skills.session = session
            skills.nodes = {'green': {'key': 'green', 'color': 'green', 'supports': []}}
            skills.origin = [0., 0.]
            skills.scale = .03
            skills.counter = [0]
            skills.bridge = SimpleNamespace(current_target=Mock(return_value=(None, {}, False)),
                place_node=Mock(side_effect=[RuntimeError('occluded'), (True, None)]))
            first = skills.execute('bridge', 'green')
            second = skills.execute('bridge', 'green')
            self.assertEqual(first['status'], 'recoverable_error')
            self.assertEqual(second['status'], 'visually_consistent')
            self.assertIs(skills.session, session)
            self.assertEqual(len((Path(folder)/'skill_results.jsonl').read_text().splitlines()), 2)

    def test_response_validation_rejects_code_or_invented_actions(self):
        response = {'action': 'execute', 'skill': 'stack', 'node': 'blue', 'plan': [], 'reason': 'continue'}
        self.assertEqual(validate_response(response), response)
        with self.assertRaises(ValueError):
            validate_response(dict(response, code='reset()'))
        with self.assertRaises(ValueError):
            validate_response(dict(response, action='reset'))


if __name__ == '__main__':
    unittest.main()
