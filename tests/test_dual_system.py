import unittest
from dataclasses import replace
from unittest.mock import patch
from aspire_so100.dual_system import DualController,Plan,JevClient,actor_state

S={'version':0,'time_s':0.,'phase':'initial_observation','visible_colors':['red','green'],'table_fit_valid':True,'grasp_confirmed':None,'reobservations_left':2,'grasp_retries_left':1}
def reply(s,p,action='reobserve',confidence=.95):
    return {'action':action,'confidence':confidence,'state_version':s['version'],'plan_version':p.version,'model':'TEST_DOUBLE'}
class Tests(unittest.TestCase):
    def test_oracle_rejected(self):
        with self.assertRaises(ValueError):actor_state(dict(S,true_pose=[0,0,0]))
    def test_no_false_visible_motion(self):
        c=DualController(lambda s,p:reply(s,p,'continue'))
        self.assertEqual(c.tick(S)['gated_action'],'stop')
    def test_escalation_requires_new_observation(self):
        c=DualController(lambda s,p:reply(s,p,'escalate'),lambda s,p,t:replace(p,version=p.version+1))
        e=c.tick(S);self.assertEqual((e['gated_action'],e['plan_after']),('reobserve',1))
        with self.assertRaises(ValueError):c.tick(S)
    def test_slow_budget_not_expandable(self):
        c=DualController(lambda s,p:reply(s,p,'escalate'),lambda s,p,t:replace(p,version=p.version+1,max_slow_calls=999))
        self.assertEqual(c.tick(S)['gated_action'],'stop');self.assertEqual(c.plan.version,0)
    def test_jev_only_no_astra(self):
        def forbidden(*a):raise AssertionError('Astra must not be called')
        c=DualController(lambda s,p:reply(s,p,'escalate'),forbidden,mode='jev_only')
        self.assertEqual(c.tick(S)['slow_calls_total'],0)
    def test_stale_reply(self):
        c=DualController(lambda s,p:dict(reply(s,p),state_version=99))
        self.assertEqual(c.tick(S)['gated_action'],'stop')
    def test_nan_confidence(self):
        c=DualController(lambda s,p:reply(s,p,confidence=float('nan')))
        self.assertEqual(c.tick(S)['gated_action'],'stop')
    def test_no_key_no_inference(self):
        with patch.dict('os.environ',{},clear=True),patch('urllib.request.urlopen') as send:
            with self.assertRaises(RuntimeError):JevClient()(S,Plan())
            send.assert_not_called()
    def test_budget_exhausted_stop_after_replan(self):
        c=DualController(lambda s,p:reply(s,p,'escalate'),lambda s,p,t:replace(p,version=p.version+1))
        self.assertEqual(c.tick(dict(S,reobservations_left=0))['gated_action'],'stop')
    def test_low_confidence_routes(self):
        c=DualController(lambda s,p:reply(s,p,confidence=.2),lambda s,p,t:replace(p,version=p.version+1))
        self.assertEqual(c.tick(S)['trigger'],'low_confidence')
    def test_visible_continue(self):
        c=DualController(lambda s,p:reply(s,p,'continue'))
        self.assertEqual(c.tick(dict(S,visible_colors=['red','green','blue']))['gated_action'],'continue')
    def test_slow_call_cap(self):
        c=DualController(lambda s,p:reply(s,p,'escalate'),lambda s,p,t:replace(p,version=p.version+1))
        for i in range(4):e=c.tick(dict(S,version=i))
        self.assertEqual(e['slow_calls_total'],2);self.assertEqual(e['gated_action'],'stop')

class IntegrationContracts(unittest.TestCase):
    def test_initial_only_session(self):
        import tempfile
        from aspire_so100.decision_session import DecisionSession
        with tempfile.TemporaryDirectory() as d:
            session=DecisionSession('jev_only',d)
            with self.assertRaises(ValueError):session.decide(dict(S,phase='verify_grasp'),0.)
    def test_session_rejects_stale_sensor_before_network(self):
        import tempfile
        from aspire_so100.decision_session import DecisionSession
        with tempfile.TemporaryDirectory() as d:
            session=DecisionSession('jev_only',d)
            with self.assertRaises(ValueError):session.decide(S,1.)
    def test_source_remains_sandbox_compatible(self):
        import runpy
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        builder=runpy.run_path(str(root/'scripts/build_dual_system_policy.py'))
        runner=runpy.run_path(str(root/'scripts/run_vision_trial.py'))
        runner['validate_source'](builder['source']())
    def test_low_confidence_can_only_reobserve_with_existing_budget(self):
        c=DualController(lambda s,p:reply(s,p,confidence=.2),mode='jev_only')
        self.assertEqual(c.tick(S)['gated_action'],'reobserve')
        c=DualController(lambda s,p:reply(s,p,confidence=.2),mode='jev_only')
        self.assertEqual(c.tick(dict(S,reobservations_left=0))['gated_action'],'stop')


class CallAccounting(unittest.TestCase):
    def test_failed_decision_retains_elapsed_time_without_secret_error_text(self):
        def fail(s,p):raise TimeoutError('sensitive provider details')
        c=DualController(fail,mode='jev_only')
        with patch('aspire_so100.dual_system.time.perf_counter',side_effect=[10.,12.,12.]):
            event=c.tick(S)
        self.assertEqual(event['decision_wall_seconds'],2.)
        self.assertEqual(event['trigger'],'provider_error:TimeoutError')
        self.assertNotIn('sensitive',str(event))

    def test_failed_astra_call_is_not_free_or_missing(self):
        from aspire_so100.astra_decision import Astra
        a=Astra()
        with patch.object(a,'_ask',side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):a.ask({'role':'test'},{})
        self.assertEqual(len(a.calls),1)
        self.assertIsNone(a.calls[0]['usage'])
        self.assertIsNone(a.calls[0]['estimated_usd'])
        self.assertEqual(a.calls[0]['error_type'],'TimeoutError')

if __name__=='__main__':unittest.main()
