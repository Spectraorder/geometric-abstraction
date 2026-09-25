import unittest
from dataclasses import replace
from unittest.mock import patch
from runtime import DualController,Plan,JevClient,actor_state

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
if __name__=='__main__':unittest.main()
