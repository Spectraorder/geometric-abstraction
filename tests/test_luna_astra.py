"""No network: verify role/effort routing and prevent accidental Jev usage."""
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch
from aspire_so100.astra_decision import Astra
from aspire_so100.decision_session import DecisionSession

STATE={'version':0,'time_s':0.,'phase':'initial_observation','visible_colors':['blue','red','green'],
 'table_fit_valid':True,'grasp_confirmed':None,'reobservations_left':2,'grasp_retries_left':1}

class FakeModel:
    def __init__(self,reasoning='low',model='gpt-6-astra'):
        self.reasoning,self.model,self.calls=reasoning,model,[]
    def replan(self,state,plan,trigger):
        self.calls.append({'role':'plan','effort':self.reasoning});return replace(plan,version=plan.version+1)
    def decide(self,state,plan):
        self.calls.append({'role':'decision','effort':self.reasoning})
        return {'action':'continue','confidence':1.,'state_version':state['version'],'plan_version':plan.version}

class Routing(unittest.TestCase):
    def test_luna_grid_routes_real_model_ids_and_efforts_without_jev(self):
        for fast in ['low','medium','high']:
            for slow in ['low','medium','high','xhigh']:
                with self.subTest(fast=fast,slow=slow),tempfile.TemporaryDirectory() as d:
                    with patch('aspire_so100.decision_session.Astra',FakeModel),patch('aspire_so100.decision_session.JevClient',side_effect=AssertionError('Jev forbidden')):
                        session=DecisionSession('luna_hybrid',d,fast_effort=fast,slow_effort=slow)
                        self.assertEqual(session.decide(STATE,0.)['action'],'continue')
                        self.assertEqual(session.decider.model,'gpt-6-luna')
                        self.assertEqual(session.decider.reasoning,fast)
                        self.assertEqual(session.planner.reasoning,slow)
                        self.assertEqual(len(session.planner.calls),1)
    def test_luna_only_never_calls_slow_model(self):
        with tempfile.TemporaryDirectory() as d,patch('aspire_so100.decision_session.Astra',FakeModel):
            s=DecisionSession('luna_only',d)
            with patch.object(s.planner,'replan',side_effect=AssertionError('Slow forbidden')):
                self.assertEqual(s.decide(STATE,0.)['action'],'continue')
                self.assertEqual(s.planner.calls,[])
    def test_pure_astra_uses_same_requested_effort_for_both_roles(self):
        with tempfile.TemporaryDirectory() as d,patch('aspire_so100.decision_session.Astra',FakeModel):
            s=DecisionSession('astra_only',d,fast_effort='xhigh',slow_effort='xhigh')
            s.decide(STATE,0.)
            self.assertEqual((s.decider.model,s.decider.reasoning,s.planner.reasoning),('gpt-6-astra','xhigh','xhigh'))
    def test_no_unrequested_model_or_effort(self):
        with self.assertRaises(ValueError):Astra(model='jev')
        with self.assertRaises(ValueError):Astra(reasoning='automatic')

if __name__=='__main__':unittest.main()
