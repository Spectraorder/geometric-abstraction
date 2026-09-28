"""Online Ring preserves current evidence, actor targets and a single episode."""
from pathlib import Path
import sys
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from online_ring_actor import validate_response, RingActor
from online_ring_scene import OnlineRingEnv
from online_ring_skills import RingSkills, LiftBoundary, choose_training_prior
from online_ring_loop import compact_geometry


def decision(action='move_pad'):
    return dict(action=action,point_m=[.1,-.25,.5] if action=='move_pad' else [],
                roll=.2,jaw=.5,seconds=1.2,frame_id='',pixels=[],reason='Use current evidence.')


class RingTests(unittest.TestCase):
    def test_schema_rejects_malformed_targets_and_inactive_queries(self):
        self.assertEqual(validate_response(decision())['point_m'],[.1,-.25,.5])
        for change in ({'point_m':[1,2]},{'point_m':[0,0,float('nan')]},{'roll':9},
                       {'pixels':[[0,0]]},{'action':'hidden_truth'},{'seconds':-1}):
            with self.assertRaises(ValueError):validate_response(dict(decision(),**change))

    def test_depth_query_is_bounded(self):
        good=dict(decision('sample_depth'),frame_id='side:1',pixels=[[1,2]])
        validate_response(good)
        with self.assertRaises(ValueError):validate_response(dict(good,pixels=[[1,2]]*17))

    def test_exact_actor_pad_target_is_not_replaced(self):
        calls=[]
        skill=object.__new__(RingSkills);skill.action_seconds=0
        skill.sensors=SimpleNamespace(robot=SimpleNamespace(move_pad=lambda *args:calls.append(args)))
        d=decision();result=skill.execute(d)
        self.assertEqual(calls,[(d['point_m'],d['roll'],d['seconds'])])
        self.assertEqual(result['submitted_point_m'],d['point_m'])

    def test_grasp_returns_before_any_learned_placement(self):
        skill=object.__new__(RingSkills);skill.grasp_attempts=0;skill.last_grasp_opening=None;skill.api=object()
        def run(api):
            skill.learned['place'](api,None,None,None,None,.4)
            self.fail('Learned placement continued past online boundary')
        skill.learned={'run':run}
        self.assertEqual(skill.grasp_lift()['status'],'lift_observed')
        self.assertEqual(skill.last_grasp_opening,.4)

    def test_reset_cannot_be_repeated(self):
        env=object.__new__(OnlineRingEnv);env.reset_count=1
        with self.assertRaises(RuntimeError):env.reset(1)

    def test_success_requires_sustained_private_evidence(self):
        env=object.__new__(OnlineRingEnv)
        env.data=SimpleNamespace(time=0.);env.episode_started=time.monotonic()
        env.last_success_check=-100;env.stable_since=env.stable_position=None
        env.success_time=env.success_evidence=None;env.monitor_rows=[]
        status=dict(on_peg=True,ring_center_m=[0,0,0],reasons=[])
        env.ring_status=lambda:status
        env._monitor_success()
        env.data.time=1.;env._monitor_success();self.assertIsNone(env.success_time)
        status['on_peg']=False;env.data.time=1.2;env._monitor_success()
        status['on_peg']=True;env.data.time=1.4;env._monitor_success()
        env.data.time=3.1;env._monitor_success();self.assertIsNone(env.success_time)
        env.data.time=3.5;env._monitor_success();self.assertIsNotNone(env.success_time)

    def test_compact_feedback_preserves_evidence_and_unavailable_state(self):
        ring=dict(status='current_estimate',time=3.,frame='world',units='meters',
                  hole=dict(center_m=[0,0,1],axis=[0,0,1],inscribed_radius_m=.02,
                            inner_boundary_m=[[1,2,3]],evidence='initialized_shape_registered_to_current_edges'),
                  height_m=.015,minimum_z_m=.9,bottom_evidence='model_inferred')
        def observe(name):
            if name=='peg':raise ValueError('not visible')
            return ring
        packet=compact_geometry(SimpleNamespace(observe=observe))
        self.assertEqual(packet['ring']['time'],3.)
        self.assertEqual(packet['ring']['hole']['evidence'],ring['hole']['evidence'])
        self.assertNotIn('inner_boundary_m',packet['ring']['hole'])
        self.assertEqual(packet['peg']['status'],'unavailable')
        self.assertNotIn('top_center_m',packet['peg'])

    def test_actor_uses_ring_schema_without_mutating_tower(self):
        from online_bridge_actor import OnlineActor, SCHEMA
        self.assertIn('point_m',RingActor.request.__globals__['SCHEMA']['required'])
        self.assertNotIn('point_m',SCHEMA['required'])
        self.assertIsNot(RingActor.request,OnlineActor.request)


if __name__=='__main__':unittest.main()
