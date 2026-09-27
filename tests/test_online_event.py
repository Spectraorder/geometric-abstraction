"""Motion parity, cooperative cancellation, bounded memory and retry evidence."""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from online_event_skills import motion_generator
from online_event_memory import RetryGate, PublicMemory, parse_plan, validate_operation
from online_bridge_skills import plain


class FakePnP:
    def __init__(self, held=False):
        self.position=np.array([0., 0., .5]); self.calls=[]; self.observations=0
        self.grasp_fraction=.4; self.release_clearance=.003; self.position_tolerance=.004
        self.held={'color':'red','confirmed':True,'park':self.position.copy()} if held else None
        self.api=SimpleNamespace(gripper=lambda *a,**kw:self.calls.append(('gripper',a,kw)),
                                 wait=lambda *a,**kw:self.calls.append(('wait',a,kw)))
    def tool(self): return self.position.copy()
    def move(self, point, seconds=1.5):
        self.position=np.asarray(point).copy(); self.calls.append(('move',self.position.tolist(),seconds))
    def observe(self, *args):
        self.observations+=1
        center=np.array([.1, .1, .415]) if self.observations==1 and self.held is None else self.tool()-[0,0,.012]
        return {'center':center,'top':center[2]+.015,'height':.03,'bottom':center[2]-.015,'extent_z':.015}


def original(name):
    tree=ast.parse((Path(__file__).resolve().parents[1]/'policies/pick_place_rgbd_shared.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    scope={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'original','exec'),scope)
    return scope[name]


class EventTests(unittest.TestCase):
    def test_segmenting_preserves_pick_and_place_commands_and_result(self):
        for name,args in [('pick',('red',)),('place',([.1,.1,.4],))]:
            old=FakePnP(name=='place');new=FakePnP(name=='place')
            expected=original(name)(old,*args)
            generator=motion_generator(name)(new,*args);boundaries=[]
            while True:
                try: boundaries.append(next(generator))
                except StopIteration as done:
                    actual=done.value;break
            self.assertGreater(len(boundaries),4)
            self.assertEqual(old.calls,new.calls)
            self.assertEqual(plain(old.held),plain(new.held))
            self.assertEqual(plain(expected),plain(actual))

    def test_cancellation_does_not_run_remaining_motion_or_reset(self):
        robot=FakePnP();gen=motion_generator('pick')(robot,'red')
        next(gen);before=list(robot.calls);gen.close()
        self.assertEqual(robot.calls,before)
        self.assertIsNone(robot.held)

    def test_interruption_after_closing_keeps_uncertain_holding_for_recovery(self):
        robot=FakePnP();gen=motion_generator('pick')(robot,'red')
        while robot.held is None:
            next(gen)
        self.assertFalse(robot.held['confirmed'])
        before=list(robot.calls);gen.close()
        self.assertEqual(robot.calls,before)
        self.assertEqual(robot.held['color'],'red')
        with self.assertRaises(RuntimeError):original('pick')(robot,'red')

    def test_failed_pick_is_not_retried_until_geometric_evidence_changes(self):
        gate=RetryGate();op=('execute','stack','red');evidence={'red':[0.,0.,.4],'blue':None}
        gate.record(op,evidence,False)
        self.assertTrue(gate.blocked(op,{'red':[.001,0.,.4],'blue':None}))
        self.assertFalse(gate.blocked(op,{'red':[0.,0.,.4],'blue':[0.,0.,.4]}))
        self.assertFalse(gate.blocked(op,{'red':[.02,0.,.4],'blue':None}))
        self.assertFalse(gate.blocked(('clear','none','red'),evidence))
        gate.record(op,evidence,True);self.assertFalse(gate.blocked(op,evidence))

    def test_release_unlocks_retry_even_without_object_translation(self):
        gate=RetryGate();op=('execute','stack','red')
        gate.record(op,{'red':[0,0,.4],'@holding':[1,0]},False)
        self.assertFalse(gate.blocked(op,{'red':[0,0,.4],'@holding':None}))

    def test_memory_is_bounded_and_does_not_store_geometry_or_errors(self):
        memory=PublicMemory()
        for i in range(20):
            memory.record({'action':'execute','skill':'bridge','node':'red'},
                          {'status':'unconfirmed','error':'private numeric details','center':[1,2,3]})
        self.assertEqual(len(memory.packet()['recent_events']),4)
        self.assertNotIn('center',str(memory.packet()))
        self.assertNotIn('private numeric',str(memory.packet()))

    def test_plan_rejects_unknown_keys_and_wrong_skill(self):
        nodes={'red':{'supports':[]},'blue':{'supports':['red']}}
        self.assertEqual(len(parse_plan(['bridge:red','stack:blue'],nodes)),2)
        for entries in (['stack:red'],['bridge:green'],['red'],['bridge:red']*6):
            with self.assertRaises(ValueError):parse_plan(entries,nodes)



class SchedulerTests(unittest.TestCase):
    def test_plan_runs_without_intermediate_model_calls_and_failure_discards_tail(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch, Mock
        import time
        import run_online_event_development as runtime
        for fail in (False, True):
            class Env:
                def __init__(self, *a, **kw):
                    self.episode_started=None;self.success_time=1.;self.success_evidence={};self.resets=0
                def reset(self, seed):self.resets+=1
                def start_episode(self, budget, realtime):
                    self.episode_started=time.monotonic();self.deadline=time.monotonic()+budget
                def advance_while_waiting(self):pass
                def timing(self):return {'reset_count':self.resets}
                def close(self):pass
            class API:
                def __init__(self, env, output):
                    self.output=output;self.camera_count=0;self.action_seconds=0.
                def task(self):return {'nodes':[]}
            class Skills:
                def __init__(self, api):
                    self.api=api;self.provenance={};self.order=['red','blue']
                    self.nodes={'red':{'supports':[]},'blue':{'supports':['red']}}
                    self.session=SimpleNamespace(time=lambda:0.)
                def evidence(self):return {'red':[0,0,0],'blue':[1,0,0]}
                def holding(self):return None
            class Actor:
                calls=0
                def __init__(self,*a):self.turn=0;self.receipts=[]
                def request(self,*a,**kw):
                    self.turn+=1;Actor.calls+=1
                    return ({'action':'execute','skill':'bridge','node':'red','plan':['stack:blue'],'reason':'build'}
                            if Actor.calls==1 else
                            {'action':'clear','skill':'none','node':'red','plan':[],'reason':'recover'})
            events=[]
            def execute(skills, operation, boundary):
                events.append(operation)
                return {'status':'unconfirmed' if fail and len(events)==1 else 'visually_consistent'}
            with TemporaryDirectory() as folder, patch.multiple(runtime,
                OnlineBridgeTowerEnv=Env,SensorAPI=API,EventSkills=Skills,OnlineActor=Actor,
                feedback=Mock(return_value=({},[])),drive=execute,
                settle=Mock(side_effect=runtime.EpisodeSuccess)):
                r=runtime.run(Path(folder)/'episode',2028100101,'basic')
            self.assertEqual(r['status'],'success')
            self.assertEqual(r['timing']['reset_count'],1)
            self.assertEqual(Actor.calls,2 if fail else 1)
            self.assertEqual(events[1]['action'],'clear' if fail else 'execute')
            if not fail:self.assertEqual(events[1]['node'],'blue')

class ValidationTests(unittest.TestCase):
    def test_initial_operation_is_validated_like_plan_tail(self):
        nodes={'red':{'supports':[]}}
        for action,skill,node in [('execute','stack','red'),('execute','none','red'),
                                  ('clear','bridge','red'),('observe','none','red')]:
            with self.assertRaises(ValueError):
                validate_operation(dict(action=action,skill=skill,node=node),nodes)
        validate_operation(dict(action='execute',skill='bridge',node='red'),nodes)

    def test_memory_does_not_grow_with_invented_object_keys(self):
        memory=PublicMemory()
        for i in range(100):
            memory.record(dict(action='execute',skill='bridge',node=str(i)),{'status':'invalid_plan'})
        self.assertEqual(len(memory.packet()['last_attempt_per_object']),5)

class ContextTests(unittest.TestCase):
    def test_context_rotation_preserves_memory_and_environment_after_invalid_actions(self):
        import run_online_event_development as runtime
        from tempfile import TemporaryDirectory
        from unittest.mock import Mock, patch
        import time
        environments=[];prompts=[]
        class Env:
            def __init__(self,*a,**kw):
                environments.append(self);self.resets=0;self.episode_started=None
                self.success_time=1.;self.success_evidence={}
            def reset(self,seed):self.resets+=1
            def start_episode(self,budget,realtime):
                self.episode_started=time.monotonic();self.deadline=time.monotonic()+budget
            def advance_while_waiting(self):pass
            def timing(self):return {'reset_count':self.resets}
            def close(self):pass
        class API:
            def __init__(self,env,output):self.output=output;self.camera_count=0;self.action_seconds=0
            def task(self):return {'nodes':[]}
        class Skills:
            def __init__(self,api):
                self.api=api;self.provenance={};self.nodes={'red':{'supports':[]}}
                self.session=SimpleNamespace(time=lambda:0.)
            def evidence(self):return {'red':None}
            def holding(self):return None
        class Actor:
            def __init__(self,*a):self.turn=0;self.receipts=[]
            def request(self,prompt,*a,**kw):
                self.turn+=1;prompts.append(prompt)
                return (dict(action='execute',skill='stack',node='red',plan=[],reason='invalid')
                        if len(prompts)<4 else
                        dict(action='wait',skill='none',node='',plan=[],reason='settle'))
        with TemporaryDirectory() as folder, patch.multiple(runtime,
            OnlineBridgeTowerEnv=Env,SensorAPI=API,EventSkills=Skills,OnlineActor=Actor,
            feedback=Mock(side_effect=lambda *a,**kw:({},[])),
            settle=Mock(side_effect=runtime.EpisodeSuccess)):
            r=runtime.run(Path(folder)/'episode',2028100101,'basic')
        self.assertEqual(r['status'],'success')
        self.assertEqual(len(environments),1)
        self.assertEqual(environments[0].resets,1)
        self.assertEqual(r['contexts'],2)
        self.assertEqual(r['model_requests_attempted'],4)
        self.assertIn('invalid_plan',prompts[-1])
        self.assertIn('Goal:',prompts[-1])
        self.assertEqual(r['dispatched_operations'],0)

if __name__=='__main__':unittest.main()
