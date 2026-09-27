"""Check actor schema isolation and the sensor-query-to-target decision path."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from online_bound_actor import BoundActor, SCHEMA, validate_response
from online_bridge_actor import OnlineActor, SCHEMA as LEGACY_SCHEMA, validate_response as legacy_validate
from online_bound_decisions import BoundDecisionLoop, reference_feedback
from online_metric_depth import MetricDepthSensorMixin


def decision(action,**kwargs):
    return dict(action=action,skill='none',node='',surface_point_world_m=[],frame_id='',pixels=[],reason='Test.',**{}) | kwargs


class Sensor:
    def __init__(self,output):self.output=Path(output);self.captures=0
    def image(self,name,view):
        self.captures+=1
        path=self.output/(name+'.png')
        rgb=np.zeros((4,4,3),np.uint8)
        Image.fromarray(rgb).save(path)
        np.savez(path.with_suffix('.npz'),rgb=rgb,depth_m=np.full((4,4),.5),
                 valid_depth=np.ones((4,4),bool),time=1.,intrinsics=np.eye(3),camera_to_world=np.eye(4))
        return path,{'time':1.}
    def proprioception(self):return {'time':1.}
    def task(self):return {'kind':'support_graph','nodes':['red','blue']}
    def wait(self,seconds):pass


class API(MetricDepthSensorMixin,Sensor):pass


class Session:
    def __init__(self):self.snapshots=0
    def time(self):return 1.
    def snapshot(self):self.snapshots+=1
    def reference(self,name):
        return dict(reference=name,position_m=[.1,.2,.4],position_fully_observed=not name.startswith('red'),
                    source='rgbd_test',observation_time=1.,vertices_m=[[1,2,3]])


class BoundDecisionTests(unittest.TestCase):
    def test_transport_reused_without_modifying_legacy_schema(self):
        self.assertIs(BoundActor.request.__code__,OnlineActor.request.__code__)
        self.assertIs(BoundActor.request.__globals__['SCHEMA'],SCHEMA)
        self.assertIs(OnlineActor.request.__globals__['SCHEMA'],LEGACY_SCHEMA)
        self.assertNotIn('pixels',LEGACY_SCHEMA['properties'])
        legacy_validate(dict(action='wait',skill='none',node='',plan=[],reason='test'))
        validate_response(decision('sample_depth',frame_id='side:1',pixels=[[1,2]]))

    def test_schema_rejects_cross_action_arguments(self):
        for value in (decision('execute',skill='stack',node='red'),
                      decision('observe',frame_id='side:1'),
                      decision('sample_depth',frame_id='side:1',pixels=[[True,2]]),
                      decision('execute',skill='stack',node='red',surface_point_world_m=[1,2,3],pixels=[[1,2]])):
            with self.assertRaises(ValueError):validate_response(value)

    def test_partial_references_do_not_expose_an_executable_point(self):
        skills=SimpleNamespace(session=Session(),nodes={'red':{'color':'red'},'blue':{'color':'blue'}})
        output=reference_feedback(skills)
        red,blue=output['objects']
        self.assertEqual(red['references']['center']['status'],'unavailable')
        self.assertNotIn('position_m',red['references']['center'])
        self.assertEqual(blue['references']['center']['source'],'rgbd_test')
        self.assertNotIn('vertices_m',blue['references']['center'])

    def test_both_interfaces_query_then_submit_the_same_depth_point(self):
        submitted={}
        for arm in ('basic','symbolic'):
            with self.subTest(arm=arm),TemporaryDirectory() as name:
                api=API(name)
                session=Session()
                skills=SimpleNamespace(session=session,holding=lambda:None,
                    nodes={'red':{'color':'red','supports':['blue']},'blue':{'color':'blue','supports':[]}},
                    bridge=SimpleNamespace(observe=lambda *a,**k:None))
                packets=[]
                class Actor:
                    def __init__(self,output):self.index=0
                    def request(self,prompt,images,*,slow,advance,deadline):
                        self.index+=1;advance()
                        packet=json.loads(prompt.split('Current feedback:\n',1)[1]);packets.append(packet)
                        if self.index==1:
                            self.assertions=(len(images)==2 and slow)
                            return decision('sample_depth',frame_id=packet['cameras']['side']['metric_depth_frame']['frame_id'],pixels=[[1,1]])
                        if self.index==2:
                            if images:raise AssertionError('A depth response must not recapture its image')
                            point=packet['last_result']['samples'][0]['world_point_m']
                            return decision('execute',skill='stack',node='red',surface_point_world_m=point)
                        return decision('observe')
                env=SimpleNamespace(deadline=time.monotonic()+100,advance_while_waiting=lambda:None)
                loop=BoundDecisionLoop(env,api,skills,arm,actor_factory=Actor)
                def execute(skills,operation,callback):
                    submitted[arm]=operation['surface_point_world_m']
                    return dict(status='unconfirmed',holding=None,surface_point_world_m=submitted[arm],
                                task_success_certified=False,private_debug_center=[99,99,99])
                with patch('online_bound_decisions.drive_bound',side_effect=execute):
                    loop.step();loop.step();loop.step()
                self.assertTrue(loop.actor.assertions)
                self.assertEqual(loop.requests,3)
                self.assertEqual(api.depth_query_count,1)
                self.assertEqual(loop.completed_operations,1)
                self.assertNotIn('private_debug_center',packets[2]['last_result'])
                self.assertEqual('geometry' in packets[0],arm=='symbolic')
                self.assertEqual(session.snapshots,3 if arm=='symbolic' else 0)
                self.assertEqual(api.captures,2 if arm=='symbolic' else 3)
        self.assertEqual(submitted['basic'],[.5,.5,.5])
        self.assertEqual(submitted['basic'],submitted['symbolic'])


if __name__=='__main__':unittest.main()
