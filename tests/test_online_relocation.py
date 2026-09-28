"""Relocation contracts, dimension transfer, and decision isolation."""
import ast
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from online_relocation_scene import recipe, RelocationEnv
from online_relocation_runtime import relocation_library, RelocationSkills, RelocationActor, RelocationLoop, CONTRACT
from online_bound_actor import SCHEMA
from online_bound_decisions import BoundDecisionLoop
from online_ring_actor import RingActor


class RelocationTests(unittest.TestCase):
    def test_goal_exposes_final_supports_without_action_order_or_coordinates(self):
        goal=recipe(1).public_goal(); roles=goal['roles']
        nodes={n['key']:n for n in goal['nodes']}
        self.assertEqual(nodes[roles['middle']]['supports'],[roles['destination']])
        self.assertEqual(nodes[roles['upper']]['supports'],[roles['middle']])
        self.assertTrue(all('grid' not in n for n in nodes.values()))
        self.assertEqual(goal['length_unit']['observed_cube'],roles['destination'])
        self.assertIn('equal size',goal['assumptions'])

    def test_dimensions_transfer_without_pose_or_table_height_bootstrap(self):
        tree=ast.parse(relocation_library())
        owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MixedSession')
        class Pnp:
            def __init__(self,api): self.models={}; self.calls=[]
            def set_observer(self,callback): pass
            def observe(self,color):
                self.calls.append((color,dict(self.models.get(color,{}))))
                if color=='yellow':self.models[color]={'height':.031,'dims':np.array([.031,.031]),'center':[99,99,99]}
                else:
                    if not self.models.get(color):raise AssertionError('Elevated cube bootstrapped against table')
                    if 'center' in self.models[color]:raise AssertionError('Reference pose was copied')
                    if color=='blue':raise ValueError('Occluded current pose')
        class Ledger:
            def __init__(self): self.missing=[]
            def unavailable(self,*args): self.missing.append(args)
        class API:
            def task(self):return {'roles':{'destination':'yellow'}}
            def proprioception(self):return {'time':4.}
        scope=dict(np=np,MixedPnPObservation=lambda *a,**kw:object(),MetricScene=Ledger,ObservedRGBDPickPlace=Pnp)
        exec(compile(ast.Module(body=[owner],type_ignores=[]),'<test>','exec'),scope)
        session=scope['MixedSession'](API(),[{'color':c} for c in ['red','blue','yellow','green']])
        self.assertEqual(session.pnp.calls[0][0],'yellow')
        self.assertEqual(session.pnp.models['blue']['height'],.031)
        self.assertEqual(session.ledger.missing[0][0],'blue')
        self.assertEqual(session.ledger.missing[0][1],4.)

    def test_goal_allows_repair_but_preserves_final_anchor_constraint(self):
        goal=recipe(2).public_goal()
        self.assertEqual(goal['foundation_translation_tolerance_m'],.005)
        self.assertIn('may be repaired',goal['foundation_constraint'])
        self.assertIn('displacement can be repaired',CONTRACT)

    def test_displaced_foundation_can_be_restored_to_a_valid_goal(self):
        env=object.__new__(RelocationEnv)
        env.roles={r:r for r in ('source','destination','middle','upper')}
        env.order=tuple(env.roles)
        positions=np.array([[0,0,.015],[.2,0,.015],[.2,0,.045],[.2,0,.075]])
        indices={name:i for i,name in enumerate(env.order)}
        env.anchor_positions={r:positions[indices[r]].copy() for r in ('source','destination')}
        env.anchor_violation=False
        env.model=SimpleNamespace(body=lambda name:SimpleNamespace(id=indices[name.split('/')[0]]),
                                  body_mass=np.ones(4))
        env.data=SimpleNamespace(xipos=positions)
        env._object_state=lambda n:dict(position=positions[indices[n]],linear_velocity=np.zeros(3),angular_velocity=np.zeros(3))
        pairs={frozenset(p) for p in (('source','table'),('destination','table'),
                                      ('destination','middle'),('middle','upper'))}
        env._objects_in_contact=lambda a,b:frozenset((a,b)) in pairs
        env.support_frame=lambda n:(np.array([0,0,1.]),positions[indices[n]]+[0,0,.015],[0,1])
        env.geometry_bounds=lambda n:(positions[indices[n]]-.015,positions[indices[n]]+.015)
        env.support_margin=lambda n,p:.015
        self.assertTrue(env.structure_status(env.order)['stable'])
        positions[0,0]=.006
        self.assertFalse(env.structure_status(env.order)['stable'])
        positions[0,0]=0.
        self.assertTrue(env.structure_status(env.order)['stable'])

    def test_actor_and_loop_do_not_mutate_frozen_module_globals(self):
        self.assertIs(RelocationActor._bound_request.__globals__['SCHEMA'],SCHEMA)
        self.assertIsNot(RingActor.request.__globals__['SCHEMA'],SCHEMA)
        self.assertEqual(RelocationLoop.step.__globals__['CONTRACT'],CONTRACT)
        self.assertNotEqual(BoundDecisionLoop.step.__globals__['CONTRACT'],CONTRACT)


if __name__=='__main__':unittest.main()
