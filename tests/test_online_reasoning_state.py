"""Evidence-preserving constraint projection and relevant retry regressions."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from online_reasoning_state import geometric_constraints, RelevantRetryGate, project_feedback, ancestors, observed_center

NODES = {'base': {'supports': []}, 'middle': {'supports': ['base']},
         'top': {'supports': ['middle']}, 'other': {'supports': []}}


def packet():
    return {'time':1.,'placement_reference':'center_xy_min_z','placement_tolerance_m':{},
            'objects':[{'node':key,'status':'current_estimate','position_fully_observed':True,
                        'orientation_compatible':True,'supports':{p:'true' for p in node['supports']},
                        'placement':{'translation_within_tolerance':True,'remaining_translation_m':[0,0,0]}}
                       for key,node in NODES.items()]}


class ConstraintTests(unittest.TestCase):
    def test_locally_aligned_child_retains_violated_ancestor(self):
        source=packet();source['objects'][0]['placement']['translation_within_tolerance']=False
        rows={r['id']:r for r in geometric_constraints(source,NODES)['objects']}
        self.assertEqual(rows['top']['local_goal_geometry'],'true')
        self.assertEqual(rows['top']['support_chain_geometry'],'false')
        self.assertEqual(rows['top']['violated_ancestors'],['base'])
        self.assertEqual(rows['other']['support_chain_geometry'],'true')

    def test_unobserved_position_never_becomes_valid_residual(self):
        source=packet();source['objects'][0]['position_fully_observed']=False
        rows={r['id']:r for r in geometric_constraints(source,NODES)['objects']}
        self.assertEqual(rows['base']['local_goal_geometry'],'unknown')
        self.assertNotIn('translation_error_m',rows['base'])
        self.assertEqual(rows['top']['support_chain_geometry'],'unknown')
        self.assertEqual(rows['top']['unknown_ancestors'],['base'])

    def test_false_support_is_not_masked_by_unknown_translation(self):
        source=packet();source['objects'][1]['position_fully_observed']=False
        source['objects'][1]['supports']['base']='false'
        rows={r['id']:r for r in geometric_constraints(source,NODES)['objects']}
        self.assertEqual(rows['middle']['local_goal_geometry'],'false')
        self.assertIn('translation',rows['middle']['unknown'])
        self.assertIn('support:base',rows['middle']['violated'])

    def test_base_never_gets_derived_geometry_and_source_is_not_mutated(self):
        raw={'holding':{'color':'base','lift_confirmed':False}}
        self.assertNotIn('geometry',project_feedback(raw,NODES))
        full=dict(raw,geometry=packet());old=deepcopy(full)
        project_feedback(full,NODES)
        self.assertEqual(full,old)

    def test_partial_pose_cannot_supply_retry_or_motion_evidence(self):
        for flag in (False, None, 'unknown'):
            self.assertIsNone(observed_center({'center':[.2,.3,.4],'fully_observed':flag}))
        self.assertEqual(observed_center({'center':[.2,.3,.4],'fully_observed':True}),[.2,.3,.4])
        self.assertIsNone(observed_center({'center':[float('nan'),0,0],'fully_observed':True}))

    def test_cycle_is_rejected(self):
        with self.assertRaises(ValueError):ancestors({'a':{'supports':['b']},'b':{'supports':['a']}},'a')


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.gate=RelevantRetryGate(NODES);self.op=('execute','stack','top')
        self.evidence={key:[0,0,0] for key in NODES};self.evidence['@holding']=None
        self.gate.record(self.op,self.evidence,False)

    def test_unrelated_motion_and_lost_visibility_do_not_unlock(self):
        changed=deepcopy(self.evidence);changed['other']=[.1,0,0];changed['top']=None
        self.assertTrue(self.gate.blocked(self.op,changed))

    def test_ancestor_motion_and_release_unlock(self):
        changed=deepcopy(self.evidence);changed['base']=[.02,0,0]
        self.assertFalse(self.gate.blocked(self.op,changed))
        changed=deepcopy(self.evidence);changed['@holding']=[1,1]
        self.assertFalse(self.gate.blocked(self.op,changed))

    def test_new_observation_unlocks_and_failed_state_is_copied(self):
        self.evidence['top']=None;self.gate.record(self.op,self.evidence,False)
        self.evidence['top']=[0,0,0]
        self.assertFalse(self.gate.blocked(self.op,self.evidence))

    def test_clear_depends_on_all_objects(self):
        op=('clear','none','top');self.gate.record(op,self.evidence,False)
        changed=deepcopy(self.evidence);changed['other']=[.02,0,0]
        self.assertFalse(self.gate.blocked(op,changed))

if __name__=='__main__':unittest.main()
