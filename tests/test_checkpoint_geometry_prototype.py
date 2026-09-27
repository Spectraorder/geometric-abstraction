"""Checkpoint deltas must preserve unknowns and avoid changing current state."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from checkpoint_geometry_prototype import compare_checkpoints


def packet(stamp, center, status='available'):
    return {'coordinate_frame':'world','units':'meters','time':stamp,
            'objects':{'a':{'status':status,'center':center,'position_fully_observed':True,
                            'change':{'estimated_translation_m':[0.,0.,0.]}}},
            'desired_states':{'a':{'center':[9.,9.,9.]}}}


class CheckpointGeometryTests(unittest.TestCase):
    def test_cross_action_change_survives_zero_internal_observation_delta(self):
        before=packet(1.,[0.,0.,.03]);after=packet(5.,[.08,0.,.03]);saved=copy.deepcopy((before,after))
        result=compare_checkpoints(before,after)
        self.assertEqual(result['objects']['a']['estimated_translation_m'],[.08,0.,0.])
        self.assertEqual((before,after),saved)
        self.assertFalse(result['poses_fused_or_predicted'])

    def test_unavailable_or_stale_pose_never_becomes_a_zero_or_predicted_delta(self):
        for status in ('unavailable','stale'):
            row=compare_checkpoints(packet(1.,[0.,0.,0.]),packet(5.,[1.,1.,1.],status))['objects']['a']
            self.assertEqual(row['status'],'unknown')
            self.assertNotIn('estimated_translation_m',row)

    def test_incompatible_frames_units_and_time_order_are_rejected(self):
        for key,value in [('coordinate_frame','camera'),('units','millimeters'),('time',0.)]:
            after=packet(5.,[.08,0.,.03]);after[key]=value
            with self.assertRaises(ValueError):compare_checkpoints(packet(1.,[0.,0.,.03]),after)

    def test_missing_entity_and_nonfinite_estimate_are_not_silently_filled(self):
        before=packet(1.,[0.,0.,0.]);after=packet(2.,[1.,0.,0.]);before['objects']={}
        self.assertEqual(compare_checkpoints(before,after)['objects']['a']['status'],'unknown')
        before=packet(1.,[0.,0.,0.]);after=packet(2.,[float('nan'),0.,0.])
        with self.assertRaises(ValueError):compare_checkpoints(before,after)
