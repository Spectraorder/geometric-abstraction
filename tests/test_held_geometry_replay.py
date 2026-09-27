"""Rigid-frame prediction and explicit guards against stale or released state."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_held_geometry import predict_center, replay_records


def pose(t, position, rotation=None):
    return {'time':t,'method':'proprioception','observation':{
        'tool_position':position,'tool_rotation':np.eye(3).tolist() if rotation is None else rotation}}


def record(t, center, action='placement_unconfirmed', status='current_estimate'):
    return {'kind':'geometry_step','record':{'object':'red','time':t,'frame':'world','units':'meters',
        'status':status,'action_status':action,'geometry':{'center_m':center,'position_fully_observed':True}}}


class HeldReplayTests(unittest.TestCase):
    def test_rotation_transforms_the_object_offset_not_just_tool_translation(self):
        a=pose(1,[0,0,0]);b=pose(2,[1,0,0],[[0,-1,0],[1,0,0],[0,0,1]])
        np.testing.assert_allclose(predict_center([1,0,0],a['observation'],b['observation']),[1,1,0])
        rows=replay_records([record(1,[1,0,0],'lift_visually_confirmed'),record(2,[1,1,0])],[a,b])
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['visual_residual_norm_m'],0)

    def test_gripper_commands_and_release_invalidate_the_attachment(self):
        a=pose(1,[0,0,0]);b=pose(3,[1,0,0]);start=record(1,[1,0,0],'lift_visually_confirmed');end=record(3,[2,0,0])
        self.assertEqual(replay_records([start,end],[a,{'time':2,'method':'gripper'},b]),[])
        self.assertEqual(replay_records([start,record(2,[1,0,0],'released'),end],[a,pose(2,[0,0,0]),b]),[])

    def test_missing_visual_or_robot_evidence_does_not_bridge_an_unknown_gap(self):
        a=pose(1,[0,0,0]);b=pose(3,[1,0,0]);start=record(1,[1,0,0],'lift_visually_confirmed');end=record(3,[2,0,0])
        self.assertEqual(replay_records([start,record(2,[1,0,0],None,'unavailable'),end],[a,b]),[])
        self.assertEqual(replay_records([start,end],[a]),[])

    def test_reflections_are_not_robot_rotations(self):
        a=pose(1,[0,0,0]);b=pose(2,[0,0,0],np.diag([1,1,-1]).tolist())
        with self.assertRaises(ValueError):predict_center([0,0,1],a['observation'],b['observation'])


if __name__=='__main__':unittest.main()
