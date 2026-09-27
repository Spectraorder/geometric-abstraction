"""Conditional prediction must not turn missing or released objects into observations."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from replay_attachment_evidence import depth_consistency, replay_trial


class AttachmentEvidenceTests(unittest.TestCase):
    def frame(self, depth=.5):
        return {'rgb':np.tile(np.array([20,20,200],np.uint8),(4,4,1)),
                'depth_m':np.full((4,4),depth), 'valid_depth':np.ones((4,4),bool),
                'intrinsics':np.array([[100.,0,1.5],[0,100.,1.5],[0,0,1.]]),
                'camera_to_world':np.eye(4), 'time':1.65}

    def test_depth_can_contradict_but_cannot_certify_attachment(self):
        plane=[0,0,.4];size=[.03,.03,.03]
        first=depth_consistency(self.frame(),'blue',[0,0,.5],size,plane)
        self.assertEqual(first['status'],'not_contradicted')
        self.assertFalse(first['attachment_certified'])
        self.assertFalse(first['position_recovered'])
        wrong=depth_consistency(self.frame(),'blue',[.2,0,.5],size,plane)
        self.assertEqual(wrong['status'],'contradicted')
        self.assertEqual(wrong['fraction_outside_margin'],1.)

    def test_background_missing_depth_and_unmasked_robot_color_stay_unknown(self):
        args=('blue',[0,0,.5],[.03,.03,.03],[0,0,.4])
        self.assertEqual(depth_consistency(self.frame(2.),*args)['status'],'unknown')
        frame=self.frame();frame['valid_depth'][:]=False
        self.assertEqual(depth_consistency(frame,*args)['status'],'unknown')
        self.assertEqual(depth_consistency(self.frame(),'green',args[1],args[2],args[3])['status'],'unknown')

    def replay(self, change=None, confirmed=True, observed=True, fresh=True):
        def pose(time, x):
            value={'tool_position':[x,0,.5],'tool_rotation':np.eye(3).tolist(),'time':time}
            return {'method':'proprioception','time':time,'params':{},'observation':value}
        calls=[{'method':'camera','time':.65,'params':{'view':'side'}},pose(.65,0)]
        if change=='gripper':
            calls.append({'method':'gripper','time':1.,'params':{'angle':.8}})
        if fresh:
            calls.append(pose(1.65,.01))
        first={'object':'blue','step':'pick','time':.65,'frame':'world','units':'meters',
               'status':'current_estimate','action_status':'lift_visually_confirmed' if confirmed else 'grasp_unconfirmed',
               'geometry':{'center_m':[0,0,.43],'dimensions_m':[.03,.03,.03],
                           'position_fully_observed':observed}}
        later={'object':'red' if change=='focus' else 'blue','step':'place','time':1.65,
               'status':'unavailable','action_status':'exception: visual evidence unavailable'}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            with np.load(ROOT/'tests/fixtures/mixed_bridge_release_rgbd.npz') as data:
                frame={k[len('initial_'):]:data[k] for k in data.files if k.startswith('initial_')}
            np.savez(path/'camera_000.npz',**frame)
            (path/'api_calls.json').write_text(json.dumps(calls))
            (path/'geometry_records.json').write_text(json.dumps([{'kind':'geometry_step','record':r} for r in (first,later)]))
            return replay_trial(path)

    def test_unknown_current_pose_preserves_only_a_conditional_prediction(self):
        records=self.replay()['records'];self.assertEqual(len(records),1)
        row=records[0]
        np.testing.assert_allclose(row['predicted_center_m'],[.01,0,.43])
        self.assertEqual(row['kind'],'conditional_prediction_not_observation')
        self.assertEqual(row['depth_evidence'],'unknown')
        self.assertIsNone(row['current_visual_center_m'])
        self.assertFalse(row['pose_recovered'])

    def test_release_focus_change_unconfirmed_or_unobserved_anchor_invalidates(self):
        for arguments in ({'change':'gripper'},{'change':'focus'},{'confirmed':False},
                          {'observed':False},{'fresh':False}):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.replay(**arguments)['records'],[])


if __name__=='__main__':
    unittest.main()
