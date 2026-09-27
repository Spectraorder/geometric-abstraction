"""Target-focused action records preserve freshness, goals and uncertainty."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'policies'))
from geometry_step_state import GeometryStepState


def fit(t, x=0., observed=True):
    return {'coordinate_frame':'world','units':'meters','observation_time':t,
            'shape':'cube','center':np.array([x,0.,.42]), 'axis_dimensions':[.03]*3,
            'normal':[0,0,1], 'position_fully_observed':observed, 'inferred_orientation':False}


class StepStateTests(unittest.TestCase):
    def test_one_object_current_geometry_and_one_step_delta(self):
        state=GeometryStepState('red')
        first=state.update('before_pick',fit(1),1)
        after=state.update('pick',fit(2,.01,False),2,action_status='lift_visually_confirmed')
        self.assertEqual(after['object'],'red')
        self.assertEqual(after['estimated_change']['center_delta_m'],[.01,0.,0.])
        self.assertFalse(after['estimated_change']['both_positions_fully_observed'])
        self.assertNotIn('facts',after)
        self.assertNotIn('objects',after)
        self.assertNotIn('history',after)
        first['geometry']['center_m'][0]=100
        self.assertEqual(state.previous['center'][0],.01)

    def test_goal_reference_is_not_confused_with_center_and_remains_independent(self):
        state=GeometryStepState('red');goal=np.array([.1,0,.4])
        state.set_goal(goal,reference='bottom_face');goal[:]=99
        record=state.update('pick',fit(1),1)
        self.assertIsNone(record['remaining_translation_m'])
        ref={'reference':'red.bottom_face','type':'plane_patch','center_m':[.08,0,.405],
             'normal':[0,0,-1],'coordinate_frame':'world','units':'meters','observation_time':2}
        after=state.update('place',fit(2,.08),2,references=[ref])
        self.assertEqual(after['remaining_translation_m'],[.02,0.,-.005])
        self.assertEqual(after['goal']['point_m'],[.1,0.,.4])
        self.assertEqual(after['geometry']['center_m'],[.08,0.,.42])
        with self.assertRaises(ValueError):
            state.update('check',fit(2),2,references=[dict(ref,reference='blue.top_face')])

    def test_unknown_and_stale_data_never_reappear_as_current_or_bridge_a_gap(self):
        state=GeometryStepState('red');state.set_goal([.1,0,.4])
        state.update('pick',fit(1),1)
        unknown=state.update('place',None,2)
        self.assertNotIn('geometry',unknown);self.assertIsNone(unknown['remaining_translation_m'])
        fresh=state.update('check',fit(3),3)
        self.assertNotIn('estimated_change',fresh)
        stale=state.update('wait',fit(3),4)
        self.assertEqual(stale['status'],'stale');self.assertNotIn('geometry',stale)
        with self.assertRaises(ValueError):state.update('backwards',fit(2),2)

    def test_requested_points_require_same_timestamp(self):
        state=GeometryStepState('red')
        ref={'reference':'red.corner_0','type':'point','position_m':[0,0,.4],
             'coordinate_frame':'world','units':'meters','observation_time':0}
        for stamp in (0, float('nan')):
            with self.assertRaises(ValueError):
                state.update('place',fit(1),1,references=[dict(ref, observation_time=stamp)])
        self.assertIsNone(state.previous)


if __name__=='__main__':unittest.main()
