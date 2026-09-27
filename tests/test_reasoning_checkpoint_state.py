"""Episode, time, uncertainty, and prediction/evidence boundaries for the prototype."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reasoning_checkpoint_state import ReasoningCheckpoints


def packet(t, xyz=(0.,0.,0.), observed=True):
    return {'coordinate_frame':'world','units':'meters','time':t,'objects':{'a':{
        'status':'available' if observed else 'unavailable','last_observed':t,'center':list(xyz),
        'rotation':np.eye(3).tolist(),'constrained_axes':[True,True,False],
        'position_fully_observed':False}}}


class ReasoningStateTests(unittest.TestCase):
    def test_expected_motion_never_becomes_current_geometry_and_residual_is_projected(self):
        state=ReasoningCheckpoints('episode'); before=packet(1); saved=copy.deepcopy(before)
        state.remember('pre_pick',before,episode='episode')
        transform=np.eye(4);transform[:3,3]=[.1,0.,.2]
        expectation=state.expect_transform('pre_pick','a',transform,'if rigidly attached without slip',episode='episode')
        expectation['center_m'][0]=99.
        after=packet(2,(.09,0.,.4)); result=state.compare('pre_pick',after,episode='episode')
        self.assertEqual(before,saved)
        self.assertEqual(result['current_estimates']['objects']['a']['center'],[.09,0.,.4])
        residual=result['conditional_effects']['a']
        np.testing.assert_allclose(residual['estimated_residual_m'],[-.01,0.,.2])
        np.testing.assert_allclose(residual['observable_residual_m'],[-.01,0.,0.])
        self.assertFalse(residual['condition_verified'])
        self.assertFalse(result['temporal_pose_fusion'])

    def test_missing_current_evidence_preserves_unknown(self):
        state=ReasoningCheckpoints('episode');state.remember('start',packet(1),episode='episode')
        state.expect_transform('start','a',np.eye(4),'if stationary',episode='episode')
        result=state.compare('start',packet(2,observed=False),episode='episode')
        self.assertEqual(result['checkpoint_changes']['objects']['a']['status'],'unknown')
        self.assertEqual(result['conditional_effects']['a']['comparison_status'],'unknown')
        self.assertNotIn('center',result['current_estimates']['objects']['a'])

    def test_episode_reversal_staleness_and_post_observation_prediction(self):
        state=ReasoningCheckpoints('episode');state.remember('start',packet(1),episode='episode')
        with self.assertRaises(ValueError):state.compare('start',packet(2),episode='another')
        stale=packet(2);stale['objects']['a']['last_observed']=1
        self.assertEqual(state.compare('start',stale,episode='episode')['current_estimates']['objects']['a']['status'],'unknown')
        with self.assertRaises(ValueError):state.compare('start',packet(1),episode='episode')
        with self.assertRaisesRegex(ValueError,'later observation'):
            state.expect_transform('start','a',np.eye(4),'if stationary',episode='episode')

    def test_eviction_removes_expectations_and_labels_cannot_be_overwritten(self):
        state=ReasoningCheckpoints('episode',capacity=1)
        state.remember('start',packet(1),episode='episode')
        state.expect_transform('start','a',np.eye(4),'if stationary',episode='episode')
        with self.assertRaises(ValueError):state.remember('start',packet(1),episode='episode')
        state.remember('next',packet(2),episode='episode')
        self.assertFalse(state.effects)
        with self.assertRaises(KeyError):state.compare('start',packet(3),episode='episode')

    def test_source_uncertainty_is_not_removed_by_a_fully_observed_endpoint(self):
        state=ReasoningCheckpoints('episode');before=packet(1)
        before['objects']['a']['constrained_axes']=[True,False,False]
        state.remember('start',before,episode='episode')
        state.expect_transform('start','a',np.eye(4),'if stationary',episode='episode')
        after=packet(2,(0.,.1,0.));after['objects']['a']['constrained_axes']=[True,True,True]
        result=state.compare('start',after,episode='episode')
        change=result['checkpoint_changes']['objects']['a']
        self.assertEqual(change['joint_constraint_rank'],1)
        np.testing.assert_allclose(change['jointly_constrained_translation_m'],[0,0,0])
        effect=result['conditional_effects']['a']
        np.testing.assert_allclose(effect['observable_residual_m'],[0,.1,0])
        np.testing.assert_allclose(effect['jointly_constrained_residual_m'],[0,0,0])
        self.assertEqual(effect['joint_constraint_rank'],1)
        self.assertFalse(effect['condition_verified'])

    def test_disjoint_translation_constraints_return_unknown_vector_not_zero(self):
        state=ReasoningCheckpoints('episode');before=packet(1)
        before['objects']['a']['constrained_axes']=[True,False,False]
        state.remember('start',before,episode='episode')
        state.expect_transform('start','a',np.eye(4),'if stationary',episode='episode')
        after=packet(2,(.1,.1,0.));after['objects']['a']['constrained_axes']=[False,True,False]
        result=state.compare('start',after,episode='episode')
        for row,key in [(result['checkpoint_changes']['objects']['a'],'jointly_constrained_translation_m'),
                        (result['conditional_effects']['a'],'jointly_constrained_residual_m')]:
            self.assertEqual(row['joint_constraint_rank'],0)
            self.assertIsNone(row[key])

    def test_effect_transports_source_constraints_with_the_rigid_transform(self):
        state=ReasoningCheckpoints('episode');before=packet(1)
        before['objects']['a']['constrained_axes']=[True,False,False]
        state.remember('start',before,episode='episode')
        transform=np.eye(4);transform[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
        state.expect_transform('start','a',transform,'if attached during rotation',episode='episode')
        after=packet(2,(0,.02,0));after['objects']['a']['constrained_axes']=[False,True,False]
        result=state.compare('start',after,episode='episode')
        self.assertEqual(result['checkpoint_changes']['objects']['a']['joint_constraint_rank'],0)
        effect=result['conditional_effects']['a']
        self.assertEqual(effect['joint_constraint_rank'],1)
        np.testing.assert_allclose(effect['jointly_constrained_residual_m'],[0,.02,0])

    def test_invalid_rigid_effect_and_uncertainty_metadata_rejected(self):
        state=ReasoningCheckpoints('episode');state.remember('start',packet(1),episode='episode')
        reflection=np.eye(4);reflection[0,0]=-1
        with self.assertRaises(ValueError):state.expect_transform('start','a',reflection,'condition',episode='episode')
        malformed=packet(2);malformed['objects']['a']['constrained_axes']=[1,1,0]
        with self.assertRaises(ValueError):state.compare('start',malformed,episode='episode')


if __name__=='__main__':unittest.main()
