"""Rigid constraint semantics, including unobserved twist and shape preservation."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rigid_goal_prototype import point_axis_goal,transform_points


class RigidGoalTests(unittest.TestCase):
    def test_random_full_frame_constraints_recover_rigid_transform(self):
        random=np.random.default_rng(19)
        for _ in range(40):
            rotation,_=np.linalg.qr(random.normal(size=(3,3)))
            if np.linalg.det(rotation)<0:rotation[:,0]*=-1
            source_basis,_=np.linalg.qr(random.normal(size=(3,3)))
            point=random.uniform(-.5,.5,3);translation=random.uniform(-.2,.2,3)
            goal=point_axis_goal(point,source_basis[:,2],rotation@point+translation,
                rotation@source_basis[:,2],source_basis[:,0],rotation@source_basis[:,0])
            np.testing.assert_allclose(goal['rotation'],rotation,atol=1e-12)
            np.testing.assert_allclose(goal['translation'],translation,atol=1e-12)
            self.assertFalse(goal['unconstrained_twist'])
            vertices=random.uniform(-.03,.03,(8,3))+point
            desired=transform_points(goal,vertices)
            np.testing.assert_allclose(np.linalg.norm(vertices[:,None]-vertices[None,:],axis=-1),
                np.linalg.norm(desired[:,None]-desired[None,:],axis=-1),atol=1e-12)

    def test_antiparallel_alignment_keeps_twist_explicitly_free(self):
        goal=point_axis_goal([.1,.2,.3],[0,0,1],[.2,.1,.4],[0,0,-1])
        self.assertTrue(goal['unconstrained_twist'])
        self.assertEqual(goal['representative_selection'],'antiparallel_minimum_angle_tie')
        self.assertLess(goal['axis_residual'],1e-12)
        self.assertLess(goal['point_residual_m'],1e-12)
        self.assertAlmostEqual(np.linalg.det(goal['rotation']),1.)

    def test_face_centers_can_mate_without_hardcoded_dimensions_or_colors(self):
        for size in [.015,.03,.06]:
            goal=point_axis_goal([.1,.2,.4-size/2],[0,0,1],[.2,.1,.5],[0,0,1])
            desired_center=transform_points(goal,[[.1,.2,.4]])[0]
            np.testing.assert_allclose(desired_center,[.2,.1,.5+size/2])
            self.assertEqual(goal['kind'],'desired_state')
            self.assertFalse(goal['execution_or_contact_certified'])

    def test_degenerate_axes_are_not_silently_repaired(self):
        with self.assertRaises(ValueError):point_axis_goal([0,0,0],[0,0,0],[0,0,0],[0,0,1])
        with self.assertRaises(ValueError):point_axis_goal([0,0,0],[0,0,1],[0,0,0],[0,0,1],[0,0,1],[1,0,0])
        with self.assertRaises(ValueError):point_axis_goal([0,0,0],[0,0,1],[0,0,0],[0,0,1],[1,0,0],None)


if __name__=='__main__':unittest.main()
