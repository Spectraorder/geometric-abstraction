"""Conditional containment geometry, independent of the running policy trial."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rigid_containment_prototype import containment_bounds, above_interval


class ContainmentBoundsTests(unittest.TestCase):
    def test_rotated_partial_surfaces_contain_the_true_center(self):
        random=np.random.default_rng(13)
        for _ in range(50):
            rotation,_=np.linalg.qr(random.normal(size=(3,3)))
            if np.linalg.det(rotation)<0:rotation[:,0]*=-1
            dimensions=random.uniform(.015,.06,3);center=random.uniform(-.4,.4,3)
            local=random.uniform(-.5,.5,(80,3))*dimensions
            local[:,0]=dimensions[0]/2
            local[:,2]=np.abs(local[:,2])  # Occlusion hides the lower half.
            points=local@rotation.T+center
            result=containment_bounds(points,rotation,dimensions,tolerance_m=0)
            expected=center@rotation
            self.assertTrue(result['consistent'])
            self.assertTrue(np.all(expected>=result['center_local_lower_m']-1e-12))
            self.assertTrue(np.all(expected<=result['center_local_upper_m']+1e-12))
            self.assertGreater(result['interval_width_m'][2],dimensions[2]/2)

    def test_full_extent_is_tight_and_more_points_never_widen_bounds(self):
        points=np.array([[-.015,-.015,-.015],[.015,.015,.015]])
        first=containment_bounds(points[:1],np.eye(3),np.repeat(.03,3))
        full=containment_bounds(points,np.eye(3),np.repeat(.03,3))
        np.testing.assert_allclose(full['interval_width_m'],.003)
        self.assertTrue(np.all(full['interval_width_m']<=first['interval_width_m']))
        np.testing.assert_allclose(full['center_world_midpoint_m'],0)

    def test_partial_faces_can_establish_height_order_without_contact(self):
        cloud=np.array([[.015,-.01,0],[.015,.01,.012]])
        lower=containment_bounds(cloud,np.eye(3),np.repeat(.03,3))
        upper=containment_bounds(cloud+[0,0,.03],np.eye(3),np.repeat(.03,3))
        self.assertEqual(above_interval(upper,np.eye(3),lower,np.eye(3))['truth'],'true')
        self.assertEqual(above_interval(lower,np.eye(3),upper,np.eye(3))['truth'],'false')
        self.assertEqual(above_interval(lower,np.eye(3),lower,np.eye(3))['truth'],'unknown')

    def test_inconsistent_points_abstain(self):
        result=containment_bounds([[0,0,0],[.1,0,0]],np.eye(3),np.repeat(.03,3))
        self.assertFalse(result['consistent'])
        self.assertIsNone(result['center_world_midpoint_m'])


if __name__=='__main__':unittest.main()
