"""Alignment metrics distinguish translation, square symmetry and yaw error."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from aspire_so100.cube_alignment import cube_edge_alignment, cube_alignment_observable

ROOT = Path(__file__).resolve().parents[1]


def cube(xy=(0.,0.), yaw=0.):
    c,s=np.cos(yaw),np.sin(yaw)
    rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    center=np.r_[xy,.45]
    corners=np.array([[x,y,1.] for x in (-1.,1.) for y in (-1.,1.)])*.015
    return dict(center=center,top_vertices=center+corners@rotation.T,shape='cube',
                normal=np.array([0.,0.,1.]),observed_face_count=3,inferred_orientation=False)


class CubeAlignmentTests(unittest.TestCase):
    def test_quarter_turn_is_equivalent_but_small_yaw_is_not(self):
        lower=cube()
        equal=cube_edge_alignment(cube(yaw=np.pi/2),lower)
        self.assertLess(equal['corner_max_m'],1e-12)
        self.assertAlmostEqual(equal['yaw_error_rad'],0.)
        rotated=cube_edge_alignment(cube(yaw=np.deg2rad(10)),lower)
        self.assertAlmostEqual(abs(rotated['yaw_error_rad']),np.deg2rad(10))
        self.assertAlmostEqual(rotated['corner_max_m'],2*np.sqrt(2)*.015*np.sin(np.deg2rad(5)))

    def test_translation_is_measured_independently_of_corner_order(self):
        upper=cube((.003,-.004))
        upper['top_vertices']=upper['top_vertices'][[2,0,3,1]]
        result=cube_edge_alignment(upper,cube())
        self.assertAlmostEqual(result['xy_error_m'],.005)
        self.assertAlmostEqual(result['corner_max_m'],.005)
        self.assertAlmostEqual(result['yaw_error_rad'],0.)

    def test_occluded_or_inferred_pose_does_not_authorize_refinement(self):
        model=cube()
        self.assertTrue(cube_alignment_observable(model))
        for change in ({'observed_face_count':1},{'inferred_orientation':True},
                       {'normal':np.array([0.,.3,.95])},{'shape':'cylinder'}):
            self.assertFalse(cube_alignment_observable(dict(model,**change)))

    def test_invalid_corners_rejected(self):
        for corners in (np.zeros((3,3)),np.full((4,3),np.nan)):
            with self.assertRaises(ValueError):cube_edge_alignment(dict(cube(),top_vertices=corners),cube())

    def test_alignment_and_column_bundles_are_sensor_only(self):
        spec=importlib.util.spec_from_file_location('build_alignment',ROOT/'scripts/build_inventory_policy.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        spec=importlib.util.spec_from_file_location('runner_alignment',ROOT/'scripts/run_vision_trial.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        for controller in ('column_policy', 'cube_alignment_policy'):
            runner.validate_source(builder.build_policy(controller))


if __name__=='__main__':unittest.main()
