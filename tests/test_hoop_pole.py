"""Hoop geometry and ordinary-contact insertion checks; no learned policy."""
import unittest
import mujoco
import numpy as np
import trimesh
from aspire_so100.hoop_pole import (get_hoop_spec,hoop_mesh,HOOP_HEIGHT,HOOP_MASS,
                                    HOOP_INNER_RADIUS,HOOP_OUTER_RADIUS,BASE_HEIGHT,POLE_HEIGHT)
from view_scene import build_scene,TABLE_HEIGHT


class HoopPoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.model,cls.data=build_scene(scene_task='hoop_pole')

    def setUp(self):
        mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        mujoco.mj_forward(self.model,self.data)

    def test_ring_mesh_and_open_collision_hole(self):
        v,f=hoop_mesh();mesh=trimesh.Trimesh(vertices=v,faces=f,process=False)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertGreater(mesh.volume,0.)
        m=get_hoop_spec().compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        self.assertAlmostEqual(m.body_mass.sum(),HOOP_MASS)
        groups=np.array([0,0,0,1,0,0],dtype=np.uint8)
        for x,expected in [(0.,-1.),((HOOP_INNER_RADIUS+HOOP_OUTER_RADIUS)/2,.1-HOOP_HEIGHT)]:
            ray=mujoco.mj_ray(m,d,np.array([x*np.cos(.05),x*np.sin(.05),.1]),np.array([0.,0.,-1.]),groups,True,-1,None)
            self.assertAlmostEqual(ray,expected,places=6)

    def test_tight_bore_changes_visual_and_collision_together(self):
        for variant,inner in [('standard',.012),('tight',.010)]:
            vertices,faces=hoop_mesh(variant=variant)
            mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
            self.assertTrue(mesh.is_watertight)
            self.assertAlmostEqual(np.linalg.norm(np.asarray(vertices)[:,:2],axis=1).min(),inner)
            m=get_hoop_spec(variant=variant).compile();d=mujoco.MjData(m)
            mujoco.mj_forward(m,d)
            self.assertAlmostEqual(m.body_mass.sum(),HOOP_MASS)
            for group in (2,3):
                groups=np.zeros(6,dtype=np.uint8);groups[group]=1
                for radius in (0.,.009,.011,.015):
                    ray=mujoco.mj_ray(m,d,np.array([radius*np.cos(.05),radius*np.sin(.05),.1]),
                        np.array([0.,0.,-1.]),groups,True,-1,None)
                    self.assertAlmostEqual(ray,.1-HOOP_HEIGHT if radius>inner else -1.,places=6)

    def test_passive_scene_settles_with_one_free_hoop_and_fixed_pole(self):
        m,d=self.model,self.data
        self.assertEqual(np.count_nonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE),1)
        self.assertEqual(m.body('red_pole/red_pole').jntnum[0],0)
        self.assertEqual(m.neq,0)
        for _ in range(round(3/m.opt.timestep)):mujoco.mj_step(m,d)
        self.assertTrue(np.isfinite(d.qpos).all())
        b=m.body('red_hoop/red_hoop').id
        self.assertAlmostEqual(d.xpos[b,2],TABLE_HEIGHT,delta=.0003)
        self.assertGreater(d.xmat[b].reshape(3,3)[2,2],.999)
        self.assertFalse(np.any(d.warning.number))

    def test_aligned_hoop_slides_over_pole_and_rests_on_base(self):
        m,d=self.model,self.data
        bolt=m.body('red_pole/red_pole').id;hoop=m.body('red_hoop/red_hoop').id
        q=m.joint('red_hoop/red_hoop_joint').qposadr[0]
        d.qpos[q:q+3]=d.xpos[bolt]+[0,0,POLE_HEIGHT+.01]
        d.qpos[q+3:q+7]=[1,0,0,0];mujoco.mj_forward(m,d)
        for _ in range(round(3/m.opt.timestep)):mujoco.mj_step(m,d)
        np.testing.assert_allclose(d.xpos[hoop,:2],d.xpos[bolt,:2],atol=.0005)
        self.assertAlmostEqual(d.xpos[hoop,2]-d.xpos[bolt,2],BASE_HEIGHT,delta=.0003)
        self.assertGreater(d.xmat[hoop].reshape(3,3)[2,2],.999)
        self.assertFalse(np.any(d.warning.number))

    def test_pole_collides_with_ring_material_when_misaligned(self):
        m,d=self.model,self.data
        bolt=m.body('red_pole/red_pole').id;hoop=m.body('red_hoop/red_hoop').id
        q=m.joint('red_hoop/red_hoop_joint').qposadr[0]
        d.qpos[q:q+3]=d.xpos[bolt]+[.011,0,.022]
        d.qpos[q+3:q+7]=[1,0,0,0];mujoco.mj_forward(m,d)
        contacts=[c for c in d.contact if {m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]}=={bolt,hoop}]
        self.assertTrue(any(c.dist<0 for c in contacts))


if __name__=='__main__':unittest.main()
