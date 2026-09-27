"""Geometry checks for the first Fasten Screws scene."""
import unittest

import mujoco
import numpy as np
import trimesh

from aspire_so100.fasteners import (
    FASTENER_XY, NUT_HEIGHT, PAIR_COLORS, _ring_mesh, _thread_mesh, get_nut_spec,
    get_screw_spec, nut_bore_radius, THREAD_PITCH, THREAD_START,
    SCREW_CORE_RADIUS, SCREW_THREAD_RADIUS, THREAD_RADIAL_CLEARANCE,
    LEGACY_SCENE_VERSION, LEGACY_NUT_BORE_RADIUS, STATIC_SCENE_VERSION,
)


class FastenerTests(unittest.TestCase):
    def test_four_fasteners_settle_upright_in_complete_scene(self):
        from view_scene import build_scene
        _, model, data = build_scene(scene_task='fasten_screws')
        for _ in range(round(5/model.opt.timestep)):
            mujoco.mj_step(model, data)
        self.assertTrue(np.isfinite(data.qpos).all())
        free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(len(free_joints), 4)
        for name, xy in FASTENER_XY.items():
            body = model.body(f'{name}/{name}').id
            np.testing.assert_allclose(data.xpos[body,:2], xy, atol=.001)
            self.assertAlmostEqual(data.xpos[body,2], .4, delta=.0005)
            self.assertGreater(data.xmat[body].reshape(3,3)[2,2], .999)

    def test_visual_meshes_are_closed_and_outward_facing(self):
        for factory in (_ring_mesh, _thread_mesh):
            with self.subTest(mesh=factory.__name__):
                vertices, faces = factory()
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                self.assertTrue(mesh.is_watertight)
                self.assertTrue(mesh.is_winding_consistent)
                self.assertGreater(mesh.volume, 0.)

    def test_female_groove_pitch_handedness_and_clearance(self):
        theta = np.linspace(0., 2*np.pi, 257)
        z = THREAD_PITCH*theta/(2*np.pi)+THREAD_PITCH
        np.testing.assert_allclose(nut_bore_radius(theta,z),
                                   SCREW_THREAD_RADIUS+THREAD_RADIAL_CLEARANCE)
        np.testing.assert_allclose(nut_bore_radius(theta,z+THREAD_PITCH/2),
                                   SCREW_CORE_RADIUS+THREAD_RADIAL_CLEARANCE)
        np.testing.assert_allclose(nut_bore_radius(theta,z),
                                   nut_bore_radius(theta,z+THREAD_PITCH))

    def test_actual_mesh_surfaces_fit_when_threaded_but_not_out_of_phase(self):
        # Ray tests inspect the actual triangulated render meshes, not only the
        # analytic radius function. They are NOT collision-solver validation.
        nut = get_nut_spec('nut', PAIR_COLORS['red'],version=STATIC_SCENE_VERSION).compile()
        screw = get_screw_spec('screw', PAIR_COLORS['red'],version=STATIC_SCENE_VERSION).compile()
        nd, sd = mujoco.MjData(nut), mujoco.MjData(screw)
        mujoco.mj_forward(nut, nd)
        mujoco.mj_forward(screw, sd)
        groups = np.array([0,0,1,0,0,0],dtype=np.uint8)
        geom = np.zeros(1,dtype=np.int32)
        rng = np.random.default_rng(18)
        wrong_phase_gaps = []
        for yaw in (0., .8, 2.4, -.5):
            # A compatible axial phase, several turns below the bolt tip.
            height = THREAD_START+6*THREAD_PITCH+THREAD_PITCH*yaw/(2*np.pi)
            for theta,z in zip(rng.uniform(-np.pi,np.pi,120),rng.uniform(.039,.043,120)):
                direction = np.array([np.cos(theta),np.sin(theta),0.])
                hit = mujoco.mj_ray(screw,sd,.03*direction+[0,0,z],-direction,
                                    groups,True,-1,geom)
                self.assertGreater(hit,0.)
                male_radius = .03-hit
                local = np.array([np.cos(theta-yaw),np.sin(theta-yaw),0.])
                for phase_error in (0., THREAD_PITCH/2):
                    female_radius = mujoco.mj_ray(nut,nd,np.array([0.,0.,z-height+phase_error]),
                                                  local,groups,True,-1,geom)
                    self.assertGreater(female_radius,0.)
                    gap = female_radius-male_radius
                    if phase_error == 0.:
                        self.assertGreater(gap,.00015)
                    else:
                        wrong_phase_gaps.append(gap)
        self.assertLess(min(wrong_phase_gaps),-.001)

    def test_legacy_mesh_keeps_original_smooth_bore(self):
        vertices,faces = _ring_mesh(version=LEGACY_SCENE_VERSION)
        self.assertEqual(len(vertices),96*8)
        np.testing.assert_allclose(np.linalg.norm(vertices[5*96:6*96,:2],axis=1),
                                   LEGACY_NUT_BORE_RADIUS)
        mesh = trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        self.assertTrue(mesh.is_watertight)
        with self.assertRaises(ValueError):
            get_nut_spec('nut',PAIR_COLORS['red'],version='unknown')

    def test_collision_bore_is_open_and_nut_wall_is_solid(self):
        model = get_nut_spec('nut', PAIR_COLORS['red'],version=STATIC_SCENE_VERSION).compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        groups = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
        geom_id = np.zeros(1, dtype=np.int32)
        # Restrict to collision wedges: the visual mesh must never fill the hole.
        direction = np.array([0., 0., -1.])
        center = mujoco.mj_ray(model, data, np.array([0., 0., .1]), direction,
                              groups, True, -1, geom_id)
        wall = mujoco.mj_ray(model, data, np.array([.018, .002, .1]), direction,
                            groups, True, -1, geom_id)
        self.assertEqual(center, -1.)
        self.assertAlmostEqual(wall, .1-NUT_HEIGHT, places=6)
        visual = model.geom('nut_surface').id
        self.assertEqual(model.geom_contype[visual], 0)
        self.assertEqual(model.geom_conaffinity[visual], 0)
