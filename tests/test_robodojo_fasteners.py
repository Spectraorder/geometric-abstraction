"""Provenance, geometry and historical replay checks for imported fasteners."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import mujoco
import numpy as np

from aspire_so100.fasteners import SCENE_VERSION, ROBODOJO_LEGACY_VERSION, ROBODOJO_VERSIONS, STATIC_SCENE_VERSION, THREADED_SCENE_VERSION, LEGACY_SCENE_VERSION
from aspire_so100.robodojo_fasteners import ASSETS, NUT_HEIGHT, get_nut_spec, get_screw_spec, load_mesh


class RoboDojoFastenerTests(unittest.TestCase):
    def test_original_assets_and_converted_geometry_are_pinned(self):
        manifest=json.loads((ASSETS/'manifest.json').read_text())
        self.assertEqual(manifest['revision'],'43dacb13d3051a9ccd421f4e7e08e4e3eb29dfab')
        for name,counts in [('factory_nut',(13089,15080)),('factory_bolt',(78366,26122))]:
            part=manifest['parts'][name]
            source=ASSETS.parent/'upstream'/part['source_path']
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),part['source_sha256'])
            self.assertEqual(hashlib.sha256((ASSETS/part['mesh_file']).read_bytes()).hexdigest(),part['mesh_sha256'])
            mesh=load_mesh(part['mesh_file'])
            self.assertEqual((len(mesh['vertices']),len(mesh['faces'])),counts)
            self.assertTrue(all(np.isfinite(a).all() for a in mesh.values()))
            np.testing.assert_allclose([mesh['vertices'].min(0),mesh['vertices'].max(0)],part['bounds_m'])
            np.testing.assert_allclose(np.linalg.norm(mesh['normals'],axis=1),1.,atol=1.e-6)

    def test_original_nut_collision_hole_remains_open(self):
        model=get_nut_spec('nut',(1,0,0,1)).compile()
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        groups=np.array([0,0,0,1,0,0],dtype=np.uint8)
        for x,expected in [(0.,-1.),(.015,.1-NUT_HEIGHT)]:
            distance=mujoco.mj_ray(model,data,np.array([x,0.,.1]),np.array([0.,0.,-1.]),
                                   groups,True,-1,None)
            self.assertAlmostEqual(distance,expected,places=6)
        self.assertEqual(model.ngeom,7)
        self.assertEqual(model.geom('nut_surface').contype[0],0)

    def test_original_screw_is_imported_and_fixed_when_requested(self):
        model=get_screw_spec('bolt',(0,0,1,1),fixed=True).compile()
        self.assertEqual(model.njnt,0)
        self.assertEqual(model.nmesh,1)
        self.assertEqual(model.mesh_facenum[0],26122)
        self.assertEqual(model.geom('bolt_surface').contype[0],0)

    def test_historical_scene_versions_keep_their_dimensions_and_dynamics(self):
        from aspire_so100.playback import recorded_env_for_result
        for version in (LEGACY_SCENE_VERSION,THREADED_SCENE_VERSION,STATIC_SCENE_VERSION,ROBODOJO_LEGACY_VERSION,SCENE_VERSION):
            result=dict(task='fasten_screws',seed=5102,scene=dict(generator=version,seed=5102,
                       pairs=['red','blue','yellow'],fixture='fixed_bolts',object_count=6))
            with self.subTest(version=version), TemporaryDirectory() as tmp:
                e=recorded_env_for_result(result,Path(tmp))
                self.assertAlmostEqual(e.thread_pitch,.003 if version in ROBODOJO_VERSIONS else .004)
                self.assertAlmostEqual(e.entry_z,.0505 if version in ROBODOJO_VERSIONS else .046)
                e._set_thread_friction('red',True)
                expected=.002 if version in (STATIC_SCENE_VERSION,*ROBODOJO_VERSIONS) else 0.
                self.assertEqual(e.model.dof_frictionloss[e.spin_dofs['red']],expected)
                e.close()
