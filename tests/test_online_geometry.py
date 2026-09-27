"""Runtime geometry preserves sensor provenance and explicit memory semantics."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import unittest
import numpy as np
from aspire_so100.online_geometry import SceneGeometry
from test_geometric_state import fit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_online_geometry import build


def measured(center=(0.,0.,.415),axes=(True,True,True)):
    result=fit(center,axes=axes)
    result['model']={'height':.03,'dims':np.array([.03,.03])}
    result['observed_face_count']=sum(axes)
    return result


class OnlineGeometryTests(unittest.TestCase):
    def initialize(self,memory=False):
        with patch('aspire_so100.online_geometry.inventory_table',return_value=np.array([0.,0.,.4])),patch('aspire_so100.online_geometry.observe_cube',return_value=measured()):
            return SceneGeometry({'time':0.},['blue'],memory)

    def test_instantaneous_does_not_fill_missing_pose_from_history(self):
        state=self.initialize()
        with patch('aspire_so100.online_geometry.observe_cube',side_effect=ValueError('occluded')):
            packet=state.refresh({'time':1.})
        self.assertEqual(packet['objects'],{})
        self.assertIn('blue',packet['unavailable'])
        with self.assertRaises(ValueError):state.get('blue')
        with self.assertRaises(ValueError):state.predict('blue',np.eye(4),2.,'if held')

    def test_memory_retains_occluded_object_as_stale_not_fresh(self):
        state=self.initialize(memory=True)
        with patch('aspire_so100.online_geometry.observe_cube',side_effect=ValueError('occluded')):
            state.refresh({'time':1.})
        self.assertEqual(state.get('blue')['source'],'remembered')
        self.assertFalse(state.get('blue')['position_fully_observed'])

    def test_same_time_views_fuse_without_temporal_memory(self):
        state=self.initialize()
        with patch('aspire_so100.online_geometry.observe_cube',return_value=measured((.1,.2,.43),(True,False,True))):state.refresh({'time':1.})
        self.assertFalse(state.get('blue')['position_fully_observed'])
        with patch('aspire_so100.online_geometry.observe_cube',return_value=measured((.1,.2,.43),(False,True,True))):state.refresh({'time':1.},'wrist')
        self.assertTrue(state.get('blue')['position_fully_observed'])

    def test_target_and_returned_arrays_do_not_change_measured_pose(self):
        state=self.initialize();original=state.get('blue')['center'].copy()
        state.target('blue',[.1,.2,.5])
        changed=state.get('blue');changed['center'][:]=9
        np.testing.assert_allclose(state.get('blue')['center'],original)
        with patch('aspire_so100.online_geometry.observe_cube',return_value=measured()):state.refresh({'time':1.})
        self.assertIn('target_delta_mm',state.packet()['objects']['blue'])
        self.assertEqual(state.reference('blue.top_face')['type'],'plane_patch')

    def test_exported_library_runs_on_real_rgbd_without_robot_api(self):
        namespace={};source=build()
        self.assertNotIn('def stack_targets(',source)
        exec(source,namespace)
        frame=dict(np.load(ROOT/'tests/fixtures/ablation_cube_initial_rgbd.npz'))
        state=namespace['GEOMETRY'].scene(frame,['blue','red','green'])
        packet=state.packet(details=True);json.dumps(packet,allow_nan=False)
        self.assertEqual(set(packet['objects']),{'blue','red','green'})
        self.assertFalse(packet['memory_enabled'])
        self.assertEqual(state.relation('contact','red','blue')['truth'],'unknown')
        self.assertEqual(state.get('blue')['vertices'].shape,(8,3))

    def test_runtime_scene_rejects_marker_without_losing_good_observation(self):
        with np.load(ROOT/'tests/fixtures/geometry_multiview_green_marker.npz') as data:
            frames={prefix:{key[len(prefix)+1:]:data[key] for key in data.files if key.startswith(prefix+'_')} for prefix in ['initial','side','wrist']}
        state=SceneGeometry(frames['initial'],['blue','red','green'])
        state.refresh(frames['side'])
        reference=state.get('green')['center'].copy()
        packet=state.refresh(frames['wrist'],'wrist')
        np.testing.assert_allclose(state.get('green')['center'],reference,atol=1e-8)
        self.assertTrue(state.get('green')['position_fully_observed'])
        self.assertTrue(any(row['object']=='green' and row['view']=='wrist' for row in packet['rejected_views']))
        self.assertIn('green/wrist',state.text())


class RobotVisualSceneRegression(unittest.TestCase):
    def test_exported_robot_mask_handles_marker_when_side_pose_is_partial(self):
        with np.load(ROOT/'tests/fixtures/geometry_robot_marker_partial.npz') as data:
            frames={prefix:{key[len(prefix)+1:]:data[key] for key in data.files if key.startswith(prefix+'_')} for prefix in ['initial','side','wrist']}
            robot=json.loads(str(data['robot_json']))
        namespace={};exec(build(),namespace)
        raw=namespace['GEOMETRY'].scene(frames['initial'],['blue','red','green'])
        raw.refresh(frames['side']);reference=raw.get('green')['center'].copy()
        raw.refresh(frames['wrist'],'wrist')
        self.assertGreater(np.linalg.norm(raw.get('green')['center']-reference),.05)
        fixed=namespace['GEOMETRY'].scene(frames['initial'],['blue','red','green'])
        fixed.refresh(frames['side'],robot_geometry=robot)
        packet=fixed.refresh(frames['wrist'],'wrist',robot_geometry=robot)
        self.assertLess(np.linalg.norm(fixed.get('green')['center']-reference),.002)
        self.assertGreater(packet['robot_geometry_exclusion']['wrist']['removed_depth_samples'],100)
        self.assertFalse(fixed.get('green')['position_fully_observed'])
        self.assertFalse(packet['task_success_certified'])
        json.dumps(packet,allow_nan=False)
