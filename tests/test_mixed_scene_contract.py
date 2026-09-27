"""Public observation contract checked against recorded RGB-D, without object truth."""
from pathlib import Path
import sys
import unittest
import numpy as np
from test_rgbd_pnp_base import PixelsOnlyAPI, ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from build_mixed_scene import build

OBJECTS = [{'color':'blue','shape':'cube'}, {'color':'yellow','shape':'cube'},
           {'color':'green','shape':'thin_prism','orientation':'flat_x'},
           {'color':'red','shape':'cylinder'}]
REQUIRED = {'shape','center','normal','height','dims','top','bottom','extent_z','model','source',
            'rotation','axis_dimensions','vertices','constrained_axes','inferred_orientation',
            'position_fully_observed','observation_time','view'}


class RecordedSceneAPI(PixelsOnlyAPI):
    def __init__(self):
        super().__init__()
        with np.load(ROOT/'tests/fixtures/mixed_bridge_release_rgbd.npz') as sensor:
            self.frame={key[8:]:sensor[key] for key in sensor.files if key.startswith('initial_')}
    def proprioception(self):
        return {'tool_position':self.position.copy(),'time':float(self.frame['time'])}
    def gripper_geometry(self):
        return {'time':float(self.frame['time']),'coordinate_frame':'world','units':'meters',
                'source':'robot_model_and_joint_encoders','visual_spheres':[]}


class PublicObservationContractTests(unittest.TestCase):
    def test_observe_and_get_supply_the_same_documented_fields_for_every_shape(self):
        for symbolic in (False, True):
            library={};exec(compile(build(symbolic),'<public-session>','exec'),library)
            api=RecordedSceneAPI();session=library['SymbolicSession' if symbolic else 'MixedSession'](api,OBJECTS)
            for obj in OBJECTS:
                fit=session.observe(obj['color']);stored=session.get(obj['color'])
                self.assertFalse(REQUIRED-set(fit),msg=str((symbolic,obj['shape'],REQUIRED-set(fit))))
                self.assertFalse(REQUIRED-set(stored))
                for key in ('center','rotation','axis_dimensions','normal','vertices','dims','constrained_axes'):
                    np.testing.assert_array_equal(fit[key],stored[key])
                self.assertEqual(fit['axis_dimensions'].shape,(3,))
                self.assertTrue(np.all(fit['axis_dimensions']>0))
                # This is the unchanged failing R2 access, using measured values.
                if obj['shape']=='cube':
                    self.assertGreater(float(np.median(np.array(fit['axis_dimensions'],dtype=float,copy=True))),0.)
                fit['axis_dimensions'][:]=10.
                self.assertLess(np.max(session.get(obj['color'])['axis_dimensions']),1.)
            self.assertEqual(api.actions,[])
