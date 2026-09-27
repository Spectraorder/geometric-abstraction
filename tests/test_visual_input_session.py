"""No supplied object coordinates reach the raw-visual planning interface."""
from pathlib import Path
import sys
import unittest
import numpy as np
from test_geometry_step_session import RecordingAPI, OBJECTS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from build_visual_input_session import library
from run_geometry_step_experiment import validate_planner

ROOT = Path(__file__).resolve().parents[1]


class VisualInputTests(unittest.TestCase):
    def test_projection_blocks_measurements_and_all_action_return_leaks(self):
        source = library('basic')
        self.assertEqual(source.replace('SURFACE_ONLY = True','SURFACE_ONLY = False'),library('symbolic'))
        ns = {}; exec(compile(source,'<visual-input-library>','exec'),ns)
        api = RecordingAPI(); session = ns['GEOMETRY'].session(api, OBJECTS)
        for name in ('observe','get','table_z','snapshot','state','history','target','goal_error','reference','relation','align','text'):
            self.assertFalse(hasattr(session,name),name)
        # Inject every geometry-bearing return channel from the shared actuator.
        detailed = {'status':'released','held':False,'observation':{'center':[1,2,3]},
                    'xy_error_m':.1,'surface_gap_m':.2,'vertices':[[1,2,3]]}
        for name,args in (('pick',(OBJECTS[0]['color'],)),('place',([0,0,.4],)),('pick_and_place',(OBJECTS[0]['color'],[0,0,.4]))):
            setattr(session.backend,name,lambda *a,**kw:detailed)
            self.assertEqual(getattr(session,name)(*args),{'status':'released','held':False})
        def forbidden(*a,**kw):
            raise AssertionError('Checkpoint must not sense or query object geometry')
        session.backend.get = forbidden; session.backend.observe = forbidden; api.camera = forbidden
        row = session.checkpoint(OBJECTS[0]['color'],'place',action_status='released')
        self.assertEqual(set(row),{'object','step','time','status','action_status'})
        self.assertEqual(api.logged[-1],{'kind':'geometry_step','record':row})
        with self.assertRaises(ValueError):session.checkpoint(OBJECTS[0]['color'],'check',refresh=True)

    def test_rich_condition_retains_the_visual_geometry_interface(self):
        ns = {}; exec(compile(library('symbolic'),'<full-input-library>','exec'),ns)
        session=ns['GEOMETRY'].session(RecordingAPI(),OBJECTS)
        self.assertIn('center',session.get(OBJECTS[0]['color']))
        self.assertIn('geometry',session.checkpoint(OBJECTS[0]['color'],'check'))

    def test_common_starter_is_valid_and_geometry_is_computed_from_pixels(self):
        source=(ROOT/'policies/support_graph_visual_starter.py').read_text()
        validate_planner(source)
        ns={};exec(compile(source,'<visual-starter>','exec'),ns)
        api=RecordingAPI();plane=ns['fit_table'](api)
        color=OBJECTS[0]['color'];before=ns['locate'](api,plane,color)
        original=api.camera
        def shifted():
            frame=dict(original());frame['camera_to_world']=np.array(frame['camera_to_world'],copy=True)
            frame['camera_to_world'][0,3]+=.1
            return frame
        api.camera=shifted
        after=ns['locate'](api,ns['fit_table'](api),color)
        np.testing.assert_allclose(after['center']-before['center'],[.1,0,0],atol=1e-8)
        def blacked_out():
            frame=dict(original());frame['valid_depth']=np.zeros_like(frame['valid_depth'],dtype=bool)
            return frame
        api.camera=blacked_out
        with self.assertRaises(RuntimeError):ns['fit_table'](api)
        with self.assertRaises(RuntimeError):ns['locate'](api,plane,color)


if __name__=='__main__':unittest.main()
