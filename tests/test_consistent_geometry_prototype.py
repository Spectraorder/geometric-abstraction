"""Regression evidence for same-time cross-view gating, outside the frozen pilot."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from consistent_geometry_prototype import fuse
from aspire_so100.ablation_geometry import observe_cube
from aspire_so100.instance_geometry import inventory_table
from test_geometric_state import fit


def item(view,center,axes):
    f=fit(center,axes=axes);f['observed_face_count']=sum(axes)
    f['inferred_orientation']=sum(axes)<2
    return {'view':view,'fit':f}


class ConsistentGeometryTests(unittest.TestCase):
    def test_real_marker_fit_cannot_overwrite_good_cube_in_either_order(self):
        with np.load(ROOT/'tests/fixtures/geometry_multiview_green_marker.npz') as data:
            frames={prefix:{key[len(prefix)+1:]:data[key] for key in data.files if key.startswith(prefix+'_')} for prefix in ['initial','side','wrist']}
        plane=inventory_table(frames['initial'])
        model=observe_cube(frames['initial'],'green',table_plane=plane)['model']
        good=observe_cube(frames['side'],'green',model=model,table_plane=plane)
        bad=observe_cube(frames['wrist'],'green',model=model,table_plane=plane)
        values=[{'view':'side','fit':good},{'view':'wrist','fit':bad}]
        for order in [values,values[::-1]]:
            output=fuse('green',order,float(frames['side']['time']))
            self.assertEqual(output['accepted'],['side'])
            self.assertEqual(output['rejected'][0]['view'],'wrist')
            np.testing.assert_allclose(output['state'].estimate('green')['center'],good['center'],atol=1e-8)

    def test_complementary_partial_views_add_observed_directions(self):
        values=[item('a',[.1,9,.415],[True,False,True]),item('b',[9,.2,.415],[False,True,True])]
        result=fuse('red',values,0.)['state'].estimate('red')
        self.assertTrue(result['position_fully_observed'])
        np.testing.assert_allclose(result['center'],[.1,.2,.415])

    def test_equally_supported_disagreement_abstains(self):
        output=fuse('blue',[item('a',[0,0,.415],[True]*3),item('b',[.1,0,.415],[True]*3)],0.)
        self.assertIsNone(output['state'])
        self.assertEqual(output['reason'],'equally_constrained_views_disagree')

    def test_consistent_noisy_views_average_in_measured_directions(self):
        output=fuse('red',[item('a',[0,0,.415],[True]*3),item('b',[.002,9,.415],[True,False,True])],0.)
        np.testing.assert_allclose(output['state'].estimate('red')['center'],[.001,0,.415],atol=1e-8)

    def test_nearly_parallel_views_do_not_extrapolate_hidden_axis(self):
        first=item('a',[.1,.2,.415],[True,False,True])
        second=item('b',[.101,.2,.415],[True,False,True])
        theta=.001;c=np.cos(theta);s=np.sin(theta)
        second['fit']['rotation']=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        value=fuse('blue',[first,second],0.)['state'].estimate('blue')
        self.assertEqual(value['observed_translation_rank'],2)
        self.assertLess(np.linalg.norm(value['center']-first['fit']['center']),.003)
