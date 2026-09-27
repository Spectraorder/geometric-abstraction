"""Text feedback derives only from the same explicitly selected sensor frames."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from geometric_feedback import make_feedback,actor_projection


class GeometricFeedbackTests(unittest.TestCase):
    def test_selected_frames_and_geometry_do_not_read_private_evaluation(self):
        frame=dict(np.load(ROOT/'tests/fixtures/ablation_cube_initial_rgbd.npz'))
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            # Invalid private files make accidental consumption fail loudly.
            for name in ['trace.json','result.json']:
                (root/name).write_text('PRIVATE MUST NOT BE READ')
            calls=[]
            for index,time in enumerate([0.,1.,2.]):
                image=dict(frame,time=np.array(time))
                np.savez_compressed(root/f'camera_{index:03d}.npz',**image)
                calls.append({'time':time,'method':'camera','params':{'view':'side'}})
            (root/'api_calls.json').write_text(json.dumps(calls))
            packet=make_feedback(root)
            self.assertEqual([p['camera_index'] for p in packet['observations']],[0,2])
            self.assertFalse(packet['memory_enabled'])
            projected=actor_projection(packet)
            encoded=json.dumps(projected,allow_nan=False)
            self.assertNotIn('sensor_file',encoded)
            self.assertNotIn('PRIVATE',encoded)
            self.assertTrue(all(set(item)=={'time','text','facts','unavailable'} for item in projected))
            self.assertTrue(all(set(fact)=={'predicate','args','truth','residuals'} for item in projected for fact in item['facts']))
            self.assertEqual(set(packet['observations'][0]['packet']['objects']),{'blue','red','green'})

    def test_no_camera_returns_no_invented_state(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'api_calls.json').write_text('[]')
            self.assertEqual(actor_projection(make_feedback(root)),[])
