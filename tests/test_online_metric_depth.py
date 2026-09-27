"""Validate metric access, provenance and invalid/old observations independently."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from online_metric_depth import MetricDepthFrames, MetricDepthSensorMixin


def frame():
    T = np.eye(4)
    T[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
    T[:3,3] = [.1,.2,.3]
    return dict(rgb=np.zeros((4,5,3),np.uint8),depth_m=np.full((4,5),.5),
                valid_depth=np.ones((4,5),bool),intrinsics=np.array([[2,0,1],[0,2,1],[0,0,1]]),
                camera_to_world=T,time=3.)


class MetricDepthTests(unittest.TestCase):
    def test_backprojection_uses_axis_depth_rotation_and_translation(self):
        frames = MetricDepthFrames()
        identifier = frames.register('side',frame())['frame_id']
        result = frames.sample(identifier,[[3,2]],current_time=3.)
        sample = result['samples'][0]
        # Camera point [0.5, 0.25, 0.5], then rotate and translate.
        np.testing.assert_allclose(sample['world_point_m'],[-.15,.7,.8])
        self.assertEqual(sample['depth_m'],.5)
        self.assertTrue(result['at_current_sensor_time'])

    def test_invalid_samples_remain_unavailable(self):
        f = frame();f['depth_m'][0,:4] = [np.nan,np.inf,0,-1];f['valid_depth'][0,4] = False
        frames = MetricDepthFrames();key=frames.register('side',f)['frame_id']
        result=frames.sample(key,[[i,0] for i in range(5)]+[[5,0]],current_time=3.)
        self.assertEqual([r['status'] for r in result['samples']],['invalid_depth']*5+['out_of_frame'])
        self.assertTrue(all('world_point_m' not in r for r in result['samples']))

    def test_later_time_is_historical_not_new_pose(self):
        frames=MetricDepthFrames();key=frames.register('side',frame())['frame_id']
        result=frames.sample(key,[[1,1]],current_time=7.)
        self.assertFalse(result['at_current_sensor_time']);self.assertEqual(result['age_seconds'],4.)
        with self.assertRaises(ValueError):frames.sample(key,[[1,1]],current_time=2.)

    def test_latest_frame_per_view_and_copy_isolation(self):
        frames=MetricDepthFrames();f=frame();first=frames.register('side',f)
        f['depth_m'][:]=99
        self.assertEqual(frames.sample(first['frame_id'],[[1,1]],current_time=3.)['samples'][0]['depth_m'],.5)
        wrist=frames.register('wrist',frame())
        frames.register('side',frame())
        self.assertEqual(frames.sample(first['frame_id'],[[1,1]],current_time=3.)['status'],'frame_unavailable')
        self.assertEqual(frames.sample(wrist['frame_id'],[[1,1]],current_time=3.)['status'],'sampled')
        self.assertEqual(len(frames.frames),2)

    def test_no_object_or_evaluator_fields_propagate(self):
        f=frame();f.update(object_pose=[99,99,99],contacts=True,goal='hidden')
        frames=MetricDepthFrames();key=frames.register('side',f)['frame_id']
        result=frames.sample(key,[[1,1]],current_time=3.)
        self.assertTrue({'object_pose','contacts','goal'}.isdisjoint(result))

    def test_rejects_bad_calibration_and_query_shape(self):
        for kind in ['mask','rotation','intrinsics']:
            f=frame()
            if kind=='mask':f['valid_depth']=np.ones((4,5),int)
            elif kind=='rotation':f['camera_to_world'][0,0]=4
            else:f['intrinsics'][2,0]=1
            with self.assertRaises(ValueError):MetricDepthFrames().register('side',f)
        frames=MetricDepthFrames();key=frames.register('side',frame())['frame_id']
        for pixels in ([],[[1,1]]*17,[[True,1]],[[1.2,1]]):
            with self.assertRaises(ValueError):frames.sample(key,pixels,current_time=3.)

    def test_mixin_reads_exact_image_archive_without_recapturing(self):
        class Sensor:
            def __init__(self,output):self.output=Path(output);self.captures=0
            def image(self,name,view):
                self.captures+=1
                path=self.output/(name+'.png')
                np.savez(path.with_suffix('.npz'),**frame())
                return path,{'time':3.}
            def proprioception(self):return {'time':4.}
        class API(MetricDepthSensorMixin,Sensor):pass
        with TemporaryDirectory() as name:
            api=API(name);_,metadata=api.image('first')
            key=metadata['metric_depth_frame']['frame_id']
            result=api.sample_depth(key,[[3,2]])
            self.assertEqual(api.captures,1)
            self.assertEqual(api.depth_query_count,1)
            self.assertFalse(result['at_current_sensor_time'])
            self.assertTrue((Path(name)/'depth_queries.jsonl').exists())
            self.assertEqual(api.sample_depth(key,[[float('nan'),1]])['status'],'invalid_query')
            self.assertEqual(api.depth_query_count,2)
            self.assertEqual(api.captures,1)


if __name__=='__main__':unittest.main()
