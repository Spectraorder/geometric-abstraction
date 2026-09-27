"""Verify shared pixel queries against archived development RGB-D and shown PNGs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from online_metric_depth import MetricDepthFrames


def verify(development, output):
    paths = sorted(development.glob('*/feedback_000_*.npz'))
    if not paths:
        raise ValueError('No archived initial development frames')
    records = []
    for path in paths:
        with np.load(path) as loaded:
            frame = {name:loaded[name] for name in loaded.files}
        image_path = path.with_suffix('.png')
        if not np.array_equal(frame['rgb'],np.asarray(Image.open(image_path))):
            raise ValueError('Displayed RGB differs from queried sensor archive')
        depth, valid = frame['depth_m'], frame['valid_depth']
        choices = np.argwhere(valid & np.isfinite(depth) & (depth>0))
        if len(choices)<16:
            raise ValueError('Insufficient valid depth for replay check')
        # Uniform indices through all valid pixels; no object labels or task truth.
        selected = choices[np.linspace(0,len(choices)-1,16,dtype=int)]
        pixels = [[int(u),int(v)] for v,u in selected]
        view = 'wrist' if path.stem.endswith('wrist') else 'side'
        responses = []
        for arm in ('basic','symbolic'):
            registry = MetricDepthFrames()
            key = registry.register(view,frame)['frame_id']
            responses.append(registry.sample(key,pixels,current_time=float(frame['time'])))
        if responses[0] != responses[1]:
            raise ValueError('Shared metric queries differ by arm')
        T, K = frame['camera_to_world'], frame['intrinsics']
        errors = []
        for pixel, sample in zip(pixels,responses[0]['samples']):
            u,v = pixel
            if sample['status']!='observed_surface_sample' or sample['depth_m']!=float(depth[v,u]):
                raise ValueError('Depth quantization or availability mismatch')
            camera = T[:3,:3].T@(np.asarray(sample['world_point_m'])-T[:3,3])
            projected = K@camera
            errors.append(float(np.max(np.abs(projected[:2]/projected[2]-pixel))))
            if not np.isclose(camera[2],depth[v,u],rtol=0,atol=1e-10):
                raise ValueError('Optical-axis depth mismatch')
        if max(errors)>1e-9:
            raise ValueError('Calibration reprojection mismatch')
        records.append(dict(frame=str(path),frame_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),samples=len(pixels),
            max_reprojection_error_pixels=max(errors),identical_arm_responses=True))
    report = dict(status='passed',frames=len(records),samples=sum(r['samples'] for r in records),
        model_requests=0,simulation_rollouts=0,records=records,
        scope='Displayed-frame correspondence, exact metric depth, calibration consistency and arm parity; '
              'not object geometry accuracy, agent capability or task success.',
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('online_metric_depth.py','verify_online_metric_depth.py')})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=verify(args.development,args.output)
    print(json.dumps({key:result[key] for key in ('status','frames','samples','model_requests','simulation_rollouts')}))
