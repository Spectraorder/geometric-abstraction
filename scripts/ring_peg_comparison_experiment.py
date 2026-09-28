"""Frozen contracts and source inventory for the paired ring study."""
from pathlib import Path
import run_ring_peg_training as legacy
import run_ring_peg_repaired_training as repaired

ROOT = legacy.ROOT
ARMS = ('basic', 'symbolic')
STARTER_PATH = ROOT/'policies/controllers/ring_peg_comparison_starter.py'
GUIDANCE_PATH = ROOT/'docs/RING_PEG_STARTER_GUIDANCE.md'


def contract(arm):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    text = repaired.CONTRACT
    begin = text.index("api.observe('ring'):")
    end = text.index('api.robot():', begin)
    full_api = text[begin:end]
    text = text[:begin]+text[end:]
    shared = '''
Both arms have the SAME raw calibrated RGB-D and robot motion interface.
api.camera(view='side'): acquire a current RGB-D record; view is 'side' or 'wrist'.
Returns numpy arrays rgb[H,W,3] uint8; depth_m[H,W] optical-axis depth in meters;
valid_depth[H,W] bool; intrinsics[3,3]; camera_to_world[4,4]; and scalar time.
Resolution is 640x480, pixel coordinates are (u=column,v=row), with camera axes
right/down/forward. Transform D*inv(K)@[u,v,1] by camera_to_world for a surface
sample. This supplies pixels and calibration, not object segmentation or pose.
Both arms may implement CPU image/depth processing with numpy/scipy/math.
Unique object colors are public labels; pixel association is not supplied to Base.
api.task() also declares geometry_available to identify the observation interface.
The shared R0 reads pixels; it additionally queries supplied geometry when available.
No actions, prior trained candidate, or completed task procedure are supplied.
The generic command-target wrappers may be edited or replaced by the policy.
Persistent actor contexts and training feedback remain separate across chains.
'''
    availability = ('\nFull-only supplied geometry:\n'+full_api if arm == 'symbolic' else
                    '\nBase: api.observe(object_id) is unavailable and raises an error. Use api.camera() and your own estimator. Full field names in the shared guidance describe conceptual geometry, not available Base observations.\n')
    return text+shared+availability+'\nShared qualitative guidance:\n'+GUIDANCE_PATH.read_text()


def sources():
    result = repaired.sources()
    files = ('ring_peg_comparison_runtime.py', 'ring_peg_comparison_worker.py',
             'run_ring_peg_comparison_trial.py', 'ring_peg_comparison_experiment.py',
             'run_ring_peg_comparison_study.py', 'experiment_parallel.py')
    for path in [ROOT/'scripts'/name for name in files]+[STARTER_PATH, GUIDANCE_PATH]:
        result[str(path.relative_to(ROOT))] = legacy.digest(path)
    for path in (ROOT/'assets').rglob('*'):
        if path.is_file() and path.suffix.lower() in ('.stl', '.obj', '.msh', '.urdf'):
            result[str(path.relative_to(ROOT))] = legacy.digest(path)
    return result
