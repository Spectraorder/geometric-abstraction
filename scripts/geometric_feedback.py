"""Sensor-only symbolic feedback from selected archived RGB-D observations.

The returned raw image references are identical for both experiment arms. Only
one arm additionally receives the derived symbolic/metric packet. This first
feedback adapter uses instantaneous fits with dimensions learned at the initial
on-table frame; it does not silently add a temporal memory treatment.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from aspire_so100.ablation_geometry import observe_cube
from aspire_so100.geometric_state import GeometricState
from aspire_so100.instance_geometry import inventory_table
from aspire_so100.symbolic_geometry import symbolic_packet


def make_feedback(trial, colors=('blue','red','green')):
    trial = Path(trial)
    # Deliberately never read result.json, trace.json, or rollout object states.
    calls = json.loads((trial/'api_calls.json').read_text())
    cameras = [call for call in calls if call['method']=='camera']
    side = [index for index,call in enumerate(cameras) if call['params'].get('view','side')=='side']
    if not side:
        return {'schema':'instantaneous_symbolic_feedback_v1','observations':[],'reason':'no side camera observations'}
    selected = list(dict.fromkeys([side[0],side[-1]]))
    initial = dict(np.load(trial/f'camera_{side[0]:03d}.npz'))
    plane = inventory_table(initial)
    models = {}
    dimension_errors = {}
    for color in colors:
        try:
            models[color] = observe_cube(initial,color,table_plane=plane)['model']
        except ValueError as error:
            dimension_errors[color] = str(error)
    observations = []
    for index in selected:
        path=trial/f'camera_{index:03d}.npz'
        frame=dict(np.load(path));stamp=float(frame['time'])
        state=GeometricState()
        errors={}
        for color in colors:
            if color not in models:
                errors[color]=dimension_errors[color]
                continue
            try:
                fit=observe_cube(frame,color,model=models[color],table_plane=plane)
                state.observe(color,fit,stamp,'side')
            except ValueError as error:
                errors[color]=str(error)
        packet=symbolic_packet(state,stamp)
        text=state.text(stamp)
        # Publish a few typed relations alongside the compact numeric summary.
        facts=[f for f in packet['facts'] if f['predicate']=='support_candidate']
        observations.append({'camera_index':index,'time':stamp,
            'image':str(trial/f'camera_{index:03d}.png'),
            'sensor_file':str(path),'sensor_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'calibration':{'intrinsics':frame['intrinsics'].tolist(),'camera_to_world':frame['camera_to_world'].tolist()},
            'text':text,'facts':facts,'unavailable':errors,'packet':packet})
    return {'schema':'instantaneous_symbolic_feedback_v1','coordinate_frame':'world',
            'dimension_source':'first side RGB-D on-table fit', 'memory_enabled':False,
            'predictions_enabled':False,'observations':observations,
            'contact_or_task_success_certified':False}


def actor_projection(feedback):
    """Only visual semantics enter text; no private result or filesystem metadata."""
    return [{'time':item['time'],'text':item['text'],
             'facts':[{'predicate':f['predicate'],'args':f['args'],'truth':f['truth'],
                       'residuals':f['residuals']} for f in item['facts']],
             'unavailable':item['unavailable']} for item in feedback['observations']]


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    packet=make_feedback(args.trial)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+'\n')
    print(json.dumps(actor_projection(packet),indent=2))
