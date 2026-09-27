"""Ground a point/axis goal in a real RGB-D checkpoint, without physics truth."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from rigid_goal_prototype import point_axis_goal,transform_points


def frame(path):
    with np.load(path) as data:return {key:data[key].copy() for key in data.files}


def record(replica,trial,checkpoint,source,destination,output):
    path=replica/'geometry.py';manifest=json.loads((replica/'manifest.json').read_text())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=manifest['geometry_sha256']:raise ValueError('Frozen library hash mismatch')
    library={};exec(compile(path.read_text(),str(path),'exec'),library)
    cameras=[c for c in json.loads((trial/'api_calls.json').read_text()) if c['method']=='camera']
    files=sorted(trial.glob('camera_*.npz'))
    if len(files)!=len(cameras):raise ValueError('Sensor record mismatch')
    runtime=json.loads((trial/'geometry_records.json').read_text())[checkpoint]
    scene=library['GEOMETRY'].scene(frame(files[0]),list(runtime['objects']))
    used=[]
    for path,call in zip(files,cameras):
        current=frame(path)
        if abs(float(current['time'])-runtime['time'])<1e-6:
            scene.refresh(current,call['params']['view']);used.append(path.name)
    for name in [source,destination]:
        if not np.allclose(np.round(scene.get(name)['center']*1000,2),runtime['objects'][name]['center_mm'],atol=.011):
            raise ValueError('Replay differs from runtime record')
    first=scene.reference(source+'.bottom_face');second=scene.reference(destination+'.top_face')
    goal=point_axis_goal(first['center_m'],first['normal'],second['center_m'],-second['normal'])
    state=scene.get(source)
    payload={'schema':'rigid_goal_sensor_example_v1','trial':str(trial),'checkpoint':checkpoint,
        'time':runtime['time'],'sensor_files':used,'geometry_library_sha256':digest,
        'symbolic_constraints':[{'predicate':'coincident','source':source+'.bottom_face.center','target':destination+'.top_face.center'},
            {'predicate':'opposed','source':source+'.bottom_face.normal','target':destination+'.top_face.normal'}],
        'source_evidence':{source:runtime['objects'][source],destination:runtime['objects'][destination]},
        'desired_transform':goal,'desired_center_m':transform_points(goal,[state['center']])[0],
        'desired_vertices_m':transform_points(goal,state['vertices']),
        'actual_policy_used_this_prototype':False,
        'limitations':['A counterfactual desired state computed after the rollout, not a predicted or observed outcome.',
            'Shape is preserved, but reachability, collision, grasp and stability are not established.',
            'Twist is not constrained by point coincidence plus a single axis; minimum-angle representative chosen.',
            'Input-reference uncertainty remains; numerical constraint residuals are not physical accuracy.']}
    def arrays(value):
        if isinstance(value,np.ndarray):return value.tolist()
        raise TypeError(str(type(value)))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(payload,default=arrays,indent=2)+'\n')
    print(json.dumps({'point_residual_m':goal['point_residual_m'],'axis_residual':goal['axis_residual'],
        'unconstrained_twist':goal['unconstrained_twist'],'desired_center_m':payload['desired_center_m'].tolist()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replica',type=Path,required=True);parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--checkpoint',type=int,required=True);parser.add_argument('--source',required=True)
    parser.add_argument('--destination',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();record(args.replica,args.trial,args.checkpoint,args.source,args.destination,args.output)
