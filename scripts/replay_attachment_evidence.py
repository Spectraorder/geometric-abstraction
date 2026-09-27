"""Replay conditional held-point predictions against current visual evidence.

Only action-boundary estimates, robot proprioception and RGB-D are read. No rollout
state, task success or contacts. A bounding-sphere check can refute an attachment
hypothesis under its assumptions; passing it cannot recover or certify a pose.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from replay_held_geometry import predict_center
from aspire_so100.instance_geometry import inventory_points, inventory_color_mask, inventory_table


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def depth_consistency(frame, color, predicted, dimensions, table_plane, margin_m=.005):
    if color == 'green':
        return {'status':'unknown','reason':'Replay does not mask the green robot marker'}
    dimensions = np.asarray(dimensions, float)
    predicted = np.asarray(predicted, float)
    if (dimensions.shape != (3,) or predicted.shape != (3,) or np.any(dimensions <= 0)
            or not np.isfinite(dimensions).all() or not np.isfinite(predicted).all()
            or not np.isfinite(margin_m) or margin_m <= 0):
        raise ValueError('Finite visual size, predicted center and positive margin required')
    points, valid = inventory_points(frame)
    plane = np.asarray(table_plane, float)
    if plane.shape != (3,) or not np.isfinite(plane).all():
        raise ValueError('Require a visually measured table plane')
    height = points[..., 2]-points[..., :2]@plane[:2]-plane[2]
    # Match the shared observer work volume, not a prediction-centered crop.
    mask = valid & inventory_color_mask(frame['rgb'], color) & (height > .004) & (height < .35)
    cloud = points[mask]
    if len(cloud) < 8:
        return {'status':'unknown','reason':'Insufficient current color/depth evidence','points':len(cloud)}
    radius = float(np.linalg.norm(dimensions)/2)
    excess = np.linalg.norm(cloud-predicted, axis=1)-radius
    outside = float(np.mean(excess > margin_m))
    return {'status':'contradicted' if outside > .5 else 'not_contradicted',
            'points':len(cloud), 'enclosing_radius_m':radius, 'margin_m':margin_m,
            'fraction_outside_margin':outside, 'median_excess_m':float(np.median(excess)),
            'position_recovered':False, 'attachment_certified':False}


def replay_trial(trial):
    record_path, call_path = trial/'geometry_records.json', trial/'api_calls.json'
    calls = json.loads(call_path.read_text())
    hashes = {str(call_path):digest(call_path)}
    if not record_path.exists():
        return {'trial':str(trial),'records':[],'invalidations':[], 'status':'no_action_boundary_geometry','input_sha256':hashes}
    records = json.loads(record_path.read_text());hashes[str(record_path)] = digest(record_path)
    first_frame = trial/'camera_000.npz'
    with np.load(first_frame) as data:
        initial = {k:data[k] for k in data.files}
    plane = inventory_table(initial);hashes[str(first_frame)] = digest(first_frame)
    poses = {round(float(c['time']),6):c['observation'] for c in calls
             if c['method']=='proprioception' and 'observation' in c}
    gripper_times = [float(c['time']) for c in calls if c['method']=='gripper']
    frames = [(trial/f'camera_{i:03d}.npz', c) for i,c in enumerate(x for x in calls if x['method']=='camera')]
    anchor = None; rows = []; invalidations = []
    for item in records:
        if item.get('kind') != 'geometry_step':
            continue
        row = item['record'];stamp=float(row['time']);key=round(stamp,6)
        action=row.get('action_status','');color=row['object']
        if anchor is not None:
            changed = any(anchor['time']+1e-6 < t <= stamp+1e-6 for t in gripper_times)
            if color != anchor['object'] or changed:
                invalidations.append({'time':stamp,'reason':'focus_changed' if color != anchor['object'] else 'gripper_command'})
                anchor=None
        if action == 'lift_visually_confirmed':
            anchor=None
            geometry=row.get('geometry',{})
            if row['status']=='current_estimate' and geometry.get('position_fully_observed') and key in poses:
                if row.get('frame')!='world' or row.get('units')!='meters':
                    raise ValueError('Require world-meter estimates')
                anchor={'object':color,'time':stamp,'center':geometry['center_m'],
                        'dimensions':geometry['dimensions_m'],'robot':poses[key]}
            continue
        if anchor is None or not row['step'].startswith('place') or key not in poses or stamp <= anchor['time']:
            continue
        predicted=predict_center(anchor['center'],anchor['robot'],poses[key])
        result={'object':color,'from_time':anchor['time'],'time':stamp,'step':row['step'],
                'action_status':action,'kind':'conditional_prediction_not_observation',
                'condition':'Rigid attachment without slip since a visually confirmed lift; not contact certified.',
                'predicted_center_m':predicted.tolist(),'current_visual_status':row['status'],
                'current_visual_center_m':None,'fully_observed_visual_residual_m':None,'depth_views':[]}
        geometry=row.get('geometry',{})
        if row['status']=='current_estimate' and geometry.get('position_fully_observed'):
            center=np.asarray(geometry['center_m'],float)
            result['current_visual_center_m']=center.tolist()
            result['fully_observed_visual_residual_m']=float(np.linalg.norm(center-predicted))
        for path,call in frames:
            if abs(float(call['time'])-stamp)>1e-6:
                continue
            with np.load(path) as data:
                frame={k:data[k] for k in data.files}
            if abs(float(frame['time'])-stamp)>1e-6:
                raise ValueError('Camera archive time mismatch')
            hashes[str(path)]=digest(path)
            check=depth_consistency(frame,color,predicted,anchor['dimensions'],plane)
            result['depth_views'].append(dict(check,view=call['params']['view'],camera_path=str(path)))
        known=[v['status'] for v in result['depth_views'] if v['status']!='unknown']
        result['depth_evidence']=('unknown' if not known else known[0] if len(set(known))==1 else 'conflicting_views')
        result['pose_recovered']=False
        rows.append(result)
        if result['depth_evidence']=='contradicted':
            anchor=None
    return {'trial':str(trial),'records':rows,'invalidations':invalidations,'input_sha256':hashes}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite diagnostic evidence')
    trials=[replay_trial(p.parent) for p in sorted(args.trials.glob('seed_*/api_calls.json'))]
    rows=[r for t in trials for r in t['records']]
    report={'scope':__doc__,'script_sha256':digest(Path(__file__)),
            'prediction_source_sha256':digest(Path(__file__).with_name('replay_held_geometry.py')),
            'visual_helpers_sha256':digest(Path(__file__).resolve().parents[1]/'src/aspire_so100/instance_geometry.py'),
            'training_trials_read':len(trials),'eligible_place_boundaries':len(rows),
            'contradicted_by_current_depth':sum(r['depth_evidence']=='contradicted' for r in rows),
            'poses_recovered':0,'trials':trials,
            'limitations':['Only a conditional kinematic hypothesis; no contact or task-success labels read.',
                           'The 5 mm margin is a diagnostic tolerance, not a calibrated uncertainty bound.',
                           'Unique-color association and previously observed size are assumed; green is left unknown without robot masking.',
                           'Passing the enclosing-sphere test cannot establish a pose or exclude slip.',
                           'No memory is fused into a controller; no learning or SR improvement is claimed.']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('trials','scope')},indent=2))
    for t in trials:
        for r in t['records']:print(Path(t['trial']).name,r)
