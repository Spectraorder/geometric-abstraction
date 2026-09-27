"""Separate sensor-only bounds replay, followed by private offline coverage scoring."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from scipy import ndimage
from rigid_containment_prototype import containment_bounds


def read_frame(path):
    with np.load(path) as data:return {key:data[key].copy() for key in data.files}


def visible_cloud(frame,color,plane,library):
    points,valid=library['rgbd_points'](frame)
    height=points[...,2]-points[...,:2]@plane[:2]-plane[2]
    labels,count=ndimage.label(library['color_mask'](frame['rgb'],color)&valid&(height>.004)&(height<.35))
    components=[points[labels==i] for i in range(1,count+1)]
    components=[cloud for cloud in components if len(cloud)>=12]
    if not components:raise ValueError('No visible component')
    return max(components,key=len)


def estimate(trial,library,frame_transform=None):
    """Read RGB-D and runtime geometry records; never physics or result files."""
    calls=[c for c in json.loads((trial/'api_calls.json').read_text()) if c['method']=='camera']
    files=sorted(trial.glob('camera_*.npz'))
    if len(files)!=len(calls):raise ValueError('Camera record count mismatch')
    records=json.loads((trial/'geometry_records.json').read_text())
    stamps=sorted(set(float(row['time']) for row in records))
    initial=read_frame(files[0])
    if frame_transform is not None:initial=frame_transform(initial)
    scene=library['GEOMETRY'].scene(initial,['blue','red','green'])
    groups={}
    for path,call in zip(files,calls):
        frame=read_frame(path);stamp=float(frame['time'])
        if stamp in stamps:
            if frame_transform is not None:frame=frame_transform(frame)
            groups.setdefault(stamp,{})[call['params']['view']]=frame
    rows=[]
    for stamp,frames in sorted(groups.items()):
        for view,frame in frames.items():scene.refresh(frame,view)
        for color in ['blue','red','green']:
            try:state=scene.get(color)
            except ValueError:continue
            row={'time':stamp,'color':color,'center_m':state['center'].tolist(),
                 'observed_rank':state['observed_translation_rank'],'rotation':state['rotation'].tolist(),
                 'observed_projector':state['observed_translation_projector'].tolist(),
                 'prior_sigma_mm':(1000*np.sqrt(np.diag(state['covariance_m2']))).tolist()}
            if state['inferred_orientation']:
                row['bounds_available']=False;row['reason']='orientation_not_observed';rows.append(row);continue
            rejected={v['view'] for v in scene.rejected_views if v['object']==color}
            clouds=[visible_cloud(frame,color,scene.table_coefficients,library) for view,frame in frames.items()
                    if view not in rejected and view in scene.current_fits[color]]
            if not clouds:
                row['bounds_available']=False;row['reason']='no_accepted_points';rows.append(row);continue
            result=containment_bounds(np.concatenate(clouds),state['rotation'],np.repeat(state['height'],3))
            row.update({key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in result.items()})
            row['bounds_available']=result['consistent'];rows.append(row)
    return {'trial':str(trial),'rows':rows}


def score(packet):
    """Private evaluation happens only after sensor-only outputs are saved."""
    import mujoco
    from aspire_so100.playback import recorded_env_for_result
    trial=Path(packet['trial']);result=json.loads((trial/'result.json').read_text())
    with np.load(trial/'rollout.npz') as record:
        states=record['states'];state_type=mujoco.mjtState(int(record['state_type']))
    with TemporaryDirectory(prefix='containment-offline-score-') as temp:
        env=recorded_env_for_result(result,Path(temp))
        try:
            for row in packet['rows']:
                index=int(np.argmin(abs(states[:,0]-row['time'])))
                if abs(states[index,0]-row['time'])>1.1*env.model.opt.timestep:raise ValueError('Time alignment failed')
                mujoco.mj_setState(env.model,env.data,states[index],state_type);mujoco.mj_forward(env.model,env.data)
                name=row['color']+'_cube';gid=env.model.geom(name+'/'+name+'_geom').id
                center=env.data.geom_xpos[gid].copy()
                row['current_center_error_mm']=float(np.linalg.norm(np.asarray(row['center_m'])-center)*1000)
                if row['bounds_available']:
                    local=center@np.asarray(row['rotation'])
                    lower=np.asarray(row['center_local_lower_m']);upper=np.asarray(row['center_local_upper_m'])
                    row['covered_by_bounds']=bool(np.all(local>=lower)&np.all(local<=upper))
                    row['max_bound_violation_mm']=float(max(0.,np.max(lower-local),np.max(local-upper))*1000)
                    row['midpoint_error_mm']=float(np.linalg.norm(np.asarray(row['center_world_midpoint_m'])-center)*1000)
        finally:env.close()
    return packet


def summarize(packets):
    rows=[row for packet in packets for row in packet['rows']];groups={}
    for label,subset in [('all',rows),('partial',[r for r in rows if r['observed_rank']<3])]:
        available=[row for row in subset if row['bounds_available']]
        groups[label]={'total':len(subset),'available':len(available),'covered':sum(r['covered_by_bounds'] for r in available)}
        if available:
            groups[label].update(median_max_interval_width_mm=float(np.median([max(r['interval_width_m'])*1000 for r in available])),
                p95_max_interval_width_mm=float(np.percentile([max(r['interval_width_m'])*1000 for r in available],95)),
                largest_bound_violation_mm=max(r['max_bound_violation_mm'] for r in available),
                paired_current_center_p95_mm=float(np.percentile([r['current_center_error_mm'] for r in available],95)),
                paired_interval_midpoint_p95_mm=float(np.percentile([r['midpoint_error_mm'] for r in available],95)))
    return {'groups':groups,'tolerance_m':.0015,'limits':[
        'Inspected training trajectories, correlated checkpoints, ideal simulated depth; not reserved evidence.',
        'Bounds assume correct instance association and orientation/dimensions within the explicit tolerance.',
        'No policy execution or training input is changed by this separate prototype.',
        'Do not interpret interval midpoint accuracy or coverage as task-success improvement.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replica',type=Path,required=True);parser.add_argument('--trials',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads((args.replica/'manifest.json').read_text());path=args.replica/'geometry.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['geometry_sha256']:raise ValueError('Frozen library mismatch')
    library={};exec(compile(path.read_text(),str(path),'exec'),library)
    args.output.mkdir(parents=True,exist_ok=False)
    for source in [Path(__file__),Path(__file__).with_name('rigid_containment_prototype.py')]:
        (args.output/source.name).write_bytes(source.read_bytes())
    packets=[estimate(trial.resolve(),library) for trial in args.trials]
    (args.output/'sensor_only_estimates.json').write_text(json.dumps(packets,indent=2)+'\n')
    scored=[score(packet) for packet in packets]
    (args.output/'offline_scored_estimates.json').write_text(json.dumps(scored,indent=2)+'\n')
    summary=summarize(scored);(args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
