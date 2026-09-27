"""Offline memory/prediction diagnostic; simulator truth is scoring-only.

Three estimators consume the same recorded RGB-D. The prediction condition is a
visual lift near a closed gripper, and SE(3) displacement uses recorded robot
encoder poses, never commanded poses or object truth. This diagnostic is not a
policy success comparison or a production attachment estimator.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import mujoco
from aspire_so100.online_geometry import SceneGeometry
from aspire_so100.playback import recorded_env_for_result

MODES=('instantaneous','memory','conditional_transport')


def transform(observation):
    result=np.eye(4)
    result[:3,:3]=np.asarray(observation['tool_rotation'])
    result[:3,3]=np.asarray(observation['tool_position'])
    return result


def estimate(trial):
    """Sensor/encoder-only phase: no result, trace or physics-record reads."""
    calls=json.loads((trial/'api_calls.json').read_text())
    cameras=[call for call in calls if call['method']=='camera']
    first=next(i for i,c in enumerate(cameras) if c['params'].get('view','side')=='side')
    initial=dict(np.load(trial/f'camera_{first:03d}.npz'))
    colors=('blue','red','green')
    scenes={mode:SceneGeometry(initial,colors,memory=mode!='instantaneous') for mode in MODES}
    initial_height={color:scenes['instantaneous'].get(color)['center'][2] for color in colors}
    rows=[];events=[];camera_index=-1;closed=False;attached=None;anchor=None;robot=None
    for call in calls:
        stamp=float(call['time']);method=call['method']
        if method=='gripper':
            if call['params']['angle']<0:
                closed=True;attached=None;anchor=None
            else:
                closed=False
                if attached is not None:
                    # Diagnostic adapter explicitly revokes the hypothesis when
                    # opening is commanded. This does not certify object release.
                    scene=scenes['conditional_transport']
                    scene.belief.objects[attached]['prediction']=None
                    scene.belief.missing(attached,stamp,'attachment_condition_revoked_by_open_command')
                    scene.latest_time=stamp
                    events.append({'kind':'condition_revoked','time':stamp,'object':attached})
                attached=None;anchor=None
        if method=='proprioception' and 'observation' in call:
            robot=call['observation'];current=transform(robot)
            if closed and attached is not None and anchor is not None:
                scene=scenes['conditional_transport']
                scene.predict(attached,current @ np.linalg.inv(anchor),float(robot['time']),
                              'if visually lifted object remains rigidly attached to closed gripper')
                anchor=current
                events.append({'kind':'conditional_prediction','time':float(robot['time']),'object':attached})
        if method!='camera':continue
        camera_index+=1
        if camera_index<first:continue
        sensor=trial/f'camera_{camera_index:03d}.npz';frame=dict(np.load(sensor));stamp=float(frame['time'])
        for mode,scene in scenes.items():
            scene.refresh(frame,call['params'].get('view','side'))
            for color in colors:
                try:value=scene.get(color)
                except ValueError:continue
                rows.append({'mode':mode,'camera_index':camera_index,'time':stamp,'color':color,
                    'center_m':value['center'].tolist(),'observed_rank':value['observed_translation_rank'],
                    'source':value['source'],'sigma_mm':(np.sqrt(np.diag(value['covariance_m2']))*1000).tolist(),
                    'sensor_sha256':hashlib.sha256(sensor.read_bytes()).hexdigest()})
        if closed and attached is None and robot is not None and abs(float(robot['time'])-stamp)<1e-6:
            candidates=[]
            for color in colors:
                try:value=scenes['instantaneous'].get(color)
                except ValueError:continue
                z=np.array([0.,0.,1.]);observed=np.linalg.norm(value['observed_translation_projector']@z-z)<.05
                distance=np.linalg.norm(value['center']-np.asarray(robot['tool_position']))
                lifted=value['center'][2]-initial_height[color]>.5*value['height']
                if observed and lifted and distance<2*value['height']:
                    candidates.append((distance,color))
            if candidates:
                attached=min(candidates)[1];anchor=transform(robot)
                events.append({'kind':'visual_lift_attachment_hypothesis','time':stamp,'object':attached})
    return {'trial':str(trial),'rows':rows,'conditional_events':events,
            'input_contract':'Only RGB-D, public color IDs, logged gripper commands and robot proprioception.'}


def score(packet):
    """Read private physics only after the complete sensor-only packet exists."""
    trial=Path(packet['trial']);result=json.loads((trial/'result.json').read_text())
    with np.load(trial/'rollout.npz') as record:
        states=record['states'];state_type=mujoco.mjtState(int(record['state_type']))
    with TemporaryDirectory(prefix='temporal-geometry-score-') as tmp:
        env=recorded_env_for_result(result,Path(tmp))
        try:
            for row in packet['rows']:
                index=int(np.argmin(abs(states[:,0]-row['time'])))
                if abs(states[index,0]-row['time'])>1.1*env.model.opt.timestep:raise ValueError('State time alignment failed')
                mujoco.mj_setState(env.model,env.data,states[index],state_type);mujoco.mj_forward(env.model,env.data)
                name=row['color']+'_cube';gid=env.model.geom(name+'/'+name+'_geom').id
                residual=(np.asarray(row['center_m'])-env.data.geom_xpos[gid])*1000
                row['center_error_mm']=float(np.linalg.norm(residual));row['residual_xyz_mm']=residual.tolist()
        finally:env.close()
    return packet


def summarize(packets):
    rows=[row for packet in packets for row in packet['rows']];groups={}
    for mode in MODES:
        groups[mode]={}
        for label,partial in [('all',None),('fully_observed',False),('partial_or_stale',True)]:
            chosen=[r for r in rows if r['mode']==mode and (partial is None or (r['observed_rank']<3)==partial)]
            values=[r['center_error_mm'] for r in chosen]
            groups[mode][label]={'count':len(values),'median_mm':float(np.median(values)) if values else None,
                'p95_mm':float(np.percentile(values,95)) if values else None,'max_mm':max(values) if values else None}
    return {'scope':'Private offline diagnostic; excluded from all actor inputs','groups':groups,
            'events':{kind:sum(e['kind']==kind for p in packets for e in p['conditional_events']) for kind in
                      ['visual_lift_attachment_hypothesis','conditional_prediction','condition_revoked']},
            'limits':['Correlated frames from development rollouts, not independent accuracy trials.',
                'No policy success or geometry-benefit claim follows from this replay.',
                'Attachment is an explicit fallible hypothesis, not contact truth.',
                'Open-command revocation is an offline adapter; public scene revocation API remains to be designed.',
                'Only centers scored; ideal simulated RGB-D, not real-sensor validation.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    packets=[estimate(path.resolve()) for path in args.trials]
    (args.output/'sensor_only_estimates.json').write_text(json.dumps(packets,indent=2)+'\n')
    scored=[score(packet) for packet in packets]
    (args.output/'offline_scored_estimates.json').write_text(json.dumps(scored,indent=2)+'\n')
    summary=summarize(scored);(args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
