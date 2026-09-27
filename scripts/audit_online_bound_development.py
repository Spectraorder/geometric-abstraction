"""Audit actual depth queries, submitted targets and terminal contacts in development."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from audit_online_reasoning_development import audit as audit_resources
from online_bound_actor import validate_response


def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def terminal_contacts(directory):
    model=mujoco.MjModel.from_binary_path(str(directory/'private_model.mjb'))
    data=mujoco.MjData(model)
    with np.load(directory/'private_terminal_state.npz') as state:
        data.time=float(state['time'])
        for name in ('qpos','qvel','act','ctrl','qacc_warmstart'):getattr(data,name)[:]=state[name]
    mujoco.mj_forward(model,data)
    def namespaces(geom):
        names={model.geom(geom).name.split('/')[0]}
        body=int(model.geom_bodyid[geom])
        while body:
            names.add(model.body(body).name.split('/')[0]);body=int(model.body_parentid[body])
        return names
    objects={model.body(i).name.split('/')[0] for i in range(model.nbody)
             if model.body(i).name.startswith(('red_','blue_','green_','yellow_','magenta_'))}
    touching=set()
    for contact in data.contact:
        a,b=namespaces(contact.geom1),namespaces(contact.geom2)
        if 'robot' in a:touching.update(b&objects)
        if 'robot' in b:touching.update(a&objects)
    return sorted(touching)


def audit(root,episode_paths=None):
    resource=audit_resources(root,episode_paths)
    reports=[]
    paths=root.glob('*/result.json') if episode_paths is None else (root/p/'result.json' for p in episode_paths)
    for path in sorted(paths):
        directory=path.parent;result=read(path)
        if result['scripted']:raise ValueError('This audit expects model-driven episodes')
        decisions={row['turn']:validate_response(row['decision']) for row in rows(directory/'model_decisions.jsonl')}
        receipts=list(sorted(directory.glob('context_000/decision_*/receipt.json')))
        if len(receipts)!=len(decisions):raise ValueError('Decision/receipt count mismatch')
        for i,receipt in enumerate(receipts):
            if read(receipt.parent/'response.json')!=decisions[i]:raise ValueError('Recorded decision differs from actual response')
        bindings=rows(directory/'target_bindings.jsonl')
        goal=read(directory/'goal.json') if (directory/'goal.json').exists() else {'nodes':[]}
        nodes={n['key']:n for n in goal['nodes']}
        attempted=[d for d in decisions.values() if d['action']=='execute']
        executions=[d for d in attempted if d['node'] in nodes and
                    (len(nodes[d['node']]['supports'])==1)==(d['skill']=='stack')]
        for turn,d in decisions.items():
            if d['action']=='execute' and d not in executions:
                outcome=next((r for r in rows(directory/'decision_results.jsonl') if r['turn']==turn),None)
                if outcome is None or outcome['result']['status']!='invalid_operation':
                    raise ValueError('Rejected execution was not recorded')
        if len(bindings)!=len(executions):raise ValueError('Execution/binding count mismatch')
        for binding,decision in zip(bindings,executions):
            for key in ('node','skill','surface_point_world_m'):
                if binding[key]!=decision[key]:raise ValueError('Submitted target was replaced')
        queries=iter(rows(directory/'depth_queries.jsonl'));query_count=0;sample_count=0;latest={}
        for feedback in rows(directory/'feedback.jsonl'):
            packet=feedback['packet']
            if result['arm']=='basic' and 'geometry' in packet:raise ValueError('Object geometry exposed to Base')
            for view,calibration in packet.get('cameras',{}).items():
                descriptor=calibration['metric_depth_frame']
                image=next(Path(p) for p in feedback['images'] if Path(p).name.endswith('_'+view+'.png'))
                with np.load(image.with_suffix('.npz')) as frame:
                    saved={name:frame[name] for name in frame.files}
                if not np.array_equal(np.asarray(Image.open(image)),saved['rgb']):raise ValueError('Displayed/query frame mismatch')
                if descriptor['observation_time']!=float(saved['time']):raise ValueError('Frame timestamp mismatch')
                latest[view]=(descriptor['frame_id'],saved)
            decision=decisions.get(feedback['turn'])
            if decision is None or decision['action']!='sample_depth':continue
            query=next(queries);query_count+=1
            if query['frame_id']!=decision['frame_id'] or query['pixels']!=decision['pixels']:
                raise ValueError('Query differs from model decision')
            frame=next((f for identifier,f in latest.values() if identifier==query['frame_id']),None)
            response=query['result']
            if frame is None:
                if response['status']!='frame_unavailable':raise ValueError('Expired frame accepted')
                continue
            if response['status']!='sampled':raise ValueError('Unexpected query status')
            if (response['observation_time']!=float(frame['time']) or
                    response['at_current_sensor_time']!=(response['current_time']==float(frame['time']))):
                raise ValueError('Query freshness mismatch')
            if len(response['samples'])!=len(query['pixels']):raise ValueError('Missing samples')
            for (u,v),sample in zip(query['pixels'],response['samples']):
                sample_count+=1
                if sample['pixel']!=[u,v]:raise ValueError('Pixel order changed')
                h,w=frame['depth_m'].shape
                if not (0<=u<w and 0<=v<h):
                    if sample['status']!='out_of_frame' or 'world_point_m' in sample:raise ValueError('Bad pixel accepted')
                    continue
                depth=float(frame['depth_m'][v,u])
                valid=bool(frame['valid_depth'][v,u]) and np.isfinite(depth) and depth>0
                if not valid:
                    if sample['status']!='invalid_depth' or 'world_point_m' in sample:raise ValueError('Bad depth accepted')
                    continue
                point=frame['camera_to_world'][:3,:3]@(depth*np.linalg.solve(frame['intrinsics'],[u,v,1.]))+frame['camera_to_world'][:3,3]
                if sample['depth_m']!=depth or not np.allclose(sample['world_point_m'],point,rtol=0,atol=1e-10):
                    raise ValueError('Metric query differs from sensor archive')
        if next(queries,None) is not None:raise ValueError('Unaccounted query')
        if query_count!=result['depth_queries']:raise ValueError('Query aggregate mismatch')
        touching=terminal_contacts(directory)
        if result['success'] and touching:raise ValueError('Successful terminal state touches robot')
        if result['timing']['reset_count']!=1 or not result['timing']['paced_actions']:
            raise ValueError('Episode reset or pacing mismatch')
        reports.append(dict(seed=result['seed'],episode=str(directory.relative_to(root)),arm=result['arm'],status=result['status'],success=result['success'],
            capped_seconds=result['capped_completion_seconds'],usage=result['usage'],
            images=sum(len(row['images']) for row in rows(directory/'feedback.jsonl')),
            completed_requests=result['model_requests_completed'],incomplete_requests=result['incomplete_requests'],
            model_seconds=result['model_seconds'],target_bindings=len(bindings),depth_queries=query_count,
            invalid_operations=len(attempted)-len(executions),
            depth_samples=sample_count,terminal_robot_contacts=touching,
            initial_state_sha256=result['private_initial_state_sha256'],
            max_simulation_lag_seconds=result['timing']['max_simulation_lag_seconds']))
    for seed in {r['seed'] for r in reports}:
        pair=[r for r in reports if r['seed']==seed]
        if len(pair)==2 and pair[0]['initial_state_sha256']!=pair[1]['initial_state_sha256']:
            raise ValueError('Initial physics not paired')
    report=dict(status='passed',episodes=reports,resource_audit=resource['status'],
        scope='Actual response/binding equality, archived metric queries, Base reference exclusion, '
              'initial physics pairing and reconstructed terminal contacts. Continuous stability uses '
              'the original private monitor. Statistical interpretation depends on the registered study size.',
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'interface_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    result=audit(parser.parse_args().root);print(json.dumps(result,indent=2))
