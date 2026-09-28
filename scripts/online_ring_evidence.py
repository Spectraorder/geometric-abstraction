"""Source snapshots and independent online Ring interface/resource checks."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import mujoco
from ring_peg_scene import RingPegEnv, RingPegGeometry
from PIL import Image
from run_online_bridge_pilot import source_paths, package_versions, client_version
from audit_online_reasoning_development import audit as resource_audit
from geometry_actor_session import DISABLED_TOOLS
from online_ring_actor import validate_response
from online_ring_loop import CONTRACT
from online_metric_depth import MetricDepthFrames
from online_ring_skills import digest

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()] if Path(path).exists() else []
def require(ok,message):
    if not ok:raise ValueError(message)


def snapshot(output):
    sources={str(p.relative_to(ROOT)):digest(p) for p in source_paths()}
    for name in sources:
        destination=output/'sources'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    (output/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    return sources


def verify_snapshot(output):
    for name,sha in read(output/'sources.json').items():
        require(digest(ROOT/name)==sha==digest(output/'sources'/name),'Source changed: '+name)


def audit_episode(root,relative):
    directory=root/relative;result=read(directory/'result.json')
    require(result['status']!='infrastructure_error' and not result.get('archive_error'),'Episode infrastructure failure')
    require(result['timing']['reset_count']==1,'Unexpected reset')
    require(result['timing']['paced_actions'] and not result['scripted'],'Expected live paced episode')
    feedback={r['turn']:r for r in rows(directory/'feedback.jsonl')}
    decisions={r['turn']:r['decision'] for r in rows(directory/'model_decisions.jsonl')}
    bindings={r['turn']:r['decision'] for r in rows(directory/'target_bindings.jsonl')}
    frame_store=MetricDepthFrames();query_rows=iter(rows(directory/'depth_queries.jsonl'))
    previous=b'';session=None;request_seconds=0.;incomplete_seconds=0.;images=0
    for index,turn in enumerate(sorted(directory.glob('context_000/decision_*'))):
        invocation=read(turn/'invocation.json');packet=feedback[index]['packet'];command=invocation['command']
        require(invocation['model']=='gpt-6-astra' and invocation['effort']=='medium','Model settings changed')
        require('model_provider="openai-http"' in command,'Expected uniform HTTP transport')
        require('--ignore-user-config' in command and invocation['repository_context'] is False,'Unexpected context')
        for tool in DISABLED_TOOLS:
            require(any(command[i:i+2]==['--disable',tool] for i in range(len(command)-1)),'Actor tool access changed')
        require(set(packet)<= {'robot','last_result','last_grasp_opening_command','cameras','geometry','remaining_wall_seconds'},'Unexpected feedback field')
        if result['arm']=='basic':require('geometry' not in packet,'Base geometry leak')
        prefix=CONTRACT+'\nTask: '+json.dumps(read(directory/'goal.json'))+'\n' if index==0 else ''
        expected=prefix+'Interface: '+result['arm']+'\nCurrent feedback:\n'+json.dumps(packet,separators=(',',':'))
        require((turn/'prompt.txt').read_text()==expected,'Prompt projection differs')
        require([r['source'] for r in invocation['images']]==feedback[index]['images'],'Image list changed')
        images+=len(invocation['images'])
        for view,calibration in packet.get('cameras',{}).items():
            path=next(Path(p) for p in feedback[index]['images'] if Path(p).name.endswith('_'+view+'.png'))
            with np.load(path.with_suffix('.npz')) as archive:frame={key:archive[key] for key in archive.files}
            require(np.array_equal(np.asarray(Image.open(path)),frame['rgb']),'Image/depth frame mismatch')
            descriptor=frame_store.register(view,frame)
            require(descriptor==calibration['metric_depth_frame'],'Displayed frame registration changed')
        if (turn/'attempt_failure.json').exists():
            require(index==len(feedback)-1 and not (turn/'receipt.json').exists(),'Incomplete request retried')
            incomplete_seconds+=read(turn/'attempt_failure.json')['wall_seconds']
            continue
        receipt=read(turn/'receipt.json');request_seconds+=receipt['wall_seconds']
        require(session is None or session==receipt['session_id'],'Persistent session changed')
        session=receipt['session_id'];transcript=(turn/'session.jsonl').read_bytes()
        require(transcript.startswith(previous),'Context was rewritten')
        new=[json.loads(x) for x in transcript[len(previous):].decode().splitlines() if x.strip()]
        users=[x['payload'] for x in new if x.get('type')=='response_item' and x.get('payload',{}).get('role')=='user']
        matched=[r for r in users if any(c.get('type')=='input_text' and c.get('text')==expected for c in r.get('content',[]))]
        require(len(matched)==1,'Prompt missing from actual model transcript')
        embedded=[c['image_url'] for c in matched[0]['content'] if c.get('type')=='input_image']
        require([hashlib.sha256(base64.b64decode(v.split(',',1)[1])).hexdigest() for v in embedded]==
                [r['sha256'] for r in invocation['images']],'Actual model images differ')
        previous=transcript
        response=validate_response(read(turn/'response.json'))
        require(response==decisions[index],'Recorded decision differs from model output')
        if response['action'] in ('grasp_lift','move_pad','release','wait'):
            require(bindings.get(index)==response,'Execution target changed')
        elif response['action']=='sample_depth':
            query=next(query_rows)
            require(query['frame_id']==response['frame_id'] and query['pixels']==response['pixels'],'Depth query changed')
            expected_query=frame_store.sample(query['frame_id'],query['pixels'],current_time=query['result'].get('current_time',0))
            require(expected_query==query['result'],'Depth samples differ from archived sensor pixels')
    require(next(query_rows,None) is None,'Unaccounted depth query')
    require(abs(request_seconds-result['model_seconds'])<1e-6,'Request time mismatch')
    require(abs(incomplete_seconds-result['incomplete_model_seconds'])<1e-6,'Incomplete time mismatch')
    replay=object.__new__(RingPegEnv)
    replay.geometry=RingPegGeometry()
    replay.model=mujoco.MjModel.from_binary_path(str(directory/'private_model.mjb'))
    replay.data=mujoco.MjData(replay.model)
    with np.load(directory/'private_terminal_state.npz') as state:
        replay.data.time=float(state['time'])
        for key in ('qpos','qvel','act','ctrl','qacc_warmstart'):getattr(replay.data,key)[:]=state[key]
    mujoco.mj_forward(replay.model,replay.data)
    terminal=replay.ring_status()
    require(terminal['on_peg']==result['private_final_status']['on_peg'],'Terminal evaluator replay differs')
    if result['success']:
        require(terminal['on_peg'],'Private terminal physics did not satisfy the task')
        require(result['private_final_status']['on_peg'],'Successful terminal state violates evaluator')
        evidence=result['private_success_evidence'];monitor=read(directory/'private_monitor.json')
        held=[r for r in monitor if evidence['start_sim_time']<=r['time']<=evidence['end_sim_time']]
        require(held and all(r['on_peg'] and r['drift_m']<=.003 for r in held),'Stability trace violation')
        require(evidence['end_sim_time']-evidence['start_sim_time']>=2.,'Insufficient stability duration')
        require(result['time_to_success_s']<=result['budget_s'],'Success after deadline')
    return dict(episode=relative,seed=result['seed'],arm=result['arm'],success=result['success'],
                capped_seconds=result['capped_completion_seconds'],usage=result['usage'],images=images,
                completed_requests=result['model_requests_completed'],incomplete_requests=result['incomplete_requests'],
                model_seconds=request_seconds,incomplete_model_seconds=incomplete_seconds,
                operations=result['operations'],depth_queries=result['depth_queries'],
                camera_acquisitions=result['camera_acquisitions'],action_seconds=result['action_seconds'],
                feedback_seconds=result['feedback_seconds'],initial_state_sha256=result['private_initial_state_sha256'],
                max_simulation_lag_seconds=result['timing']['max_simulation_lag_seconds'])


def audit(root,paths):
    resource_audit(root,paths)
    records=[audit_episode(root,p) for p in paths]
    for seed in {r['seed'] for r in records}:
        paired=[r for r in records if r['seed']==seed]
        require(len(paired)==2 and len({r['initial_state_sha256'] for r in paired})==1,'Initial states not paired')
    report=dict(status='passed',episodes=records,
                scope='Sources, tokens, actual prompt/image delivery, Base geometry exclusion, submitted targets, depth and continuous success traces')
    (root/'interface_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
