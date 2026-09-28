"""Audit relocation inputs, target bindings, resources, and terminal task geometry."""
import json
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np
from audit_online_bound_development import audit as bound_audit
from audit_online_compact_prompts import audit_prompts
from run_online_ring_checked_study import installed
from online_ring_evidence import read, require
from online_relocation_scene import RelocationEnv
from online_relocation_runtime import CONTRACT
from online_compact_geometry import compact_prompt


def audit(root, slots):
    with installed():
        report=bound_audit(root,[s['path'] for s in slots])
    audit_prompts(root,slots)
    for row in report['episodes']:
        episode=root/row['episode']; result=read(episode/'result.json')
        require(result['kind'] in ('relocation_development','relocation_reserved'),'Unexpected episode kind')
        require(result['scenario']=='initial_three_cube_column_and_separate_destination','Wrong initial scenario')
        goal=read(episode/'goal.json')
        feedback=[json.loads(s) for s in (episode/'feedback.jsonl').read_text().splitlines()]
        explained=False
        for index,turn in enumerate(sorted(episode.glob('context_000/decision_*'))):
            invocation=read(turn/'invocation.json')
            require(invocation['model']=='gpt-6-astra' and invocation['effort']=='medium','Model changed')
            require('model_provider="openai-http"' in invocation['command'],'Expected HTTP-only transport')
            packet=feedback[index]['packet']
            require(set(packet)<= {'time','holding','last_result','memory','cameras','geometry','remaining_wall_seconds'},'Unexpected feedback field')
            prefix=CONTRACT+'\nGoal: '+json.dumps(goal)+'\n' if index==0 else ''
            raw=prefix+'Interface: '+result['arm']+'\nCurrent feedback:\n'+json.dumps(packet,separators=(',',':'))
            expected,encoded=compact_prompt(raw,explain=not explained)
            require((turn/'prompt.txt').read_text()==expected,'Actual task/feedback prompt differs')
            if (turn/'receipt.json').exists() and encoded:explained=True
        env=object.__new__(RelocationEnv)
        env.model=mujoco.MjModel.from_binary_path(str(episode/'private_model.mjb'))
        env.data=mujoco.MjData(env.model)
        with np.load(episode/'private_terminal_state.npz') as saved:
            env.data.time=float(saved['time'])
            for key in ('qpos','qvel','act','ctrl','qacc_warmstart'):getattr(env.data,key)[:]=saved[key]
        mujoco.mj_forward(env.model,env.data)
        env.roles={key:color+'_cube' for key,color in goal['roles'].items()}
        env.order=tuple(env.roles[r] for r in ('source','destination','middle','upper'))
        env.anchor_positions={key:np.asarray(value) for key,value in read(episode/'private_initial_anchors.json').items()}
        env.anchor_violation=False
        terminal=env.structure_status(env.order)
        monitor=read(episode/'private_monitor.json')
        if result['success']:
            require(terminal['stable'],'Successful terminal structure invalid')
            hold=result['private_success_evidence']
            require(hold['duration_sim_seconds']>=5. and hold['maximum_final_displacement_m']<=.003,'Invalid stability evidence')
            observed=[r for r in monitor if r['time']>=hold['stable_start_sim_time']]
            require(len(observed)>=90 and all(r['stable_since']==hold['stable_start_sim_time'] and not r['anchor_violation'] for r in observed),'Stability hold interrupted')
        row.update({key:result[key] for key in ('camera_acquisitions','action_seconds','feedback_seconds','depth_query_seconds','incomplete_model_seconds')})
        row['terminal_structure']=terminal
    report['scope']+=' Relocation task prompt, HTTP routing, final support geometry, anchor positions and continuous hold records.'
    (root/'relocation_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
