"""Post-run audit of recovery prompts, image evidence, counters and pairing."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from audit_online_bridge_pilot import require, records, read, digest
from geometry_actor_session import DISABLED_TOOLS
from online_bridge_actor import audit_decision_events, validate_response
from run_online_event_development import CONTRACT
from run_online_bridge_pilot import verify_sources, write
from summarize_online_recovery import summarize
from verified_token_usage import turn_usage


def audit(root, partial=False):
    root=root.resolve();protocol=verify_sources(root);identities=set();reports=[]
    for slot in protocol['slots']:
        folder=root/slot['path']
        if not (folder/'result.json').exists():
            require(partial,'Study is not complete')
            continue
        result=read(folder/'result.json')
        require(result['seed']==slot['seed'] and result['arm']==slot['arm'],'Slot mismatch')
        require(result['budget_s']==protocol['budget_s'],'Budget mismatch')
        fixture=read(folder/'fixture.json')
        if fixture['status']=='prefix_failed':
            require(not list(folder.glob('context_*')),'Actor called after setup failure')
            reports.append(dict(slot,status='setup_failed'));continue
        require(fixture['kind']=='foundation' and fixture['axis']=='y','Intervention mismatch')
        require(fixture['displacement_m']==.025 and len(fixture['moved_private'])==1,'Intervention size mismatch')
        moved=fixture['moved_private'][0]
        delta=[a-b for a,b in zip(moved['after'],moved['before'])]
        require(all(abs(x-y)<1e-12 for x,y in zip(delta,[0,.025,0,0,0,0,0])),'Unexpected intervention change')
        require(fixture['reset_count']==result['timing']['reset_count']==1,'Scene reset')
        require(result['timing']['paced_actions'],'Unpaced evaluation')
        if result['success']:
            evidence=result['private_success_evidence']
            require(evidence['duration_sim_seconds']>=5.,'Success interval too short')
            require(evidence['maximum_final_displacement_m']<=.003,'Success drift too large')
            require(result['private_final_structure']['stable'],'Invalid final structure')
            require(result['time_to_success_s']<=protocol['budget_s']+.05,'Success outside budget')
        feedback=records(folder/'feedback.jsonl');global_index=0;receipts=[]
        for context_index,context in enumerate(sorted(folder.glob('context_*'))):
            require(context.name==f'context_{context_index:03d}','Context sequence gap')
            require(not (context/'actor_home/auth.json').exists(),'Temporary credentials retained')
            previous=b'';counts_before=[];totals_before={};session_id=None
            turns=sorted(context.glob('decision_*'));require(len(turns)<=3,'Conversation window exceeded')
            for local_index,directory in enumerate(turns):
                require(directory.name==f'decision_{local_index:03d}','Decision sequence gap')
                invocation=read(directory/'invocation.json');command=invocation['command']
                model=invocation['model'];effort='medium' if model=='gpt-6-astra' else 'low'
                require(model in ('gpt-6-astra','gpt-6-luna') and invocation['effort']==effort,'Model mismatch')
                require(global_index!=0 or model=='gpt-6-astra','Initial planner mismatch')
                require(invocation['session_before']==session_id,'Session lineage mismatch')
                for flag in DISABLED_TOOLS:
                    require(any(command[i:i+2]==['--disable',flag] for i in range(len(command)-1)),'Tool enabled')
                for pair in (['--sandbox','read-only'],['-c','project_doc_max_bytes=0'],['-c','web_search="disabled"']):
                    require(any(command[i:i+2]==pair for i in range(len(command)-1)),'Isolation mismatch')
                require('--ignore-user-config' in command,'Uncontrolled client configuration')
                row=feedback[global_index];packet=row['packet']
                require(('geometry' in packet)==(slot['arm']=='symbolic'),'Treatment mismatch')
                require([v['source'] for v in invocation['images']]==row['images'],'Image list mismatch')
                for im in invocation['images']:
                    image_path=Path(im['source'])
                    require(image_path.is_relative_to(folder) and digest(image_path)==im['sha256'],'Image provenance mismatch')
                prefix=CONTRACT+'\nGoal: '+json.dumps(read(folder/'goal.json'))+'\n' if local_index==0 else ''
                expected=prefix+'Interface: '+slot['arm']+'\nCurrent feedback:\n'+json.dumps(packet,separators=(',',':'))
                prompt=(directory/'prompt.txt').read_text();require(prompt==expected,'Prompt projection mismatch')
                if (directory/'attempt_failure.json').exists():
                    require(global_index==len(feedback)-1,'Incomplete request retried')
                    global_index+=1;continue
                receipt=read(directory/'receipt.json');events=records(directory/'model.jsonl')
                raw=audit_decision_events(events,model,local_index>0)
                transcript=(directory/'session.jsonl').read_bytes()
                require(transcript.startswith(previous),'Conversation history rewritten')
                if session_id is None:
                    session_id=receipt['session_id'];require(session_id not in identities,'Shared model context')
                    identities.add(session_id)
                require(session_id==receipt['session_id'],'Conversation changed')
                added=[json.loads(line) for line in transcript[len(previous):].decode().splitlines() if line.strip()]
                users=[e['payload'] for e in added if e.get('type')=='response_item' and e.get('payload',{}).get('role')=='user']
                matching=[u for u in users if any(c.get('type')=='input_text' and c.get('text')==prompt for c in u.get('content',[]))]
                require(len(matching)==1,'Prompt missing from actual conversation')
                embedded=[c['image_url'] for c in matching[0]['content'] if c.get('type')=='input_image']
                hashes=[hashlib.sha256(base64.b64decode(v.split(',',1)[1])).hexdigest() for v in embedded]
                require(hashes==[im['sha256'] for im in invocation['images']],'Actual conversation images mismatch')
                archive=[json.loads(line) for line in transcript.decode().splitlines() if line.strip()]
                counts=[e['payload']['info'] for e in archive if e.get('type')=='event_msg'
                        and e.get('payload',{}).get('type')=='token_count' and e['payload'].get('info') is not None]
                require(counts[:len(counts_before)]==counts_before and counts,'Token history mismatch')
                totals=counts[-1]['total_token_usage'];usage=turn_usage(totals_before,totals,raw,counts[len(counts_before):])
                require(usage==receipt['usage'],'Token delta mismatch')
                response=validate_response(read(directory/'response.json'))
                messages=[e['item']['text'] for e in events if e.get('type')=='item.completed' and e.get('item',{}).get('type')=='agent_message']
                require(messages and json.loads(messages[-1])==response,'Decision differs from model output')
                previous,counts_before,totals_before=transcript,counts,totals
                receipts.append(receipt);global_index+=1
        require(global_index==len(feedback)==result['model_requests_attempted'],'Request count mismatch')
        reports.append(dict(slot,status='verified',completed_requests=len(receipts),
                            result_sha256=digest(folder/'result.json')))
    complete=len(reports)==len(protocol['slots'])
    summary=summarize(root) if complete else None
    if complete:
        require(summary==read(root/'summary.json'),'Stored summary mismatch')
    return {'status':'passed' if complete else 'partial','episodes':reports,
            'timing_or_pairing_violations':summary['audit_violations'] if complete else None,
            'protocol_sha256':digest(root/'protocol.json'),'auditor_sha256':digest(Path(__file__))}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--partial',action='store_true')
    a=p.parse_args();report=audit(a.root,a.partial);write(a.root/('audit_partial.json' if a.partial else 'audit.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2))
