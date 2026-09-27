"""Independently aggregate paired recovery outcomes and receipt-based costs."""
import argparse
import json
from pathlib import Path
from statistics import mean
from run_online_bridge_pilot import write, verify_sources


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def summarize(root):
    protocol=verify_sources(root); records=[];state_pairs={};violations=[]
    for slot in protocol['slots']:
        d=root/slot['path'];r=json.loads((d/'result.json').read_text())
        fixture=json.loads((d/'fixture.json').read_text()) if (d/'fixture.json').exists() else {}
        setup_failed=fixture.get('status')=='prefix_failed'
        if fixture.get('status')=='ready':
            state_pairs.setdefault(slot['seed'],{})[slot['arm']]=fixture['after_sha256']
            if r['timing']['reset_count']!=1:violations.append(slot['path']+': resets')
            if r['timing']['max_simulation_lag_seconds']>protocol['max_timing_lag_for_realtime_claim_s']:
                violations.append(slot['path']+': timing lag')
        receipts=[json.loads(p.read_text()) for p in sorted(d.glob('context_*/decision_*/receipt.json'))]
        usage={key:sum(x['usage'].get(key,0) for x in receipts)
               for key in ('input_tokens','cached_input_tokens','output_tokens','total_tokens')}
        if usage!=r['usage']:raise ValueError('Receipt usage mismatch: '+str(d))
        incomplete=len(list(d.glob('context_*/decision_*/attempt_failure.json')))
        if incomplete!=r['incomplete_requests']:raise ValueError('Incomplete usage mismatch')
        feedback=rows(d/'feedback.jsonl');scheduler=rows(d/'scheduler.jsonl')
        decisions=rows(d/'model_decisions.jsonl')
        for row in feedback:
            packet=row['packet']
            if 'moved_private' in json.dumps(packet) or 'scripted_plan' in json.dumps(packet):
                raise ValueError('Private fixture data in public feedback')
            if slot['arm']=='basic' and 'geometry' in packet:raise ValueError('Geometry exposed to Base')
        records.append(dict(slot,success=r['success'],status='setup_failed' if setup_failed else r['status'],
            capped_seconds=r['capped_completion_seconds'],usage=usage,
            completed_requests=len(receipts),incomplete_requests=incomplete,
            model_seconds=sum(x['wall_seconds'] for x in receipts),
            first_request_seconds=receipts[0]['wall_seconds'] if receipts else None,
            first_decision=decisions[0]['decision'] if decisions else None,
            images=sum(len(row['images']) for row in feedback),
            retry_suppressed=sum(row['event']=='retry_suppressed' for row in scheduler),
            first_dispatch_seconds=next((row['time']-fixture['setup_sim_seconds'] for row in scheduler
                                        if row['event']=='operation_started'),None),
            dispatched=r['dispatched_operations'],action_seconds=r.get('action_seconds',0),
            camera_acquisitions=r.get('camera_acquisitions',0),timing=r.get('timing')))
    for seed,pair in state_pairs.items():
        if len(pair)==2 and len(set(pair.values()))!=1:violations.append(str(seed)+': mismatched event state')
    arms={}
    for arm in ('basic','symbolic'):
        rr=[r for r in records if r['arm']==arm]
        arms[arm]={'successes':sum(r['success'] for r in rr),'episodes':len(rr),
            'setup_failures':sum(r['status']=='setup_failed' for r in rr),
            'mean_capped_seconds':mean(r['capped_seconds'] for r in rr),
            'completed_requests':sum(r['completed_requests'] for r in rr),
            'incomplete_requests':sum(r['incomplete_requests'] for r in rr),
            'usage_complete':all(r['incomplete_requests']==0 for r in rr),
            'usage':{key:sum(r['usage'][key] for r in rr) for key in records[0]['usage']},
            'images':sum(r['images'] for r in rr),
            'model_seconds':sum(r['model_seconds'] for r in rr),
            'mean_first_request_seconds':mean(r['first_request_seconds'] for r in rr if r['first_request_seconds'] is not None)}
    return {'arms':arms,'records':records,'audit_violations':violations,
            'interpretation':'Five paired layouts; exploratory foundation recovery only. '
                             'Setup failures retained. Incomplete request usage is unknown, not zero.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();summary=summarize(a.root.resolve());write(a.root/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
