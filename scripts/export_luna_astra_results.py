"""Export measured attempts, including failures, into a shareable static report."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[1]
WEB=ROOT/'web/recovery/luna-astra'

def read(path,default):
    return json.loads(path.read_text()) if path.exists() else default

def reference_actions(state):
    if state['table_fit_valid'] and set(state['visible_colors'])=={'red','green','blue'}:return {'continue'}
    return {'reobserve'} if state['reobservations_left']>0 else {'stop','escalate'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    manifest=read(a.root/'campaign.json',{});runs=[]
    (WEB/'evidence').mkdir(parents=True,exist_ok=True);(WEB/'media').mkdir(exist_ok=True)
    for job in manifest['jobs']:
        folder=a.root/job['id'];terminal=read(a.root/(job['id']+'.terminal.json'),None)
        if terminal is None:continue
        result=read(folder/'result.json',{});trace=read(folder/'decision_trace.json',{})
        events=trace.get('records',[]);calls=trace.get('planner_calls',[])+trace.get('fast_calls',[])
        actions=read(folder/'api_calls.json',[])
        motion=sum(x['method'] in {'move_tool','move_joints','gripper'} for x in actions)
        model_errors=sum(c.get('status')=='failed' for c in calls)
        abstained=bool(events and events[-1]['gated_action']=='stop' and motion==0 and not result.get('success') and model_errors==0)
        passed=terminal['status']=='completed' and (abstained if job['endpoint']=='correct_abstention' else result.get('success',False))
        safe_id=job['id'].replace('=','-')
        item=dict(job,status=terminal['status'],endpoint_pass=bool(passed),tower_complete=bool(result.get('success',False)),
            correct_abstention=abstained if job['endpoint']=='correct_abstention' else None,
            motion_calls=motion,trial_wall_seconds=terminal['wall_seconds'],sim_seconds=result.get('policy_sim_seconds'),
            model_calls=len(calls),model_errors=model_errors,model_wait_seconds=sum(c['wall_seconds'] for c in calls),
            fast_wait_seconds=sum(c['wall_seconds'] for c in trace.get('fast_calls',[])),
            slow_wait_seconds=sum(c['wall_seconds'] for c in trace.get('planner_calls',[])),
            repair_calls=max(0,len(trace.get('planner_calls',[]))-1),
            input_tokens=sum((c.get('usage') or {}).get('input_tokens',0) for c in calls),
            output_tokens=sum((c.get('usage') or {}).get('output_tokens',0) for c in calls),
            cached_input_tokens=sum((c.get('usage') or {}).get('cached_input_tokens',0) for c in calls),
            usage_complete=bool(calls) and all(c.get('usage') is not None for c in calls),
            decisions=len(events),rule_agreement=sum(e.get('proposed') in reference_actions(e['state']) for e in events),
            guard_interventions=sum(e.get('trigger') is not None for e in events),
            candidate_sha256=result.get('candidate_sha256'),estimated_usd=None,
            evidence='evidence/'+safe_id+'.json')
        # No auth, environment, raw CLI stderr, or temporary files are public.
        evidence={'configuration':job,'terminal':terminal,'result':result,'decision_trace':trace,'harness':read(folder/'harness.json',{})}
        (WEB/item['evidence']).write_text(json.dumps(evidence,indent=2)+'\n')
        for source,key,extension in [('demo.mp4','video','.mp4'),('outcome.png','poster','.png')]:
            if (folder/source).exists():
                item[key]='media/'+safe_id+extension
                shutil.copy2(folder/source,WEB/item[key])
                item[key+'_sha256']=hashlib.sha256((folder/source).read_bytes()).hexdigest()
        runs.append(item)
    groups=[]
    for config in manifest['groups']:
        rr=[r for r in runs if r['group']==config['group']]
        tower=[r for r in rr if r['endpoint']=='tower_completion'];stop=[r for r in rr if r['endpoint']=='correct_abstention']
        groups.append(dict(config,completed=len(rr),scheduled=3,passed=sum(r['endpoint_pass'] for r in rr),
            tower_passed=sum(r['endpoint_pass'] for r in tower),tower_n=len(tower),
            abstention_passed=sum(r['endpoint_pass'] for r in stop),abstention_n=len(stop),
            median_model_wait_seconds=statistics.median(r['model_wait_seconds'] for r in rr) if rr else None,
            total_model_wait_seconds=sum(r['model_wait_seconds'] for r in rr),input_tokens=sum(r['input_tokens'] for r in rr),
            output_tokens=sum(r['output_tokens'] for r in rr),model_calls=sum(r['model_calls'] for r in rr),
            model_errors=sum(r['model_errors'] for r in rr),repair_calls=sum(r['repair_calls'] for r in rr),
            decisions=sum(r['decisions'] for r in rr),rule_agreement=sum(r['rule_agreement'] for r in rr)))
    report={'schema':'luna-astra-results/1','scope':manifest['scope'],'split':'development','scheduled':len(manifest['jobs']),
        'completed':len(runs),'complete':len(runs)==len(manifest['jobs']),'concurrent_trials':2,
        'groups':groups,'runs':runs,'limitations':manifest['limitations'],
        'accuracy_note':'Endpoint passes over three heterogeneous pilot cases. Rule agreement is not independent decision accuracy.',
        'estimated_usd':None,'no_jev_calls':True,'manifest_sha256':hashlib.sha256((a.root/'campaign.json').read_bytes()).hexdigest()}
    (WEB/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'docs/evaluations/luna_astra_results.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copy2(a.root/'campaign.json',WEB/'campaign.json')
    guide=(ROOT/'docs/LUNA_ASTRA_README.md').read_text()
    guide=guide.replace('(evaluations/luna_astra_results.json)','(results.json)').replace('(../benchmarks/luna_astra_campaign.json)','(campaign.json)')
    for doc in ['TASK_CATALOG','POLICY_BENCHMARK']:
        guide=guide.replace(f'({doc}.md)',f'(../docs/{doc}.md)')
    guide=guide.replace('(evaluations/luna_astra_audit.json)','(audit.json)')
    (WEB/'README.md').write_text(guide)
    audit=ROOT/'docs/evaluations/luna_astra_audit.json'
    if audit.exists():shutil.copy2(audit,WEB/'audit.json')
    print(f'Exported {len(runs)}/{len(manifest["jobs"])} actual attempts')

if __name__=='__main__':main()
