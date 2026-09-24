"""Export only model/physics result evidence, never credentials or environment files."""
import argparse,json,hashlib,shutil,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'outputs/dual_system_live');a=p.parse_args()
web=ROOT/'web/recovery';runs=[]
for trial in sorted(a.root.iterdir()):
 if not (trial/'result.json').exists():continue
 result=json.loads((trial/'result.json').read_text());trace=json.loads((trial/'decision_trace.json').read_text())
 events=trace['records'];jev=[e['decision'] for e in events if e.get('decision') and e['decision'].get('model','').startswith('jev')]
 astra=trace['planner_calls']+trace['astra_decision_calls']
 # Include failed decision waits; older successful traces use provider timing.
 times=[x.get('decision_wall_seconds',(x.get('decision') or {}).get('wall_seconds',0)) for x in events]+[x['wall_seconds'] for x in trace['planner_calls']]
 reported=[x.get('usage') for x in jev+astra]
 v2=trial.name.startswith('v2_');group=trace['mode']
 finding='Current-state conditional plan; initial gate only.' if v2 else 'Original protocol: retained unsuccessful integration attempt.'
 item={'id':trial.name,'label':('Protocol 2' if v2 else 'Original protocol')+' / '+group.replace('_',' '),'protocol':2 if v2 else 1,'group':group,'seed':result['seed'],'success':result['success'],'sim_seconds':result['policy_sim_seconds'],'model_calls':len(events)+len(trace['planner_calls']),'successful_reported_calls':len(jev)+sum(x.get('status')!='failed' for x in astra),'model_wait_seconds':sum(times),'input_tokens':sum(u.get('input_tokens',0) for u in reported if u),'output_tokens':sum(u.get('output_tokens',0) for u in reported if u),'jev_estimated_usd':sum(x['estimated_usd'] for x in jev if x['estimated_usd'] is not None),'astra_usd':None,'usage_complete':len(jev)+len(astra)==len(events)+len(trace['planner_calls']) and all(u is not None for u in reported),'finding':finding,'candidate_sha256':result['candidate_sha256'],'evidence':'evidence/'+trial.name+'_decision.json'}
 evidence={'result':result,'decision_trace':trace}
 if (trial/'harness.json').exists(): evidence['harness']=json.loads((trial/'harness.json').read_text())
 (web/item['evidence']).write_text(json.dumps(evidence,indent=2)+'\n')
 if (trial/'demo.mp4').exists():
  item['video']='media/'+trial.name+'.mp4';shutil.copy2(trial/'demo.mp4',web/item['video']);item['video_sha256']=hashlib.sha256((trial/'demo.mp4').read_bytes()).hexdigest()
 runs.append(item)
report={'schema':'dual-system-integration/1','scope':'One known development seed, initial-observation gate only. Original and corrected protocols are separate; these are not heldout benchmark scores. Costs for Astra remain unknown.','runs':runs,'task_accuracy':None,'cost_advantage':None}
(ROOT/'docs/evaluations/dual_system_live.json').write_text(json.dumps(report,indent=2)+'\n');(web/'live_results.json').write_text(json.dumps(report,indent=2)+'\n')
for name in ['TASK_CATALOG','POLICY_BENCHMARK','DUAL_SYSTEM','PARAMETER_NAMING']:
 source=(ROOT/f'docs/{name}.md').read_text()
 # Repository evidence links must remain usable from the standalone static site.
 source=re.sub(r'\]\(\.\./([^)]*)\)',r'](https://github.com/Spectraorder/aspire_so100/blob/codex/dual-system-benchmark-english/\1)',source)
 (web/f'docs/{name}.md').write_text(source)
# A local copy makes the relative evidence link in the served harness document work.
(web/'docs/evaluations').mkdir(exist_ok=True);shutil.copy2(ROOT/'docs/evaluations/dual_system_live.json',web/'docs/evaluations/dual_system_live.json')
print('Exported',len(runs),'actual model/physics attempts')
