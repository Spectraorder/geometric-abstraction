"""Recorded-observation interface exercise, NOT a counterfactual robot rollout."""
import argparse,json
from pathlib import Path
from dataclasses import replace
from runtime import DualController,Plan,JevClient
from astra import Astra
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['hybrid','astra_only','jev_only'],default='hybrid');p.add_argument('--live',action='store_true');p.add_argument('--limit',type=int,default=3);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():p.error('Output must be new')
if not 1<=a.limit<=20:p.error('limit must be 1..20')
if a.live:
 import os
 if a.mode!='astra_only' and not os.environ.get('TYPESAFE_API_KEY'):p.error('Configure TYPESAFE_API_KEY before live calls')
cases=json.loads(Path(__file__).with_name('cases.json').read_text())['cases'][:a.limit]
a.output.mkdir(parents=True);records=[]
for case in cases:
 # Independent decision probes: never treat later recorded frames as model-controlled futures.
 planner=Astra(reasoning='high');decider=Astra(reasoning='low');plan=Plan()
 if a.live:
  decide=decider.decide if a.mode=='astra_only' else JevClient()
  repair=planner.replan if a.mode!='jev_only' else None
  if repair:plan=repair(case['state'],plan,'initial_plan')
 else:
  def decide(s,p):return {'action':'escalate','confidence':.4,'state_version':s['version'],'plan_version':p.version,'model':'TEST_DOUBLE_NOT_JEV'}
  def repair(s,p,t):return replace(p,version=p.version+1)
 controller=DualController(decide,repair,plan,mode=a.mode)
 event=controller.tick(case['state'])
 records.append({'case':case['id'],'event':event,'planner_calls':planner.calls,'decision_calls':decider.calls,'reference_action':case['reference_action'],'reference_type':case['reference_type']})
report={'mode':a.mode,'live':a.live,'kind':'independent_recorded_observation_probes','robot_execution':False,'task_success_rate':None,'accuracy':None,'records':records}
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('Saved',len(records),'probes; robot success and accuracy are not measured')
