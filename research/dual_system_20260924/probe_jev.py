"""One official Jev decision; dry-run exports the exact actor state without a call."""
import argparse,json
from pathlib import Path
from runtime import JevClient,Plan,actor_state
p=argparse.ArgumentParser();p.add_argument('--live',action='store_true');p.add_argument('--case',default='dev-000');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():p.error('Output must be new')
cases=json.loads(Path(__file__).with_name('cases.json').read_text())['cases'];case=next(c for c in cases if c['id']==a.case)
state=actor_state(case['state']);record={'case':a.case,'state':state,'live':a.live,'robot_execution':False}
if a.live:
 try:record['decision']=JevClient()(state,Plan());record['status']='model_response_received'
 except Exception as e:record.update(status='not_evaluated',error_type=type(e).__name__)
else:record['status']='dry_run_no_model_call'
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record,indent=2)+'\n');print(record['status'])
