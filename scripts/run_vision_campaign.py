"""Continue a sensor-only fresh-model campaign; retain video for every rollout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone


def run(command, log):
    with log.open('w') as output:
        subprocess.run([sys.executable,*command],stdout=output,stderr=subprocess.STDOUT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--start-round',type=int,default=1);p.add_argument('--rounds',type=int,default=0);p.add_argument('--model',default='gpt-6-astra');a=p.parse_args()
    if a.rounds<0 or a.start_round<1:p.error('rounds must be >=0 and start-round must be >=1')
    root=a.root;state=json.loads((root/'campaign.json').read_text());state.setdefault('history',[])
    index=a.start_round
    while a.rounds==0 or index<a.start_round+a.rounds:
        r=root/f'round_{index:03d}'
        state.update(status='proposing',active_round=index);(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
        if not (r/'candidate.py').exists():
            if r.exists():raise RuntimeError('Existing unfinished proposal; inspect rather than overwrite')
            previous=root/f'round_{index-1:03d}'
            initial=root/'initial.png'
            if not initial.exists():initial=root/'infrastructure/smoke_03/initial.png'
            command=['scripts/propose_vision_policy.py','--round-dir',str(r),'--image',str(initial),'--model',a.model]
            if index>1:
                command+=['--previous',str(previous)]
                for path in [previous/'trial_81/outcome.png',*sorted((previous/'trial_81').glob('camera_*.png'))[-2:]]:
                    if path.exists():command+=['--image',str(path)]
            run(command,root/f'proposal_{index:03d}.log')
        # Audit that no model-side tools were called in its fresh context.
        events=[json.loads(x) for x in (r/'model.jsonl').read_text().splitlines() if x.startswith('{')]
        forbidden=[e for e in events if e.get('item',{}).get('type') in {'command_execution','mcp_tool_call','web_search','file_change'}]
        if forbidden:raise RuntimeError('Actor used tools; isolation audit failed')
        trials=[]
        for seed in [81,82,83]:
            trial=r/f'trial_{seed}'
            state.update(status='executing',active_seed=seed);(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
            print(f'Round {index}: executing seed {seed}',flush=True)
            if not (trial/'result.json').exists():
                run(['scripts/run_vision_trial.py','--candidate',str(r/'candidate.py'),'--output',str(trial),'--seed',str(seed)],r/f'trial_{seed}.log')
            result=json.loads((trial/'result.json').read_text());trials.append(result)
            state.update(status='recording');(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
            if not (trial/'video.json').exists():
                run(['scripts/render_vision_trial.py','--trial',str(trial)],r/f'video_{seed}.log')
            print(f'Round {index}, seed {seed}: success={result["success"]}; error={result["policy_error"]}; video saved',flush=True)
            if not result['success']:break
        entry={'round':index,'candidate':str(r/'candidate.py'),'sha256':hashlib.sha256((r/'candidate.py').read_bytes()).hexdigest(),'hypothesis':json.loads((r/'response.json').read_text())['hypothesis'],'trials':trials,'successes':sum(t['success'] for t in trials),'model_tool_calls':len(forbidden)}
        state['history']=[e for e in state['history'] if e['round']!=index]+[entry];(r/'summary.json').write_text(json.dumps(entry,indent=2)+'\n')
        if len(trials)>1:
            feedback={'trials':[json.loads((r/f'trial_{s["seed"]}/actor_feedback.json').read_text())|{'seed':s['seed']} for s in trials]}
            (r/'trial_81/actor_feedback.json').write_text(json.dumps(feedback,indent=2)+'\n')
        if len(trials)==3 and all(t['success'] for t in trials):
            shutil.copyfile(r/'candidate.py',root/'best_policy.py')
            state.update(status='debug_passed',best_round=index,frozen_at=datetime.now(timezone.utc).isoformat(),best_sha256=entry['sha256'])
            (root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
            print('3/3 debug passed. Frozen policy ready for held-out validation.',flush=True)
            return
        index+=1
        (root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
    state['status']='round_limit';(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')

if __name__=='__main__':main()
