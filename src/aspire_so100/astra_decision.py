"""Optional Astra adapter through the installed Codex CLI; no call on import."""
import json,os,shutil,subprocess,tempfile,time
from dataclasses import replace
from pathlib import Path
from aspire_so100.dual_system import ACTIONS,actor_state

class Astra:
    def __init__(self,reasoning='low',timeout=120,model='gpt-6-astra'):
        if model not in {'gpt-6-astra','gpt-6-luna'}:raise ValueError('Unsupported model')
        if reasoning not in {'low','medium','high','xhigh'}:raise ValueError('Unsupported effort')
        self.model=model
        self.reasoning,self.timeout=reasoning,timeout;self.calls=[]
    def _ask(self,payload,properties):
        started=time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='aspire-decision-') as folder:
            work=Path(folder);home=work/'codex_home';home.mkdir(mode=0o700)
            auth=Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex')))/'auth.json'
            if auth.exists():shutil.copyfile(auth,home/'auth.json');(home/'auth.json').chmod(0o600)
            schema={'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
            (work/'schema.json').write_text(json.dumps(schema))
            cmd=['codex','exec','--ignore-user-config','--ephemeral','--sandbox','read-only','--skip-git-repo-check','--cd',folder,'--model',self.model,'-c','project_doc_max_bytes=0','-c','model_reasoning_effort='+json.dumps(self.reasoning),'--enable','skip_host_skill_discovery']
            for feature in ['shell_tool','unified_exec','apps','browser_use','computer_use','skill_search']:cmd+=['--disable',feature]
            cmd+=['--output-schema',str(work/'schema.json'),'--output-last-message',str(work/'response.json'),'--json','-']
            prompt='Decide only from the following sensor-derived state. No tools, files, external knowledge of this episode, or hidden object states. Return only the requested JSON.\n'+json.dumps(payload)
            proc=subprocess.run(cmd,input=prompt,text=True,capture_output=True,timeout=self.timeout,env=dict({k:v for k,v in os.environ.items() if k not in {'TYPESAFE_API_KEY'}},CODEX_HOME=str(home)),check=True)
            answer=json.loads((work/'response.json').read_text())
            usage=None
            for line in proc.stdout.splitlines():
                try:event=json.loads(line)
                except ValueError:continue
                if event.get('type')=='turn.completed':usage=event.get('usage')
        self.calls.append({'requested_model':self.model,'reasoning':self.reasoning,'wall_seconds':time.perf_counter()-started,'usage':usage,'request':payload,'response':answer,'estimated_usd':None,'cost_basis':'Codex subscription; no API invoice available'})
        return answer
    def ask(self,payload,properties):
        started=time.perf_counter()
        try:
            return self._ask(payload,properties)
        except Exception as exc:
            self.calls.append({'requested_model':self.model,'reasoning':self.reasoning,
                'wall_seconds':time.perf_counter()-started,'usage':None,'request':payload,
                'response':None,'estimated_usd':None,'status':'failed',
                'error_type':type(exc).__name__,'cost_basis':'Unknown; failed calls may be billed'})
            raise
    def decide(self,state,plan):
        answer=self.ask({'role':'decision','state':actor_state(state),'plan':plan.description,'actions':ACTIONS}, {'action':{'type':'string','enum':list(ACTIONS)},'confidence':{'type':'number','minimum':0,'maximum':1}})
        return dict(answer,state_version=state['version'],plan_version=plan.version,model=self.model,**{k:v for k,v in self.calls[-1].items() if k not in {'requested_model'}})
    def replan(self,state,plan,trigger):
        answer=self.ask({'role':'plan_or_repair','state':actor_state(state),'current_plan':plan.description,'trigger':trigger,'allowed_actions':ACTIONS,'constraint':'Write a conditional, phase-specific rule, not a statement that the current snapshot stays true. CURRENT observation always overrides old snapshot facts. At initial_observation, all colors visible plus valid table permits continue even if grasp_confirmed is null. Do not change budgets or execution code. A repaired plan requires a fresh observation before motion.'}, {'plan':{'type':'string'}})
        return replace(plan,version=plan.version+1,description=answer['plan'])
