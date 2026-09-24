"""Host service for a bounded initial-observation model gate."""
import json,math,time,hashlib
from pathlib import Path
from aspire_so100.dual_system import DualController,Plan,JevClient,actor_state
from aspire_so100.astra_decision import Astra

class DecisionSession:
    def __init__(self,mode,output,confidence=.8,max_decisions=4):
        self.mode=mode;self.output=Path(output);self.max_decisions=max_decisions
        self.planner=Astra(reasoning='high');self.decider=Astra(reasoning='low')
        self.plan=Plan(confidence_threshold=confidence);self.controller=None;self.records=[]
        self.started=time.perf_counter()
    def decide(self,state,simulation_time):
        state=actor_state(state)
        if state['phase']!='initial_observation':raise ValueError('Only initial-observation integration is implemented')
        if not math.isfinite(simulation_time) or abs(state['time_s']-simulation_time)>.01:raise ValueError('Stale sensor time')
        if len(self.records)>=self.max_decisions:raise ValueError('High-level decision budget exhausted')
        if self.controller is None:
            if self.mode!='jev_only':self.plan=self.planner.replan(state,self.plan,'initial_plan')
            self.controller=DualController(self.decider.decide if self.mode=='astra_only' else JevClient(),
                self.planner.replan if self.mode!='jev_only' else None,self.plan,mode=self.mode)
        before=time.perf_counter();event=self.controller.tick(state)
        event['state']=state;event['service_seconds']=time.perf_counter()-before
        self.records.append(event);self.write()
        return {'action':event['gated_action'],'state_version':state['version'],
                'plan_version':self.controller.plan.version}
    def write(self):
        data={'mode':self.mode,'scope':'initial_observation_gate_then_frozen_visual_skill',
              'simulation_paused_during_model_calls':True,'records':self.records,
              'planner_calls':self.planner.calls,'astra_decision_calls':self.decider.calls,
              'threshold':self.plan.confidence_threshold,'threshold_calibrated':False}
        (self.output/'decision_trace.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
