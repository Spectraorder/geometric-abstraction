"""Sensor-only decision contracts and official TypeSafe Jev client."""
import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass

ACTIONS = {
    'continue': 'At initial_observation: table_fit_valid is true and all three colors are visible NOW. At verify_grasp: grasp_confirmed is true. Start or resume the frozen visual skill.',
    'reobserve': 'Acquire fresh RGB-D when evidence is missing and observation budget remains.',
    'retry_grasp': 'Only at verify_grasp, when grasp_confirmed is explicitly false and retries remain. Never choose this during initial_observation.',
    'escalate': 'Request a revised plan when evidence conflicts, budget is exhausted, or no action fits.',
    'stop': 'Stop this task when recovery is unavailable.'
}


def actor_state(raw):
    """Strict whitelist, rejecting extra fields rather than forwarding arbitrary dictionaries."""
    fields = {'version', 'time_s', 'phase', 'visible_colors', 'table_fit_valid',
              'grasp_confirmed', 'reobservations_left', 'grasp_retries_left'}
    if set(raw) != fields:
        raise ValueError('Actor state schema mismatch')
    s = dict(raw)
    if type(s['version']) is not int or s['version'] < 0:
        raise ValueError('Invalid state version')
    if isinstance(s['time_s'],bool) or not isinstance(s['time_s'],(int,float)) or not math.isfinite(s['time_s']):
        raise ValueError('Invalid timestamp')
    if s['phase'] not in {'initial_observation', 'verify_grasp'}:
        raise ValueError('Unimplemented phase')
    if type(s['table_fit_valid']) is not bool or (s['grasp_confirmed'] is not None and type(s['grasp_confirmed']) is not bool):
        raise ValueError('Invalid perception flag')
    if not isinstance(s['visible_colors'], list) or any(type(x) is not str or x not in {'red','green','blue'} for x in s['visible_colors']):
        raise ValueError('Invalid observed colors')
    for k in ('reobservations_left', 'grasp_retries_left'):
        if type(s[k]) is not int or not 0 <= s[k] <= 2:
            raise ValueError('Invalid budget')
    return s


def permitted(s):
    """Mechanical guards; models cannot make missing evidence true."""
    s = actor_state(s)
    allowed = {'escalate','stop'}
    if s['reobservations_left'] > 0: allowed.add('reobserve')
    visible = s['table_fit_valid'] and set(s['visible_colors']) == {'red','green','blue'}
    if s['phase'] == 'initial_observation' and visible: allowed.add('continue')
    if s['phase'] == 'verify_grasp':
        if s['grasp_confirmed'] is True: allowed.add('continue')
        if s['grasp_confirmed'] is False and visible and s['grasp_retries_left'] > 0: allowed.add('retry_grasp')
    return allowed


@dataclass(frozen=True)
class Plan:
    version: int = 0
    max_slow_calls: int = 2
    confidence_threshold: float = .8  # Development placeholder, NOT calibrated.
    description: str = 'Build blue-red-green tower. Verify visual evidence before existing skills; reobserve uncertainty, retry failed grasps, escalate conflicts.'


class JevClient:
    endpoint = 'https://api.typesafe.ai/v1/systemone'
    def __init__(self, model='jev-1.13.0', timeout=20):
        self.model, self.timeout = model, timeout

    def __call__(self, state, plan):
        key = os.environ.get('TYPESAFE_API_KEY')
        if not key: raise RuntimeError('TYPESAFE_API_KEY is not configured; no model call made')
        payload = {'model':self.model, 'state':{'observation':actor_state(state), 'plan':plan.description},
                   'questions':{'next_action':{'type':'choice','instructions':'Choose exactly one next action from the CURRENT observation. Current state overrides stale factual statements in the plan. The plan is conditional guidance, not additional perception. At initial_observation, grasp_confirmed=null is normal and must not block continue when the table and all three colors are visible. Never exceed the remaining budgets. Do not infer hidden object state.', 'criteria':ACTIONS}}}
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}, method='POST')
        started=time.perf_counter()
        # No automatic retries: timed-out requests may still be billed; benchmark counts every attempt.
        with urllib.request.urlopen(request,timeout=self.timeout) as response:
            result=json.load(response)
        elapsed=time.perf_counter()-started
        answer=result['answers']['next_action']
        probs=answer['probabilities']
        if answer['type']!='choice' or set(probs)!=set(ACTIONS): raise ValueError('Unexpected Jev output schema')
        if any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in probs.values()) or abs(sum(probs.values())-1)>.001:
            raise ValueError('Invalid Jev distribution')
        if answer['choice'] not in probs or probs[answer['choice']] < max(probs.values())-1e-6: raise ValueError('Choice is not a maximum')
        usage=result.get('usage',{})
        tokens=usage.get('input_tokens')
        estimate=tokens*.042/1_000_000 if type(tokens) is int and tokens>=0 else None
        return {'action':answer['choice'],'confidence':answer['confidence'], 'state_version':state['version'],
                'plan_version':plan.version,'model':result['model'],'wall_seconds':elapsed,
                'usage':usage,'estimated_usd':estimate,'cost_basis':'published input rate 2026-09-24; not invoice',
                'probabilities':probs}


class DualController:
    """State-machine interface: decider(state, plan), replanner(state, plan, trigger).

    tick never executes a skill. Every returned action requires a separate fresh-state
    executor check; a revised plan always returns reobserve/stop, never immediate motion.
    """
    def __init__(self, decider, replanner=None, plan=None, mode='hybrid'):
        if mode not in {'hybrid','astra_only','jev_only','luna_hybrid','luna_only'}: raise ValueError('Unknown mode')
        self.decider,self.replanner,self.plan,self.mode=decider,replanner,plan or Plan(),mode
        self.slow_calls=0;self.last_version=-1;self.trace=[]

    def tick(self, raw):
        s=actor_state(raw)
        if s['version']<=self.last_version: raise ValueError('Fresh observation version required')
        self.last_version=s['version']; started=time.perf_counter();reply=None;trigger=None
        try:
            reply=self.decider(s,self.plan)
            if reply['state_version']!=s['version'] or reply['plan_version']!=self.plan.version: raise ValueError('Stale reply')
            c=reply['confidence']
            if type(c) not in (int,float) or not math.isfinite(c) or not 0<=c<=1: raise ValueError('Invalid confidence')
            if reply['action'] not in ACTIONS: raise ValueError('Unknown action')
            if reply['action'] not in permitted(s): trigger='precondition_violation'
            elif c<self.plan.confidence_threshold: trigger='low_confidence'
            elif reply['action']=='escalate': trigger='requested_escalation'
        except Exception as e:
            # Never log arbitrary exception strings that could contain provider credentials.
            trigger='provider_error:'+type(e).__name__
        decision_wall_seconds=time.perf_counter()-started
        action=reply['action'] if trigger is None else ('reobserve' if trigger=='low_confidence' and 'reobserve' in permitted(s) else 'stop')
        old_version=self.plan.version
        old_description=self.plan.description
        if trigger and self.mode not in {'jev_only','luna_only'} and self.replanner and self.slow_calls<self.plan.max_slow_calls:
            self.slow_calls+=1
            try:
                update=self.replanner(s,self.plan,trigger)
                if not isinstance(update,Plan) or update.version!=self.plan.version+1: raise ValueError('Bad plan version')
                if update.max_slow_calls!=self.plan.max_slow_calls or update.confidence_threshold!=self.plan.confidence_threshold:
                    raise ValueError('A model cannot expand evaluation budgets or lower gates')
                if not isinstance(update.description,str) or not 0<len(update.description)<=8000: raise ValueError('Invalid plan text')
                self.plan=update
                action='reobserve' if 'reobserve' in permitted(s) else 'stop'
            except Exception as e:trigger+=';replan_error:'+type(e).__name__
        event={'state_version':s['version'],'plan_before':old_version,'plan_after':self.plan.version,
               'proposed':None if reply is None else reply.get('action'),'executed':False,
               'gated_action':action,'trigger':trigger,'slow_calls_total':self.slow_calls,
               'wall_seconds':time.perf_counter()-started,'decision_wall_seconds':decision_wall_seconds,'decision':reply,'plan_text_before':old_description,'plan_text_after':self.plan.description}
        self.trace.append(event)
        return event
