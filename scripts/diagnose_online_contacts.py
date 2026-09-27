"""Evaluator-only development contact audit; no model calls or policy truth access."""
import argparse
import json
from pathlib import Path
import numpy as np
from online_bridge_scene import OnlineBridgeTowerEnv, EpisodeSuccess
from online_bridge_skills import SensorAPI, append, plain
from online_reasoning_skills import ReasoningSkills
from online_reasoning_state import observed_center
from online_event_skills import drive


class ContactAuditEnv(OnlineBridgeTowerEnv):
    def __init__(self, *args, **kwargs):
        self.audit_active = None; self.audit_baseline = {}; self.audit_seen = set()
        self.audit_last = -1.
        super().__init__(*args, **kwargs)

    def _step(self):
        super()._step()
        if self.audit_active is None or self.data.time-self.audit_last < .01:
            return
        self.audit_last = float(self.data.time)
        for name in self.order:
            if name == self.audit_active:
                continue
            position = self._object_state(name)['position']
            displacement = float(np.linalg.norm(position-self.audit_baseline[name]))
            pairs = []
            for c in self.data.contact:
                bodies = [self.model.body(int(self.model.geom_bodyid[g])).name for g in (c.geom1,c.geom2)]
                if any(body.startswith(name+'/') for body in bodies):
                    others = [body for body in bodies if not body.startswith(name+'/')]
                    if any(body.startswith(('robot/', self.audit_active+'/')) for body in others):
                        pairs.append(bodies)
            contact = any(any(body.startswith('robot/') for body in pair) for pair in pairs)
            for kind, active in (('robot_contact',contact),('active_object_contact',bool(pairs) and not contact),('displaced_over_8mm',displacement>.008)):
                key = (name, kind)
                if active and key not in self.audit_seen:
                    self.audit_seen.add(key)
                    append(self.output_dir/'private_contacts.jsonl', {
                        'active_object':self.audit_active, 'other_object':name,
                        'event':kind, 'time':float(self.data.time),
                        'displacement_m':displacement, 'robot_contact':contact, 'contact_body_pairs':pairs})


def run(output, seed):
    output.mkdir(parents=True,exist_ok=False)
    env=ContactAuditEnv(output,scene_seed=seed)
    result={'kind':'private_development_diagnostic','seed':seed,'sequence':[]}
    try:
        env.reset(seed);env.start_episode(180,realtime=False)
        api=SensorAPI(env,output);skills=ReasoningSkills(api)
        for index,key in enumerate(skills.order):
            env.audit_active=env.order[index]
            env.audit_baseline={name:env._object_state(name)['position'].copy() for name in env.order}
            env.audit_seen=set()
            baseline=skills.evidence();last_check=skills.session.time()
            def guard(event):
                nonlocal last_check
                if skills.holding() is None or event['time']-last_check<2.:return False
                last_check=event['time']
                for parent in skills.nodes[key]['supports']:
                    fit=skills.bridge.observe(skills.session,api,parent,attempts=1)
                    current=observed_center(fit);old=baseline[parent]
                    if old is not None and current is not None and np.linalg.norm(np.array(current)-old)>.01:
                        return True
                return False
            op={'action':'execute','skill':'stack' if len(skills.nodes[key]['supports'])==1 else 'bridge','node':key}
            outcome=drive(skills,op,guard);result['sequence'].append(outcome)
            if outcome['status']!='visually_consistent':break
        result['private_final_structure']=env.structure_status(list(env.order))
        api.image('diagnostic_end')
    except EpisodeSuccess:
        result['success']=True
    finally:
        result['reset_count']=env.reset_count
        env.close();(output/'result.json').write_text(json.dumps(plain(result),indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=range(2028100101,2028100104),default=2028100102)
    a=p.parse_args();print(json.dumps(plain(run(a.output,a.seed)),indent=2))
