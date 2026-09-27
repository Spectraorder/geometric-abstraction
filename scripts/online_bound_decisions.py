"""Development decision loop for common depth queries and explicit target skills."""
import json
import math
import time
from online_bound_actor import BoundActor, validate_response
from online_bound_target_skills import BoundTargetSkills, drive_bound
from online_bridge_skills import append, plain
from online_event_memory import PublicMemory
from online_reasoning_state import observed_center

CONTRACT = '''Build the public support graph in this continuing scene using shared bridge and stack skills.
The actor binds a world XYZ support-surface point in meters for each execute action.
This is the object's desired center XY and bottom Z, not its desired center XYZ.
The controller executes that intent; it does not select or correct the task destination.
Use bridge for foundations or a beam, and stack for a node with exactly one support.
Execute may also relocate an object to a caller-chosen free table point for recovery.
There is one placement per decision, no queued node-only plan and no reset after failure.
Both interfaces may sample_depth on a displayed frame_id with up to 16 integer [u,v]
pixels (origin top-left). Queries return metric optical-axis depth and backprojected
visible surface points. They do not identify object centers or infer hidden geometry.
A point retains its observation timestamp; new physics does not update old samples.
An observe action provides fresh side and wrist images with calibration and frame IDs.
Base normally receives a side image after an action. Full receives current typed
object references; initial images and requested images remain available to both.
Full reference source, observability and time qualify every geometric estimate.
Unknown is not false. A submitted point is an intent, not an observation.
Keep a confirmed held object when continuing its placement; release opens at the
current tool location. Failed placement may leave the object held. A changed support
can interrupt motion without resetting or silently recomputing the target.
A visually_consistent outcome is relative to the submitted point, not certified
completion of the task. The same private task monitor determines episode success.
Return all required JSON fields; inactive arrays are [] and inactive strings empty.
Only sample_depth uses frame_id/pixels; only execute uses skill/node/surface_point_world_m.
Keep reason to one sentence. No external tools or generated code are available.
'''


class ExplicitSpatialSkills(BoundTargetSkills):
    def initialize_workspace(self):
        # Explicit targets make automatic worksite selection unnecessary. The
        # observed unit is used only for shared local placement tolerances.
        key=self.task['length_unit']['observed_cube']
        fit=self.bridge.observe(self.session,self.api,self.nodes[key]['color'],attempts=2)
        if fit is None:
            raise ValueError('Current reference dimensions unavailable')
        self.scale=self.bridge.reference_side(self.nodes[key],fit)
        if self.scale is None:
            raise ValueError('Current reference dimensions unavailable')

    def run_steps(self,skill,key):
        if getattr(self,'_bound_target',None) is None:
            raise ValueError('This development loop requires an explicit placement target')
        return (yield from super().run_steps(skill,key))


def reference_feedback(skills):
    """Expose supported current references, without automatic task goal values."""
    session=skills.session
    session.snapshot()
    records=[]
    for key,node in skills.nodes.items():
        row={'object':key,'references':{}}
        for role in ('center','top_face','bottom_face'):
            try:
                ref=plain(session.reference(node['color']+'.'+role))
                if ref.get('position_fully_observed') is not True:
                    row['references'][role]={'status':'unavailable','reason':'Position is not fully observed'}
                else:
                    # Keep evidence and reference meaning; omit dense boundary vertices.
                    row['references'][role]={k:v for k,v in ref.items() if k!='vertices_m'}
            except (ValueError,RuntimeError,KeyError):
                row['references'][role]={'status':'unavailable'}
        records.append(row)
    return {'time':session.time(),'objects':records,'semantics':'Sensor-grounded references, not task destinations'}


class BoundDecisionLoop:
    """Own model decisions inside an already-started, never-reset episode."""
    def __init__(self,env,api,skills,arm,*,actor_factory=BoundActor):
        if arm not in ('basic','symbolic'):
            raise ValueError('Unknown interface')
        self.env,self.api,self.skills,self.arm=env,api,skills,arm
        self.actor_factory=actor_factory
        self.actor=None
        self.requests=0
        self.memory=PublicMemory()
        self.last_result=None
        self.previous_action=None
        self.feedback_seconds=0.
        self.completed_operations=0
        self.dispatched_operations=0

    def _support_positions(self,node):
        positions={}
        for key in self.skills.nodes[node]['supports']:
            fit=self.skills.bridge.observe(self.skills.session,self.api,self.skills.nodes[key]['color'],attempts=1)
            positions[key]=observed_center(fit)
        return positions

    def step(self):
        self.env.advance_while_waiting()
        initial=self.requests==0
        start=time.perf_counter()
        packet={'time':self.skills.session.time(),'holding':self.skills.holding(),
                'last_result':self.last_result,'memory':self.memory.packet()}
        images=[]
        refresh=initial or self.previous_action=='observe'
        # A query response refers to the already displayed frame. Do not replace
        # that image merely to deliver its samples to the next model decision.
        if refresh or (self.arm=='basic' and self.previous_action!='sample_depth'):
            for view in (('side','wrist') if refresh else ('side',)):
                path,calibration=self.api.image(f'feedback_{self.requests:03d}_{view}',view)
                images.append(path)
                packet.setdefault('cameras',{})[view]=calibration
        if self.arm=='symbolic':
            packet['geometry']=reference_feedback(self.skills)
        packet['remaining_wall_seconds']=max(0.,self.env.deadline-time.monotonic())
        self.feedback_seconds+=time.perf_counter()-start
        if self.actor is None:
            context=self.api.output/'context_000'
            context.mkdir()
            self.actor=self.actor_factory(context)
        prefix=CONTRACT+'\nGoal: '+json.dumps(self.api.task())+'\n' if initial else ''
        prompt=prefix+'Interface: '+self.arm+'\nCurrent feedback:\n'+json.dumps(plain(packet),separators=(',',':'))
        append(self.api.output/'feedback.jsonl',{'turn':self.requests,'packet':packet,'images':[str(p) for p in images]})
        decision=self.actor.request(prompt,images,slow=True,advance=self.env.advance_while_waiting,deadline=self.env.deadline)
        self.requests+=1
        validate_response(decision)
        append(self.api.output/'model_decisions.jsonl',{'turn':self.requests-1,'decision':decision})
        action=decision['action']
        if action=='sample_depth':
            outcome=self.api.sample_depth(decision['frame_id'],decision['pixels'])
        elif action=='execute':
            key=decision['node']
            if key not in self.skills.nodes or (len(self.skills.nodes[key]['supports'])==1)!=(decision['skill']=='stack'):
                outcome={'status':'invalid_operation'}
            else:
                baseline=self._support_positions(key)
                checked=self.skills.session.time()
                def boundary(event):
                    nonlocal checked
                    self.env.advance_while_waiting()
                    if self.skills.holding() is None or event['time']-checked<2.:
                        return False
                    checked=event['time']
                    current=self._support_positions(key)
                    return any(old is not None and current[k] is not None
                               and math.dist(old,current[k])>.01 for k,old in baseline.items())
                self.dispatched_operations+=1
                append(self.api.output/'scheduler.jsonl',{'event':'operation_started','decision':decision,
                                                         'time':self.skills.session.time()})
                outcome=drive_bound(self.skills,decision,boundary)
                self.completed_operations+=1
        elif action=='observe':
            outcome={'status':'observation_requested'}
        elif action=='release':
            outcome=self.skills.release()
        else:
            self.api.wait(5.)
            outcome={'status':'settled'}
        self.memory.record(decision,outcome)
        # Depth samples are a common sensor result. Other numeric skill internals
        # are not public feedback, including in the Base interface.
        self.last_result=(plain(outcome) if action=='sample_depth' else
            {k:plain(outcome[k]) for k in ('status','holding','surface_point_world_m','task_success_certified') if k in outcome})
        self.previous_action=action
        append(self.api.output/'decision_results.jsonl',{'turn':self.requests-1,'action':action,'result':self.last_result})
        return self.last_result

    def run(self):
        while True:
            self.step()
