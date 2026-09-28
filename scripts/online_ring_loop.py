"""Compact current geometry or visual feedback between explicit Ring actions."""
import json
import time
import numpy as np
from online_ring_actor import RingActor, validate_response
from online_metric_depth import MetricDepthFrames
from online_bridge_skills import append
from ring_peg_development import DevelopmentSensors
from ring_peg_runtime import plain

CONTRACT = '''Place the ring over the fixed upright peg and release it in this continuing scene.
You have 200 seconds including all sensing, model requests, actions and settling. No reset is allowed.
Both arms have the same learned grasp_lift skill, robot encoders, calibrated images and metric pixel queries.
The shared grasp skill uses local RGB-D to choose a wall pinch and lift. It does NOT transport to the peg,
align the opening or release. Its internal object geometry is not returned to Base.
Actions:
- grasp_lift: run the common grasp/lift prefix. It may return incomplete or unavailable. Never assume success.
- move_pad: submit point_m=[x,y,z] in world meters for the FIXED finger pad, roll in radians and seconds.
  The actuator executes exactly this target with robot-only IK. It does not replace your target.
- release: open to jaw radians, retreat the fixed pad vertically 0.06 m and wait briefly. It does not align first.
- observe: request fresh side and wrist images with calibration and frame IDs.
- sample_depth: query up to 16 [u,v] integer pixels from a displayed frame_id, top-left origin.
  Samples are visible surfaces at the archived image time, not inferred centers. The hole center pixel
  may show the table behind the hole. Sample ring material or several boundary points to infer its center.
- wait: advance seconds (0.1 to 10), allowing natural settling.
All required fields must be returned. Non-motion point_m=[], inactive frame_id='', pixels=[].
Use finite numeric roll/jaw/seconds even when inactive; keep reason to one sentence.
Shared skill experience: an empty opening is not a grasp surface. A lifted ring can tilt or slip.
After grasping, use current evidence to determine the hole position. First raise above the peg before
horizontal transport. Preserve roll while carrying unless evidence motivates a change. Use current
fixed_pad_m plus the desired translation of the observed ring, never the hole center as a pad target.
Keep the entire ring AND jaws clear of the peg during transport. Once aligned, lower cautiously,
check alignment again, then release near the peg top and retreat. Small corrections can be repeated.
The closing jaw range is -0.174 (closed) to 1.75 (wide open). Avoid opening into the opposite ring wall.
A shared grasp opening, when reported, is command history only. Use images or current geometry to assess progress.
Actions may fail, leave the ring held, drop it or move it unexpectedly. Recover from the resulting scene.
No object attachment is assumed from a previous command. Robot values come from encoders and public geometry.
Full receives compact sensor-grounded hole/peg estimates with times and evidence; unavailable means unknown.
Base receives side RGB after physical actions. Both receive initial/requested images and the same depth queries.
Private physical success monitoring terminates the episode; action status alone does not certify task success.
'''


class OnlineRingSensors(DevelopmentSensors):
    def __init__(self, env, output):
        super().__init__(env, output)
        self.metric_frames = MetricDepthFrames()
        self.camera_seconds = self.depth_query_seconds = 0.
        self.depth_query_count = 0

    def camera(self, view='side'):
        start=time.monotonic()
        try:
            return super().camera(view)
        finally:
            self.camera_seconds += time.monotonic()-start

    def image(self, view):
        index=self.frames
        frame=self.camera(view)
        descriptor=self.metric_frames.register(view,frame)
        path=self.output/f'camera_{index:03d}_{view}.png'
        return path, dict(metric_depth_frame=descriptor,
                          intrinsics=plain(frame['intrinsics']),camera_to_world=plain(frame['camera_to_world']))

    def sample_depth(self, frame_id, pixels):
        start=time.monotonic();self.depth_query_count+=1
        result=self.metric_frames.sample(frame_id,pixels,current_time=float(self.env.data.time))
        self.depth_query_seconds+=time.monotonic()-start
        append(self.output/'depth_queries.jsonl',dict(frame_id=frame_id,pixels=pixels,result=result))
        return result


def compact_geometry(api):
    result={}
    for name in ('ring','peg'):
        try:
            value=api.observe(name)
            common={key:value[key] for key in ('status','time','frame','units') if key in value}
            if name=='ring':
                common.update(hole={k:v for k,v in value['hole'].items() if k!='inner_boundary_m'},
                              height_m=value['height_m'],minimum_z_m=value['minimum_z_m'],
                              bottom_evidence=value['bottom_evidence'])
            else:
                common.update({k:value[k] for k in ('top_center_m','axis','radius_m','evidence') if k in value})
            result[name]=common
        except (ValueError,RuntimeError):
            result[name]={'status':'unavailable','reason':'Insufficient current visual evidence'}
    return plain(result)


def robot_packet(api):
    robot=api.robot.current()
    vertices=np.asarray(robot['jaw_vertices_m'])
    return {**{k:plain(robot[k]) for k in ('time','fixed_pad_m','moving_pad_m','closing_axis','roll','jaw','source')},
            'minimum_jaw_z_m':float(vertices[:,2].min())}


class RingLoop:
    def __init__(self,env,api,skills,arm):
        if arm not in ('basic','symbolic'):raise ValueError('Unknown arm')
        self.env,self.api,self.skills,self.arm=env,api,skills,arm
        context=api.output/'context_000';context.mkdir()
        self.actor=RingActor(context)
        self.last_result=None;self.previous_action=None;self.turn=0
        self.feedback_seconds=0.;self.operations=0

    def step(self):
        self.env.advance_while_waiting()
        start=time.monotonic()
        initial=self.turn==0
        packet={'robot':robot_packet(self.api),'last_result':self.last_result,
                'last_grasp_opening_command':self.skills.last_grasp_opening}
        images=[]
        refresh=initial or self.previous_action=='observe'
        if refresh or (self.arm=='basic' and self.previous_action!='sample_depth'):
            for view in (('side','wrist') if refresh else ('side',)):
                path,descriptor=self.api.image(view)
                images.append(path);packet.setdefault('cameras',{})[view]=descriptor
        if self.arm=='symbolic' and (initial or self.previous_action not in ('sample_depth',)):
            packet['geometry']=compact_geometry(self.api)
        packet['remaining_wall_seconds']=max(0.,self.env.deadline-time.monotonic())
        self.feedback_seconds+=time.monotonic()-start
        prefix=CONTRACT+'\nTask: '+json.dumps(self.api.task())+'\n' if initial else ''
        prompt=prefix+'Interface: '+self.arm+'\nCurrent feedback:\n'+json.dumps(plain(packet),separators=(',',':'))
        append(self.api.output/'feedback.jsonl',dict(turn=self.turn,packet=packet,images=[str(p) for p in images]))
        decision=self.actor.request(prompt,images,slow=True,advance=self.env.advance_while_waiting,deadline=self.env.deadline)
        validate_response(decision)
        append(self.api.output/'model_decisions.jsonl',dict(turn=self.turn,decision=decision))
        action=decision['action']
        if action=='observe':outcome={'status':'observation_requested'}
        elif action=='sample_depth':outcome=self.api.sample_depth(decision['frame_id'],decision['pixels'])
        else:
            self.operations+=1
            append(self.api.output/'target_bindings.jsonl',dict(turn=self.turn,decision=decision))
            outcome=self.skills.execute(decision)
        self.last_result=plain(outcome);self.previous_action=action
        append(self.api.output/'decision_results.jsonl',dict(turn=self.turn,result=self.last_result))
        self.turn+=1

    def run(self):
        while True:self.step()
