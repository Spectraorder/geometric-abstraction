"""Sensor-only hoop insertion task with a private sustained-placement evaluator."""
from dataclasses import dataclass,asdict
import json
import math
import mujoco
import numpy as np
from aspire_so100.stacking_env import StackingEnv,TABLE_HEIGHT,_jsonable
from aspire_so100.hoop_pole import BASE_HEIGHT,HOOP_HEIGHT,HOOP_VARIANTS,POLE_RADIUS,POLE_HEIGHT,LEARNING_SCENE_VERSIONS

CONTROLLER='robot_gravity_ff_v1'


@dataclass
class HoopMetrics:
    success: bool
    score: int
    seated: bool
    released: bool
    tool_clear: bool
    stable_seconds: float
    center_error_m: float
    bottom_error_m: float
    tilt_degrees: float
    milestones: dict
    failure: str | None


class HoopPoleEnv(StackingEnv):
    pixel_aligned_rgbd=True

    def __init__(self,output_dir,render=False,*,scene_seed=6101,hoop_variant="standard"):
        self.inner_radius=HOOP_VARIANTS[hoop_variant]
        self.scene_seed=scene_seed
        self.objects=('red_hoop','red_pole')
        self.order=('red_hoop',)
        super().__init__(output_dir,render,scene_task='hoop_pole_learning',scene_seed=scene_seed,hoop_variant=hoop_variant)
        self.scene_metadata=dict(generator=LEARNING_SCENE_VERSIONS[hoop_variant],seed=scene_seed,
            objects=list(self.objects),object_count=2,controller=CONTROLLER,evaluator='hoop_seated_v1')
        if hoop_variant != "standard":
            self.scene_metadata.update(hoop_variant=hoop_variant,hoop_inner_radius_m=self.inner_radius)
        m=self.model
        self.hoop=m.body('red_hoop/red_hoop').id
        self.pole=m.body('red_pole/red_pole').id
        self.base_geom=m.geom('red_pole/red_pole_base').id
        self.hoop_dof=int(m.joint('red_hoop/red_hoop_joint').dofadr[0])
        self.jaw_address=int(m.joint('robot/Jaw').qposadr[0])
        self.jaws={m.body('robot/'+n).id for n in ('Fixed_Jaw','Moving_Jaw')}
        self.robot_bodies={i for i in range(m.nbody) if (m.body(i).name or '').startswith('robot/')}
        self.gravity_data=mujoco.MjData(m)
        self.gravity_force=np.zeros(m.nv)
        self._clear()

    def _clear(self):
        self.failure=None
        self.stable_seconds=0.
        self.progress=dict(approach=False,grasp=False,lift=False,align=False,insert=False,release=False,stable=False)

    def reset(self,seed,layout='sampled'):
        if seed!=self.scene_seed or layout!='sampled':
            raise ValueError('Hoop task requires its constructed seed and sampled layout')
        self.seed=seed;self.traces=[];self._clear()
        mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        mujoco.mj_forward(self.model,self.data)
        self._step_for(.65)
        return self.observation()

    def observation(self):
        return dict(time=float(self.data.time),tool_position=self.data.site_xpos[self.site_id].copy())

    def public_goal(self):
        return dict(kind='hoop_pole',hoop='red_hoop',pole='red_pole',
            instruction='Lift the red hoop over the matching fixed red pole; lower it around the pole, '
                        'release it onto the base, open the gripper and withdraw. No twisting is needed.')

    def placement_status(self):
        m,d=self.model,self.data
        contacts=set();base_contact=False
        for c in d.contact:
            if c.dist>.0001:continue
            a,b=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            if self.hoop not in (a,b):continue
            contacts.add(b if a==self.hoop else a)
            base_contact |= self.base_geom in (c.geom1,c.geom2)
        delta=d.xpos[self.hoop]-d.xpos[self.pole]
        xy=float(np.linalg.norm(delta[:2]));bottom=float(delta[2]-BASE_HEIGHT)
        tilt=math.acos(float(np.clip(d.xmat[self.hoop].reshape(3,3)[2,2],-1,1)))
        seated=bool(xy<=self.inner_radius-POLE_RADIUS-.001 and abs(bottom)<=.001
                    and tilt<=math.radians(10) and base_contact)
        released=bool(d.qpos[self.jaw_address]>=.6 and not (contacts & self.robot_bodies))
        tool_clear=bool(np.linalg.norm(d.site_xpos[self.site_id]-(d.xpos[self.hoop]+[0,0,HOOP_HEIGHT/2]))>=.04)
        still=bool(np.linalg.norm(d.qvel[self.hoop_dof:self.hoop_dof+3])<.005
                   and np.linalg.norm(d.qvel[self.hoop_dof+3:self.hoop_dof+6])<.15)
        return dict(seated=seated,released=released,tool_clear=tool_clear,still=still,
                    center_error_m=xy,bottom_error_m=bottom,tilt_degrees=math.degrees(tilt),
                    grasped=self.jaws<=contacts)

    def _step(self):
        # Robot-only gravity feedforward, through the existing actuator limits.
        m,d=self.model,self.data;scratch=self.gravity_data
        scratch.qpos[self.arm_qpos_addresses]=d.qpos[self.arm_qpos_addresses]
        scratch.qpos[self.jaw_address]=d.qpos[self.jaw_address]
        mujoco.mj_kinematics(m,scratch);mujoco.mj_comPos(m,scratch);mujoco.mj_comVel(m,scratch)
        mujoco.mj_rne(m,scratch,0,self.gravity_force)
        command=d.ctrl[:5].copy()
        d.ctrl[:5]+=self.gravity_force[self.arm_dof_addresses]/m.actuator_gainprm[:5,0]
        try:super()._step()
        finally:d.ctrl[:5]=command
        s=self.placement_status();p=self.progress
        p['approach'] |= bool(np.linalg.norm(d.site_xpos[self.site_id]-d.xpos[self.hoop])<.05)
        p['grasp'] |= bool(s['grasped'])
        p['lift'] |= bool(p['grasp'] and d.xpos[self.hoop,2]>TABLE_HEIGHT+.025)
        p['align'] |= bool(p['lift'] and s['center_error_m']<self.inner_radius-POLE_RADIUS
                           and d.xpos[self.hoop,2]>d.xpos[self.pole,2]+POLE_HEIGHT)
        p['insert'] |= bool(p['align'] and s['seated'])
        p['release'] |= bool(p['insert'] and s['released'] and s['tool_clear'])
        valid=s['seated'] and s['released'] and s['tool_clear'] and s['still']
        self.stable_seconds=self.stable_seconds+m.opt.timestep if valid else 0.
        p['stable'] |= self.stable_seconds>=2.

    def evaluate(self):
        s=self.placement_status()
        success=bool(self.stable_seconds>=2. and s['seated'] and s['released']
                     and s['tool_clear'] and s['still'] and self.failure is None)
        return HoopMetrics(success,100 if success else 0,s['seated'],s['released'],s['tool_clear'],
            float(self.stable_seconds),s['center_error_m'],s['bottom_error_m'],s['tilt_degrees'],
            {k:int(v) for k,v in self.progress.items()},self.failure)

    def check_structure(self,order=None,duration=5.):
        self.wait(duration)

    def write_trace(self,metrics,error=None):
        path=self.output_dir/'trace.json'
        path.write_text(json.dumps(_jsonable(dict(task='hoop_pole',seed=self.seed,error=error,
            metrics=asdict(metrics),primitives=self.traces)),indent=2)+'\n')
        return path
