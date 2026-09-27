"""Private fastening environment with versioned helical engagement.

This is an explicit simplified contact model, not microscopic thread contact.
Current scenes use solver constraints; historical scenes retain their springs.
Neither teleports the nut or attaches it to the hand.
"""
from dataclasses import dataclass
from dataclasses import asdict
import json
import math

import mujoco
import numpy as np

from aspire_so100.fasteners import (PAIR_COLORS, SCREW_HEIGHT as LEGACY_SCREW_HEIGHT,
                                    THREAD_PITCH as LEGACY_THREAD_PITCH,
                                    THREAD_START, SCENE_VERSION, LEGACY_SCENE_VERSION, STATIC_SCENE_VERSION,
                                    check_version, ROBODOJO_VERSIONS)
from aspire_so100.robodojo_fasteners import BOLT_HEIGHT, BOLT_HEAD_HEIGHT, THREAD_PITCH, THREAD_PHASE
from aspire_so100.stacking_env import StackingEnv
from aspire_so100.stacking_env import TABLE_HEIGHT
from aspire_so100.grasp_quality import GraspQuality, GRASP_QUALITY_VERSION

COLORS = dict(PAIR_COLORS, yellow=(.95,.73,.04,1.))
REQUIRED_TURNS = 3.
REQUIRED_DEPTH = REQUIRED_TURNS*THREAD_PITCH
ENTRY_Z = BOLT_HEIGHT-.002
LEAD = THREAD_PITCH/(2*math.pi)
# Demonstration parameters, not calibrated hardware friction. Solver friction
# supports a nonzero holding torque at zero velocity; damping alone cannot.
THREAD_HOLD_TORQUE = .002  # N m
THREAD_SPIN_DAMPING = .006  # N m s / rad, integrated implicitly by MuJoCo
TRACKING_CONTROLLER = 'robot_gravity_ff_v1'
CONTROLLERS = ('legacy', TRACKING_CONTROLLER)


def learning_layout(seed, pair_count=3):
    if pair_count not in (1, 3):
        raise ValueError("Fastening supports one or three pairs")
    colors = tuple(COLORS) if pair_count == 3 else ("red",)
    slots = (-.11, 0., .11) if pair_count == 3 else (0.,)
    rng = np.random.default_rng(seed)
    positions = {}
    for color,x in zip(colors,slots):
        positions[color+'_screw'] = tuple(np.array([x,-.22])+rng.uniform(-.003,.003,2))
    for color,x in zip(rng.permutation(list(colors)),slots):
        positions[color+'_nut'] = tuple(np.array([x,-.33])+rng.uniform(-.007,.007,2))
    return positions


def benchmark_score(count, released, home):
    if not released or count == 0:
        return 0
    if count == 3 and home:
        return 100
    return 50 if count >= 2 else 20


@dataclass
class FastenMetrics:
    success: bool
    score: int
    fastened_pairs: int
    gripper_open: bool
    returned_home: bool
    pairs: dict
    milestones: dict
    failure: str | None
    pickup: dict
    grasp_quality: dict


class FastenScrewsEnv(StackingEnv):
    pixel_aligned_rgbd = True

    def __init__(self, output_dir, render=False, *, scene_seed=5101, scene_version=SCENE_VERSION,
                 controller='legacy', learning_stage='fasten', grasp_quality='legacy', pair_count=3):
        if pair_count not in (1, 3):
            raise ValueError("Fastening supports one or three pairs")
        self.pair_count = pair_count
        self.colors = tuple(COLORS) if pair_count == 3 else ("red",)
        check_version(scene_version)
        if controller not in CONTROLLERS:
            raise ValueError(f'Unknown robot controller: {controller}')
        self.controller = controller
        if learning_stage not in ('pickup', 'fasten'):
            raise ValueError('Unknown fastening learning stage')
        self.learning_stage = learning_stage
        if grasp_quality not in ('legacy',GRASP_QUALITY_VERSION):
            raise ValueError('Unknown grasp-quality evaluator version')
        self.grasp_quality_version = grasp_quality
        self.scene_version = scene_version
        original_assets = scene_version in ROBODOJO_VERSIONS
        self.entry_z = ENTRY_Z if original_assets else LEGACY_SCREW_HEIGHT-.002
        self.thread_pitch = THREAD_PITCH if original_assets else LEGACY_THREAD_PITCH
        self.thread_lead = self.thread_pitch/(2*math.pi)
        self.thread_phase = THREAD_PHASE if original_assets else THREAD_START
        self.thread_floor = BOLT_HEAD_HEIGHT+.001 if original_assets else .009
        self.required_depth = REQUIRED_TURNS*self.thread_pitch
        self.scene_seed = scene_seed
        self.order = tuple(self.colors)
        self.objects = tuple(learning_layout(scene_seed, pair_count))
        self.scene_metadata = dict(generator=scene_version, seed=scene_seed,
                                   pairs=list(self.colors), fixture='fixed_bolts', object_count=2*pair_count)
        super().__init__(output_dir, render, scene_task='fasten_screws_learning', scene_seed=scene_seed,
                         fastener_version=scene_version, fasten_pair_count=pair_count)
        if pair_count != 3:
            self.scene_metadata["pair_count"] = pair_count
        if controller != 'legacy':
            self.scene_metadata['controller'] = controller
        if learning_stage != 'fasten':
            self.scene_metadata['learning_stage'] = learning_stage
        if grasp_quality != 'legacy':
            self.scene_metadata['grasp_quality'] = grasp_quality
        # Feedforward uses the robot model and measured robot joint positions
        # only, with zero velocity. Object poses/contacts never enter this path.
        self.gravity_data = mujoco.MjData(self.model)
        self.gravity_force = np.zeros(self.model.nv)
        self.nuts = {c:self.model.body(f'{c}_nut/{c}_nut').id for c in self.colors}
        self.bolts = {c:self.model.body(f'{c}_screw/{c}_screw').id for c in self.colors}
        self.jaws = {self.model.body('robot/'+n).id for n in ('Fixed_Jaw','Moving_Jaw')}
        self.jaw_address = int(self.model.joint('robot/Jaw').qposadr[0])
        self.home = self.model.key_qpos[0,self.arm_qpos_addresses].copy()
        self.spin_dofs = {c:int(self.model.joint(f'{c}_nut/{c}_nut_joint').dofadr[0])+5
                          for c in self.colors}
        self.free_spin_properties = {
            c:(float(self.model.dof_frictionloss[dof]),float(self.model.dof_damping[dof]))
            for c,dof in self.spin_dofs.items()}
        if self.scene_version in (*ROBODOJO_VERSIONS, STATIC_SCENE_VERSION):
            for dof in self.spin_dofs.values():
                # Friction constraint regularization: high impedance bounds
                # numerical creep, without changing contact/robot constraints.
                self.model.dof_solimp[dof] = (.9999,.9999,.001,.5,2.)
                self.model.dof_solref[dof] = (.004,1.)
        self.thread_constraints = None
        if scene_version == SCENE_VERSION:
            from aspire_so100.thread_constraint import ThreadConstraints
            self.thread_constraints = ThreadConstraints(self)
        self._clear()

    def _clear(self):
        self.threads = {}
        if self.thread_constraints is not None:
            self.thread_constraints.reset()
        for color in self.colors:
            self._set_thread_friction(color,False)
        self.progress = {c:dict(approach=False,grasp=False,lift=False,align=False,
                                engage=False,rotate=False,depth=False,fasten=False) for c in self.colors}
        self.held_seconds = {c:0. for c in self.colors}
        self.failure = None
        self.pickup_grasp_seconds = 0.
        self.pickup_grasp_max_seconds = 0.
        self.pickup_hold_seconds = 0.
        self.pickup_hold_max_seconds = 0.
        self.pickup_max_lift = 0.
        self.pickup_hold_anchor = None
        self.grasp_monitors = {c:GraspQuality() for c in self.colors}

    def _set_thread_friction(self, color, engaged):
        """Dry friction about the nut's local Z axis, only on a fixed bolt.

        An upright nut's local Z is its threaded axis. Friction is resolved with
        applied/contact forces in the solver, not capped by the pre-step speed.
        No object pose or velocity is overwritten. V1/V2 remain reproducible.
        """
        if self.scene_version not in (*ROBODOJO_VERSIONS, STATIC_SCENE_VERSION):
            return
        dof = self.spin_dofs[color]
        friction,damping = self.free_spin_properties[color]
        self.model.dof_frictionloss[dof] = friction+(THREAD_HOLD_TORQUE if engaged else 0.)
        self.model.dof_damping[dof] = damping+(THREAD_SPIN_DAMPING if engaged else 0.)

    def reset(self, seed, layout='sampled'):
        if seed != self.scene_seed or layout != 'sampled':
            raise ValueError('Fastening requires the constructed seed and sampled layout')
        self.seed = seed
        self.traces = []
        self._clear()
        mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        mujoco.mj_forward(self.model,self.data)
        self._step_for(.65)
        return self.observation()

    def observation(self):
        # Used for local traces, never as a privileged actor RPC.
        return {'time':float(self.data.time),
                'tool_position':self.data.site_xpos[self.site_id].copy()}

    def public_goal(self):
        if self.learning_stage == 'pickup':
            return {'kind':'fasten_pickup', 'target':'red_nut',
                    'instruction':'Locate the red nut from RGB-D, grasp with both jaws, '
                                  'lift its bottom at least 30 mm above the table, and hold '
                                  'upright and stationary for 2 s. Finish holding it. '
                                  'Do not approach a bolt or attempt screwing yet.'}
        return {'kind':'fasten_screws','pairs':[{'nut':c+'_nut','bolt':c+'_screw'} for c in self.colors],
                'instruction':'Match colors; grasp nut, align upright above fixed bolt, turn clockwise '
                              'viewed from above while feeding downward; release and return home.'}

    def write_trace(self, metrics, error=None):
        from aspire_so100.stacking_env import _jsonable
        path=self.output_dir/'trace.json'
        path.write_text(json.dumps(_jsonable(dict(task='fasten_screws',seed=self.seed,
                        error=error,metrics=asdict(metrics),primitives=self.traces)),indent=2)+'\n')
        return path

    def _contacts(self, body):
        contacts = self.data.contact
        a = self.model.geom_bodyid[contacts.geom1]
        b = self.model.geom_bodyid[contacts.geom2]
        close = contacts.dist <= .0001
        return set(b[(a==body)&close]) | set(a[(b==body)&close])

    def _pose(self,color,bolt_color):
        n,b = self.nuts[color],self.bolts[bolt_color]
        delta = self.data.xpos[n]-self.data.xpos[b]
        rotation = self.data.xmat[n].reshape(3,3)
        return delta, rotation, math.atan2(rotation[1,0],rotation[0,0])

    def _thread_forces(self):
        """Capture an aligned, actually grasped nut; constrain z to net twist."""
        dt = self.model.opt.timestep
        for color,nut in self.nuts.items():
            self.data.xfrc_applied[nut] = 0.
            self._set_thread_friction(color,False)
            contacts = self._contacts(nut)
            grasped = self.jaws <= contacts
            p = self.progress[color]
            p['approach'] |= bool(np.linalg.norm(self.data.site_xpos[self.site_id]-self.data.xpos[nut]) < .05)
            p['grasp'] |= grasped
            p['lift'] |= bool(p['grasp'] and self.data.xpos[nut,2] > .43)
            for bolt in self.colors:
                delta,rotation,yaw = self._pose(color,bolt)
                aligned = np.linalg.norm(delta[:2]) < .002 and rotation[2,2] > math.cos(math.radians(6))
                if bolt == color:
                    p['align'] |= bool(p['lift'] and aligned and .03 < delta[2] < .09)
                if color not in self.threads and grasped and aligned and abs(delta[2]-self.entry_z) < .002:
                    if any(t['bolt']==bolt for t in self.threads.values()):
                        continue
                    # Seat onto the nearest compatible helical phase. V1 used
                    # an arbitrary yaw reference, which cannot match real meshes.
                    entry_z = self.entry_z
                    if self.scene_version != LEGACY_SCENE_VERSION:
                        phase = self.thread_phase + self.thread_lead*yaw
                        entry_z = phase + round((self.entry_z-phase)/self.thread_pitch)*self.thread_pitch
                    self.threads[color] = dict(bolt=bolt,yaw=yaw,twist=0.,entry_z=entry_z)
                    if self.thread_constraints is not None:
                        self.thread_constraints.capture(color,bolt,entry_z,yaw)
                    if bolt == color: p['engage'] = True
            if color not in self.threads:
                continue
            thread = self.threads[color]
            entry_z = thread['entry_z']
            delta,rotation,yaw = self._pose(color,thread['bolt'])
            increment = math.atan2(math.sin(yaw-thread['yaw']),math.cos(yaw-thread['yaw']))
            thread['twist'] += increment
            thread['yaw'] = yaw
            # Counterclockwise withdrawal releases the thread; gross dislocation
            # breaks engagement instead of pulling a remotely located object back.
            if (thread['twist'] > .4 or np.linalg.norm(delta[:2]) > .012 or
                    rotation[2,2] < .8 or abs(delta[2]-(entry_z+self.thread_lead*thread['twist'])) > .02):
                if self.thread_constraints is not None:
                    self.thread_constraints.release(color,thread['bolt'])
                del self.threads[color]
                continue
            self._set_thread_friction(color,True)
            if self.thread_constraints is not None:
                if thread['bolt'] == color:
                    p['rotate'] |= bool(-thread['twist'] >= math.pi/2)
                    p['depth'] |= bool(entry_z-delta[2] >= self.required_depth-.001)
                continue
            address = int(self.model.joint(f'{color}_nut/{color}_nut_joint').dofadr[0])
            linear = self.data.qvel[address:address+3]
            # Free-joint angular velocity is body-local; applied torque is world.
            angular = rotation @ self.data.qvel[address+3:address+6]
            target_z = max(self.thread_floor, entry_z+self.thread_lead*thread['twist'])
            force = -1200.*np.r_[delta[:2],delta[2]-target_z]-3.*linear
            force[2] += 3.*self.thread_lead*angular[2]
            torque = .15*np.cross(rotation[:,2],np.array([0.,0.,1.]))-.001*angular
            if self.scene_version in (*ROBODOJO_VERSIONS, STATIC_SCENE_VERSION):
                # Helical axial reaction remains physical. Dry friction and
                # spin damping are handled implicitly by MuJoCo, including the
                # static case where angular speed is zero but load is nonzero.
                torque[2] = -self.thread_lead*force[2]
            else:
                # Historical V1/V2 model: retains its known gravity creep for
                # old trials; never mix those scores into a V3 campaign.
                inertia=float(np.min(self.model.body_inertia[nut]))
                stop_torque=inertia*abs(angular[2])/dt
                drag=min(.006*abs(angular[2])+.002,stop_torque)
                torque[2] = -self.thread_lead*force[2]-math.copysign(drag,angular[2])
            self.data.xfrc_applied[nut,:3] = np.clip(force,-30.,30.)
            self.data.xfrc_applied[nut,3:] = np.clip(torque,-.15,.15)
            if thread['bolt'] == color:
                p['rotate'] |= bool(-thread['twist'] >= math.pi/2)
                p['depth'] |= bool(entry_z-delta[2] >= self.required_depth-.001)

    def pair_status(self, color):
        thread = self.threads.get(color)
        delta,rotation,_ = self._pose(color,color)
        turns = -thread['twist']/(2*math.pi) if thread else 0.
        entry_z = thread['entry_z'] if thread else self.entry_z
        depth = entry_z-delta[2]
        touching = bool(self._contacts(self.nuts[color]) & self.jaws)
        valid = bool(thread and thread['bolt']==color and turns >= REQUIRED_TURNS-.05
                     and depth >= self.required_depth-.001 and np.linalg.norm(delta[:2]) <= .002
                     and rotation[2,2] >= math.cos(math.radians(6)) and not touching
                     and abs(delta[2]-(entry_z+self.thread_lead*thread['twist'])) < .0015)
        return dict(valid=valid,turns=float(turns),depth_m=float(depth),
                    aligned=bool(np.linalg.norm(delta[:2])<=.002),touching_gripper=touching,
                    engaged=thread is not None,matching=bool(thread and thread['bolt']==color))

    def _step(self):
        self._thread_forces()
        command = self.data.ctrl[:5].copy()
        if self.controller == TRACKING_CONTROLLER:
            scratch = self.gravity_data
            scratch.qpos[self.arm_qpos_addresses] = self.data.qpos[self.arm_qpos_addresses]
            scratch.qpos[self.jaw_address] = self.data.qpos[self.jaw_address]
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_comPos(self.model, scratch)
            mujoco.mj_comVel(self.model, scratch)
            mujoco.mj_rne(self.model, scratch, 0, self.gravity_force)
            # Position-servo reference offset realizes a gravity feedforward
            # torque through the EXISTING actuators and their force limits.
            # Do not add external forces, change gains, or override joint state.
            self.data.ctrl[:5] += (self.gravity_force[self.arm_dof_addresses]
                                   / self.model.actuator_gainprm[:5, 0])
        try:
            super()._step()
        finally:
            self.data.ctrl[:5] = command
        for color in self.colors:
            valid = self.pair_status(color)['valid']
            self.held_seconds[color] = self.held_seconds[color]+self.model.opt.timestep if valid else 0.
            self.progress[color]['fasten'] |= self.held_seconds[color] >= 2.
        self._monitor_pickup()
        if self.grasp_quality_version != 'legacy':
            tool_position = self.data.site_xpos[self.site_id]
            tool_rotation = self.data.site_xmat[self.site_id].reshape(3,3)
            for color,body in self.nuts.items():
                contacts = self._contacts(body)
                self.grasp_monitors[color].update(self.model.opt.timestep,
                    self.data.xpos[body],self.data.xmat[body].reshape(3,3),tool_position,tool_rotation,
                    airborne=bool(self.data.xpos[body,2] > TABLE_HEIGHT+.01),
                    touching=bool(contacts & self.jaws),both_jaws=self.jaws <= contacts,
                    unsupported=contacts <= self.jaws,engaged=color in self.threads)

    def _monitor_pickup(self):
        """Private curriculum checks; never exposed during actor execution."""
        body = self.nuts['red']
        contacts = self._contacts(body)
        grasped = self.jaws <= contacts
        dt = self.model.opt.timestep
        self.pickup_grasp_seconds = self.pickup_grasp_seconds+dt if grasped else 0.
        self.pickup_grasp_max_seconds = max(self.pickup_grasp_max_seconds, self.pickup_grasp_seconds)
        position = self.data.xpos[body]
        lift = float(position[2]-TABLE_HEIGHT)
        if grasped:
            self.pickup_max_lift = max(self.pickup_max_lift, lift)
        upright = self.data.xmat[body].reshape(3,3)[2,2] >= math.cos(math.radians(15))
        dof = self.spin_dofs['red']-5
        still = np.linalg.norm(self.data.qvel[dof:dof+3]) < .01
        eligible = (grasped and contacts <= self.jaws and lift >= .03 and upright and still
                    and self.pickup_grasp_seconds >= .25)
        if eligible and self.pickup_hold_anchor is None:
            self.pickup_hold_anchor = position.copy()
        if eligible and np.linalg.norm(position-self.pickup_hold_anchor) > .003:
            eligible = False
        self.pickup_hold_seconds = self.pickup_hold_seconds+dt if eligible else 0.
        if not eligible:
            self.pickup_hold_anchor = None
        self.pickup_hold_max_seconds = max(self.pickup_hold_max_seconds, self.pickup_hold_seconds)

    def pickup_status(self):
        return dict(target='red_nut', success=bool(self.pickup_hold_seconds >= 2. and self.failure is None),
                    grasp=bool(self.progress['red']['grasp']),
                    sustained_grasp=bool(self.pickup_grasp_max_seconds >= .25),
                    lift=bool(self.pickup_max_lift >= .03),
                    max_lift_m=self.pickup_max_lift,
                    max_grasp_seconds=self.pickup_grasp_max_seconds,
                    hold_seconds=self.pickup_hold_seconds,
                    max_hold_seconds=self.pickup_hold_max_seconds)

    def check_structure(self, order=None, duration=5.):
        self.wait(duration)

    def evaluate(self):
        pairs = {c:dict(self.pair_status(c),stable_seconds=self.held_seconds[c]) for c in self.colors}
        count = sum(p['valid'] and p['stable_seconds'] >= 2. for p in pairs.values())
        released = bool(self.data.qpos[self.jaw_address] >= .6 and
                        all(not p['touching_gripper'] for p in pairs.values()))
        home = bool(np.max(abs(self.data.qpos[self.arm_qpos_addresses]-self.home)) < .12)
        score = benchmark_score(count,released,home)
        milestones = {k:sum(int(p[k]) for p in self.progress.values()) for k in next(iter(self.progress.values()))}
        pickup = self.pickup_status()
        success = pickup['success'] if self.learning_stage == 'pickup' else (count == self.pair_count and released
                    and home and self.failure is None)
        quality = ({c:monitor.status() for c,monitor in self.grasp_monitors.items()}
                   if self.grasp_quality_version != 'legacy' else {})
        return FastenMetrics(success,score,count,released,home,pairs,milestones,self.failure,pickup,quality)
