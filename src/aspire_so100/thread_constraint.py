"""Solver-resolved, ideal screw joint for the simplified fastening task.

A collisionless guide has axial slide and spin DOFs coupled by pitch. A weld
connects the engaged nut to that guide, never to the robot. Only the guide's
initial state is set at capture; nut poses and velocities are never overwritten.
This remains a reduced-order thread model, not resolved thread-face contact.
"""
import math
import mujoco
import numpy as np

from aspire_so100.robodojo_fasteners import THREAD_PITCH, BOLT_HEIGHT

SOLREF = (.004, 1.)
SOLIMP = (.9999, .9999, .0001, .5, 2.)
ENTRY_RADIUS = .0145


def add_thread_constraints(spec, positions, table_height):
    """Add guide DOFs after the ordinary bodies, preserving their state indices."""
    colors = [name.removesuffix('_nut') for name in positions if name.endswith('_nut')]
    for color in colors:
        x, y = positions[color+'_screw']
        # A 1 g guide avoids ill-conditioning from a near-massless auxiliary
        # body; its weight is compensated, but its small inertia remains.
        guide = spec.worldbody.add_body(name=color+'_thread_guide', pos=(x,y,table_height),
            mass=.001, inertia=(1.e-7,1.e-7,1.e-7), explicitinertial=True, gravcomp=1.)
        guide.add_joint(name=color+'_thread_z', type=mujoco.mjtJoint.mjJNT_SLIDE,
                        axis=(0,0,1), limited=False)
        guide.add_joint(name=color+'_thread_yaw', type=mujoco.mjtJoint.mjJNT_HINGE,
                        axis=(0,0,1), limited=False)
        spec.add_equality(name=color+'_thread_weld', type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY, name1=f'{color}_nut/{color}_nut',
            name2=color+'_thread_guide', active=False,
            data=(0,0,0, 0,0,0, 1,0,0,0, 1.), solref=SOLREF, solimp=SOLIMP)
        spec.add_equality(name=color+'_thread_helix', type=mujoco.mjtEq.mjEQ_JOINT,
            name1=color+'_thread_z', name2=color+'_thread_yaw', active=False,
            data=(0,THREAD_PITCH/(2*math.pi),0,0,0,0,0,0,0,0,1),
            solref=SOLREF, solimp=SOLIMP)
        # Entry shoulder blocks an unengaged nut from sliding down the smooth
        # shaft. Bit 2 collides with nut proxies only, not with the robot/table.
        bolt=spec.body(f'{color}_screw/{color}_screw')
        bolt.add_geom(name=color+'_thread_entry', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=(0,0,BOLT_HEIGHT-.003), size=(ENTRY_RADIUS,.001), mass=0,
            contype=2, conaffinity=0, group=3, rgba=(0,0,0,0),
            solref=SOLREF, solimp=SOLIMP, friction=(.3,.001,.0001), priority=2)
        for geom in spec.body(f'{color}_nut/{color}_nut').geoms:
            if geom.contype:
                geom.conaffinity |= 2
    for key in spec.keys:
        key.qpos = np.r_[key.qpos, np.zeros(2*len(colors))]
        if len(key.qvel):
            key.qvel = np.r_[key.qvel, np.zeros(2*len(colors))]


class ThreadConstraints:
    def __init__(self, env):
        self.env = env
        m=env.model
        self.guides={c:m.body(c+'_thread_guide').id for c in env.colors}
        self.weld={c:m.equality(c+'_thread_weld').id for c in env.colors}
        self.helix={c:m.equality(c+'_thread_helix').id for c in env.colors}
        self.entry={c:m.geom(c+'_thread_entry').id for c in env.colors}
        self.joints={c:tuple(m.joint(c+'_thread_'+k).id for k in ('z','yaw')) for c in env.colors}

    def reset(self):
        e=self.env
        for c in e.colors:
            e.data.eq_active[self.weld[c]]=False
            e.data.eq_active[self.helix[c]]=False
            e.model.geom_contype[self.entry[c]]=2

    def capture(self, color, bolt, entry_z, yaw):
        e=self.env;m,d=e.model,e.data
        # Initialize only the collisionless guide at the observed nut pose.
        # Solver impulses enforce the ideal joint; no nut state is assigned.
        m.body_pos[self.guides[color]]=d.xpos[e.bolts[bolt]]
        jz,jyaw=self.joints[color]
        dof=e.spin_dofs[color]-5
        delta,rotation,_=e._pose(color,bolt)
        d.qpos[m.jnt_qposadr[jz]]=delta[2]
        d.qpos[m.jnt_qposadr[jyaw]]=yaw
        d.qvel[m.jnt_dofadr[jz]]=d.qvel[dof+2]
        d.qvel[m.jnt_dofadr[jyaw]]=(rotation@d.qvel[dof+3:dof+6])[2]
        m.eq_data[self.helix[color],0]=entry_z-e.thread_lead*yaw
        d.eq_active[self.weld[color]]=True
        d.eq_active[self.helix[color]]=True
        m.geom_contype[self.entry[bolt]]=0

    def release(self, color, bolt):
        e=self.env
        e.data.eq_active[self.weld[color]]=False
        e.data.eq_active[self.helix[color]]=False
        e.model.geom_contype[self.entry[bolt]]=2
