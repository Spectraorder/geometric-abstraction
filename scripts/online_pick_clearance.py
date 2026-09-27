"""Robot-only sampled jaw envelopes and sensor-grounded worksite screening."""
import itertools
import mujoco
import numpy as np
from online_bridge_skills import SensorAPI

SIGNS = np.asarray(list(itertools.product((-1.,1.),repeat=3)))


class ClearanceSensorAPI(SensorAPI):
    def pickup_envelopes(self, fit, grasp_fraction=.4):
        """Sample robot jaw geometry; never read object state from the model."""
        model=self.env.model
        scratch=mujoco.MjData(model)
        jaw_address=int(model.joint('robot/Jaw').qposadr[0])
        ids=[]
        for i in range(model.ngeom):
            body=model.body(int(model.geom_bodyid[i])).name or ''
            if body in ('robot/Fixed_Jaw','robot/Moving_Jaw') and (model.geom_contype[i] or model.geom_conaffinity[i]):
                ids.append(i)
        if not ids:raise ValueError('No public jaw collision geometry')
        points=[]
        source=np.asarray(fit['center'],float).copy()
        source[2]+=grasp_fraction*fit['height']
        for dz in (0.,.025,.06):
            target=source+np.array([0.,0.,dz])
            plan=self.plan_tool(target)
            if plan['position_error']>.005:
                raise ValueError('A source pickup is outside the robot IK tolerance')
            scratch.qpos[self.env.arm_qpos_addresses]=plan['joint_angles']
            for jaw in (-.174,.3,.8):
                scratch.qpos[jaw_address]=jaw
                mujoco.mj_kinematics(model,scratch)
                for i in ids:
                    geom=model.geom(i)
                    if geom.type[0]==mujoco.mjtGeom.mjGEOM_MESH:
                        mesh=int(geom.dataid[0]);start=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
                        local=model.mesh_vert[start:start+count]
                    elif geom.type[0]==mujoco.mjtGeom.mjGEOM_BOX:
                        local=SIGNS*geom.size
                    else:
                        # A bounding sphere conservatively encloses unsupported
                        # primitive types in this development screen.
                        local=SIGNS*model.geom_rbound[i]
                    vertices=local@scratch.geom_xmat[i].reshape(3,3).T+scratch.geom_xpos[i]
                    points.append((vertices.min(axis=0),vertices.max(axis=0)))
        return np.asarray(points)


def overlaps(bounds, envelope, margin=.005):
    low,high=np.asarray(bounds)
    return bool(np.any(np.all((high+margin>envelope[:,0]) & (low-margin<envelope[:,1]),axis=1)))


def select_workspace(bridge, api, session, nodes, order, geometry, region, unit, envelopes):
    """Preserve later pickup space when positioning earlier construction parts."""
    roots=[k for k in order if not nodes[k]['supports']]
    grids=np.asarray([nodes[k]['grid'] for k in order],float)
    centers=np.asarray([geometry[k]['center'][:2] for k in order]);middle=centers.mean(axis=0)
    base=middle-unit*grids.mean(axis=0)
    candidates=[geometry[k]['center'][:2]-unit*np.asarray(nodes[k]['grid']) for k in roots]
    candidates += [base+unit*np.array([dx,dy]) for dx in np.arange(-7.5,7.51,1.5) for dy in np.arange(-7.5,7.51,1.5)]
    ranked=[];rejected=0
    for origin in candidates:
        layout=bridge.planned_layout(session,nodes,order,geometry,origin,unit)
        valid=True;score=0.
        for index,key in enumerate(order):
            goal=layout[key];fit=geometry[key];half=fit['footprint']/2
            if not bridge.inside_table(region,goal['xy'],half,.35*unit):valid=False;break
            distance=float(np.linalg.norm(goal['xy']-fit['center'][:2]));score+=distance/unit
            if key in roots and distance<.12*unit:score-=3.
            box=np.array([np.r_[goal['xy']-half,goal['bottom']],np.r_[goal['xy']+half,goal['top']]])
            for later in order[index+1:]:
                src=geometry[later]
                object_overlap=(np.all(np.abs(goal['xy']-src['center'][:2])<(fit['footprint']+src['footprint'])/2+.4*unit)
                                and goal['bottom']<src['top']+.25*unit and goal['top']>src['bottom'])
                if object_overlap or overlaps(box,envelopes[later]):
                    rejected+=1;valid=False;break
            if not valid:break
        if valid:
            score+=.35*float(np.linalg.norm(np.mean([layout[k]['xy'] for k in order],axis=0)-middle))/unit
            ranked.append((score,origin.copy(),layout))
    ranked.sort(key=lambda row:row[0])
    for _,origin,layout in ranked[:12]:
        if all(reachable_placement(api,np.r_[layout[k]['xy'],layout[k]['bottom']+1.2*geometry[k]['height']+.003]) for k in order):
            return origin,layout,{'valid_candidates':len(ranked),'overlap_rejections':rejected}
    return None,None,{'valid_candidates':len(ranked),'overlap_rejections':rejected}


def conditional_center_box(points, rotation, dimensions, center, constrained_axes, slack=.002):
    """Finite center bounds conditional on association, shape and orientation.

    Each surface sample lies inside the initialized oriented shape box. These
    bounds are for conservative planning, not a newly observed executable pose.
    """
    points=np.asarray(points,float);rotation=np.asarray(rotation,float);dimensions=np.asarray(dimensions,float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<8 or rotation.shape!=(3,3) or dimensions.shape!=(3,):
        raise ValueError('Insufficient conditional-bound inputs')
    if (not np.isfinite(points).all() or not np.isfinite(rotation).all() or
            not np.isfinite(dimensions).all() or not np.all(dimensions>0) or
            not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or
            len(constrained_axes)!=3 or np.asarray(center).shape!=(3,) or
            not np.isfinite(center).all() or not np.isfinite(slack) or slack<0):
        raise ValueError('Invalid conditional-bound inputs')
    local=points@rotation;low=local.max(axis=0)-dimensions/2-slack;high=local.min(axis=0)+dimensions/2+slack
    projected=np.asarray(center)@rotation
    for axis,known in enumerate(constrained_axes):
        if known:
            low[axis]=max(low[axis],projected[axis]-slack);high[axis]=min(high[axis],projected[axis]+slack)
    if np.any(low>high):raise ValueError('Visible samples contradict initialized shape bounds')
    midpoint=(low+high)/2;nominal=rotation@midpoint
    radius=np.abs(rotation)@((high-low)/2)
    return nominal,radius,{'local_lower_m':low,'local_upper_m':high,'rotation':rotation,
                           'conditional_on':'current color association, initialized dimensions and fitted orientation',
                           'pose_available':False,'planning_only':True}


def reachable_placement(api, point, max_rotation_degrees=8.):
    """Screen placement orientation as well as positional IK error."""
    desired=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    plan=api.plan_tool(point)
    angle=np.arccos(np.clip((np.trace(plan['tool_rotation']@desired.T)-1)/2,-1.,1.))
    return plan['position_error']<=.003 and angle<=np.deg2rad(max_rotation_degrees)
