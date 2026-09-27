"""Manual execution-ceiling probe; never provide this planner to study actors.

Targets use only public support-graph roles and RGB-D measurements. This is a
high-level development planner, not part of the geometry abstraction library.
"""
import numpy as np
from scipy.spatial import ConvexHull


def _visible_table(frame):
    rgb=np.asarray(frame['rgb'],float);r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
    depth=np.asarray(frame['depth_m'],float)
    mask=np.asarray(frame['valid_depth'],bool)&np.isfinite(depth)&(depth>0)
    mask&=(r>95)&(g>65)&(r>1.15*g)&(r<1.65*g)&(g>1.35*b)
    v,u=np.nonzero(mask);stride=max(1,len(u)//2500);v,u=v[::stride],u[::stride]
    rays=np.column_stack([u,v,np.ones(len(u))])@np.linalg.inv(frame['intrinsics']).T
    transform=np.asarray(frame['camera_to_world'])
    return (rays*depth[v,u,None])@transform[:3,:3].T+transform[:3,3]


def run(api):
    goal=api.task()
    if goal['kind']!='support_graph':raise ValueError('Require a public support graph')
    nodes=goal['nodes'];by_key={n['key']:n for n in nodes}
    if len({n['color'] for n in nodes})!=len(nodes):raise ValueError('Unique colors required for this probe')
    pnp=RGBDPickPlace(api)
    initial={n['key']:pnp.observe(n['color']) for n in nodes}
    unit=float(initial[goal['length_unit']['observed_cube']]['height'])
    relative_bottom={}
    for node in nodes:
        relative_bottom[node['key']]=max((relative_bottom[k]+initial[k]['height'] for k in node['supports']),default=0.)
    cloud=_visible_table(api.camera('side'))
    cloud=cloud[np.array([abs(p[2]-pnp.table_z(p[:2]))<.004 for p in cloud])]
    if len(cloud)<30:raise ValueError('Insufficient visible table')
    hull=ConvexHull(cloud[:,:2]);equations=hull.equations
    norms=np.linalg.norm(equations[:,:2],axis=1)
    anchor=np.mean([v['center'][:2] for v in initial.values()],axis=0)
    candidates=[]
    for dx in np.linspace(-4,4,9):
        for dy in np.linspace(-4,4,9):
            origin=anchor+unit*np.array([dx,dy]);clear=True
            for node in nodes:
                xy=origin+unit*np.asarray(node['grid'])
                radius=float(np.linalg.norm(initial[node['key']]['dims'])/2)
                if np.any((equations[:,:2]@xy+equations[:,2])/norms>-(radius+.01)):
                    clear=False;break
                if any(np.linalg.norm(xy-value['center'][:2])<radius+np.linalg.norm(value['dims'])/2+.008 for value in initial.values()):
                    clear=False;break
            if clear:candidates.append(origin)
    candidates.sort(key=lambda xy:float(np.linalg.norm(xy-anchor)))
    feasible=[]
    for origin in candidates[:18]:
        errors=[]
        for node in nodes:
            xy=origin+unit*np.asarray(node['grid']);support_z=pnp.table_z(xy)+relative_bottom[node['key']]
            for z in [support_z+1.2*initial[node['key']]['height']+.003,support_z+.085]:
                target=np.r_[xy,z]
                plan=api.plan_tool(target)
                if plan['position_error']>.003:plan=api.plan_tool(target,orientation=plan['tool_rotation'])
                errors.append(float(plan['position_error']))
                if errors[-1]>.003:break
            if max(errors)>.003:break
        if max(errors)<=.003:
            feasible.append((float(np.linalg.norm(origin-anchor)+10*max(errors)),origin))
    if not feasible:
        api.note('Development probe: no clear site passed executor-matched IK checks');return
    origin=min(feasible,key=lambda item:item[0])[1]
    api.record_geometry({'kind':'manual_development_plan','origin_m':origin.tolist(),'measured_unit_m':unit,
        'initial':{key:{'center_m':value['center'].tolist(),'height_m':value['height'],'dims_m':value['dims'].tolist()} for key,value in initial.items()},
        'source':'public_graph_and_RGBD_only'})
    for node in nodes:
        if not node['supports']:
            xy=origin+unit*np.asarray(node['grid']);surface=np.r_[xy,pnp.table_z(xy)]
        else:
            supports=[pnp.observe(by_key[key]['color']) for key in node['supports']]
            offset=np.asarray(node['grid'])-np.mean([by_key[key]['grid'] for key in node['supports']],axis=0)
            xy=np.mean([v['center'][:2] for v in supports],axis=0)+unit*offset
            surface=np.r_[xy,max(v['top'] for v in supports)]
        result=pnp.pick_and_place(node['color'],surface)
        api.note(node['key']+': '+result['status'])
        for _ in range(2):
            if result['status']=='released' or not result.get('held'):break
            result=pnp.place(surface);api.note(node['key']+' retry: '+result['status'])
        api.record_geometry({'kind':'manual_development_stage','key':node['key'],'surface_m':surface.tolist(),'result':{key:result[key] for key in ['status','held','xy_error_m','surface_gap_m'] if key in result}})
        if result['status']!='released':return
        api.wait(.5)
    api.note('All graph nodes released; private task success is not assumed')
