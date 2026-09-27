"""Original RoboDojo surfaces, with explicitly approximate MuJoCo contact.

USDZ provenance and conversion details live in assets/robodojo_fasteners.
The original PhysX SDF collision model is NOT reproduced by this adapter.
"""
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parents[2]/'assets/robodojo_fasteners/converted'
VERSION = 'fasten_robodojo_v1'
BOLT_HEIGHT = .0525
BOLT_HEAD_HEIGHT = .0125
NUT_HEIGHT = .019110513385385275
# Nominal helix measured from the original bolt mesh, not the old 4 mm asset.
THREAD_PITCH = .003
# Approximate axial phase difference between original male/female grooves.
THREAD_PHASE = .00035
COLORS = {'red':(1.,0.,0.,1.), 'blue':(0.,0.,1.,1.),
          'yellow':(.98039216,.7137255,.003921569,1.)}


@lru_cache(maxsize=8)
def load_mesh(filename):
    with np.load(ASSETS/filename, allow_pickle=False) as data:
        return {key:data[key] for key in data.files}


def add_mesh(spec, body, name, filename, color, *, collision=False, mass=0.):
    mesh = load_mesh(filename)
    kwargs = dict(name=name, uservert=mesh['vertices'].ravel().tolist(),
                  userface=mesh['faces'].ravel().tolist())
    if len(mesh['normals']):
        kwargs.update(usernormal=mesh['normals'].ravel().tolist(),
                      userfacenormal=mesh['normal_faces'].ravel().tolist())
    spec.add_mesh(**kwargs)
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                  rgba=color, group=3 if collision else 2, mass=mass,
                  contype=int(collision), conaffinity=int(collision),
                  friction=(1.,.005,.0001), solref=(.006,1.))


def get_nut_spec(name, color):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f'{name}_joint')
    add_mesh(spec,body,f'{name}_surface','factory_nut.npz',color)
    for i in range(6):
        add_mesh(spec,body,f'{name}_contact_{i:02d}',f'factory_nut_edge_cube_{i}.npz',
                 color,collision=True,mass=.03/6)
    body.add_site(name=f'{name}_bore',pos=(0,0,NUT_HEIGHT/2),size=(.001,),rgba=(0,0,0,0))
    return spec


def get_screw_spec(name, color, *, fixed=False):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    if not fixed:
        body.add_freejoint(name=f'{name}_joint')
    add_mesh(spec,body,f'{name}_surface','factory_bolt.npz',color)
    # MuJoCo convexifying the original concave threaded surface would fill its
    # valleys. These hidden support/shaft proxies accompany the helical model.
    body.add_geom(name=f'{name}_head_contact',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=(0,0,BOLT_HEAD_HEIGHT/2),size=(.0178,BOLT_HEAD_HEIGHT/2),
                  rgba=color,mass=.025,group=3,friction=(1.,.005,.0001),
                  priority=1,solref=(.008,1.),solimp=(.99,.99,.001,.5,2.))
    body.add_geom(name=f'{name}_shaft_contact',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=(0,0,(BOLT_HEAD_HEIGHT+BOLT_HEIGHT)/2),
                  size=(.0095,(BOLT_HEIGHT-BOLT_HEAD_HEIGHT)/2),rgba=color,
                  mass=.025,group=3,friction=(1.,.005,.0001))
    body.add_site(name=f'{name}_tip',pos=(0,0,BOLT_HEIGHT),size=(.001,),rgba=(0,0,0,0))
    return spec
