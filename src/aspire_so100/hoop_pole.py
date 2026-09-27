"""A passive hoop-on-pole insertion scene with ordinary rigid-body contacts."""
import math
import mujoco
import numpy as np

SCENE_VERSION = 'hoop_pole_v1'
HOOP_INNER_RADIUS = .012
HOOP_VARIANTS = {"standard": HOOP_INNER_RADIUS, "tight": .010}
HOOP_OUTER_RADIUS = .020
HOOP_HEIGHT = .012
HOOP_MASS = .020
POLE_RADIUS = .006
POLE_HEIGHT = .051
BASE_RADIUS = .024
BASE_HEIGHT = .006
HOOP_COLOR = (.9,.08,.055,1.)
POLE_COLOR = HOOP_COLOR
POSITIONS = {'red_hoop':(-.045,-.33), 'red_pole':(.05,-.22)}


def hoop_mesh(segments=96, *, variant="standard"):
    """Closed annulus; a single mesh is used for visuals only, never convexified."""
    inner_radius = HOOP_VARIANTS[variant]
    profile=((inner_radius,0.),(HOOP_OUTER_RADIUS,0.),
             (HOOP_OUTER_RADIUS,HOOP_HEIGHT),(inner_radius,HOOP_HEIGHT))
    vertices=[(r*math.cos(a),r*math.sin(a),z)
              for a in np.arange(segments)*2*math.pi/segments for r,z in profile]
    faces=[]
    for i in range(segments):
        for j in range(4):
            a=4*i+j;b=4*((i+1)%segments)+j
            c=4*((i+1)%segments)+(j+1)%4;d=4*i+(j+1)%4
            faces.extend(((a,b,c),(a,c,d)))
    return vertices,faces


def _mesh(spec,body,name,vertices,faces,*,collision=False,mass=0.):
    spec.add_mesh(name=name,uservert=np.asarray(vertices).ravel().tolist(),
                  userface=np.asarray(faces).ravel().tolist())
    body.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=name,
                  rgba=HOOP_COLOR,group=3 if collision else 2,mass=mass,
                  contype=int(collision),conaffinity=int(collision),
                  friction=(.8,.005,.0001),solref=(.006,1.))


def get_hoop_spec(*, variant="standard"):
    inner_radius = HOOP_VARIANTS[variant]
    name='red_hoop';spec=mujoco.MjSpec();body=spec.worldbody.add_body(name=name)
    body.add_freejoint(name=name+'_joint')
    vertices,faces=hoop_mesh(variant=variant)
    _mesh(spec,body,name+'_surface',vertices,faces)
    # Each convex wedge occupies only the ring material, preserving the hole.
    count=32
    faces=[(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
           (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    for i in range(count):
        a=2*math.pi*i/count;b=2*math.pi*(i+1)/count
        xy=[(r*math.cos(t),r*math.sin(t)) for r,t in
            ((inner_radius,a),(HOOP_OUTER_RADIUS,a),
             (HOOP_OUTER_RADIUS,b),(inner_radius,b))]
        vertices=[(x,y,z) for z in (0.,HOOP_HEIGHT) for x,y in xy]
        _mesh(spec,body,f'{name}_contact_{i:02d}',vertices,faces,
              collision=True,mass=HOOP_MASS/count)
    return spec


def get_pole_spec():
    spec=mujoco.MjSpec();body=spec.worldbody.add_body(name='red_pole')
    common=dict(rgba=POLE_COLOR,group=2,friction=(.6,.005,.0001),solref=(.006,1.))
    body.add_geom(name='red_pole_base',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=(0,0,BASE_HEIGHT/2),size=(BASE_RADIUS,BASE_HEIGHT/2),**common)
    tip_center=POLE_HEIGHT-POLE_RADIUS
    body.add_geom(name='red_pole_shaft',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=(0,0,(BASE_HEIGHT+tip_center)/2),
                  size=(POLE_RADIUS,(tip_center-BASE_HEIGHT)/2),**common)
    body.add_geom(name='red_pole_tip',type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  pos=(0,0,tip_center),size=(POLE_RADIUS,),**common)
    return spec


LEARNING_SCENE_VERSION = 'hoop_pole_learning_v1'
LEARNING_SCENE_VERSIONS = {'standard': LEARNING_SCENE_VERSION,
                           'tight': 'hoop_pole_learning_tight_v1'}


def learning_layout(seed):
    """Mild pose variation; the passive inspection layout stays unchanged."""
    rng=np.random.default_rng(seed)
    return {name:tuple(np.asarray(xy)+rng.uniform(-.005,.005,2))
            for name,xy in POSITIONS.items()}
