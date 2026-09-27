"""Matching right-handed male/female thread surfaces.

Thread faces are render geometry, not individual collision surfaces. Convex nut
sectors and a smooth shaft handle bulk contact; FastenScrewsEnv supplies the
explicitly simplified helical coupling. Version 1 preserves historical replays.
"""

import math

import mujoco
import numpy as np

PAIR_COLORS = {'red': (.88, .08, .06, 1.), 'blue': (.06, .28, .92, 1.)}
SCREW_HEAD_RADIUS = .022
SCREW_HEAD_HEIGHT = .008
SCREW_CORE_RADIUS = .0085
SCREW_THREAD_RADIUS = .0105
SCREW_HEIGHT = .048
THREAD_PITCH = .004
THREAD_START = SCREW_HEAD_HEIGHT + .002
THREAD_HALF_WIDTH = .001
THREAD_RADIAL_CLEARANCE = .0003
THREAD_AXIAL_CLEARANCE = .00015
NUT_RADIUS = .024
LEGACY_NUT_BORE_RADIUS = .0125
NUT_BORE_RADIUS = SCREW_THREAD_RADIUS + THREAD_RADIAL_CLEARANCE
NUT_HEIGHT = .016
SCENE_VERSION = 'fasten_robodojo_v2'
ROBODOJO_LEGACY_VERSION = 'fasten_robodojo_v1'
ROBODOJO_VERSIONS = (ROBODOJO_LEGACY_VERSION, SCENE_VERSION)
STATIC_SCENE_VERSION = 'fasten_helical_v3'
THREADED_SCENE_VERSION = 'fasten_helical_v2'
LEGACY_SCENE_VERSION = 'fasten_helical_v1'


def check_version(version):
    if version not in (SCENE_VERSION, ROBODOJO_LEGACY_VERSION, STATIC_SCENE_VERSION, THREADED_SCENE_VERSION, LEGACY_SCENE_VERSION):
        raise ValueError(f'Unknown fastening scene version: {version}')


def nut_bore_radius(theta, z):
    """Female groove complementary to the bolt, phase zero in nut coordinates.

    A groove (larger bore radius), not an overlapping second male ridge. Both
    surfaces follow z = phase + pitch * theta/(2*pi). Clearance is radial and
    axial; these are intentionally coarse demonstration threads, not ISO parts.
    """
    phase = np.asarray(z) - THREAD_PITCH * np.asarray(theta)/(2*math.pi)
    distance = abs((phase+THREAD_PITCH/2) % THREAD_PITCH-THREAD_PITCH/2)
    profile = np.clip((THREAD_HALF_WIDTH+THREAD_AXIAL_CLEARANCE-distance)
                      / THREAD_HALF_WIDTH, 0., 1.)
    return SCREW_CORE_RADIUS + THREAD_RADIAL_CLEARANCE + (
        SCREW_THREAD_RADIUS-SCREW_CORE_RADIUS)*profile
# Origins are on bottom faces. The colors are crossed to make matching visible.
FASTENER_XY = {
    'red_screw': (-.085, -.22), 'blue_screw': (.085, -.22),
    'blue_nut': (-.085, -.34), 'red_nut': (.085, -.34),
}


def _mesh(spec, body, name, vertices, faces, color, *, collision=False, mass=0.):
    spec.add_mesh(name=name, uservert=np.asarray(vertices).ravel().tolist(),
                  userface=np.asarray(faces, dtype=int).ravel().tolist())
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                  rgba=color, mass=mass, group=3 if collision else 2,
                  contype=int(collision), conaffinity=int(collision),
                  friction=(1., .005, .0001), solref=(.006, 1.))


def _hex_radius(theta, radius):
    sector = theta % (math.pi / 3) - math.pi / 6
    return radius * math.cos(math.pi / 6) / np.cos(sector)


def _ring_mesh(segments=192, *, version=SCENE_VERSION):
    """Watertight beveled hex shell including its internal helical groove."""
    check_version(version)
    if version == LEGACY_SCENE_VERSION:
        segments = 96
    theta = np.arange(segments) * 2 * math.pi / segments
    outside = _hex_radius(theta, NUT_RADIUS)
    bevel = .0012
    profile = [
        (outside-bevel, 0.), (outside, bevel),
        (outside, NUT_HEIGHT-bevel), (outside-bevel, NUT_HEIGHT),
    ]
    if version == LEGACY_SCENE_VERSION:
        profile += [
            (np.full(segments, LEGACY_NUT_BORE_RADIUS+bevel), NUT_HEIGHT),
            (np.full(segments, LEGACY_NUT_BORE_RADIUS), NUT_HEIGHT-bevel),
            (np.full(segments, LEGACY_NUT_BORE_RADIUS), bevel),
            (np.full(segments, LEGACY_NUT_BORE_RADIUS+bevel), 0.),
        ]
    else:
        # Sample the bore as one continuous surface, with chamfered entrances.
        # 32 axial samples/pitch and 192/revolution avoid a block-built spiral.
        for z in np.linspace(NUT_HEIGHT, 0., round(NUT_HEIGHT/THREAD_PITCH*32)+1):
            radius = nut_bore_radius(theta, z)
            end_distance = min(z, NUT_HEIGHT-z)
            blend = max(0., 1.-end_distance/bevel)
            radius = (1.-blend)*radius + blend*(NUT_BORE_RADIUS+bevel)
            profile.append((radius, z))
    vertices = np.concatenate([
        np.column_stack((r*np.cos(theta), r*np.sin(theta), np.full(segments,z)))
        for r,z in profile
    ])
    faces = []
    for ring in range(len(profile)):
        following = (ring+1) % len(profile)
        for i in range(segments):
            j = (i+1) % segments
            a,b,c,d = ring*segments+i,ring*segments+j,following*segments+j,following*segments+i
            faces.extend(((a,b,c),(a,c,d)))
    return vertices, faces


def _thread_mesh():
    """A continuous triangular helical ridge."""
    start, end = THREAD_START, SCREW_HEIGHT-.003
    turns = (end-start) / THREAD_PITCH
    steps = math.ceil(turns*96)
    vertices = []
    for i in range(steps+1):
        theta = turns*2*math.pi*i/steps
        z = start+(end-start)*i/steps
        for radius, dz in ((SCREW_CORE_RADIUS,-THREAD_HALF_WIDTH),
                           (SCREW_THREAD_RADIUS,0.), (SCREW_CORE_RADIUS,THREAD_HALF_WIDTH)):
            vertices.append((radius*math.cos(theta),radius*math.sin(theta),z+dz))
    faces = [(0,2,1)]
    for i in range(steps):
        for j in range(3):
            a,b = 3*i+j,3*i+(j+1)%3
            c,d = b+3,a+3
            faces.extend(((a,b,c),(a,c,d)))
    faces.append((3*steps,3*steps+1,3*steps+2))
    return vertices,[(a,c,b) for a,b,c in faces]


def get_screw_spec(name, color, *, fixed=False, version=SCENE_VERSION):
    """Free upright bolt, resting on its hexagonal head with shaft upward."""
    check_version(version)
    if version in ROBODOJO_VERSIONS:
        from aspire_so100.robodojo_fasteners import get_screw_spec as original_screw
        return original_screw(name,color,fixed=fixed)
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    if not fixed:
        body.add_freejoint(name=f'{name}_joint')
    theta = np.arange(6)*math.pi/3
    vertices = [(SCREW_HEAD_RADIUS*math.cos(a),SCREW_HEAD_RADIUS*math.sin(a),z)
                for z in (0.,SCREW_HEAD_HEIGHT) for a in theta]
    faces = []
    for i in range(1,5):
        faces.extend(((0,i+1,i),(6,6+i,6+i+1)))
    for i in range(6):
        j = (i+1)%6
        faces.extend(((i,j,j+6),(i,j+6,i+6)))
    _mesh(spec,body,f'{name}_head',vertices,faces,color)
    # An inscribed support cylinder avoids mesh/box resting-contact jitter.
    body.add_geom(name=f'{name}_head_contact',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=(0,0,SCREW_HEAD_HEIGHT/2),
                  size=(SCREW_HEAD_RADIUS*math.cos(math.pi/6),SCREW_HEAD_HEIGHT/2),
                  rgba=color,mass=.025,group=3,friction=(1.,.005,.0001),
                  priority=1,solref=(.008,1.),solimp=(.99,.99,.001,.5,2.))
    body.add_geom(name=f'{name}_shaft',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=(0,0,(SCREW_HEAD_HEIGHT+SCREW_HEIGHT)/2),
                  size=(SCREW_CORE_RADIUS,(SCREW_HEIGHT-SCREW_HEAD_HEIGHT)/2),
                  rgba=color,mass=.015,group=2,friction=(1.,.005,.0001))
    vertices,faces = _thread_mesh()
    lighter = tuple(min(1.,c*1.12+.04) for c in color[:3])+(1.,)
    _mesh(spec,body,f'{name}_thread',vertices,faces,lighter)
    body.add_site(name=f'{name}_tip',pos=(0,0,SCREW_HEIGHT),size=(.001,),rgba=(0,0,0,0))
    return spec


def get_nut_spec(name, color, *, version=SCENE_VERSION):
    """Internally threaded hex nut with simplified convex contact sectors."""
    check_version(version)
    if version in ROBODOJO_VERSIONS:
        from aspire_so100.robodojo_fasteners import get_nut_spec as original_nut
        return original_nut(name,color)
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f'{name}_joint')
    vertices,faces = _ring_mesh(version=version)
    _mesh(spec,body,f'{name}_surface',vertices,faces,color)
    # One concave mesh would become a solid convex collision hull in MuJoCo.
    count = 24
    bore_radius = LEGACY_NUT_BORE_RADIUS if version == LEGACY_SCENE_VERSION else NUT_BORE_RADIUS
    faces = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),
             (0,1,5),(0,5,4),(1,2,6),(1,6,5),
             (2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    for i in range(count):
        a,b = 2*math.pi*i/count,2*math.pi*(i+1)/count
        radii = (_hex_radius(a,NUT_RADIUS),_hex_radius(b,NUT_RADIUS))
        xy = [(bore_radius*math.cos(a),bore_radius*math.sin(a)),
              (radii[0]*math.cos(a),radii[0]*math.sin(a)),
              (radii[1]*math.cos(b),radii[1]*math.sin(b)),
              (bore_radius*math.cos(b),bore_radius*math.sin(b))]
        vertices = [(x,y,z) for z in (0.,NUT_HEIGHT) for x,y in xy]
        _mesh(spec,body,f'{name}_contact_{i:02d}',vertices,faces,color,
              collision=True,mass=.03/count)
    body.add_site(name=f'{name}_bore',pos=(0,0,NUT_HEIGHT/2),size=(.001,),rgba=(0,0,0,0))
    return spec
