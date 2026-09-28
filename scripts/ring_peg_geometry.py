"""Sensor-only planar outer/inner contours for a simple perforated extrusion.

No named polygon, mesh, simulator pose, object dimension or oracle is used.
Complete resolved boundaries are required; missing current evidence is rejected.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from aspire_so100.instance_geometry import inventory_points, inventory_table, inventory_color_mask
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'prototypes/polygon_prism'))
from geometry import prism_simplify, prism_fit_edges, prism_contour_points


def polygon_centroid(vertices):
    q = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0]*q[:, 1]-q[:, 0]*vertices[:, 1]
    area = np.sum(cross)/2
    if area <= 1e-8:
        raise ValueError('Unresolved polygon area')
    return np.sum((vertices+q)*cross[:, None], axis=0)/(6*area)


def cap(frame, color, table):
    points, valid = inventory_points(frame)
    elevation = points[..., 2]-points[..., :2]@table[:2]-table[2]
    mask = inventory_color_mask(frame['rgb'], color) & valid & (elevation > .004)
    dx, dy = points[1:-1, 2:]-points[1:-1, :-2], points[2:, 1:-1]-points[:-2, 1:-1]
    n = np.cross(dx, dy)
    lengths = np.linalg.norm(n, axis=2)
    interior = mask[1:-1, 1:-1] & mask[1:-1, 2:] & mask[1:-1, :-2] & mask[2:, 1:-1] & mask[:-2, 1:-1]
    good = interior & (lengths > 1e-10) & (abs(n[..., 2])/np.maximum(lengths, 1e-12) > .7)
    samples = points[1:-1, 1:-1][good]
    if len(samples) < 8:
        raise ValueError('Insufficient current cap samples')
    origin = np.mean(samples, axis=0)
    _, singular, vh = np.linalg.svd(samples-origin, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    rms = np.sqrt(np.mean(((samples-origin)@normal)**2))
    if normal[2] < .7 or singular[1] < .001 or rms > .0007:
        raise ValueError('No single resolved upward cap plane')
    top = mask & (abs((points-origin)@normal) < max(.0003, 4*rms))
    x = np.array([1., 0., 0.])-normal*normal[0]
    x /= np.linalg.norm(x)
    basis = np.column_stack((x, np.cross(normal, x)))
    pixel = float(np.median(np.linalg.norm(dx[good], axis=1))/2)
    return points, valid, top, origin, normal, basis, pixel


def boundary(frame, mask, origin, normal, basis, pixel):
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError('Clipped contour')
    cloud = prism_contour_points(frame, mask, normal, float(origin@normal))
    planar = (cloud-origin)@basis
    hull = planar[ConvexHull(planar).vertices]
    polygon = prism_simplify(hull, max(.00025, 1.3*pixel))
    if not 3 <= len(polygon) <= 12:
        raise ValueError('Unresolved polygon contour')
    polygon = prism_fit_edges(planar, polygon, pixel)
    return polygon, origin+polygon@basis.T


def observe_ring(frame, color, table, height=None):
    points, valid, top, origin, normal, basis, pixel = cap(frame, color, table)
    filled = ndimage.binary_fill_holes(top)
    labels, count = ndimage.label(filled & ~top)
    holes = [labels == i for i in range(1, count+1) if np.count_nonzero(labels == i) >= 12]
    if len(holes) != 1:
        raise ValueError('Require one resolved closed inner boundary')
    hole = holes[0]
    # A hole requires observed depth below the cap, not merely missing color/depth.
    inside = ndimage.binary_erosion(hole)
    if np.count_nonzero(inside & valid & (((points-origin)@normal) < -.003)) < .6*np.count_nonzero(inside):
        raise ValueError('Inner region lacks current depth evidence of an opening')
    outer2, outer = boundary(frame, filled, origin, normal, basis, pixel)
    inner2, inner = boundary(frame, hole, origin, normal, basis, pixel)
    center2 = polygon_centroid(inner2)
    center = origin+basis@center2
    edges = np.roll(inner2, -1, axis=0)-inner2
    outward = np.column_stack((edges[:, 1], -edges[:, 0]))/np.linalg.norm(edges, axis=1)[:, None]
    radius = float(np.min(np.sum((inner2-center2)*outward, axis=1)))
    if radius < 3*pixel:
        raise ValueError('Opening too small to resolve')
    if height is None:
        table_normal = np.r_[-table[:2], 1.]
        height = float((table_normal@center-table[2])/(table_normal@normal))
        if not .004 < height < .10 or normal[2] < .97:
            raise ValueError('Initialize height only from an upright table-supported ring')
    outer_edges = np.roll(outer, -1, axis=0)-outer
    directions = np.cross(outer_edges, normal)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    walls = [{'outer_mid_m': ((a+b)/2-normal*height/2).tolist(), 'outward_normal': n.tolist()}
             for a,b,n in zip(outer, np.roll(outer,-1,axis=0),directions)]
    return {'status': 'current_estimate', 'time': float(frame['time']), 'frame': 'world', 'units': 'meters',
            'hole': {'center_m': center.tolist(), 'axis': normal.tolist(), 'inscribed_radius_m': radius,
                     'inner_boundary_m': inner.tolist(), 'evidence': 'current_rgbd_cap_boundary'},
            'outer_boundary_m': outer.tolist(), 'height_m': height, 'walls': walls,
            'bottom_evidence': 'inferred_initialized_constant_extrusion',
            'boundary_resolution_m': pixel, 'center_m': (center-normal*height/2).tolist(),
            'minimum_z_m': float(min(outer[:,2].min(), (outer-normal*height)[:,2].min()))}


def observe_peg(frame, color, table):
    _, _, top, origin, normal, basis, pixel = cap(frame, color, table)
    if normal[2] < .97:
        raise ValueError('Require an upright peg cap')
    filled = ndimage.binary_fill_holes(top)
    cloud = prism_contour_points(frame, filled, normal, float(origin@normal))
    uv = (cloud-origin)@basis
    a = np.column_stack((2*uv, np.ones(len(uv))))
    fit = np.linalg.lstsq(a, np.sum(uv**2, axis=1), rcond=None)[0]
    radius = float(np.sqrt(fit[2]+fit[:2]@fit[:2]))
    residual = np.sqrt(np.mean((np.linalg.norm(uv-fit[:2], axis=1)-radius)**2))
    if radius < 2*pixel or residual > pixel:
        raise ValueError('Unresolved circular peg cap')
    center = origin+basis@fit[:2]
    return {'status': 'current_estimate', 'time': float(frame['time']),
            'top_center_m': center.tolist(), 'axis': normal.tolist(), 'radius_m': radius,
            'evidence': 'current_rgbd_circular_cap', 'boundary_resolution_m': pixel}


def initialized_model(record):
    normal=np.asarray(record['hole']['axis']);x=np.array([1.,0.,0.])-normal*normal[0];x/=np.linalg.norm(x)
    basis=np.column_stack((x,np.cross(normal,x)));center=np.asarray(record['hole']['center_m'])
    return {'outer':(np.asarray(record['outer_boundary_m'])-center)@basis,
            'inner':(np.asarray(record['hole']['inner_boundary_m'])-center)@basis,
            'height':record['height_m'], 'resolution':record['boundary_resolution_m']}


def register_ring(frame,color,table,model):
    """Register an initialized annular footprint to independently visible current edges."""
    from scipy.optimize import least_squares
    points,valid,top,origin,normal,basis,pixel=cap(frame,color,table)
    level=float(origin@normal);rows=[]
    for axis in (0,1):
        a=(slice(None,-1),slice(None)) if axis==0 else (slice(None),slice(None,-1))
        b=(slice(1,None),slice(None)) if axis==0 else (slice(None),slice(1,None))
        different=top[a]!=top[b]
        outside=np.where(top[a][...,None],points[b],points[a])
        outside_valid=np.where(top[a],valid[b],valid[a])
        selected=different & outside_valid & ((outside@normal-level)<-.002)
        v,u=np.where(selected)
        rows.extend(np.column_stack((u+(axis==1)*.5,v+(axis==0)*.5,np.ones(len(u)))))
    if len(rows)<20:raise ValueError('Insufficient unoccluded current boundary samples')
    pixels=np.asarray(rows);pose=np.asarray(frame['camera_to_world'])
    rays=pixels@np.linalg.inv(frame['intrinsics']).T@pose[:3,:3].T
    cloud=pose[:3,3]+((level-pose[:3,3]@normal)/(rays@normal))[:,None]*rays
    observed=(cloud-origin)@basis
    outer,inner=np.asarray(model['outer']),np.asarray(model['inner'])
    starts=np.vstack((outer,inner));ends=np.vstack((np.roll(outer,-1,axis=0),np.roll(inner,-1,axis=0)))
    def rotate(theta):return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    def residual(params):
        rot=rotate(params[0]);a=starts@rot.T+params[1:];edges=(ends-starts)@rot.T
        delta=observed[:,None]-a
        fractions=np.sum(delta*edges,axis=2)/np.sum(edges**2,axis=1)
        distances=np.linalg.norm(delta-np.clip(fractions,0,1)[...,None]*edges,axis=2)
        return np.min(distances,axis=1)
    fits=[]
    for angle in np.linspace(-np.pi,np.pi,36,endpoint=False):
        fitted=least_squares(residual,np.r_[angle,np.mean(observed,axis=0)],max_nfev=80,
                             loss='soft_l1',f_scale=max(.0007,pixel*.5))
        rms=float(np.sqrt(np.mean(residual(fitted.x)**2)))
        if rms<max(.0013,pixel*.65):fits.append((rms,fitted))
    if not fits:raise ValueError('No current boundary registration fits initialized shape')
    fits.sort(key=lambda pair:pair[0]);best=fits[0][1]
    eig=np.linalg.eigvalsh(best.jac.T@best.jac)
    if eig[0]<.002 or eig[-1]/eig[0]>1e6:raise ValueError('Boundary pose registration is underconstrained')
    # Multiple candidates are acceptable only if they reconstruct the same shape.
    transformed=lambda fit: starts@rotate(fit.x[0]).T+fit.x[1:]
    expected=transformed(best)
    for rms,fit in fits[1:]:
        if rms>fits[0][0]+.0002:continue
        other=transformed(fit)
        if np.max(np.min(np.linalg.norm(expected[:,None]-other[None],axis=2),axis=1))>2*max(pixel,model['resolution']):
            raise ValueError('Current boundary admits materially different completions')
    def inside(uv,polygon):
        edge=np.roll(polygon,-1,axis=0)-polygon;d=uv[:,None]-polygon
        return np.all(edge[None,:,0]*d[:,:,1]-edge[None,:,1]*d[:,:,0]>=0,axis=1)
    rot=rotate(best.x[0]);out2=outer@rot.T+best.x[1:];in2=inner@rot.T+best.x[1:]
    # Reject a footprint that covers observed deeper background. Foreground
    # occluders remain unknown; they are not fabricated ring surface evidence.
    uv=(points-origin)@basis;uv=uv.reshape(-1,2)
    expected_cap=inside(uv,out2)&~inside(uv,in2)
    below=((points-origin)@normal).ravel()<-.003
    valid_flat=valid.ravel()
    if np.count_nonzero(expected_cap & below & valid_flat)>max(5,.10*np.count_nonzero(expected_cap & valid_flat)):
        raise ValueError('Registered shape covers visible background')
    height=float(model['height']);outer3=origin+out2@basis.T;inner3=origin+in2@basis.T
    center=origin+basis@best.x[1:];edges=np.roll(outer3,-1,axis=0)-outer3
    directions=np.cross(edges,normal);directions/=np.linalg.norm(directions,axis=1)[:,None]
    iedges=np.roll(inner,-1,axis=0)-inner
    inormals=np.column_stack((iedges[:,1],-iedges[:,0]))/np.linalg.norm(iedges,axis=1)[:,None]
    return {'status':'current_estimate','time':float(frame['time']),'frame':'world','units':'meters',
            'hole':{'center_m':center.tolist(),'axis':normal.tolist(),
                    'inscribed_radius_m':float(np.min(np.sum(inner*inormals,axis=1))),
                    'inner_boundary_m':inner3.tolist(),'evidence':'initialized_shape_registered_to_current_edges'},
            'outer_boundary_m':outer3.tolist(),'height_m':height,
            'walls':[{'outer_mid_m':((a+b)/2-normal*height/2).tolist(),'outward_normal':n.tolist()}
                     for a,b,n in zip(outer3,np.roll(outer3,-1,axis=0),directions)],
            'bottom_evidence':'inferred_initialized_constant_extrusion',
            'boundary_resolution_m':pixel,'registration_rms_m':fits[0][0],'current_edge_samples':len(rows),
            'center_m':(center-normal*height/2).tolist(),
            'minimum_z_m':float(min(outer3[:,2].min(),(outer3-normal*height)[:,2].min()))}
