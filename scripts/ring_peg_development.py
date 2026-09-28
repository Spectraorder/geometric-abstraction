"""Opt-in ring development repairs; historical training sources remain unchanged.

The registration implementation follows ring_peg_geometry, but tests background
at the intersection of each camera ray with the cap plane. All updates still
require current surface evidence. Installation is process-local and not thread-safe.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import numpy as np
from scipy.optimize import minimize_scalar
import ring_peg_geometry as geometry
import ring_peg_runtime as runtime
from ring_peg_robot import PadActuator, vector
from aspire_so100.instance_geometry import inventory_color_mask as original_mask
from aspire_so100.rgbd import OverheadRGBD


def color_mask(rgb, color):
    if color != 'magenta':
        return original_mask(rgb, color)
    r, g, b = np.asarray(rgb, float).transpose(2, 0, 1)
    return (r > 70) & (b > 65) & (r > 1.5*g) & (b > 1.5*g) & (r < 1.8*b) & (b < 1.8*r)


def project_pixels_to_plane(frame, normal, level):
    """Intersect calibrated camera rays, not measured background points, with a plane."""
    v, u = np.indices(frame['depth_m'].shape)
    pose = np.asarray(frame['camera_to_world'])
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(frame['intrinsics']).T @ pose[:3, :3].T
    denominator = rays@normal
    scale = np.divide(level-pose[:3, 3]@normal, denominator,
                      out=np.full_like(denominator, np.nan), where=abs(denominator)>1e-8)
    scale[scale <= 0] = np.nan
    return pose[:3, 3]+scale[..., None]*rays


class CorrectedPadActuator(PadActuator):
    def plan_grasp(self, fixed_pad_m, closing_direction, jaw):
        """Validate orientation at the actual fixed-pad solution, including its offset."""
        point, axis = vector(fixed_pad_m), vector(closing_direction)
        length = np.linalg.norm(axis)
        if length < 1e-8:
            raise ValueError('Closing direction must be nonzero')
        axis = axis/length
        current = float(self.env.data.qpos[self.env.arm_qpos_addresses[-1]])
        coarse = []
        for roll in np.linspace(-2.79, 2.79, 57):
            try:
                result = self.preview(point, float(roll), jaw)
                coarse.append((float(result['closing_axis']@axis), float(roll)))
            except (RuntimeError, ValueError):
                continue
        candidates = []
        def loss(roll):
            try:
                return 1-float(self.plan_pad(point, float(roll), jaw)['closing_axis']@axis)
            except (RuntimeError, ValueError):
                return 2.
        for _, roll in sorted(coarse, reverse=True)[:4]:
            fitted = minimize_scalar(loss, bounds=(max(-2.79, roll-.2), min(2.79, roll+.2)),
                                     method='bounded', options={'xatol': .0005, 'maxiter': 24})
            try:
                plan = self.plan_pad(point, float(fitted.x), jaw)
            except (RuntimeError, ValueError):
                continue
            if float(plan['closing_axis']@axis) >= np.cos(np.deg2rad(6)):
                plan['fixed_pad_error_m'] = float(np.linalg.norm(plan['fixed_pad_m']-point))
                plan['closing_alignment'] = float(plan['closing_axis']@axis)
                candidates.append((abs(plan['roll']-current), plan))
        if not candidates:
            raise ValueError('No reachable roll aligns the requested closing direction')
        return min(candidates, key=lambda row: row[0])[1]


class DevelopmentSensors(runtime.RingSensors):
    def __init__(self, env, output):
        super().__init__(env, output)
        self.robot = CorrectedPadActuator(env)

    def camera(self, view='side'):
        if self.env._overhead_rgbd is None:
            self.env._overhead_rgbd = OverheadRGBD(self.env.model, width=640, height=480, pixel_aligned=True)
        return super().camera(view)


@contextmanager
def installed():
    """Install only inside a dedicated development process."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(geometry, 'inventory_color_mask', color_mask))
        stack.enter_context(patch.object(runtime, 'register_ring', register_ring))
        yield


def register_ring(frame,color,table,model):
    """Register an initialized annular footprint to independently visible current edges."""
    from scipy.optimize import least_squares
    points,valid,top,origin,normal,basis,pixel=geometry.cap(frame,color,table)
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
    projected = project_pixels_to_plane(frame, normal, level)
    uv = ((projected-origin)@basis).reshape(-1,2)
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
