"""Common starter with policy-owned perception adapted from stack_three_blocks_vision.

Both arms start from this exact code. Coordinates are computed from camera pixels,
not supplied by Session. Only shared pick/place actuation is delegated. The original
cube visual estimator is retained; it is not a validated mixed-shape rigid fitter.
"""
import numpy as np
from scipy import ndimage

def capture(api):
    s = api.camera()
    rgb = np.asarray(s['rgb'], dtype=float)
    d = np.asarray(s['depth_m'])
    valid = np.asarray(s['valid_depth'], dtype=bool) & np.isfinite(d) & (d > 0)
    v, u = np.indices(d.shape)
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(np.asarray(s['intrinsics'])).T
    T = np.asarray(s['camera_to_world'])
    xyz = rays * d[..., None] @ T[:3, :3].T + T[:3, 3]
    return (rgb, xyz, valid)

def masks(rgb):
    r, g, b = (rgb[..., 0], rgb[..., 1], rgb[..., 2])
    return {'red': (r > 75) & (r > 1.65 * g) & (r > 1.55 * b), 'green': (g > 65) & (g > 1.45 * r) & (g > 1.3 * b), 'blue': (b > 65) & (b > 1.55 * r) & (b > 1.2 * g), 'table': (r > 95) & (g > 65) & (r > 1.15 * g) & (r < 1.65 * g) & (g > 1.35 * b), 'yellow': (r > 100) & (g > 90) & (r < 1.5 * g) & (g > 2 * b)}

def locate(api, plane, name, size=None, near=None):
    rgb, xyz, valid = capture(api)
    labels, count = ndimage.label(masks(rgb)[name] & valid)
    candidates = []
    for label in range(1, count + 1):
        p = xyz[labels == label]
        if len(p) < 7:
            continue
        h = p[:, 2] - (p[:, 0] * plane[0] + p[:, 1] * plane[1] + plane[2])
        p = p[(h > 0.003) & (h < 0.35)]
        if len(p) < 7:
            continue
        distance = 0.0 if near is None else np.linalg.norm(np.median(p, axis=0) - near)
        if near is not None and distance > 0.1:
            continue
        candidates.append((len(p) / (1 + 80 * distance), p))
    if not candidates:
        raise RuntimeError('No reliable visible ' + name + ' block.')
    p = max(candidates, key=lambda item: item[0])[1]
    z = float(np.percentile(p[:, 2], 90))
    top = p[np.abs(p[:, 2] - z) < 0.003]
    if len(top) < 5:
        raise RuntimeError('Insufficient top surface for ' + name)
    xy = (np.percentile(top[:, :2], 3, axis=0) + np.percentile(top[:, :2], 97, axis=0)) / 2
    z = float(np.median(top[:, 2]))
    if size is None:
        size = z - table_z(plane, xy)
        if not 0.009 < size < 0.065:
            raise RuntimeError('Unreliable block size.')
    center = np.array([xy[0], xy[1], z - size / 2])
    return {'center': center, 'top': z, 'size': float(size)}

def fit_table(api):
    rgb, xyz, valid = capture(api)
    points = xyz[masks(rgb)['table'] & valid]
    if len(points) < 100:
        raise RuntimeError('Insufficient tabletop depth')
    points = points[::max(1, len(points)//4000)]
    A = np.column_stack((points[:, :2], np.ones(len(points))))
    keep = np.abs(points[:, 2]-np.median(points[:, 2])) < .025
    for _ in range(4):
        if np.count_nonzero(keep) < 50:
            raise RuntimeError('Table fit unreliable')
        plane = np.linalg.lstsq(A[keep], points[keep, 2], rcond=None)[0]
        keep = np.abs(points[:, 2]-A@plane) < .003
    return plane


def table_z(plane, xy):
    return float(plane[0]*xy[0]+plane[1]*xy[1]+plane[2])


def run(api):
    goal = api.task(); nodes = goal['nodes']; by_key = {node['key']: node for node in nodes}
    session = Session(api, nodes)
    plane = fit_table(api)
    initial = {node['key']: locate(api, plane, node['color']) for node in nodes}
    unit_key = goal['length_unit']['observed_cube']; unit = initial[unit_key]['size']
    anchor = initial[unit_key]['center'][:2]-unit*np.asarray(by_key[unit_key]['grid'])
    for node in nodes:
        if node['key'] == unit_key:
            continue
        if not node['supports']:
            xy = anchor+unit*np.asarray(node['grid']); point = np.r_[xy, table_z(plane, xy)]
        else:
            supports = [locate(api, plane, by_key[key]['color'], initial[key]['size']) for key in node['supports']]
            offset = np.asarray(node['grid'])-np.mean([by_key[key]['grid'] for key in node['supports']], axis=0)
            xy = np.mean([fit['center'][:2] for fit in supports], axis=0)+unit*offset
            point = np.r_[xy, max(fit['top'] for fit in supports)]
        session.checkpoint(node['color'], 'before_pick')
        picked = session.pick(node['color'], transit_z=float(point[2]+.085))
        session.checkpoint(node['color'], 'pick', action_status=picked['status'])
        if not picked['held']:
            api.note(node['key']+': '+picked['status'])
            return
        result = session.place(point)
        session.checkpoint(node['color'], 'place', action_status=result['status'])
        api.note(node['key']+': '+result['status'])
        if result['status'] != 'released':
            return
        api.wait(.5)
        session.checkpoint(node['color'], 'settled')
