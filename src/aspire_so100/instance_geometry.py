"""CPU-only RGB-D instances and shape evidence; no simulator or catalog imports.

Initial shape classification assumes upright objects. Dimensions are measured
from the visible surfaces and tabletop; hidden geometry is explicitly inferred.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linear_sum_assignment


def inventory_color_mask(rgb, color):
    r, g, b = np.asarray(rgb, float).transpose(2, 0, 1)
    if color == 'red':
        return (r > 75) & (r > 1.65*g) & (r > 1.55*b) & (g < 1.5*b+10)
    if color == 'yellow':
        return (r > 100) & (g > 90) & (r < 1.5*g) & (g > 2*b)
    if color == 'green':
        return (g > 65) & (g > 1.45*r) & (g > 1.3*b)
    if color == 'blue':
        return (b > 65) & (b > 1.55*r) & (b > 1.2*g)
    raise ValueError('Unknown inventory color')


def inventory_points(frame):
    depth = np.asarray(frame['depth_m'], float)
    valid = np.asarray(frame['valid_depth'], bool) & np.isfinite(depth) & (depth > 0)
    v, u = np.indices(depth.shape)
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(frame['intrinsics']).T
    pose = np.asarray(frame['camera_to_world'])
    points = (rays*np.where(valid, depth, 0)[..., None]) @ pose[:3, :3].T + pose[:3, 3]
    return points, valid


def inventory_table(frame, points=None, valid=None):
    if points is None:
        points, valid = inventory_points(frame)
    r, g, b = np.asarray(frame['rgb'], float).transpose(2, 0, 1)
    table = valid & (r > 95) & (g > 65) & (r > 1.15*g) & (r < 1.65*g) & (g > 1.35*b)
    p = points[table]
    p = p[::max(1, len(p)//4000)]
    if len(p) < 100:
        raise ValueError('Insufficient tabletop depth')
    design = np.column_stack((p[:, :2], np.ones(len(p))))
    keep = abs(p[:, 2]-np.median(p[:, 2])) < .025
    for _ in range(4):
        if np.count_nonzero(keep) < 50:
            raise ValueError('Unreliable tabletop plane')
        plane = np.linalg.lstsq(design[keep], p[keep, 2], rcond=None)[0]
        keep = abs(p[:, 2]-design @ plane) < .003
    return plane


def depth_components(mask, points, max_step=.009, min_pixels=8):
    """Split even touching image regions when their 3-D depth is discontinuous."""
    pixels = np.flatnonzero(mask)
    if not len(pixels):
        return []
    indices = np.full(mask.shape, -1, int)
    indices.flat[pixels] = np.arange(len(pixels))
    rows, cols = [], []
    for a, b in ((indices[:, :-1], indices[:, 1:]), (indices[:-1], indices[1:])):
        if a.shape[0] == mask.shape[0]:
            distance = np.linalg.norm(points[:, :-1]-points[:, 1:], axis=2)
        else:
            distance = np.linalg.norm(points[:-1]-points[1:], axis=2)
        adjacent = (a >= 0) & (b >= 0) & (distance < max_step)
        rows.append(a[adjacent]); cols.append(b[adjacent])
    row, col = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(row)), (row, col)), shape=(len(pixels), len(pixels)))
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=count)
    result = []
    for label in np.flatnonzero(sizes >= min_pixels):
        component = np.zeros(mask.shape, bool)
        component.flat[pixels[labels == label]] = True
        result.append(component)
    return result


def instance_evidence(points, mask, plane):
    cloud = points[mask]
    lo, hi = np.percentile(cloud, [1, 99], axis=0)
    center = (lo+hi)/2
    bottom = float(center[:2] @ plane[:2]+plane[2])
    height = float(hi[2]-bottom)
    dimensions = np.r_[hi[:2]-lo[:2], height]
    center[2] = bottom+height/2
    dx = points[1:-1, 2:]-points[1:-1, :-2]
    dy = points[2:, 1:-1]-points[:-2, 1:-1]
    normal = np.cross(dx, dy)
    length = np.linalg.norm(normal, axis=2)
    interior = (mask[1:-1, 1:-1] & mask[1:-1, 2:] & mask[1:-1, :-2]
                & mask[2:, 1:-1] & mask[:-2, 1:-1] & (length > 1e-10))
    normals = normal[interior]/length[interior, None]
    normals = normals[::max(1, len(normals)//250)]
    nz = abs(normals[:, 2])
    sloped = float(np.mean((nz > .4) & (nz < .88))) if len(nz) else 0.
    sides = normals[nz < .3]
    planar = 0.
    remaining = sides.copy()
    for _ in range(2):
        if not len(remaining):
            break
        votes = abs(remaining @ remaining.T) > np.cos(np.deg2rad(9))
        chosen = votes[int(np.argmax(votes.sum(axis=1)))]
        planar += np.count_nonzero(chosen)/max(1, len(sides))
        remaining = remaining[~chosen]
    width = float(max(dimensions[:2]))
    ratio = height/max(width, .001)
    thin_ratio = float(min(dimensions[:2])/max(width, .001))
    # Scores expose competing hypotheses; low-information observations abstain.
    scores = {
        'cube': abs(ratio-1.)+.7*abs(thin_ratio-1.)+2*max(0., sloped-.25),
        'prism': abs(ratio-2.)+.7*abs(thin_ratio-1.)+2*max(0., sloped-.25),
        'thin_prism': abs(ratio-2.)+1.4*abs(thin_ratio-.5)+2*max(0., sloped-.25),
        'cylinder': abs(ratio-2.)+.7*abs(thin_ratio-1.)+4*max(0., planar-.55),
        'triangular_prism': abs(ratio-1.)+2*max(0., .7-sloped),
    }
    if len(sides) >= 8:
        scores['prism'] += 4*max(0., .65-planar)
    else:
        scores['cylinder'] += .6
    ordered = sorted(scores, key=scores.get)
    margin = float(scores[ordered[1]]-scores[ordered[0]])
    shape = ordered[0] if margin > .12 and scores[ordered[0]] < .75 and len(normals) >= 6 else 'unknown'
    return {'center': center, 'observed_center': np.median(cloud, axis=0),
            'dimensions': dimensions, 'half': np.maximum((hi[:2]-lo[:2])/2, .008),
            'shape': shape, 'shape_scores': scores, 'confidence_margin': margin,
            'features': {'height_ratio': ratio, 'width_ratio': thin_ratio, 'sloped_fraction': sloped,
                         'side_planarity': planar, 'side_samples': len(sides), 'normal_samples': len(normals)},
            'pixel_count': len(cloud)}


def inventory_instances(frame, plane=None, include_partial=False):
    points, valid = inventory_points(frame)
    if plane is None:
        plane = inventory_table(frame, points, valid)
    heights = points[:, :, 2]-points[:, :, :2] @ plane[:2]-plane[2]
    objects = []
    for color in ('red', 'yellow', 'blue', 'green'):
        mask = inventory_color_mask(frame['rgb'], color) & valid & (heights > .004) & (heights < .35)
        for component in depth_components(mask, points):
            estimate = instance_evidence(points, component, plane)
            # New objects need enough area to reject glints/tool markers. Existing
            # tracks may use narrower surfaces exposed beside the gripper.
            eligible = min(estimate['dimensions'][:2]) >= .006 and estimate['pixel_count'] >= 10
            if not eligible and (not include_partial or min(estimate['dimensions'][:2]) < .002):
                continue
            objects.append(dict(estimate, color=color, mask=component, initial_eligible=eligible))
    return objects, plane


class InstanceTracker:
    """Persistent IDs, explicit motion predictions, and bounded visual association."""
    def __init__(self, plane=None):
        self.plane = plane
        self.tracks = {}
        self.next_id = 0

    def update(self, frame, predictions=None):
        detections, self.plane = inventory_instances(frame, self.plane, include_partial=True)
        predictions = {} if predictions is None else predictions
        ids = list(self.tracks)
        costs = np.full((len(ids), len(detections)), 1e6)
        for i, key in enumerate(ids):
            track = self.tracks[key]
            predicted = np.asarray(predictions.get(key, track['observed_center']))
            gate = .065 if key in predictions else .035
            for j, detection in enumerate(detections):
                distance = np.linalg.norm(detection['observed_center']-predicted)
                if track['color'] == detection['color'] and distance < gate:
                    costs[i, j] = distance
        matched = set()
        for track in self.tracks.values():
            track['visible'] = False
        if costs.size:
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
                if costs[i, j] >= 1e6:
                    continue
                other_detections = np.delete(costs[i], j)
                other_tracks = np.delete(costs[:, j], i)
                if (np.any(other_detections < costs[i, j]+.008)
                        or np.any(other_tracks < costs[i, j]+.008)):
                    # Abstain instead of silently switching an ID under overlap.
                    continue
                key = ids[i]
                old = self.tracks[key]
                fresh = dict(detections[j], id=key, visible=True)
                # Retain shape identity through grasp-induced rotation/occlusion.
                if old['shape'] != 'unknown':
                    fresh['shape'] = old['shape']
                self.tracks[key] = fresh
                matched.add(j)
        for j, detection in enumerate(detections):
            if j not in matched and detection.get('initial_eligible', True):
                key = detection['color']+'_'+str(self.next_id)
                self.next_id += 1
                self.tracks[key] = dict(detection, id=key, visible=True)
        return [track for track in self.tracks.values() if track['visible']]

    def select(self, color, shape):
        matches = [t for t in self.tracks.values() if t['visible'] and t['color'] == color and t['shape'] == shape]
        if len(matches) != 1:
            raise ValueError('No unique visible '+color+' '+shape)
        return matches[0]['id']

    def isolated_frame(self, frame, key):
        track = self.tracks[key]
        if not track['visible']:
            raise ValueError('Selected instance is occluded: '+key)
        result = dict(frame)
        result['rgb'] = np.where(track['mask'][..., None], frame['rgb'], 0)
        result['valid_depth'] = np.asarray(frame['valid_depth']) & track['mask']
        return result
