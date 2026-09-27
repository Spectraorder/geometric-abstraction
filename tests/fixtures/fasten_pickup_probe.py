"""Hand-written infrastructure diagnostic, NOT an LLM-learned policy.

Only RGB-D, robot kinematics and public commands are used. No object-state oracle.
The learning campaign must generate its own policy after this gate is validated.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def detect(frame):
    rgb = np.asarray(frame['rgb'], float)/255.
    depth = frame['depth_m']
    valid = frame['valid_depth'] & np.isfinite(depth) & (depth > 0)
    v, u = np.indices(depth.shape)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(frame['intrinsics']).T
    transform = frame['camera_to_world']
    xyz = (rays*depth[..., None]) @ transform[:3, :3].T + transform[:3, 3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    table_pixels = valid & (r > .35) & (g > .20) & (r > 1.25*g) & (g > 1.45*b)
    heights = xyz[..., 2][table_pixels]
    if len(heights) < 30:
        raise RuntimeError('Table not visible')
    bins, counts = np.unique(np.round(heights/.002), return_counts=True)
    peak = bins[np.argmax(counts)]*.002
    table = float(np.median(heights[abs(heights-peak) < .004]))
    foreground = valid & (xyz[..., 2] > table+.003) & (xyz[..., 2] < table+.20)
    masks = {'red': (r > .30) & (r > 1.65*g) & (r > 1.5*b),
             'blue': (b > .28) & (b > 1.6*r) & (b > 1.35*g),
             'yellow': (r > .35) & (g > .24) & (g > .55*r) & (g < 1.3*r)
                       & (b < .5*np.minimum(r, g))}
    objects = {}
    for color, mask in masks.items():
        points = xyz[foreground & mask]
        if len(points) < 8:
            objects[color] = []
            continue
        # Cluster in world XY so disconnected shaft/head pixels belong to one
        # object rather than being mistaken for a second, lower object (a nut).
        edges = cKDTree(points[:, :2]).query_pairs(.009, output_type='ndarray')
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                           shape=(len(points), len(points)))
        count, labels = connected_components(graph, directed=False)
        regions = []
        for k in range(count):
            p = points[labels == k]
            if len(p) < 8:
                continue
            top = float(np.percentile(p[:, 2], 95))
            upper = p[p[:, 2] > top-.003]
            lo, hi = np.min(upper[:, :2], axis=0), np.max(upper[:, :2], axis=0)
            regions.append(dict(xy=(lo+hi)/2, top=top, span=hi-lo, pixels=len(p)))
        objects[color] = regions
    return table, objects


def classify(regions, table):
    if len(regions) < 2:
        raise RuntimeError('Need two separate objects of the same color')
    bolt = max(regions, key=lambda r: r['top'])
    nuts = [r for r in regions if r['top'] < bolt['top']-.01
            and .006 < r['top']-table < .03
            and np.linalg.norm(r['xy']-bolt['xy']) > .045]
    if len(nuts) != 1:
        raise RuntimeError('Ambiguous nut/bolt classification')
    return nuts[0], bolt


def run(api):
    home = np.array(api.proprioception()['joint_angles'])
    api.gripper(.8, seconds=.4)
    api.wait(.5)
    table, scene = detect(api.camera())
    pairs = {c: classify(scene[c], table) for c in ('red', 'blue', 'yellow')}
    for c, (nut, bolt) in pairs.items():
        api.note(c+' visual nut='+str(nut['xy'].tolist())+' bolt='+str(bolt['xy'].tolist()))

    def plan(target, preferred=None):
        default = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        orientations = [] if preferred is None else [preferred]
        for a in np.linspace(-np.pi, np.pi, 8, endpoint=False):
            c, s = np.cos(a), np.sin(a)
            orientations.append(np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ default)
        candidates = []
        for rotation in orientations:
            p = api.plan_tool(target.tolist(), orientation=np.asarray(rotation).tolist())
            tilt = np.arccos(np.clip(abs(np.array(p['tool_rotation'])[2, 1]), 0., 1.))
            if p['position_error'] < .002 and tilt < np.deg2rad(15):
                candidates.append((p['position_error']+.03*tilt, p))
        if not candidates:
            raise RuntimeError('No precise, upright robot-only IK plan')
        return min(candidates, key=lambda item: item[0])[1]

    def execute(p, seconds=1.):
        api.move_joints(np.asarray(p['joint_angles']).tolist(), seconds=seconds)
        api.wait(.4)
        actual = api.proprioception()
        # Distinguish servo tracking from the IK approximation, instead of
        # conflating them and relaxing the combined tolerance to hide error.
        tracking = np.linalg.norm(np.array(actual['tool_position'])-p['tool_position'])
        api.note('IK error=%.5f tracking error=%.5f' % (p['position_error'], tracking))
        if tracking > .003:
            raise RuntimeError('Robot tracking gate failed before closing')
        return actual

    # One real pickup is enough for this diagnostic. Test all color identities
    # across shuffled layouts rather than implementing the full screw task.
    nut, bolt = pairs['red']
    # The robot tool marker is above the finger tips, not at their midpoint.
    # A midpoint-height marker target drives the fixed finger into the table.
    center = np.r_[nut['xy'], nut['top']+.007]
    grip = plan(center)
    rotation = np.array(grip['tool_rotation'])
    execute(plan(center+[0., 0., .04], rotation), 1.5)
    for dz in (.03, .02, .01, 0.):
        execute(plan(center+[0., 0., dz], rotation), .6)
    api.note('Diagnostic grasp: closing jaws')
    api.gripper(-.174, seconds=1.)
    api.wait(.5)
    execute(plan(center+[0., 0., .05], rotation), 1.5)
    api.wait(2.)
    api.camera()
    api.note('Diagnostic completed: evaluator must verify real lift and hold; commands alone are not success')
