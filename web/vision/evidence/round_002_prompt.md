You are the policy learner for a new SO-100 three-block manipulation experiment.
No prior policy or skill library is supplied. Use only this task contract, attached sensor images,
and previous attempts from THIS experiment when supplied below. Never use tools or inspect files.
Return JSON with hypothesis and code; code defines run(api). Only numpy, math, and scipy imports.
No file/network/process/introspection. Do not access private attributes. All diagnostics via api.note.
TASK: blue block stays on table; pick red and place on blue; then pick green and place on red.
All are free rigid bodies. Objects must be grasped, physically transported, released, and robot withdrawn.
After placing red, leave it untouched for at least 3 seconds before picking green. After green, withdraw
and wait at least 6 seconds. A separate evaluator observes support, alignment, release, and drift.
The policy cannot query evaluator or read any object ground truth. You receive only success/failure AFTER a trial.

SENSOR API:
api.camera() -> rgb uint8[240,320,3], depth_m float32[240,320], valid_depth bool,
intrinsics 3x3, camera_to_world 4x4, time. Ideal side-mounted calibrated RGB-D.
Depth is optical-axis z in meters, NOT range. For pixel (u,v):
world=R @ (inv(K) @ [u,v,1] * depth[v,u]) + t. Camera axes right,down,forward.
Use RGB segmentation and point clouds to estimate object positions/heights; NO preknown object coordinates or sizes.
The table and objects are visible in the attached initial image. Object colors are blue, red, green.
api.proprioception() -> joint_angles length5, tool_position world XYZ meters (robot forward kinematics), time.

ACTUATION API (robot SDK, no object perception or hidden grasp heuristics):
api.move_tool([x,y,z], seconds=1.5): position with soft preference for downward gripper,
box-constrained robot-model IK; may tilt near reach limits. It moves joints along a smooth interpolation.
Returns no object state or grasp result. Site is an approximate gripper center, not a guaranteed object center.
api.move_joints(five_angles, seconds=1.5) optional raw control; joint limits radians
[-1.92,1.92],[-3.32,.174],[-.174,3.14],[-1.66,1.66],[-2.79,2.79].
api.gripper(angle, seconds=.6): jaw angle in [-.174,1.75], larger is more open; .8 is open.
api.wait(seconds): advances physics, max10s per call. api.note(text): logs your perception/decision (no state oracle).
All motion durations 0.1..10s. All locations MUST come from current camera measurements, not constants keyed
by seed or object names. Known camera calibration and measured robot joint/tool pose are allowed.
No api.get_observation(), no api.check_structure(), no privileged object state, no forced attachment.
Pixel and pointcloud operations may use numpy and scipy.ndimage. Do not print; use api.note for diagnostics.
There is a 120s wall-time, 600 RPC call limit per trial. Robot base is fixed; workspace actions should be conservative.
Develop a physically plausible policy and adjust it using subsequent rendered outcomes and your logged estimates.

Previous attempt from this experiment:
import numpy as np
from scipy import ndimage

def run(api):
    def capture():
        s = api.camera()
        rgb = np.asarray(s['rgb'], dtype=float)
        d = np.asarray(s['depth_m'])
        valid = np.asarray(s['valid_depth'], dtype=bool) & np.isfinite(d) & (d > 0)
        v, u = np.indices(d.shape)
        rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(np.asarray(s['intrinsics'])).T
        T = np.asarray(s['camera_to_world'])
        xyz = (rays * d[..., None]) @ T[:3, :3].T + T[:3, 3]
        return rgb, xyz, valid

    def masks(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        return {
            'red': (r > 75) & (r > 1.65*g) & (r > 1.55*b),
            'green': (g > 65) & (g > 1.45*r) & (g > 1.3*b),
            'blue': (b > 65) & (b > 1.55*r) & (b > 1.2*g),
            'table': (r > 95) & (g > 65) & (r > 1.15*g) & (r < 1.65*g) & (g > 1.35*b)
        }

    rgb, xyz, valid = capture()
    tp = xyz[masks(rgb)['table'] & valid]
    if len(tp) < 100:
        api.note('Insufficient tabletop depth; stopping.')
        return
    tp = tp[::max(1, len(tp)//4000)]
    keep = np.abs(tp[:, 2] - np.median(tp[:, 2])) < 0.025
    for _ in range(4):
        q = tp[keep]
        A = np.column_stack((q[:, 0], q[:, 1], np.ones(len(q))))
        plane = np.linalg.lstsq(A, q[:, 2], rcond=None)[0]
        residual = tp[:, 2] - np.column_stack((tp[:, 0], tp[:, 1], np.ones(len(tp)))) @ plane
        keep = np.abs(residual) < 0.003
    def table_z(xy):
        return float(plane[0]*xy[0] + plane[1]*xy[1] + plane[2])

    def locate(name, size=None, near=None):
        rgb, xyz, valid = capture()
        mask = masks(rgb)[name] & valid
        labels, count = ndimage.label(mask)
        candidates = []
        for label in range(1, count+1):
            p = xyz[labels == label]
            if len(p) < 7:
                continue
            h = p[:, 2] - (p[:, 0]*plane[0] + p[:, 1]*plane[1] + plane[2])
            p = p[(h > 0.003) & (h < 0.35)]
            if len(p) < 7:
                continue
            score = float(len(p))
            if near is not None:
                distance = np.linalg.norm(np.median(p, axis=0) - near)
                if distance > 0.09:
                    continue
                score /= 1 + 100*distance
            candidates.append((score, p))
        if not candidates:
            raise RuntimeError('No reliable visible ' + name + ' block')
        p = max(candidates, key=lambda item: item[0])[1]
        ztop = float(np.percentile(p[:, 2], 90))
        top = p[np.abs(p[:, 2]-ztop) < 0.003]
        if len(top) < 5:
            top = p[p[:, 2] >= np.percentile(p[:, 2], 65)]
        xy = (np.percentile(top[:, :2], 3, axis=0) + np.percentile(top[:, :2], 97, axis=0))/2
        ztop = float(np.median(top[:, 2]))
        if size is None:
            size = ztop - table_z(xy)
            if not 0.009 < size < 0.065:
                raise RuntimeError('Unreliable measured block height')
        center = np.array([xy[0], xy[1], ztop-size/2])
        api.note(name + ' center=' + str(np.round(center, 4).tolist()) + ' size=' + str(round(float(size), 4)))
        return {'center': center, 'top': ztop, 'size': float(size)}

    def move(p, seconds=1.5):
        api.move_tool(np.asarray(p, dtype=float).tolist(), seconds=seconds)

    try:
        blocks = {name: locate(name) for name in ('blue', 'red', 'green')}
        home = np.asarray(api.proprioception()['tool_position'], dtype=float)
        safe_z = max(v['top'] for v in blocks.values()) + 0.12
        park = home.copy()
        park[2] = max(home[2], safe_z)
        api.gripper(0.8, seconds=0.6)
        move(park, 1.5)

        def stack(source_name, support_name):
            source = locate(source_name, blocks[source_name]['size'])
            support = locate(support_name, blocks[support_name]['size'])
            size = source['size']
            grasp = source['center'].copy()
            grasp[2] += 0.002
            above = grasp.copy()
            above[2] = max(safe_z, support['top'] + size + 0.075)
            api.gripper(0.8, seconds=0.5)
            move(above, 2.0)
            pre = grasp.copy()
            pre[2] += 0.025
            move(pre, 1.2)
            move(grasp, 1.2)
            api.gripper(-0.174, seconds=1.0)
            api.wait(0.4)
            lift = grasp.copy()
            lift[2] += 0.055
            move(lift, 1.6)
            api.wait(0.4)
            tool = np.asarray(api.proprioception()['tool_position'], dtype=float)
            held = locate(source_name, size, near=tool)
            rise = held['center'][2] - source['center'][2]
            if rise < 0.022:
                api.note('Lift was not confirmed; opening and stopping.')
                move(grasp, 1.5)
                api.gripper(0.8, seconds=0.7)
                move(above, 1.5)
                return False
            offset = held['center'] - tool
            api.note('Measured carried block offset=' + str(np.round(offset, 4).tolist()))
            destination = support['center'].copy()
            destination[2] = support['top'] + size/2 + 0.002
            target = destination - offset
            transit = tool.copy()
            transit[2] = max(above[2], target[2] + 0.065)
            move(transit, 1.5)
            overhead = target.copy()
            overhead[2] = transit[2]
            move(overhead, 2.2)
            preplace = target.copy()
            preplace[2] += 0.024
            move(preplace, 1.5)
            move(target, 1.8)
            api.wait(0.5)
            api.gripper(0.35, seconds=0.7)
            api.gripper(0.8, seconds=0.7)
            api.wait(0.4)
            retreat = target.copy()
            retreat[2] += 0.07
            move(retreat, 1.8)
            move(park, 2.0)
            return True

        if not stack('red', 'blue'):
            return
        api.wait(3.5)
        red = locate('red', blocks['red']['size'])
        blue = locate('blue', blocks['blue']['size'])
        alignment = np.linalg.norm(red['center'][:2] - blue['center'][:2])
        gap = red['top'] - red['size'] - blue['top']
        if alignment > 0.3*min(red['size'], blue['size']) or abs(gap) > 0.008:
            api.note('Red placement appears unstable or misaligned; leaving it untouched.')
            return
        if not stack('green', 'red'):
            return
        api.wait(6.5)
        api.note('Placement sequence completed; robot withdrawn and final settling wait finished.')
    except RuntimeError as error:
        api.note(str(error))

Trial feedback:
{
  "success": false,
  "policy_error": null,
  "policy_sim_seconds": 26.999999999995552,
  "notes": [
    "blue center=[0.0902, -0.2532, 0.4249] size=0.0499",
    "red center=[-0.116, -0.2139, 0.4199] size=0.0399",
    "green center=[-0.0994, -0.3207, 0.4199] size=0.0399",
    "red center=[-0.1158, -0.2107, 0.4199] size=0.0399",
    "blue center=[0.0913, -0.2535, 0.4249] size=0.0499",
    "red center=[-0.1061, -0.1996, 0.4675] size=0.0399",
    "Measured carried block offset=[0.0084, 0.0098, -0.0062]",
    "red center=[0.081, -0.2603, 0.4696] size=0.0399",
    "blue center=[0.1186, -0.2545, 0.4249] size=0.0499",
    "Red placement appears unstable or misaligned; leaving it untouched."
  ]
}

Design the next policy. Return complete code, not a patch.