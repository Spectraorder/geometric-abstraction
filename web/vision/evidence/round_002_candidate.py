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

    def move(p, seconds=1.5):
        api.move_tool(np.asarray(p, dtype=float).tolist(), seconds=seconds)

    def tool():
        return np.asarray(api.proprioception()['tool_position'], dtype=float)

    try:
        rgb, xyz, valid = capture()
        points = xyz[masks(rgb)['table'] & valid]
        if len(points) < 100:
            api.note('Insufficient tabletop depth.')
            return
        points = points[::max(1, len(points)//4000)]
        A = np.column_stack((points[:, :2], np.ones(len(points))))
        keep = np.abs(points[:, 2] - np.median(points[:, 2])) < 0.025
        for _ in range(4):
            if np.count_nonzero(keep) < 50:
                raise RuntimeError('Table fit unreliable.')
            plane = np.linalg.lstsq(A[keep], points[keep, 2], rcond=None)[0]
            keep = np.abs(points[:, 2] - A @ plane) < 0.003

        def table_z(xy):
            return float(plane[0]*xy[0] + plane[1]*xy[1] + plane[2])

        def locate(name, size=None, near=None):
            rgb, xyz, valid = capture()
            labels, count = ndimage.label(masks(rgb)[name] & valid)
            candidates = []
            for label in range(1, count+1):
                p = xyz[labels == label]
                if len(p) < 7:
                    continue
                h = p[:, 2] - (p[:, 0]*plane[0] + p[:, 1]*plane[1] + plane[2])
                p = p[(h > 0.003) & (h < 0.35)]
                if len(p) < 7:
                    continue
                distance = 0.0 if near is None else np.linalg.norm(np.median(p, axis=0)-near)
                if near is not None and distance > 0.10:
                    continue
                candidates.append((len(p)/(1+80*distance), p))
            if not candidates:
                raise RuntimeError('No reliable visible ' + name + ' block.')
            p = max(candidates, key=lambda item: item[0])[1]
            z = float(np.percentile(p[:, 2], 90))
            top = p[np.abs(p[:, 2]-z) < 0.003]
            if len(top) < 5:
                raise RuntimeError('Insufficient top surface for ' + name)
            xy = (np.percentile(top[:, :2], 3, axis=0) + np.percentile(top[:, :2], 97, axis=0))/2
            z = float(np.median(top[:, 2]))
            if size is None:
                size = z-table_z(xy)
                if not 0.009 < size < 0.065:
                    raise RuntimeError('Unreliable block size.')
            center = np.array([xy[0], xy[1], z-size/2])
            return {'center': center, 'top': z, 'size': float(size)}

        blocks = {name: locate(name) for name in ('blue', 'red', 'green')}
        for name, block in blocks.items():
            api.note(name + ' initial center=' + str(np.round(block['center'], 4).tolist()) + ' size=' + str(round(block['size'], 4)))
        home = tool()
        safe_z = max(b['top'] for b in blocks.values()) + 0.12
        park = home.copy()
        park[2] = max(park[2], safe_z)
        api.gripper(0.8, seconds=0.6)
        move(park)

        def stack(name, support):
            source = locate(name, blocks[name]['size'])
            size = source['size']
            grasp = source['center'].copy()
            grasp[2] += 0.002
            above = grasp.copy()
            above[2] = max(safe_z, support['top']+size+0.075)
            api.gripper(0.8, seconds=0.5)
            move(above, 1.8)
            pre = grasp.copy()
            pre[2] += 0.025
            move(pre, 1.1)
            move(grasp, 1.2)
            api.gripper(-0.174, seconds=1.0)
            api.wait(0.4)
            lift = grasp.copy()
            lift[2] += 0.055
            move(lift, 1.6)
            api.wait(0.4)
            held = locate(name, size, tool())
            if held['center'][2]-source['center'][2] < 0.022:
                api.note('Grasp not confirmed; returning and opening.')
                move(grasp)
                api.gripper(0.8, seconds=0.7)
                move(above)
                return False
            offset = held['center']-tool()
            desired = support['center'].copy()
            desired[2] = support['top']+size/2
            target = desired-offset
            transit = tool()
            transit[2] = max(above[2], target[2]+0.065)
            move(transit)
            overhead = target.copy()
            overhead[2] = transit[2]
            move(overhead, 2.0)
            preplace = target.copy()
            preplace[2] += 0.027
            move(preplace, 1.6)

            # Re-measure near the destination: grasp offset can change with IK tilt.
            for clearance, rounds in ((0.024, 3), (0.009, 3), (0.0035, 2)):
                goal = desired.copy()
                goal[2] += clearance
                for _ in range(rounds):
                    api.wait(0.18)
                    current_tool = tool()
                    measured = locate(name, size, current_tool)
                    error = goal-measured['center']
                    api.note(name + ' placement correction=' + str(np.round(error, 4).tolist()))
                    if np.linalg.norm(error[:2]) < 0.0025 and abs(error[2]) < 0.002:
                        break
                    delta = error.copy()
                    lateral = np.linalg.norm(delta[:2])
                    if lateral > 0.018:
                        delta[:2] *= 0.018/lateral
                    delta[2] = np.clip(delta[2], -0.018, 0.018)
                    # Keep lateral correction above contact when alignment is poor.
                    if lateral > 0.006 and clearance < 0.012:
                        delta[2] = max(delta[2], desired[2]+0.014-measured['center'][2])
                    move(current_tool+delta, 1.1)

            measured = locate(name, size, tool())
            lateral = np.linalg.norm(measured['center'][:2]-desired[:2])
            gap = measured['center'][2]-size/2-support['top']
            api.note(name + ' release alignment=' + str(round(float(lateral), 4)) + ' gap=' + str(round(float(gap), 4)))
            if lateral > 0.18*min(size, support['size']) or not -0.002 < gap < 0.014:
                api.note('Placement confidence insufficient; returning carried block to its measured source.')
                offset = measured['center']-tool()
                retreat = tool()
                retreat[2] = above[2]
                move(retreat)
                back = source['center']-offset
                high = back.copy()
                high[2] = above[2]
                move(high, 1.8)
                back[2] += 0.004
                move(back, 1.8)
                api.gripper(0.8, seconds=0.8)
                move(high)
                move(park)
                return False
            api.gripper(0.35, seconds=0.8)
            api.gripper(0.8, seconds=0.8)
            api.wait(0.4)
            retreat = tool()
            retreat[2] += 0.075
            move(retreat, 1.8)
            move(park, 1.8)
            return True

        # Capture blue while its complete top is visible; later occlusion biases its bounds.
        blue_support = locate('blue', blocks['blue']['size'])
        if not stack('red', blue_support):
            return
        api.wait(3.5)
        red_a = locate('red', blocks['red']['size'])
        api.wait(0.6)
        red_b = locate('red', blocks['red']['size'])
        drift = np.linalg.norm(red_b['center']-red_a['center'])
        gap = red_b['top']-red_b['size']-blue_support['top']
        alignment = np.linalg.norm(red_b['center'][:2]-blue_support['center'][:2])
        api.note('Settled red: drift=' + str(round(float(drift), 4)) + ' gap=' + str(round(float(gap), 4)) + ' anchor alignment=' + str(round(float(alignment), 4)))
        if drift > 0.004 or abs(gap) > 0.007 or alignment > 0.38*min(red_b['size'], blue_support['size']):
            api.note('Red support uncertain; leaving it untouched.')
            return
        if not stack('green', red_b):
            return
        api.wait(6.5)
        api.note('Both blocks transported and released; robot withdrawn and final settling wait completed.')
    except RuntimeError as error:
        api.note(str(error))
