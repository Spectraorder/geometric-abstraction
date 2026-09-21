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
