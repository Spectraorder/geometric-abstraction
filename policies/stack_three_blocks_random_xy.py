import numpy as np
from scipy import ndimage


def run(api):
    def tool():
        return np.asarray(api.proprioception()['tool_position'], dtype=float)

    def move(p, seconds=1.5):
        api.move_tool(np.asarray(p, dtype=float).tolist(), seconds=seconds)

    def capture():
        s = api.camera()
        rgb = np.asarray(s['rgb'], dtype=float)
        d = np.asarray(s['depth_m'], dtype=float)
        valid = np.asarray(s['valid_depth'], dtype=bool) & np.isfinite(d) & (d > 0)
        v, u = np.indices(d.shape)
        rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(np.asarray(s['intrinsics'])).T
        T = np.asarray(s['camera_to_world'])
        xyz = (rays*d[..., None]) @ T[:3, :3].T + T[:3, 3]
        return rgb, xyz, valid

    def masks(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        return {
            'red': (r > 75) & (r > 1.65*g) & (r > 1.55*b),
            'green': (g > 65) & (g > 1.45*r) & (g > 1.3*b),
            'blue': (b > 65) & (b > 1.55*r) & (b > 1.2*g),
            'table': (r > 95) & (g > 65) & (r > 1.15*g) & (r < 1.65*g) & (g > 1.35*b)
        }

    try:
        rgb, xyz, valid = capture()
        p = xyz[masks(rgb)['table'] & valid]
        if len(p) < 100:
            raise RuntimeError('Insufficient tabletop depth.')
        p = p[::max(1, len(p)//4000)]
        A = np.column_stack((p[:, :2], np.ones(len(p))))
        keep = np.abs(p[:, 2]-np.median(p[:, 2])) < 0.025
        for _ in range(4):
            if np.count_nonzero(keep) < 50:
                raise RuntimeError('Table fit unreliable.')
            plane = np.linalg.lstsq(A[keep], p[keep, 2], rcond=None)[0]
            keep = np.abs(p[:, 2]-A @ plane) < 0.003

        def table_z(xy):
            return float(np.dot(plane[:2], xy)+plane[2])

        def locate(name, model=None, near=None):
            rgb, xyz, valid = capture()
            labels, count = ndimage.label(masks(rgb)[name] & valid)
            choices = []
            for k in range(1, count+1):
                m = labels == k
                q = xyz[m]
                if len(q) < 8:
                    continue
                h = q[:, 2]-q[:, :2] @ plane[:2]-plane[2]
                if np.median(h) < 0.004 or np.median(h) > 0.32:
                    continue
                distance = 0.0 if near is None else np.linalg.norm(np.median(q, axis=0)-near)
                if distance > 0.12:
                    continue
                choices.append((len(q)/(1+60*distance), m))
            if not choices:
                raise RuntimeError('No visible '+name+' block.')
            m = max(choices, key=lambda item: item[0])[1]
            q = xyz[m]
            dx = xyz[1:-1, 2:]-xyz[1:-1, :-2]
            dy = xyz[2:, 1:-1]-xyz[:-2, 1:-1]
            n = np.cross(dx, dy)
            lengths = np.linalg.norm(n, axis=2)
            n /= np.maximum(lengths[..., None], 1e-12)
            n = np.where(n[..., 2:3] < 0, -n, n)
            interior = m[1:-1, 1:-1] & m[1:-1, 2:] & m[1:-1, :-2] & m[2:, 1:-1] & m[:-2, 1:-1]
            good = interior & (lengths > 1e-10) & (n[..., 2] > 0.78)
            seeds = xyz[1:-1, 1:-1][good]
            if len(seeds) >= 5:
                normal = np.median(n[good], axis=0)
                normal /= np.linalg.norm(normal)
                level = float(np.median(seeds @ normal))
                face = q[np.abs(q @ normal-level) < 0.0022]
                for _ in range(2):
                    if len(face) < 6:
                        break
                    mean = np.mean(face, axis=0)
                    _, _, vh = np.linalg.svd(face-mean, full_matrices=False)
                    fitted = vh[-1]
                    if fitted[2] < 0:
                        fitted = -fitted
                    if fitted[2] < 0.78:
                        break
                    normal = fitted
                    level = float(mean @ normal)
                    face = q[np.abs(q @ normal-level) < 0.0022]
            else:
                z = np.percentile(q[:, 2], 90)
                face = q[np.abs(q[:, 2]-z) < 0.0025]
                normal = np.array([0.0, 0.0, 1.0])
                level = float(np.median(face[:, 2])) if len(face) else float(z)
            if len(face) < 5:
                raise RuntimeError('Top face occluded for '+name)
            ex = np.array([1.0, 0.0, 0.0])-normal*normal[0]
            ex /= np.linalg.norm(ex)
            ey = np.cross(normal, ex)
            uv = np.column_stack((face @ ex, face @ ey))
            lo, hi = np.percentile(uv, [2, 98], axis=0)
            middle = (lo+hi)/2
            top_center = middle[0]*ex+middle[1]*ey+level*normal
            if model is None:
                height = float(top_center[2]-table_z(top_center[:2]))
                dims = hi-lo
                if not 0.009 < height < 0.065 or np.min(dims) < 0.012:
                    raise RuntimeError('Unreliable dimensions for '+name)
                model = {'height': height, 'dims': dims}
            height = model['height']
            center = top_center-normal*height/2
            extent = float((abs(ex[2])*model['dims'][0]+abs(ey[2])*model['dims'][1]+normal[2]*height)/2)
            return {'center': center, 'top': float(top_center[2]), 'normal': normal,
                    'bottom': float(center[2]-extent), 'extent_z': extent,
                    'height': height, 'dims': model['dims'], 'model': model}

        blocks = {name: locate(name) for name in ('blue', 'red', 'green')}
        safe_z = max(b['top'] for b in blocks.values())+0.12
        park = tool()
        park[2] = max(park[2], safe_z)
        api.gripper(0.8, seconds=0.6)
        move(park)
        for name in blocks:
            try:
                blocks[name] = locate(name)
            except RuntimeError:
                pass
            b = blocks[name]
            api.note(name+' measured center='+str(np.round(b['center'], 4).tolist())+' dimensions='+str(np.round(np.r_[b['dims'], b['height']], 4).tolist()))
        safe_z = max(b['top'] for b in blocks.values())+0.12

        def perceive(name, near=None):
            for attempt in range(3):
                try:
                    return locate(name, blocks[name]['model'], near)
                except RuntimeError:
                    if attempt == 2:
                        raise
                    api.wait(0.15)

        def stack(name, support):
            source = perceive(name)
            grasp = source['center'].copy()
            grasp[2] += 0.002
            above = grasp.copy()
            above[2] = max(safe_z, support['top']+source['height']+0.075)
            api.gripper(0.8, seconds=0.5)
            move(above, 1.8)
            pre = grasp.copy()
            pre[2] += 0.025
            move(pre, 1.1)
            move(grasp, 1.2)
            api.gripper(-0.174, seconds=1.0)
            api.wait(0.35)
            lift = grasp.copy()
            lift[2] += 0.06
            move(lift, 1.7)
            held = perceive(name, tool())
            if held['center'][2]-source['center'][2] < 0.025:
                move(grasp)
                api.gripper(0.8, seconds=0.5)
                move(above)
                move(park)
                api.note('Grasp not confirmed; withdrawn.')
                return False
            offset = held['center']-tool()
            target = support['center'].copy()
            target[2] = support['top']+held['extent_z']+0.03
            command = target-offset
            high = tool()
            high[2] = above[2]
            move(high)
            high[:2] = command[:2]
            move(high, 2.0)
            move(command, 1.6)
            for clearance, rounds in ((0.024, 4), (0.011, 3), (0.003, 5)):
                for _ in range(rounds):
                    api.wait(0.15)
                    current = tool()
                    measured = perceive(name, current)
                    error_xy = support['center'][:2]-measured['center'][:2]
                    lateral = float(np.linalg.norm(error_xy))
                    gap = measured['bottom']-support['top']
                    api.note(name+' alignment='+str(round(lateral, 4))+' gap='+str(round(gap, 4)))
                    if clearance == 0.003 and lateral < 0.0035 and -0.003 <= gap <= 0.006:
                        break
                    error_z = clearance-gap
                    if lateral < 0.002 and abs(error_z) < 0.0015:
                        break
                    delta = np.r_[error_xy, error_z]
                    if lateral > 0.015:
                        delta[:2] *= 0.015/lateral
                    delta[2] = np.clip(delta[2], -0.008 if clearance == 0.003 else -0.014, 0.014)
                    if lateral > 0.005 and gap < 0.012:
                        delta[:2] = 0.0
                        delta[2] = min(0.014, 0.014-gap)
                    move(current+delta, 1.1)
            measured = perceive(name, tool())
            displacement = measured['center'][:2]-support['center'][:2]
            gap = measured['bottom']-support['top']
            api.note(name+' release offset='+str(np.round(displacement, 4).tolist())+' gap='+str(round(gap, 4)))
            if np.linalg.norm(displacement) > 0.0055 or not -0.0035 <= gap <= 0.010:
                offset = measured['center']-tool()
                high = tool()
                high[2] = above[2]
                move(high)
                back = source['center']-offset
                high[:2] = back[:2]
                move(high, 1.8)
                back[2] += measured['extent_z']-source['height']/2+0.005
                move(back, 1.8)
                api.gripper(0.8, seconds=0.3)
                move(high)
                move(park)
                api.note('Placement uncertain; returned payload to source.')
                return False
            api.gripper(0.8, seconds=0.25)
            api.wait(0.45)
            retreat = tool()
            retreat[2] += 0.085
            move(retreat, 1.8)
            move(park, 1.8)
            return True

        blue = perceive('blue')
        if not stack('red', blue):
            return
        api.wait(3.5)
        red_a = perceive('red')
        api.wait(0.6)
        red = perceive('red')
        drift = float(np.linalg.norm(red['center']-red_a['center']))
        old_blue_top = blue['top']
        try:
            fresh_blue = perceive('blue')
            blue['top'] = fresh_blue['top']
            api.note('Remeasured blue top change='+str(round(blue['top']-old_blue_top, 4)))
        except RuntimeError:
            api.note('Blue top occluded; retaining preplacement support measurement.')
        gap = red['bottom']-blue['top']
        displacement = np.abs(red['center'][:2]-blue['center'][:2])
        stable = np.all(displacement < blue['dims']/2-0.004)
        api.note('Settled red drift='+str(round(drift, 4))+' gap='+str(round(gap, 4))+' normal_z='+str(round(float(red['normal'][2]), 4))+' offset='+str(np.round(displacement, 4).tolist()))
        if drift > 0.003 or not -0.012 <= gap <= 0.006 or red['normal'][2] < 0.985 or not stable:
            api.note('Red support uncertain; leaving it untouched.')
            return
        if not stack('green', red):
            return
        api.wait(6.5)
        api.note('Both blocks physically transported and released; robot withdrawn; final settling wait completed.')
        final = perceive('green')
        api.note('Observed final green center='+str(np.round(final['center'], 4).tolist()))
    except RuntimeError as error:
        api.note(str(error))