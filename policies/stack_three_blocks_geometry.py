"""Cube geometry from calibrated RGB-D only; hidden corners are model estimates.

This module deliberately uses only numerical dependencies so it can be bundled
into the isolated policy worker without granting access to simulator state.
"""
import numpy as np
from scipy import ndimage


def rgbd_points(frame):
    depth = np.asarray(frame['depth_m'], dtype=float)
    valid = np.asarray(frame['valid_depth'], dtype=bool) & np.isfinite(depth) & (depth > 0)
    v, u = np.indices(depth.shape)
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(frame['intrinsics']).T
    pose = np.asarray(frame['camera_to_world'], dtype=float)
    points = (rays * np.where(valid, depth, 0)[..., None]) @ pose[:3, :3].T + pose[:3, 3]
    return points, valid


def color_mask(rgb, color):
    r, g, b = np.asarray(rgb, dtype=float).transpose(2, 0, 1)
    if color == 'red':
        return (r > 75) & (r > 1.65 * g) & (r > 1.55 * b)
    if color == 'green':
        return (g > 65) & (g > 1.45 * r) & (g > 1.3 * b)
    if color == 'blue':
        return (b > 65) & (b > 1.55 * r) & (b > 1.2 * g)
    raise ValueError('Expected red, green, or blue')


def support_margin(point, top_vertices, normal):
    """Signed distance to the nearest edge of an inferred square support face."""
    corners = np.asarray(top_vertices, dtype=float)
    center = np.mean(corners, axis=0)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    x = corners[0] - center
    x = x / np.linalg.norm(x)
    y = np.cross(normal, x)
    uv = np.column_stack(((corners - center) @ x, (corners - center) @ y))
    order = np.argsort(np.arctan2(uv[:, 1], uv[:, 0]))
    corners = corners[order]
    edges = np.roll(corners, -1, axis=0) - corners
    inward = np.cross(normal, edges)
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    return float(np.min(np.sum((np.asarray(point) - corners) * inward, axis=1)))


def estimate_cube_geometry(frame, color, side_m, near=None, table_plane=None):
    """Fit visible face planes and infer a solid cube's centroid and 8 vertices.

    side_m is a positive side length previously measured from an unobstructed
    RGB-D view. No object pose, segmentation IDs, or scene dimensions are used.
    Three perpendicular observed planes constrain all three center coordinates.
    With fewer faces, missing coordinates use visible extents and are explicitly
    marked unreliable under occlusion. Returned vertices are always inferred.
    """
    if not np.isfinite(side_m) or side_m <= 0:
        raise ValueError('A positive measured cube side is required')
    points, valid = rgbd_points(frame)
    if table_plane is None:
        r, g, b = np.asarray(frame['rgb'], dtype=float).transpose(2, 0, 1)
        table = valid & (r > 95) & (g > 65) & (r > 1.15*g) & (r < 1.65*g) & (g > 1.35*b)
        sample = points[table][::max(1, int(np.count_nonzero(table)) // 3000)]
        if len(sample) < 50:
            raise ValueError('Insufficient tabletop depth')
        design = np.column_stack((sample[:, :2], np.ones(len(sample))))
        keep = abs(sample[:, 2] - np.median(sample[:, 2])) < .025
        for _ in range(4):
            if np.count_nonzero(keep) < 30:
                raise ValueError('Unreliable tabletop plane')
            table_plane = np.linalg.lstsq(design[keep], sample[keep, 2], rcond=None)[0]
            keep = abs(sample[:, 2] - design @ table_plane) < .003
    height = points[..., 2] - points[..., :2] @ np.asarray(table_plane)[:2] - table_plane[2]
    labels, count = ndimage.label(color_mask(frame['rgb'], color) & valid & (height > .004) & (height < .35))
    candidates = []
    for label in range(1, count + 1):
        mask = labels == label
        sample = points[mask]
        if len(sample) < 12:
            continue
        distance = 0.0 if near is None else float(np.linalg.norm(np.median(sample, axis=0) - near))
        if distance < .12:
            candidates.append((len(sample) / (1 + 60 * distance), mask))
    if not candidates:
        raise ValueError('No usable visible cube: ' + color)
    mask = max(candidates, key=lambda item: item[0])[1]
    cloud = points[mask]
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    normals = np.cross(dx, dy)
    length = np.linalg.norm(normals, axis=2)
    interior = (mask[1:-1, 1:-1] & mask[1:-1, 2:] & mask[1:-1, :-2]
                & mask[2:, 1:-1] & mask[:-2, 1:-1] & (length > 1e-10))
    samples = points[1:-1, 1:-1][interior]
    normals = normals[interior] / length[interior, None]
    # Bound fitting work even for a close-up image or a spurious large mask.
    stride = max(1, int(np.ceil(len(samples) / 500)))
    samples, normals = samples[::stride], normals[::stride]
    if len(samples) < 6:
        raise ValueError('Too few interior depth samples: ' + color)
    camera = np.asarray(frame['camera_to_world'])[:3, 3]
    normals *= np.where(np.sum(normals * (camera - samples), axis=1) >= 0, 1, -1)[:, None]

    # Dominant normal clusters, followed by a plane fit. Silhouette/depth-edge
    # pixels are excluded above; blended normals at cube edges are outliers.
    faces = []
    available = np.ones(len(samples), dtype=bool)
    for _ in range(6):
        indices = np.flatnonzero(available)
        if len(indices) < 4:
            break
        directions = normals[indices]
        votes = directions @ directions.T > np.cos(np.deg2rad(12))
        seed = int(np.argmax(np.sum(votes, axis=1)))
        members = indices[votes[seed]]
        available[members] = False
        if len(members) < 4:
            continue
        face = samples[members]
        center = np.mean(face, axis=0)
        _, singular, vh = np.linalg.svd(face - center, full_matrices=False)
        if singular[1] < .0005:
            continue
        normal = vh[-1]
        if np.dot(normal, camera - center) < 0:
            normal = -normal
        rms = float(np.sqrt(np.mean(((face - center) @ normal) ** 2)))
        if rms > .0008:
            continue
        if any(abs(np.dot(normal, f['normal'])) > .3 for f in faces):
            continue
        faces.append({'normal': normal, 'point': center, 'samples': len(face), 'rms_m': rms})
        if len(faces) == 3:
            break
    if not faces:
        raise ValueError('No reliable face planes: ' + color)

    axes = [f['normal'] for f in faces]
    if len(axes) == 1:
        seed_axis = np.eye(3)[int(np.argmin(np.abs(axes[0])))]
        axis = seed_axis - np.dot(seed_axis, axes[0]) * axes[0]
        axes.append(axis / np.linalg.norm(axis))
    if len(axes) == 2:
        axes.append(np.cross(axes[0], axes[1]))
    basis = np.column_stack(axes)
    u, _, vh = np.linalg.svd(basis)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    local = cloud @ rotation
    center_local = (np.percentile(local, 1, axis=0) + np.percentile(local, 99, axis=0)) / 2
    constrained = np.zeros(3, dtype=bool)
    plane_rms = []
    for face in faces:
        axis = int(np.argmax(np.abs(rotation.T @ face['normal'])))
        sign = float(np.sign(np.dot(rotation[:, axis], face['normal'])))
        aligned = normals @ (sign * rotation[:, axis]) > np.cos(np.deg2rad(15))
        level = float(np.median(samples[aligned] @ rotation[:, axis]))
        center_local[axis] = level - sign * side_m / 2
        constrained[axis] = True
        plane_rms.append(float(np.sqrt(np.mean((samples[aligned] @ rotation[:, axis] - level) ** 2))))
    centroid = rotation @ center_local
    signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    vertices = centroid + (signs * side_m / 2) @ rotation.T
    residual = np.min(np.abs(np.abs((cloud - centroid) @ rotation) - side_m / 2), axis=1)
    inside = np.all(np.abs((cloud - centroid) @ rotation) <= side_m / 2 + .002, axis=1)
    inlier_fraction = float(np.mean((residual < .0015) & inside))
    reliable = bool(np.all(constrained) and max(plane_rms) < .0008 and inlier_fraction > .8)
    up_axis = int(np.argmax(np.abs(rotation[2])))
    up = rotation[:, up_axis] * np.sign(rotation[2, up_axis])
    top_center = centroid + up * side_m / 2
    return {
        'centroid': centroid, 'surface_centroid': np.mean(cloud, axis=0),
        'vertices': vertices, 'vertices_kind': 'inferred_cube_model',
        'rotation': rotation, 'side_m': float(side_m),
        'top_center': top_center, 'top_normal': up,
        'top_vertices': vertices[signs[:, up_axis] == np.sign(rotation[2, up_axis])],
        'bottom_z': float(np.min(vertices[:, 2])), 'top_z': float(np.max(vertices[:, 2])),
        'observed_face_count': len(faces), 'constrained_axes': constrained,
        'plane_rms_m': float(max(plane_rms)), 'inlier_fraction': inlier_fraction,
        'reliable': reliable, 'pixel_count': len(cloud),
    }


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


        legacy_locate = locate

        def locate(name, model=None, near=None):
            # The baseline measures side length from the initial RGB-D top/table
            # separation. Reuse that measurement, never the simulator dimensions.
            estimate = legacy_locate(name, model, near)
            try:
                geometry = estimate_cube_geometry(api.camera(), name,
                    estimate['model']['height'], near=near, table_plane=plane)
            except ValueError:
                return estimate
            estimate['geometry_reliable'] = geometry['reliable']
            if not geometry['reliable']:
                return estimate
            side = geometry['side_m']
            estimate.update(center=geometry['centroid'], top=geometry['top_center'][2],
                normal=geometry['top_normal'], bottom=geometry['bottom_z'],
                extent_z=(geometry['top_z']-geometry['bottom_z'])/2,
                dims=np.array([side, side]), vertices=geometry['vertices'],
                top_vertices=geometry['top_vertices'])
            return estimate

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
        release_centers = {}
        contact_gap_min = -0.005
        api.note('Testing consistent final contact gap tolerance of -5 mm.')

        def perceive(name, near=None):
            for attempt in range(3):
                try:
                    return locate(name, blocks[name]['model'], near)
                except RuntimeError:
                    if attempt == 2:
                        raise
                    api.wait(0.15)

        def stack(name, support, placement_bias=None):
            source = perceive(name)
            aim_xy = support['center'][:2].copy()
            if placement_bias is not None:
                aim_xy += placement_bias
            grasp = source['center'].copy()
            grasp[2] += 0.30*source['height']
            api.note(name+' elevated grasp offset='+str(round(0.30*source['height'], 4)))
            api.note(name+' placement bias='+str(np.round(aim_xy-support['center'][:2], 4).tolist()))
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
            target[:2] = aim_xy
            target[2] = support['top']+held['extent_z']+0.03
            command = target-offset
            high = tool()
            high[2] = above[2]
            move(high)
            high[:2] = command[:2]
            move(high, 2.0)
            move(command, 1.6)
            for clearance, rounds in ((0.024, 4), (0.011, 3), (0.001, 7)):
                for _ in range(rounds):
                    api.wait(0.15)
                    current = tool()
                    measured = perceive(name, current)
                    error_xy = aim_xy-measured['center'][:2]
                    lateral = float(np.linalg.norm(error_xy))
                    gap = measured['bottom']-support['top']
                    api.note(name+' alignment to aim='+str(round(lateral, 4))+' gap='+str(round(gap, 4)))
                    if clearance == 0.001 and lateral < 0.003 and contact_gap_min <= gap <= 0.0025:
                        break
                    error_z = clearance-gap
                    if lateral < 0.002 and abs(error_z) < 0.0015:
                        break
                    delta = np.r_[error_xy, error_z]
                    if lateral > 0.015:
                        delta[:2] *= 0.015/lateral
                    delta[2] = np.clip(delta[2], -0.004 if clearance == 0.001 else -0.014, 0.014)
                    if lateral > 0.005 and gap < 0.012:
                        delta[:2] = 0.0
                        delta[2] = min(0.014, 0.014-gap)
                    move(current+delta, 1.3)
            measured = perceive(name, tool())
            displacement = measured['center'][:2]-support['center'][:2]
            aim_error = measured['center'][:2]-aim_xy
            gap = measured['bottom']-support['top']
            api.note(name+' release offset='+str(np.round(displacement, 4).tolist())+' gap='+str(round(gap, 4)))
            unsafe_overlap = placement_bias is not None and np.any(np.abs(displacement) > support['dims']/2-0.004)
            if support.get('geometry_reliable', False):
                unsafe_overlap = support_margin(measured['center'], support['top_vertices'], support['normal']) < .004
            if np.linalg.norm(aim_error) > 0.0055 or not contact_gap_min <= gap <= 0.004 or unsafe_overlap:
                offset = measured['center']-tool()
                high = tool()
                high[2] = above[2]
                move(high)
                back = source['center']-offset
                high[:2] = back[:2]
                move(high, 1.8)
                back[2] += measured['extent_z']-source['height']/2+0.005
                move(back, 1.8)
                api.gripper(0.8, seconds=0.8)
                move(high)
                move(park)
                api.note('Placement uncertain; returned payload to source.')
                return False
            release_centers[name] = measured['center'].copy()
            api.gripper(0.8, seconds=1.5)
            api.wait(0.8)
            api.gripper(1.4, seconds=1.0)
            api.wait(1.2)
            try:
                opened = perceive(name)
                api.note(name+' after opening displacement='+str(np.round(opened['center']-measured['center'], 4).tolist()))
            except RuntimeError:
                api.note(name+' obscured after opening.')
            retreat = tool()
            retreat[2] += 0.025
            move(retreat, 2.0)
            api.wait(0.5)
            retreat = tool()
            retreat[2] += 0.06
            move(retreat, 2.0)
            move(park, 1.8)
            try:
                withdrawn = perceive(name)
                api.note(name+' after withdrawal center='+str(np.round(withdrawn['center'], 4).tolist()))
            except RuntimeError:
                api.note(name+' obscured after withdrawal.')
            return True

        blue = perceive('blue')
        if not stack('red', blue):
            return
        api.wait(3.5)
        red_a = perceive('red')
        api.wait(0.6)
        red = perceive('red')
        drift = float(np.linalg.norm(red['center']-red_a['center']))
        api.note('Retaining unobstructed preplacement blue support measurement.')
        gap = red['bottom']-blue['top']
        displacement = np.abs(red['center'][:2]-blue['center'][:2])
        stable = np.all(displacement < blue['dims']/2-0.004)
        api.note('Settled red drift='+str(round(drift, 4))+' gap='+str(round(gap, 4))+' normal_z='+str(round(float(red['normal'][2]), 4))+' offset='+str(np.round(displacement, 4).tolist()))
        if drift > 0.003 or not -0.012 <= gap <= 0.006 or red['normal'][2] < 0.985 or not stable:
            api.note('Red support uncertain; leaving it untouched.')
            return

        release_shift = red['center'][:2]-release_centers['red'][:2]
        bias_limit = 0.30*float(np.min(np.r_[red['dims'], blocks['green']['dims']]))
        bias = -1.35*release_shift.copy()
        magnitude = float(np.linalg.norm(bias))
        if magnitude > bias_limit:
            bias *= bias_limit/magnitude
        api.note('Measured red release-to-settled XY shift='+str(np.round(release_shift, 4).tolist())+'; amplified green compensation='+str(np.round(bias, 4).tolist()))
        if not stack('green', red, bias):
            return
        api.wait(6.5)
        api.note('Both blocks physically transported and released; robot withdrawn; final settling wait completed.')
        final = perceive('green')
        api.note('Observed final green center='+str(np.round(final['center'], 4).tolist()))
    except RuntimeError as error:
        api.note(str(error))
