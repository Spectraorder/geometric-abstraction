"""CPU instance-based inventory stacking using only visual observations and robot state.

Target: red thin prism; support: blue cube. Initial objects are upright and
axis aligned. Ambiguous roles stop before grasping. Triangular prisms are
obstacles, but are not selected for relocation by this controller.
"""
import numpy as np
from scipy import ndimage
from aspire_so100.cube_geometry import estimate_cube_geometry, support_margin
from aspire_so100.cuboid_geometry import estimate_cuboid_geometry
from aspire_so100.gripper_geometry import estimate_gripper_geometry
from aspire_so100.instance_geometry import InstanceTracker, inventory_color_mask as color_mask

def run(api):

    def tool():
        return np.asarray(api.proprioception()['tool_position'], dtype=float)

    def move(p, seconds=1.5):
        api.move_tool(np.asarray(p, dtype=float).tolist(), seconds=seconds)
    last_frame = [None]
    last_robot = [None]
    tracker = InstanceTracker()
    grasp_anchors = {}

    def remember_grasp(name, robot=None):
        robot = api.proprioception() if robot is None else robot
        grasp_anchors[name] = (tracker.tracks[name]['observed_center'].copy(), robot)

    def predicted_grasps(robot=None):
        if not grasp_anchors:
            return {}
        robot = api.proprioception() if robot is None else robot
        position = np.asarray(robot['tool_position'])
        predictions = {}
        for name, (center, previous) in grasp_anchors.items():
            rotation = np.asarray(robot['tool_rotation']) @ np.asarray(previous['tool_rotation']).T
            predictions[name] = position + rotation @ (center - np.asarray(previous['tool_position']))
        return predictions

    def release_track(name):
        if name in grasp_anchors:
            tracker.tracks[name]['observed_center'] = predicted_grasps()[name]
            grasp_anchors.pop(name)


    def capture():
        s = api.camera()
        last_frame[0] = s
        if tracker.plane is not None:
            robot = api.proprioception() if grasp_anchors else None
            last_robot[0] = robot
            tracker.update(s, predicted_grasps(robot))
            for name in grasp_anchors:
                if tracker.tracks[name]['visible']:
                    remember_grasp(name, robot)
        rgb = np.asarray(s['rgb'], dtype=float)
        d = np.asarray(s['depth_m'], dtype=float)
        valid = np.asarray(s['valid_depth'], dtype=bool) & np.isfinite(d) & (d > 0)
        v, u = np.indices(d.shape)
        rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(np.asarray(s['intrinsics'])).T
        T = np.asarray(s['camera_to_world'])
        xyz = rays * d[..., None] @ T[:3, :3].T + T[:3, 3]
        return (rgb, xyz, valid)

    def masks(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        result = {name: color_mask(rgb,name) for name in ('red','yellow','blue','green')}
        result['table'] = (r > 95) & (g > 65) & (r > 1.15*g) & (r < 1.65*g) & (g > 1.35*b)
        return result

    try:
        rgb, xyz, valid = capture()
        p = xyz[masks(rgb)['table'] & valid]
        if len(p) < 100:
            raise RuntimeError('Insufficient tabletop depth.')
        p = p[::max(1, len(p) // 4000)]
        A = np.column_stack((p[:, :2], np.ones(len(p))))
        keep = np.abs(p[:, 2] - np.median(p[:, 2])) < 0.025
        for _ in range(4):
            if np.count_nonzero(keep) < 50:
                raise RuntimeError('Table fit unreliable.')
            plane = np.linalg.lstsq(A[keep], p[keep, 2], rcond=None)[0]
            keep = np.abs(p[:, 2] - A @ plane) < 0.003

        tracker.plane = plane

        def table_z(xy):
            return float(np.dot(plane[:2], xy) + plane[2])

        def locate(name, model=None, near=None):
            rgb, xyz, valid = capture()
            try:
                selected = tracker.isolated_frame(last_frame[0], name)
            except ValueError as error:
                raise RuntimeError(str(error)) from error
            labels, count = ndimage.label(selected['valid_depth'])
            choices = []
            for k in range(1, count + 1):
                m = labels == k
                q = xyz[m]
                if len(q) < 8:
                    continue
                h = q[:, 2] - q[:, :2] @ plane[:2] - plane[2]
                if np.median(h) < 0.004 or np.median(h) > 0.32:
                    continue
                distance = 0.0 if near is None else np.linalg.norm(np.median(q, axis=0) - near)
                if distance > 0.12:
                    continue
                choices.append((len(q) / (1 + 60 * distance), m))
            if not choices:
                raise RuntimeError('No visible ' + name + ' block.')
            m = max(choices, key=lambda item: item[0])[1]
            q = xyz[m]
            dx = xyz[1:-1, 2:] - xyz[1:-1, :-2]
            dy = xyz[2:, 1:-1] - xyz[:-2, 1:-1]
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
                face = q[np.abs(q @ normal - level) < 0.0022]
                for _ in range(2):
                    if len(face) < 6:
                        break
                    mean = np.mean(face, axis=0)
                    _, _, vh = np.linalg.svd(face - mean, full_matrices=False)
                    fitted = vh[-1]
                    if fitted[2] < 0:
                        fitted = -fitted
                    if fitted[2] < 0.78:
                        break
                    normal = fitted
                    level = float(mean @ normal)
                    face = q[np.abs(q @ normal - level) < 0.0022]
            else:
                z = np.percentile(q[:, 2], 90)
                face = q[np.abs(q[:, 2] - z) < 0.0025]
                normal = np.array([0.0, 0.0, 1.0])
                level = float(np.median(face[:, 2])) if len(face) else float(z)
            if len(face) < 5:
                raise RuntimeError('Top face occluded for ' + name)
            ex = np.array([1.0, 0.0, 0.0]) - normal * normal[0]
            ex /= np.linalg.norm(ex)
            ey = np.cross(normal, ex)
            uv = np.column_stack((face @ ex, face @ ey))
            lo, hi = np.percentile(uv, [2, 98], axis=0)
            middle = (lo + hi) / 2
            top_center = middle[0] * ex + middle[1] * ey + level * normal
            if model is None:
                height = float(top_center[2] - table_z(top_center[:2]))
                dims = hi - lo
                if not 0.009 < height < 0.065 or np.min(dims) < 0.006:
                    raise RuntimeError('Unreliable dimensions for ' + name)
                model = {'height': height, 'dims': dims}
            height = model['height']
            center = top_center - normal * height / 2
            extent = float((abs(ex[2]) * model['dims'][0] + abs(ey[2]) * model['dims'][1] + normal[2] * height) / 2)
            return {'center': center, 'top': float(top_center[2]), 'normal': normal, 'bottom': float(center[2] - extent), 'extent_z': extent, 'height': height, 'dims': model['dims'], 'model': model}
        legacy_locate = locate

        def locate(name, model=None, near=None):
            estimate = legacy_locate(name, model, near)
            if tracker.tracks[name]['shape'] in ('thin_prism', 'prism'):
                try:
                    geometry = estimate_cuboid_geometry(tracker.isolated_frame(last_frame[0], name), tracker.tracks[name]['color'], np.r_[estimate['model']['dims'], estimate['model']['height']], near=near, table_plane=plane)
                except ValueError:
                    return estimate
                if geometry['reliable']:
                    estimate.update(center=geometry['centroid'], top=geometry['top_center'][2], normal=geometry['top_normal'], bottom=geometry['bottom_z'], extent_z=(geometry['top_z'] - geometry['bottom_z']) / 2, vertices=geometry['vertices'], top_vertices=geometry['top_vertices'])
                return estimate
            if tracker.tracks[name]['shape'] == 'cylinder':
                return estimate
            try:
                geometry = estimate_cube_geometry(tracker.isolated_frame(last_frame[0], name), tracker.tracks[name]['color'], estimate['model']['height'], near=near, table_plane=plane)
            except ValueError:
                return estimate
            estimate['geometry_reliable'] = geometry['reliable']
            if not geometry['reliable']:
                return estimate
            side = geometry['side_m']
            estimate.update(center=geometry['centroid'], top=geometry['top_center'][2], normal=geometry['top_normal'], bottom=geometry['bottom_z'], extent_z=(geometry['top_z'] - geometry['bottom_z']) / 2, dims=np.array([side, side]), vertices=geometry['vertices'], top_vertices=geometry['top_vertices'])
            return estimate
        initial_tool = tool()
        scan = initial_tool.copy()
        scan[2] += 0.05
        move(scan, 1.0)
        capture()
        try:
            base_id = tracker.select('blue', 'cube')
            target_id = tracker.select('red', 'thin_prism')
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        for track in tracker.tracks.values():
            api.note('INSTANCE id=' + track['id'] + ' color=' + track['color'] + ' shape=' + track['shape'] +
                     ' center=' + str(np.round(track['center'], 4).tolist()) +
                     ' dimensions=' + str(np.round(track['dimensions'], 4).tolist()))
        api.note('ROLES target=' + target_id + ' base=' + base_id)
        blocks = {name: locate(name) for name in (base_id, target_id)}
        safe_z = max((b['top'] for b in blocks.values())) + 0.085
        park = initial_tool.copy()
        park[2] = max(park[2], safe_z)
        api.gripper(0.8, seconds=0.6)
        move(park)
        for name in blocks:
            try:
                blocks[name] = locate(name)
            except RuntimeError:
                pass
            b = blocks[name]
            api.note(name + ' measured center=' + str(np.round(b['center'], 4).tolist()) + ' dimensions=' + str(np.round(np.r_[b['dims'], b['height']], 4).tolist()))
        safe_z = max((b['top'] for b in blocks.values())) + 0.085
        contact_gap_min = -0.005
        api.note('Visual release gate: bottom gap between -5 and +5 mm.')
        held_models = {}

        def partial_held(name):
            previous, robot_before = held_models[name]
            if 'vertices' not in previous:
                # A prior visible top plane plus previously measured dimensions
                # defines a box model. Hidden vertices remain inferred.
                up = previous['normal']
                ex = np.array([1., 0., 0.]) - up*up[0]
                ex /= np.linalg.norm(ex)
                rotation = np.column_stack((ex, np.cross(up, ex), up))
                signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
                vertices = previous['center'] + (signs*np.r_[previous['dims'], previous['height']]/2) @ rotation.T
                previous = dict(previous, vertices=vertices, top_vertices=vertices[signs[:, 2] > 0])
            robot = last_robot[0]
            rotation = np.asarray(robot['tool_rotation']) @ np.asarray(robot_before['tool_rotation']).T
            position = np.asarray(robot['tool_position'])
            old_position = np.asarray(robot_before['tool_position'])
            predicted = position + rotation @ (previous['center'] - old_position)
            geometry = estimate_cuboid_geometry(tracker.isolated_frame(last_frame[0], name), tracker.tracks[name]['color'], np.r_[blocks[name]['model']['dims'], blocks[name]['height']], near=predicted, table_plane=plane)
            if geometry['observed_face_count'] < 2:
                raise ValueError('Insufficient visible held geometry')
            axes = geometry['rotation'][:, geometry['constrained_axes']]
            correction = axes @ (axes.T @ (geometry['centroid'] - predicted))
            if np.linalg.norm(correction) > 0.008:
                raise ValueError('Payload no longer agrees with visually confirmed grasp')
            vertices = position + (previous['vertices'] - old_position) @ rotation.T + correction
            fresh = dict(previous)
            center = predicted + correction
            normal = rotation @ previous['normal']
            if 'top_vertices' in previous:
                fresh['top_vertices'] = position + (previous['top_vertices'] - old_position) @ rotation.T + correction
            fresh.update(center=center, normal=normal, vertices=vertices, bottom=float(np.min(vertices[:, 2])), extent_z=float(np.ptp(vertices[:, 2]) / 2), top=float((center + normal * previous['height'] / 2)[2]))
            held_models[name] = (fresh, robot)
            api.note(name + ' partial depth update: unseen axes propagated from last visual pose and robot motion')
            return fresh

        def perceive(name, near=None):
            for attempt in range(3):
                try:
                    estimate = locate(name, blocks[name]['model'], near)
                    if name in held_models:
                        held_models[name] = (estimate, last_robot[0])
                    return estimate
                except RuntimeError:
                    if near is not None and name in held_models and tracker.tracks[name]['shape'] in ('cube', 'thin_prism', 'prism'):
                        try:
                            return partial_held(name)
                        except ValueError:
                            pass
                    if attempt == 2:
                        raise
                    if near is not None:
                        current = tool()
                        current[2] += .015
                        if name not in held_models:
                            current[0] += .015 if attempt == 0 else -.03
                        if api.plan_tool(current)['position_error'] < .003:
                            move(current, .7)
                    else:
                        current = tool()
                        if current[2] >= safe_z - 0.008:
                            current[0] += 0.035 if attempt == 0 else -0.07
                            preview = api.plan_tool(current)
                            if preview['position_error'] < 0.003:
                                move(current, 0.7)
                    api.wait(0.15)

        def refresh_support(name, prior):
            try:
                capture()
                geometry = estimate_cube_geometry(tracker.isolated_frame(last_frame[0], name), tracker.tracks[name]['color'], blocks[name]['height'], near=prior['center'], table_plane=plane)
            except ValueError:
                return prior
            if geometry['observed_face_count'] < 2:
                return prior
            axes = geometry['rotation'][:, geometry['constrained_axes']]
            center = prior['center'] + axes @ (axes.T @ (geometry['centroid'] - prior['center']))
            if np.linalg.norm(center - prior['center']) > 0.004:
                return prior
            shift = center - geometry['centroid']
            fresh = dict(prior)
            fresh.update(center=center, top=float(geometry['top_center'][2] + shift[2]), bottom=float(geometry['bottom_z'] + shift[2]), normal=geometry['top_normal'], top_vertices=geometry['top_vertices'] + shift, geometry_reliable=geometry['reliable'])
            return fresh

        def landing_center(block):
            center = block['center'].copy()
            normal = block['normal']
            axis = np.cross(normal, np.array([0.0, 0.0, 1.0]))
            skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
            rotation = np.eye(3) + skew + skew @ skew / max(1e-06, 1.0 + normal[2])
            if 'vertices' in block:
                pivot = block['vertices'][np.argmin(block['vertices'][:, 2])]
                predicted = pivot + rotation @ (center - pivot)
                center[:2] = predicted[:2]
            else:
                center[:2] -= normal[:2] * block['extent_z']
            return center
        retry_shift = [0.0]

        def stack_once(name, support, placement_bias=None):
            source = perceive(name)
            aim_xy = support['center'][:2].copy()
            if placement_bias is not None:
                aim_xy += placement_bias
            grasp = source['center'].copy()
            grasp[2] += (0.6 if source['height'] > 0.045 else 0.9) * source['height']
            api.note(name + ' elevated grasp offset=' + str(round((0.6 if source['height'] > 0.045 else 0.9) * source['height'], 4)))
            api.note(name + ' placement bias=' + str(np.round(aim_xy - support['center'][:2], 4).tolist()))
            above = grasp.copy()
            above[2] = max(safe_z, support['top'] + source['height'] + 0.045)
            api.gripper(0.8, seconds=0.5)
            move(above, 1.8)
            if source['height'] > 0.045:
                inspection = source['center'].copy()
                inspection[2] += 0.5 * source['height'] + 0.03
                move(inspection, 1.0)
            try:
                jaws = estimate_gripper_geometry(api.camera(), tool())
                drop = jaws['fingertip_drop_m']
                offset = np.clip(drop + (0.12 if source['height'] > 0.045 else 0.25) * source['height'], (0.35 if source['height'] > 0.045 else 0.75) * source['height'], (0.7 if source['height'] > 0.045 else 0.98) * source['height'])
                if tracker.tracks[name]['shape'] == 'thin_prism':
                    offset = max(offset, 0.4 * source['height'])
                if tracker.tracks[name]['shape'] == 'cylinder':
                    offset = max(offset, 0.5 * source['height'])
                grasp[2] = source['center'][2] + offset
                api.note(name + ' visible finger reach=' + str(round(drop, 4)) + '; geometry grasp offset=' + str(round(float(offset), 4)))
            except ValueError:
                api.note('Gripper occluded; using previously verified grasp height.')
            grasp[2] += retry_shift[0]
            pre = grasp.copy()
            pre[2] += 0.025
            move(pre, 1.1)
            move(grasp, 1.2)
            api.gripper(-0.174, seconds=1.0)
            api.wait(0.35)
            lift = grasp.copy()
            lift[2] += 0.06
            remember_grasp(name)
            move(lift, 1.7)
            held = perceive(name, tool())
            if held['center'][2] - source['center'][2] < 0.025:
                move(grasp)
                release_track(name)
                api.gripper(0.8, seconds=0.5)
                move(above)
                move(park)
                api.note('Grasp not confirmed; withdrawn.')
                return False
            try:
                jaws = estimate_gripper_geometry(api.camera(), tool(), held['bottom'])
                api.note(name + ' visible finger clearance above cube bottom=' + str(round(jaws['visible_clearance_m'], 4)))
            except ValueError:
                api.note('Held gripper depth occluded.')
            wrist = api.camera('wrist')
            pose = np.asarray(wrist['camera_to_world'])
            local = (held['center'] - pose[:3, 3]) @ pose[:3, :3]
            projected = np.asarray(wrist['intrinsics']) @ local
            uv = projected[:2] / max(local[2], 1e-06)
            rows, cols = np.nonzero(color_mask(wrist['rgb'], tracker.tracks[name]['color']))
            radius = max(10.0, float(wrist['intrinsics'][0, 0]) * held['height'] / max(local[2], 0.01))
            visible = np.count_nonzero((cols - uv[0]) ** 2 + (rows - uv[1]) ** 2 < radius ** 2) >= 8
            api.note(name + ' wrist grasp corroboration=' + str(bool(visible)))
            if not visible:
                api.wait(0.15)
                held = perceive(name, tool())
                if held['center'][2] - source['center'][2] < 0.025:
                    release_track(name)
                    api.gripper(0.8, seconds=0.4)
                    move(above)
                    move(park)
                    return False
            held_models[name] = (held, api.proprioception())
            offset = landing_center(held) - tool()
            target = support['center'].copy()
            target[:2] = aim_xy
            target[2] = support['top'] + held['extent_z'] + 0.03
            command = target - offset
            high = tool()
            high[2] = above[2]
            move(high)
            high[:2] = command[:2]
            move(high, 2.0)
            move(command, 1.6)
            table_relocation = support.get('on_table', False) and name != base_id
            alignment_tolerance = 0.008 if table_relocation else 0.001
            release_tolerance = 0.01 if table_relocation else 0.0055
            servo_command = tool()
            for clearance, rounds in ((0.024, 4), (0.011, 3), (0.003, 16)):
                for iteration in range(rounds):
                    api.wait(0.15)
                    support = refresh_support(base_id, support) if name == target_id and (not support.get('on_table', False)) else support
                    aim_xy = support['center'][:2].copy()
                    if placement_bias is not None:
                        aim_xy += placement_bias
                    current = tool()
                    measured = perceive(name, current)
                    # A recovery observation may have repositioned the wrist.
                    # Rebase the integrated command on the resulting robot pose.
                    recovered = last_robot[0]
                    if recovered is not None:
                        shift = np.asarray(recovered['tool_position'])-current
                        if np.linalg.norm(shift) > .003:
                            servo_command += shift
                    error_xy = aim_xy - landing_center(measured)[:2]
                    lateral = float(np.linalg.norm(error_xy))
                    gap = measured['bottom'] - support['top']
                    if iteration == 0 or iteration == rounds - 1:
                        api.note(name + ' alignment to aim=' + str(round(lateral, 4)) + ' gap=' + str(round(gap, 4)))
                    if clearance == 0.003 and lateral < alignment_tolerance and (contact_gap_min <= gap <= 0.005):
                        break
                    error_z = clearance - gap
                    if lateral < 0.001 and abs(error_z) < 0.0015:
                        break
                    delta = np.r_[error_xy, error_z]
                    if lateral > 0.015:
                        delta[:2] *= 0.015 / lateral
                    delta[2] = np.clip(delta[2], -0.004 if clearance == 0.003 else -0.014, 0.014)
                    if lateral > 0.005 and gap < 0.012:
                        delta[:2] = 0.0
                        delta[2] = min(0.014, 0.014 - gap)
                    servo_command += 1.0 * delta
                    move(servo_command, 0.65)
            measured = perceive(name, tool())
            displacement = landing_center(measured)[:2] - support['center'][:2]
            aim_error = landing_center(measured)[:2] - aim_xy
            gap = measured['bottom'] - support['top']
            api.note(name + ' release offset=' + str(np.round(displacement, 4).tolist()) + ' gap=' + str(round(gap, 4)))
            unsafe_overlap = placement_bias is not None and np.any(np.abs(displacement) > support['dims'] / 2 - 0.004)
            if support.get('geometry_reliable', False):
                unsafe_overlap = support_margin(landing_center(measured), support['top_vertices'], support['normal']) < 0.004
            if np.linalg.norm(aim_error) > release_tolerance or not contact_gap_min <= gap <= 0.005 or unsafe_overlap:
                offset = measured['center'] - tool()
                high = tool()
                high[2] = above[2]
                move(high)
                back = source['center'] - offset
                high[:2] = back[:2]
                move(high, 1.8)
                back[2] += measured['extent_z'] - source['height'] / 2 + 0.005
                move(back, 1.8)
                held_models.pop(name, None)
                release_track(name)
                api.gripper(0.8, seconds=0.8)
                move(high)
                move(park)
                api.note('Placement uncertain; returned payload to source.')
                return False
            held_models.pop(name, None)
            release_track(name)
            api.gripper(1.4, seconds=0.35)
            api.wait(0.3)
            try:
                opened = perceive(name)
                api.note(name + ' after opening displacement=' + str(np.round(opened['center'] - measured['center'], 4).tolist()))
            except RuntimeError:
                api.note(name + ' obscured after opening.')
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
                api.note(name + ' after withdrawal center=' + str(np.round(withdrawn['center'], 4).tolist()))
            except RuntimeError:
                api.note(name + ' obscured after withdrawal.')
            return True

        def stack(name, support, placement_bias=None):
            for attempt in range(2):
                retry_shift[0] = -0.002 * attempt
                if stack_once(name, support, placement_bias):
                    return True
                api.note(name + ' retry after visually unconfirmed grasp or placement.')
            return False

        def objects_from_depth():
            capture()
            # A hidden obstacle is not evidence of free space. Retain its last
            # observed footprint until it is seen again, including after motion.
            return [dict(id=t['id'], color=t['color'], shape=t['shape'], center=t['center'][:2], half=t['half'])
                    for t in tracker.tracks.values()]

        def blocked(xy, objects, ignore=(), pad=0.055):
            return [o for o in objects if o['id'] not in ignore and np.all(np.abs(xy - o['center']) < o['half'] + pad)]

        def parking(name, worksite, objects):
            options = []
            frame = last_frame[0]
            pose = np.asarray(frame['camera_to_world'])
            for dx in (-0.2, -0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16, 0.2):
                for dy in (-0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12):
                    xy = park[:2] + np.array([dx, dy])
                    if np.linalg.norm(xy - worksite) < 0.1 or blocked(xy, objects, ignore=(name,), pad=0.055):
                        continue
                    # Require observed tabletop at the parking cell, rather than
                    # treating camera/robot occlusion as verified free space.
                    local = (np.r_[xy, table_z(xy)]-pose[:3, 3]) @ pose[:3, :3]
                    if local[2] <= 0:
                        continue
                    uv = np.asarray(frame['intrinsics']) @ local
                    u, v = np.rint(uv[:2]/uv[2]).astype(int)
                    if not (0 <= v < frame['depth_m'].shape[0] and 0 <= u < frame['depth_m'].shape[1]):
                        continue
                    if not frame['valid_depth'][v, u] or abs(frame['depth_m'][v, u]-local[2]) > .015:
                        continue
                    high = api.plan_tool(np.r_[xy, safe_z])
                    if high['position_error'] > 0.003:
                        continue
                    low = api.plan_tool(np.r_[xy, table_z(xy) + blocks[name]['height'] * 0.9])
                    if low['position_error'] > 0.003:
                        continue
                    tilt = float(np.arccos(np.clip(np.asarray(low['tool_rotation'])[2, 1], -1.0, 1.0)))
                    options.append((np.linalg.norm(xy - blocks[name]['center'][:2]) + 0.5 * tilt, xy))
            return min(options, key=lambda item: item[0])[1] if options else None
        objects = objects_from_depth()
        options = []
        for dx in (-0.06, -0.03, 0.0, 0.03, 0.06):
            for dy in (-0.02, 0.0, 0.02):
                xy = park[:2] + np.array([dx, dy])
                preview = api.plan_tool(np.r_[xy, table_z(xy) + blocks[base_id]['height'] + 0.85 * blocks[target_id]['height']])
                tilt = float(np.arccos(np.clip(np.asarray(preview['tool_rotation'])[2, 1], -1.0, 1.0)))
                if preview['position_error'] > 0.003 or tilt > 0.18:
                    continue
                blockers = blocked(xy, objects, ignore=(base_id,))
                names = list(dict.fromkeys((o['id'] for o in blockers)))
                if len(names) > 2 or any(tracker.tracks[n]['shape'] not in ('cube', 'prism', 'thin_prism', 'cylinder') for n in names):
                    continue
                cost = 0.1 * len(names) + 0.12 * (target_id in names) + tilt + 0.5 * np.linalg.norm(xy - park[:2])
                options.append((cost, xy, names))
        if not options:
            api.note('No worksite with at most two movable blockers; stopping before stacking.')
            return
        _, worksite, names = min(options, key=lambda item: item[0])
        api.note('CLUTTER PLAN worksite=' + str(np.round(worksite, 4).tolist()) + ' blockers=' + str(names))
        for name in names:
            if name not in blocks:
                blocks[name] = locate(name)
            objects = objects_from_depth()
            destination = parking(name, worksite, objects)
            if destination is None:
                api.note('No observed reachable parking cell for ' + name)
                return
            support = {'center': np.r_[destination, table_z(destination)], 'top': table_z(destination), 'dims': np.array([0.1, 0.1]), 'on_table': True}
            api.note('CLEAR blocker=' + name + ' destination=' + str(np.round(destination, 4).tolist()))
            if not stack(name, support):
                return
            moved = perceive(name)
            if np.linalg.norm(moved['center'][:2] - destination) > 0.02:
                api.note('Blocker move not visually confirmed; stopping.')
                return
            if name == target_id and moved['top'] - table_z(moved['center'][:2]) < 0.8 * blocks[name]['height']:
                api.note('Target no longer visibly upright after relocation; stopping.')
                return
            api.note('CLEAR confirmed ' + name)
        objects = objects_from_depth()
        if blocked(worksite, objects, ignore=(base_id, target_id)):
            api.note('Worksite still obstructed after relocation; stopping.')
            return
        table = {'center': np.r_[worksite, table_z(worksite)], 'top': table_z(worksite), 'dims': np.array([0.1, 0.1]), 'on_table': True}
        if not stack(base_id, table):
            return
        blue = perceive(base_id)
        if not stack(target_id, blue):
            return
        api.wait(6.5)
        final = perceive(target_id)
        visible_height = final['top'] - blue['top']
        if not 0.85 * blocks[target_id]['height'] <= visible_height <= 1.15 * blocks[target_id]['height']:
            api.note('Final visual height inconsistent with an upright prism on the base.')
        else:
            api.note('Prism released; final visual normal=' + str(np.round(final['normal'], 4).tolist()))
    except RuntimeError as error:
        api.note(str(error))
