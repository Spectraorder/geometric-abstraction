"""CPU visual construction of task-defined single columns.

This extends the frozen two-object inventory controller. The symbolic task
specifies identities and order; dimensions and poses still come from RGB-D.
Triangular caps are representable goals but require a separate grasp skill.
"""
import numpy as np
from scipy import ndimage
from aspire_so100.cube_geometry import estimate_cube_geometry, support_margin
from aspire_so100.cuboid_geometry import estimate_cuboid_geometry
from aspire_so100.gripper_geometry import estimate_gripper_geometry
from aspire_so100.column_tracking import ColumnTracker
from aspire_so100.column_geometry import ColumnGeometryConflict, column_size_uncertainty, estimate_column_geometry, update_partial_column_geometry, select_column_levels
from aspire_so100.instance_geometry import InstanceTracker, inventory_color_mask as color_mask

def run(api):

    def tool():
        return np.asarray(api.proprioception()['tool_position'], dtype=float)

    def move(p, seconds=1.5):
        p = np.asarray(p, dtype=float).tolist()
        try:
            api.move_tool(p, seconds=seconds)
        except RuntimeError as error:
            if 'unreachable' not in str(error):
                raise
            # At a tall transport waypoint the soft default orientation may
            # trade away position accuracy. Reuse the robot-only IK preview's
            # achievable orientation; payload tracking observes the rotation.
            first = api.plan_tool(p)
            second = api.plan_tool(p, orientation=first['tool_rotation'])
            if second['position_error'] > .003:
                raise
            api.move_tool(p, seconds=seconds, orientation=first['tool_rotation'])
    last_frame = [None]
    last_robot = [None]
    tracker = ColumnTracker()
    visual_models = {}
    grasp_anchors = {}
    warm_grasps = {}
    held_models = {}

    def remember_grasp(name, robot=None):
        robot = api.proprioception() if robot is None else robot
        grasp_anchors[name] = (tracker.tracks[name]['observed_center'].copy(), robot)
        if name not in warm_grasps and name in visual_models:
            warm_grasps[name] = (dict(visual_models[name]), robot)

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
        warm_grasps.pop(name, None)
        if name in grasp_anchors:
            tracker.tracks[name]['observed_center'] = predicted_grasps()[name]
            grasp_anchors.pop(name)


    def capture():
        s = api.camera()
        last_frame[0] = s
        if tracker.plane is not None:
            robot = api.proprioception() if grasp_anchors else None
            last_robot[0] = robot
            surface_models = dict(visual_models)
            for name in grasp_anchors:
                if name not in visual_models:
                    continue
                if name in held_models:
                    previous, before = held_models[name]
                else:
                    # Jaw closure can shift a source before its first visual
                    # grasp confirmation. Do not crop that evidence away.
                    surface_models.pop(name, None)
                    continue
                rotation = np.asarray(robot['tool_rotation']) @ np.asarray(before['tool_rotation']).T
                fresh = dict(previous)
                fresh['center'] = np.asarray(robot['tool_position']) + rotation @ (previous['center'] - np.asarray(before['tool_position']))
                fresh['normal'] = rotation @ previous['normal']
                if 'rotation' in previous:
                    fresh['rotation'] = rotation @ previous['rotation']
                for field in ('vertices', 'top_vertices'):
                    if field in previous:
                        fresh[field] = np.asarray(robot['tool_position']) + (previous[field] - np.asarray(before['tool_position'])) @ rotation.T
                surface_models[name] = fresh
            tracker.surface_models = surface_models
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
        goal = api.task()
        levels = goal.get('levels', [])
        if goal.get('kind') != 'single_column' or not 2 <= len(levels) <= 4:
            raise RuntimeError('Expected a single-column goal with 2..4 levels')
        if any(level['shape'] == 'triangular_prism' for level in levels):
            raise RuntimeError('Triangular cap grasp is not implemented; no object will be moved')
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
                if tracker.tracks[name]['shape'] == 'cube':
                    dims = np.repeat(height, 2)
                model = {'height': height, 'dims': dims}
                if tracker.tracks[name]['shape'] in ('prism', 'thin_prism'):
                    try:
                        model.update(column_size_uncertainty(
                            tracker.isolated_frame(last_frame[0], name),
                            tracker.tracks[name]['color'], model, plane))
                    except ValueError as error:
                        raise RuntimeError(str(error)) from error
            height = model['height']
            center = top_center - normal * height / 2
            extent = float((abs(ex[2]) * model['dims'][0] + abs(ey[2]) * model['dims'][1] + normal[2] * height) / 2)
            return {'shape': tracker.tracks[name]['shape'], 'center': center, 'top': float(top_center[2]), 'normal': normal, 'bottom': float(center[2] - extent), 'extent_z': extent, 'height': height, 'dims': model['dims'], 'model': model}
        legacy_locate = locate

        def locate(name, model=None, near=None):
            if tracker.tracks[name]['shape'] in ('cube', 'cylinder', 'prism', 'thin_prism'):
                capture()
                shape = tracker.tracks[name]['shape']
                if model is not None and shape == 'cube':
                    model = dict(model, dims=np.repeat(model['height'], 2))
                try:
                    selected = tracker.isolated_frame(last_frame[0], name)
                    if name in held_models:
                        previous, robot_before = held_models[name]
                        return update_partial_column_geometry(selected, tracker.tracks[name]['color'], shape,
                                                              model, previous, robot_before, last_robot[0], plane)
                    prior = None
                    if near is not None and name in warm_grasps:
                        previous, before = warm_grasps[name]
                        robot = last_robot[0]
                        rotation = np.asarray(robot['tool_rotation']) @ np.asarray(before['tool_rotation']).T
                        prior = dict(previous)
                        prior['center'] = np.asarray(robot['tool_position']) + rotation @ (previous['center'] - np.asarray(before['tool_position']))
                        prior['normal'] = rotation @ previous['normal']
                        if 'rotation' in previous:
                            prior['rotation'] = rotation @ previous['rotation']
                        for field in ('vertices', 'top_vertices'):
                            if field in previous:
                                prior[field] = np.asarray(robot['tool_position']) + (previous[field] - np.asarray(before['tool_position'])) @ rotation.T
                    try:
                        return estimate_column_geometry(selected, tracker.tracks[name]['color'], shape,
                                                        model=model, table_plane=plane)
                    except ValueError:
                        if prior is None:
                            raise
                        return estimate_column_geometry(selected, tracker.tracks[name]['color'], shape,
                                                        model=model, table_plane=plane, prior=prior)
                except ColumnGeometryConflict:
                    raise
                except ValueError as error:
                    if shape == 'cylinder':
                        raise RuntimeError(str(error)) from error
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
            level_ids = select_column_levels(tracker, levels, last_frame[0])
            if len(set(level_ids)) != len(level_ids):
                raise ValueError('Each tower level must resolve to a distinct instance')
            base_id = level_ids[0]
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        for track in tracker.tracks.values():
            api.note('INSTANCE id=' + track['id'] + ' color=' + track['color'] + ' shape=' + track['shape'] +
                     ' center=' + str(np.round(track['center'], 4).tolist()) +
                     ' dimensions=' + str(np.round(track['dimensions'], 4).tolist()))
        api.note('COLUMN bottom_to_top=' + str(level_ids))
        active_support = [None]
        blocks = {name: locate(name) for name in level_ids}
        visual_models.update(blocks)
        tall_column = sum(b['height'] for b in blocks.values()) > .155
        safe_z = max((b['top'] for b in blocks.values())) + 0.085
        park = initial_tool.copy()
        park[2] = max(park[2], safe_z)
        api.gripper(0.8, seconds=0.6)
        move(park)
        for name in blocks:
            try:
                blocks[name] = locate(name)
                visual_models[name] = blocks[name]
            except RuntimeError:
                pass
            b = blocks[name]
            api.note(name + ' measured center=' + str(np.round(b['center'], 4).tolist()) + ' dimensions=' + str(np.round(np.r_[b['dims'], b['height']], 4).tolist()))
        safe_z = max((b['top'] for b in blocks.values())) + 0.085
        contact_gap_min = -0.005
        api.note('Visual release gate: bottom gap between -5 and +5 mm.')
        release_drifts = {}

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
            if 'rotation' in previous:
                fresh['rotation'] = rotation @ previous['rotation']
            center = predicted + correction
            normal = rotation @ previous['normal']
            if 'top_vertices' in previous:
                fresh['top_vertices'] = position + (previous['top_vertices'] - old_position) @ rotation.T + correction
            fresh.update(center=center, normal=normal, vertices=vertices, bottom=float(np.min(vertices[:, 2])), extent_z=float(np.ptp(vertices[:, 2]) / 2), top=float((center + normal * previous['height'] / 2)[2]))
            held_models[name] = (fresh, robot)
            api.note(name + ' partial depth update: unseen axes propagated from last visual pose and robot motion')
            return fresh

        def remember_model(name, estimate):
            # Later views may resolve the initial pixel-sized dimension interval.
            # Persist that visual model without replacing the source location.
            old_dimensions = np.r_[blocks[name]['dims'], blocks[name]['height']]
            dimensions = np.r_[estimate['dims'], estimate['height']]
            if np.max(abs(dimensions - old_dimensions)) > .0001:
                api.note(name + ' visual dimensions refined=' + str(np.round(dimensions, 5).tolist()))
            visual_models[name] = estimate
            blocks[name].update(model=estimate['model'], dims=estimate['dims'],
                                height=estimate['height'])

        def perceive(name, near=None):
            for attempt in range(3):
                try:
                    estimate = locate(name, blocks[name]['model'], near)
                    remember_model(name, estimate)
                    if name in held_models:
                        held_models[name] = (estimate, last_robot[0])
                    return estimate
                except RuntimeError:
                    if near is not None and name in held_models and tracker.tracks[name]['shape'] in ('cube', 'thin_prism', 'prism'):
                        try:
                            fresh = partial_held(name)
                            remember_model(name, fresh)
                            return fresh
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
                selected = tracker.isolated_frame(last_frame[0], name)
                fresh = estimate_column_geometry(selected, tracker.tracks[name]['color'],
                                                 tracker.tracks[name]['shape'], blocks[name]['model'],
                                                 plane, prior)
            except ColumnGeometryConflict:
                raise
            except ValueError:
                return prior
            if np.linalg.norm(fresh['center'] - prior['center']) > .008:
                raise RuntimeError('Supporting level moved outside its trusted visual pose')
            if fresh['normal'][2] < .97 or abs(fresh['top'] - prior['top']) > .005:
                raise RuntimeError('Supporting level is no longer visibly upright at its expected height')
            remember_model(name, fresh)
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

        placed_models = {}
        balance_history = {}
        balance_velocity = {}

        def balanced_aim(name, support, refresh=False):
            if name not in level_ids or not placed_models:
                return None
            prefix = level_ids[:level_ids.index(name)]
            if not prefix or any(tracker.tracks[n]['shape'] != 'cube' for n in prefix+[name]):
                return None
            if refresh:
                for previous_name in prefix:
                    placed_models[previous_name] = refresh_support(previous_name, placed_models[previous_name])
                    center = placed_models[previous_name]['center'][:2].copy()
                    timestamp = float(last_frame[0]['time'])
                    previous = balance_history.get(previous_name)
                    if previous is None:
                        balance_history[previous_name] = (timestamp,center)
                    elif timestamp-previous[0]>.5:
                        velocity = (center-previous[1])/(timestamp-previous[0])
                        velocity = .5*velocity+.5*balance_velocity.get(previous_name,np.zeros(2))
                        speed = np.linalg.norm(velocity)
                        if speed>.0005:
                            velocity *= .0005/speed
                        balance_velocity[previous_name] = velocity
                        balance_history[previous_name] = (timestamp,center)
            centers = np.array([placed_models[n]['center'][:2]+6.*balance_velocity.get(n,np.zeros(2)) for n in prefix])
            masses = np.array([blocks[n]['height']**3 for n in prefix])
            new_mass = blocks[name]['height']**3
            targets = []
            weights = []
            for i, center in enumerate(centers):
                above_mass = masses[i+1:]
                moment = np.sum(above_mass[:,None]*centers[i+1:], axis=0)
                target = ((new_mass+above_mass.sum())*center-moment)/new_mass
                targets.append(target)
                weights.append(float(len(centers)-i)**2)
            target = np.average(targets,axis=0,weights=weights)
            shift = release_drifts.get('cube',np.zeros(2))
            compensation = -.8*shift
            length = np.linalg.norm(compensation)
            if length>.003:
                compensation *= .003/length
            target += compensation
            offset = target-support['center'][:2]
            length = np.linalg.norm(offset)
            if length>.006:
                target = support['center'][:2]+offset*.006/length
            if refresh:
                api.note('BALANCE centers='+str(np.round(centers,4).tolist())+' target='+str(np.round(target,4).tolist()))
            return target

        def stack_once(name, support, placement_bias=None):
            source = perceive(name)
            aim_xy = support['center'][:2].copy()
            if placement_bias is not None:
                aim_xy += placement_bias
                balanced = balanced_aim(name, support, refresh=True)
                if balanced is not None:
                    aim_xy = balanced
            grasp = source['center'].copy()
            grasp[2] += (0.6 if source['height'] > 0.045 else 0.9) * source['height']
            api.note(name + ' elevated grasp offset=' + str(round((0.6 if source['height'] > 0.045 else 0.9) * source['height'], 4)))
            api.note(name + ' placement bias=' + str(np.round(aim_xy - support['center'][:2], 4).tolist()))
            above = grasp.copy()
            # Approach a remote source at its own clearance height. Requiring
            # the final tower's clearance here can exceed the arm's reach.
            above[2] = source['top'] + .075
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
            grasp[2] -= .009 if tracker.tracks[name]['shape'] in ('cube', 'cylinder') else 0.
            if tall_column and source['height'] > .045:
                grasp[2] = source['center'][2] + .25*source['height']
            if source['shape'] == 'cube':
                grasp[2] = max(grasp[2], source['center'][2] + .6*source['height'])
            grasp[2] += retry_shift[0]
            api.note(name + ' applied grasp offset=' + str(round(float(grasp[2]-source['center'][2]), 4)))
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
            api.note(name + ' payload tilt=' + str(np.round(held['normal'], 4).tolist()))
            held_models[name] = (held, api.proprioception())
            offset = landing_center(held) - tool()
            target = support['center'].copy()
            target[:2] = aim_xy
            target[2] = support['top'] + held['extent_z'] + 0.03
            command = target - offset
            high = tool()
            transport_z = max(high[2], command[2]+.025)
            high[2] = transport_z
            preview = api.plan_tool(high)
            if preview['position_error'] > .003:
                preview = api.plan_tool(high, orientation=preview['tool_rotation'])
            if preview['position_error'] > .003:
                # Move partway toward the worksite before raising. The waypoint
                # remains outside the column's gripper-clearance neighborhood.
                waypoint = tool()
                direction = waypoint[:2]-command[:2]
                distance = np.linalg.norm(direction)
                if distance < .10:
                    raise RuntimeError('No reachable transport clearance above the column')
                waypoint[:2] = command[:2] + direction/distance*max(.10, distance/2)
                move(waypoint)
                high = tool()
                high[2] = transport_z
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
                    support = refresh_support(active_support[0], support) if active_support[0] is not None and not support.get('on_table', False) else support
                    aim_xy = support['center'][:2].copy()
                    if placement_bias is not None:
                        aim_xy += placement_bias
                        balanced = balanced_aim(name, support, refresh=iteration==0)
                        if balanced is not None:
                            aim_xy = balanced
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
            if support.get('shape') == 'cylinder':
                unsafe_overlap = np.linalg.norm(displacement) > min(support['dims'])/2-.004
            if support.get('geometry_reliable', False):
                unsafe_overlap = support_margin(landing_center(measured), support['top_vertices'], support['normal']) < 0.004
            if np.linalg.norm(aim_error) > release_tolerance or not contact_gap_min <= gap <= 0.005 or unsafe_overlap:
                offset = measured['center'] - tool()
                high = tool()
                # Clear the current column before returning toward the source.
                # A source's approach height can be below a tall destination.
                high[2] = max(high[2], above[2], support['top'] + measured['height'] + .035)
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
            except (RuntimeError, ColumnGeometryConflict):
                # The payload is already released. Reject the obscured view,
                # then withdraw the empty gripper before observing again.
                api.note(name + ' geometry unconfirmed after opening.')
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
                drift = withdrawn['center'][:2] - landing_center(measured)[:2]
                if np.linalg.norm(drift) < .012:
                    release_drifts[tracker.tracks[name]['shape']] = drift
            except (RuntimeError, ColumnGeometryConflict):
                api.note(name + ' geometry unconfirmed after withdrawal.')
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
        # Prefer the established extra approach clearance. If every worksite
        # fails, retry at the observed nominal cube grasp height: half a cube
        # above its support plus the controller's 0.6-height tool offset.
        cube_heights = (1.4, 1.1) if any(blocks[n]['shape'] == 'cube' for n in level_ids) else (1.4,)
        for cube_tool_height in cube_heights:
            for dx in (-0.06, -0.03, 0.0, 0.03, 0.06):
                for dy in (-0.02, 0.0, 0.02):
                    xy = park[:2] + np.array([dx, dy])
                    # Preview every placement height, not only the final level.
                    cumulative = 0.
                    tilts = []
                    for level_id in level_ids:
                        height = blocks[level_id]['height']
                        tool_height = cumulative + ((.75 if tall_column else .9) if height > .045 else cube_tool_height)*height
                        preview = api.plan_tool(np.r_[xy, table_z(xy) + tool_height])
                        if tall_column and preview['position_error'] > .003:
                            preview = api.plan_tool(np.r_[xy, table_z(xy) + tool_height], orientation=preview['tool_rotation'])
                        if preview['position_error'] > .003:
                            break
                        tilts.append(float(np.arccos(np.clip(np.asarray(preview['tool_rotation'])[2, 1], -1., 1.))))
                        cumulative += height
                    if len(tilts) != len(level_ids) or max(tilts) > (.85 if tall_column else .28):
                        continue
                    tilt = max(tilts)
                    blockers = blocked(xy, objects, ignore=(base_id,))
                    names = list(dict.fromkeys((o['id'] for o in blockers)))
                    if len(names) > 2 or any(tracker.tracks[n]['shape'] not in ('cube', 'prism', 'thin_prism', 'cylinder') for n in names):
                        continue
                    cost = 0.1 * len(names) + 0.12 * any(n in level_ids[1:] for n in names) + tilt + 0.5 * np.linalg.norm(xy - park[:2])
                    options.append((cost, xy, names))
            if options:
                break
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
            if name in level_ids[1:] and moved['top'] - table_z(moved['center'][:2]) < 0.8 * blocks[name]['height']:
                api.note('Target no longer visibly upright after relocation; stopping.')
                return
            api.note('CLEAR confirmed ' + name)
        objects = objects_from_depth()
        if blocked(worksite, objects, ignore=(base_id,)):
            api.note('Worksite still obstructed after relocation; stopping.')
            return
        table = {'center': np.r_[worksite, table_z(worksite)], 'top': table_z(worksite), 'dims': np.array([0.1, 0.1]), 'on_table': True}
        if not stack(base_id, table):
            return
        support = perceive(base_id)
        column_axis = support['center'][:2].copy()
        placed_models[base_id] = support
        for index, name in enumerate(level_ids[1:], start=1):
            active_support[0] = level_ids[index-1]
            safe_z = max(safe_z, support['top'] + blocks[name]['height'] + .035)
            # Open jaws extend below the tool reference. Keep the parked arm
            # well above the completed level rather than descending onto it.
            park[2] = max(safe_z, support['top'] + blocks[name]['height'] + .10)
            move(park)
            # The base axis is the default for mixed shapes. Cube prefixes
            # refine this aim using predicted whole-column load balance.
            placement_bias = column_axis - support['center'][:2]
            shift = release_drifts.get(tracker.tracks[name]['shape'], np.zeros(2))
            compensation = -.8*shift
            limit = .15*float(np.min(support['dims']))
            if np.linalg.norm(compensation) > limit:
                compensation *= limit/np.linalg.norm(compensation)
            placement_bias += compensation
            if not stack(name, support, placement_bias=placement_bias):
                return
            api.wait(3.5)
            first = perceive(name)
            api.wait(.6)
            measured = perceive(name)
            if np.linalg.norm(measured['center']-first['center']) > .003:
                api.note('Placed level is still drifting; stopping before adding load')
                return
            visible_height = measured['top'] - support['top']
            if not .8*blocks[name]['height'] <= visible_height <= 1.2*blocks[name]['height']:
                api.note('Placed level has inconsistent visible support height; stopping')
                return
            support = measured
            placed_models[name] = measured
            api.note('LEVEL confirmed ' + str(index+1) + '/' + str(len(level_ids)) + ' id=' + name)
        api.wait(6.5)
        api.note('Column released; all levels visually placed in task order')
    except (RuntimeError, ColumnGeometryConflict) as error:
        api.note(str(error))
