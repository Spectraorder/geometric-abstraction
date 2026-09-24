"""CPU visual assembly of low-rise support graphs.

The manipulation core is derived from the frozen column controller. Geometry
helpers are shared; the historical column source and bundle remain unchanged.
"""
import numpy as np
from scipy import ndimage
from aspire_so100.cube_geometry import estimate_cube_geometry, support_margin
from aspire_so100.cuboid_geometry import estimate_cuboid_geometry
from aspire_so100.gripper_geometry import estimate_gripper_geometry
from aspire_so100.column_tracking import ColumnTracker
from aspire_so100.column_geometry import ColumnGeometryConflict, column_size_uncertainty, estimate_column_geometry, update_partial_column_geometry, select_column_levels
from aspire_so100.instance_geometry import InstanceTracker, inventory_color_mask as color_mask
from aspire_so100.beam_geometry import select_structure_nodes, flat_beam_aligned, estimate_flat_beam_geometry
from aspire_so100.structure_geometry import convex_polygon, model_footprint, support_equilibrium

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
        levels = goal.get('nodes', [])
        if goal.get('kind') != 'support_graph' or not 3 <= len(levels) <= 8:
            raise RuntimeError('Expected a support graph with 3..8 nodes')
        if goal.get('axes') != 'world_xy':
            raise RuntimeError('Expected world XY construction axes')
        node_keys = [node['key'] for node in levels]
        if len(set(node_keys)) != len(node_keys):
            raise RuntimeError('Duplicate structure node')
        for index, node in enumerate(levels):
            if node.get('orientation') not in ('upright', 'flat_x') or node['shape'] == 'triangular_prism':
                raise RuntimeError('Unsupported structure shape or desired orientation')
            if not set(node['supports']) <= set(node_keys[:index]):
                raise RuntimeError('Support graph must be in construction order')
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
                    if (node_for_id.get(name, {}).get('orientation') == 'flat_x'
                            and name not in held_models and name not in warm_grasps):
                        try:
                            return estimate_flat_beam_geometry(selected, tracker.tracks[name]['color'], plane, model)
                        except ValueError:
                            if model is None:
                                raise
                    if name in held_models:
                        previous, robot_before = held_models[name]
                        return update_partial_column_geometry(selected, tracker.tracks[name]['color'], shape,
                                                              model, previous, robot_before, last_robot[0], plane)
                    flat_target = node_for_id.get(name, {}).get('orientation') == 'flat_x'
                    prior = visual_models.get(name) if flat_target else None
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
                                                        model=model, table_plane=plane, prior=prior if flat_target else None)
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
            level_ids = select_structure_nodes(tracker, levels, last_frame[0])
            if len(set(level_ids)) != len(level_ids):
                raise ValueError('Each tower level must resolve to a distinct instance')
            base_id = level_ids[0]
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        for track in tracker.tracks.values():
            api.note('INSTANCE id=' + track['id'] + ' color=' + track['color'] + ' shape=' + track['shape'] +
                     ' center=' + str(np.round(track['center'], 4).tolist()) +
                     ' dimensions=' + str(np.round(track['dimensions'], 4).tolist()))
        api.note('STRUCTURE construction_order=' + str(level_ids))
        active_support = [[]]
        node_for_id = dict(zip(level_ids, levels))
        id_for_key = dict(zip(node_keys, level_ids))
        parents = {name: [id_for_key[key] for key in node['supports']] for name, node in node_for_id.items()}
        blocks = {name: locate(name) for name in level_ids}
        visual_models.update(blocks)
        unit_id = id_for_key[goal['length_unit']['observed_cube']]
        if blocks[unit_id]['shape'] != 'cube' or parents[unit_id]:
            raise RuntimeError('Length reference must be an observed foundation cube')
        unit = float(blocks[unit_id]['height'])
        has_flat_beam = any(node['orientation'] == 'flat_x' for node in levels)
        relative_xy = {name: unit*np.asarray(node['grid'], float) for name, node in node_for_id.items()}
        tops = {}
        for name in level_ids:
            parent_tops = [tops[parent] for parent in parents[name]]
            if parent_tops and np.ptp(parent_tops) > .002:
                raise RuntimeError('Required supports do not have equal measured heights')
            tops[name] = (max(parent_tops) if parent_tops else 0.)+blocks[name]['height']
        tall_column = max(tops.values()) > .155
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
                if node_for_id.get(name, {}).get('orientation') == 'flat_x':
                    fresh = estimate_flat_beam_geometry(selected, tracker.tracks[name]['color'], plane, blocks[name]['model'])
                else:
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
        structure_origin = [None]

        def mass_proxy(model):
            factor = np.pi/4 if model['shape'] == 'cylinder' else 1.
            return float(np.prod(model['dims'])*model['height']*factor)

        def graph_balance(name, model):
            if name not in parents:
                return True
            models = {key: dict(value, mass=mass_proxy(value)) for key, value in placed_models.items()}
            models[name] = dict(model, mass=mass_proxy(model))
            return support_equilibrium(models, {key: parents[key] for key in models}, margin=.002)['feasible']

        def refresh_active_support(prior):
            supports = active_support[0]
            if not supports:
                return prior
            observed = []
            for name in supports:
                placed_models[name] = refresh_support(name, placed_models[name])
                observed.append(placed_models[name])
            heights = np.array([item['top'] for item in observed])
            if np.ptp(heights) > .003:
                raise RuntimeError('Multiple supports have inconsistent visible heights')
            points = np.concatenate([model_footprint(item) for item in observed])
            polygon = convex_polygon(points)
            # Each support contributes its observed displacement from the goal.
            origins = [item['center'][:2]-relative_xy[name] for name, item in zip(supports, observed)]
            xy = np.mean(origins, axis=0)+relative_xy[active_node[0]]
            height = float(np.max(heights))
            vertices = np.column_stack((polygon, np.repeat(height, len(polygon))))
            return {'center': np.r_[xy, height], 'top': height, 'dims': np.ptp(polygon, axis=0),
                    'normal': np.array([0., 0., 1.]), 'top_vertices': vertices,
                    'geometry_reliable': True, 'shape': 'support_union'}

        active_node = [None]

        def stack_once(name, support, placement_bias=None):
            source = perceive(name)
            flat_beam = node_for_id.get(name, {}).get('orientation') == 'flat_x'
            if flat_beam and not flat_beam_aligned(source):
                raise RuntimeError('Flat beam must be visibly lying along X before pickup')
            aim_xy = support['center'][:2].copy()
            if placement_bias is not None:
                aim_xy += placement_bias
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
            if source['height'] > 0.045 or flat_beam:
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
                if flat_beam:
                    # The thin slab needs an absolute observed fingertip reach;
                    # clamping reach to slab thickness would strike the table.
                    offset = drop + .5*source['height']
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
            table_relocation = support.get('relocation', False)
            alignment_tolerance = 0.008 if table_relocation else 0.001
            release_tolerance = 0.01 if table_relocation else 0.0055
            servo_command = tool()
            for clearance, rounds in ((0.024, 4), (0.011, 3), (0.003, 16)):
                for iteration in range(rounds):
                    api.wait(0.15)
                    support = refresh_active_support(support) if active_support[0] and not support.get('on_table', False) else support
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
            if support.get('shape') == 'cylinder':
                unsafe_overlap = np.linalg.norm(displacement) > min(support['dims'])/2-.004
            if support.get('geometry_reliable', False):
                unsafe_overlap = support_margin(landing_center(measured), support['top_vertices'], support['normal']) < 0.004
            if name in parents and not support.get('relocation', False):
                landing = landing_center(measured)
                normal = measured['normal']
                axis = np.cross(normal, np.array([0., 0., 1.]))
                skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
                settled_rotation = np.eye(3)+skew+skew @ skew/max(1e-6, 1.+normal[2])
                landed = dict(measured, center=landing, normal=np.array([0., 0., 1.]))
                for field in ('vertices', 'top_vertices', 'bottom_vertices'):
                    if field in measured:
                        landed[field] = landing+(measured[field]-measured['center']) @ settled_rotation.T
                unsafe_overlap = unsafe_overlap or not graph_balance(name, landed)
            if flat_beam and not flat_beam_aligned(measured, tolerance_degrees=8.):
                unsafe_overlap = True
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
            destination_rotation = None
            if name in relative_xy:
                destination_xy = worksite+relative_xy[name]
                h = blocks[name]['height']
                target_z = table_z(destination_xy)+tops[name]-h+(1.1 if h < .045 else .85)*h
                preview = api.plan_tool(np.r_[destination_xy, target_z])
                if preview['position_error'] < .003:
                    destination_rotation = np.asarray(preview['tool_rotation'])
            for dx in (-0.2, -0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16, 0.2):
                for dy in (-0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12):
                    xy = park[:2] + np.array([dx, dy])
                    if min(np.linalg.norm(xy-worksite-relative_xy[n]) for n in level_ids) < .10 or blocked(xy, objects, ignore=(name,), pad=.07 if has_flat_beam else .055):
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
                    # A staging cell is also a future grasp location. Favor
                    # robot attitudes close to the eventual placement attitude
                    # to reduce large wrist rotations after the second grasp.
                    attitude_change = 0.
                    if destination_rotation is not None:
                        relative_rotation = destination_rotation.T @ np.asarray(low['tool_rotation'])
                        attitude_change = float(np.arccos(np.clip((np.trace(relative_rotation)-1.)/2., -1., 1.)))
                    options.append((np.linalg.norm(xy - blocks[name]['center'][:2]) + .5*tilt + .12*attitude_change, xy))
            return min(options, key=lambda item: item[0])[1] if options else None
        objects = objects_from_depth()
        options = []
        for dx in ((-.09, -.06, -.03, 0., .03, .06, .09) if has_flat_beam else (-.06, -.03, 0., .03, .06)):
            for dy in ((-.04, -.02, 0., .02, .04, .06, .08) if has_flat_beam else (-.02, 0., .02)):
                origin = park[:2]+np.array([dx, dy])
                tilts = []
                blockers = set()
                for name in level_ids:
                    xy = origin+relative_xy[name]
                    height = blocks[name]['height']
                    tool_height = tops[name]-height+(.9 if height > .045 else 1.4)*height
                    preview = api.plan_tool(np.r_[xy, table_z(xy)+tool_height])
                    if preview['position_error'] > .003:
                        break
                    tilts.append(float(np.arccos(np.clip(np.asarray(preview['tool_rotation'])[2, 1], -1., 1.))))
                    blockers.update(item['id'] for item in blocked(xy, objects, pad=.055))
                if len(tilts) != len(level_ids) or max(tilts) > .28:
                    continue
                # A foundation at its source is still repositioned precisely;
                # reserve an empty footprint before the assembly begins.
                names = sorted(blockers)
                if len(names) > 3 or any(tracker.tracks[n]['shape'] not in ('cube', 'prism', 'thin_prism', 'cylinder') for n in names):
                    continue
                cost = .1*len(names)+max(tilts)+.5*np.linalg.norm(origin-park[:2])
                options.append((cost, origin, names))
        if not options:
            api.note('No reachable structure footprint with at most three movable blockers')
            return
        if has_flat_beam:
            # A nominally clearable footprint is useful only if each blocker
            # has an observed, reachable staging cell. Preview alternatives
            # before moving anything; reserve proposed cells for later blockers.
            chosen = None
            for option in sorted(options, key=lambda item: item[0])[:10]:
                reserved = [dict(item) for item in objects]
                destinations = {}
                for blocker in option[2]:
                    if blocker not in blocks:
                        blocks[blocker] = locate(blocker)
                    destination = parking(blocker, option[1], reserved)
                    if destination is None:
                        break
                    destinations[blocker] = destination
                    for item in reserved:
                        if item['id'] == blocker:
                            item['center'] = destination
                else:
                    chosen = option
                    break
            if chosen is None:
                raise RuntimeError('No reachable workspace with feasible blocker staging')
            _, worksite, names = chosen
        else:
            _, worksite, names = min(options, key=lambda item: item[0])
        structure_origin[0] = worksite.copy()
        api.note('STRUCTURE PLAN origin='+str(np.round(worksite, 4).tolist())+' unit='+str(round(unit, 5))+' blockers='+str(names))
        for name in names:
            if name not in blocks:
                blocks[name] = locate(name)
            objects = objects_from_depth()
            destination = destinations[name] if has_flat_beam else parking(name, worksite, objects)
            if destination is None:
                api.note('No observed reachable parking cell for ' + name)
                return
            support = {'center': np.r_[destination, table_z(destination)], 'top': table_z(destination), 'dims': np.array([0.1, 0.1]), 'on_table': True, 'relocation': True}
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
        for index, name in enumerate(level_ids):
            active_node[0] = name
            active_support[0] = parents[name]
            for previous_name in placed_models:
                placed_models[previous_name] = refresh_support(previous_name, placed_models[previous_name])
            xy = structure_origin[0]+relative_xy[name]
            if parents[name]:
                support = refresh_active_support(None)
            else:
                # Previously placed neighboring foundations are intentional.
                objects = objects_from_depth()
                if blocked(xy, objects, ignore=tuple(placed_models)+(name,), pad=.055):
                    raise RuntimeError('Foundation workspace is obstructed')
                support = {'center': np.r_[xy, table_z(xy)], 'top': table_z(xy),
                           'dims': np.array([.1, .1]), 'on_table': True}
            safe_z = max(safe_z, support['top']+blocks[name]['height']+.035)
            park[2] = max(safe_z, support['top']+blocks[name]['height']+.08)
            move(park)
            shift = release_drifts.get(tracker.tracks[name]['shape'], np.zeros(2))
            compensation = -.8*shift
            length = np.linalg.norm(compensation)
            if length > .002:
                compensation *= .002/length
            if not stack(name, support, placement_bias=compensation):
                return
            api.wait(3.5)
            first = perceive(name)
            api.wait(.6)
            measured = perceive(name)
            if np.linalg.norm(measured['center']-first['center']) > .003:
                raise RuntimeError('Placed node is still drifting')
            if node_for_id[name]['orientation'] == 'flat_x' and not flat_beam_aligned(measured):
                raise RuntimeError('Placed beam has the wrong visible orientation')
            if abs(measured['top']-support['top']-blocks[name]['height']) > .004:
                raise RuntimeError('Placed node has inconsistent visible support height')
            if np.linalg.norm(measured['center'][:2]-support['center'][:2]) > .006:
                raise RuntimeError('Placed node has inconsistent visible XY position')
            if not graph_balance(name, measured):
                raise RuntimeError('Observed structure has no balanced support reactions')
            placed_models[name] = measured
            if index == 0:
                structure_origin[0] = measured['center'][:2]-relative_xy[name]
            api.note('NODE confirmed '+node_for_id[name]['key']+' '+str(index+1)+'/'+str(len(level_ids)))
        api.wait(6.5)
        api.note('Structure released; all nodes visually placed according to the support graph')
    except (RuntimeError, ColumnGeometryConflict) as error:
        api.note(str(error))
