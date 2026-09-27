import numpy as np
import math
from scipy.spatial import cKDTree, ConvexHull
from scipy.optimize import least_squares


def run(api):
    PAD = .0214
    DEFAULT = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    api.note(str(api.task()))

    def observe():
        c = api.camera(view='side')
        rgb = np.asarray(c['rgb'], dtype=float)
        d = np.asarray(c['depth_m'], dtype=float)
        valid = np.asarray(c['valid_depth'], dtype=bool) & np.isfinite(d) & (d > 0)
        v, u = np.indices(d.shape)
        rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(np.asarray(c['intrinsics'])).T
        T = np.asarray(c['camera_to_world'])
        w = (rays * np.where(valid, d, 0)[..., None]) @ T[:3, :3].T + T[:3, 3]
        return rgb, w, valid

    rgb, world, valid = observe()
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    wood = valid & (r > 70) & (r > 1.15*g) & (g > .38*r) & (b < .75*g)
    z = world[..., 2][wood]
    if len(z) < 100:
        api.note('Table depth unavailable')
        return
    bins, counts = np.unique(np.floor(z/.002).astype(int), return_counts=True)
    peak = (bins[np.argmax(counts)]+.5)*.002
    table = float(np.median(z[np.abs(z-peak) < .004]))

    def red_points(rgb, w, valid):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mask = valid & (r > 65) & (r > 1.55*g) & (r > 1.45*b) & (w[..., 2] > table+.0003)
        return w[mask]

    def groups(points):
        if len(points) == 0:
            return []
        tree = cKDTree(points[:, :2])
        unused = np.ones(len(points), dtype=bool)
        out = []
        for i in range(len(points)):
            if not unused[i]:
                continue
            ids = [i]
            unused[i] = False
            for j in ids:
                for k in tree.query_ball_point(points[j, :2], .011):
                    if unused[k]:
                        unused[k] = False
                        ids.append(k)
            p = points[ids]
            if len(p) >= 4:
                lo, hi = np.quantile(p[:, :2], [.03, .97], axis=0)
                out.append(dict(p=p, xy=(lo+hi)/2, width=hi-lo, top=float(np.max(p[:, 2]))))
        return out

    def circle(points):
        xy = np.asarray(points)[:, :2]
        center = (np.min(xy, axis=0)+np.max(xy, axis=0))/2
        radius = float(np.max(np.linalg.norm(xy-center, axis=1)))
        if len(xy) >= 6:
            try:
                edge = xy[ConvexHull(xy).vertices]
                fit = least_squares(lambda x: np.linalg.norm(edge-x[:2], axis=1)-x[2],
                                    np.r_[center, radius], loss='soft_l1', f_scale=.001)
                if np.linalg.norm(fit.x[:2]-center) < .004 and .65*radius < fit.x[2] < 1.4*radius:
                    center, radius = fit.x[:2], float(fit.x[2])
            except Exception:
                pass
        return center, radius

    pts = red_points(rgb, world, valid)
    gs = groups(pts)
    api.note('Measured red components: '+str([(x['xy'].tolist(), x['top']-table, len(x['p'])) for x in gs]))
    if len(gs) < 2:
        return
    pole = max(gs, key=lambda x: x['top'])
    tip = pole['top']
    shaft = pole['p'][pole['p'][:, 2] > table+.55*(tip-table)]
    if len(shaft) < 3:
        return
    pole_xy, shaft_radius = circle(shaft)
    candidates = [x for x in gs if x['top'] < tip-.006
                  and np.linalg.norm(x['xy']-pole_xy) > .022
                  and np.max(x['width']) < .065]
    if len(candidates) != 1:
        api.note('Cannot distinguish hoop from pole and base')
        return
    hoop = candidates[0]
    top = hoop['top']
    thickness = top-table
    if not .001 < thickness < .020:
        return
    upper = hoop['p'][hoop['p'][:, 2] > top-.0015]
    hoop_xy, outer_radius = circle(upper if len(upper) >= 5 else hoop['p'])
    distance = np.linalg.norm(world[..., :2]-hoop_xy, axis=-1)
    hole_mask = valid & (np.abs(world[..., 2]-table) < .0007) & (distance < .70*outer_radius)
    hole_samples = world[hole_mask]
    if len(hole_samples):
        hole_radius = float(np.max(np.linalg.norm(hole_samples[:, :2]-hoop_xy, axis=1)))
        api.note('Annular evidence: %d table samples; visible hole radius %.4f; shaft radius %.4f' % (len(hole_samples), hole_radius, shaft_radius))
        if hole_radius < shaft_radius+.001:
            api.note('Measured hole clearance insufficient')
            return
    else:
        api.note('Hole interior unresolved; using isolated low rim geometry')
    radial = np.linalg.norm(pts[:, :2]-pole_xy, axis=1)
    base_points = pts[(radial < max(.018, outer_radius*1.5)) &
                      (radial > shaft_radius+.001) &
                      (pts[:, 2] < table+.40*(tip-table))]
    if len(base_points) < 3:
        api.note('Pole base height unresolved')
        return
    base_top = float(np.quantile(base_points[:, 2], .85))
    api.note('Table %.5f, hoop top %.5f radius %.4f, pole tip %.5f, base top %.5f' %
             (table, top, outer_radius, tip, base_top))

    def tilt(rot):
        return math.acos(float(np.clip(rot[2, 1], -1., 1.)))

    def yaw_rotation(yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ DEFAULT

    last_R = DEFAULT.copy()
    last_q = np.asarray(api.proprioception()['joint_angles'])
    jaw_span = max(.010, outer_radius)

    def choose(pos, limit=10., floor=None):
        pos = np.asarray(pos, dtype=float)
        radial_yaw = math.atan2(pos[1], pos[0])
        rotations = [last_R, yaw_rotation(radial_yaw), yaw_rotation(radial_yaw+math.pi/2)]
        rotations += [yaw_rotation(a) for a in np.linspace(-math.pi, math.pi, 12, endpoint=False)]
        best = None
        tried = []
        min_angle = math.pi
        min_error = float('inf')
        for rot in rotations:
            if any(np.linalg.norm(rot-old) < .001 for old in tried):
                continue
            tried.append(rot)
            p = api.plan_tool(pos.tolist(), orientation=rot.tolist())
            actual = np.asarray(p['tool_position'])
            rr = np.asarray(p['tool_rotation'])
            q = np.asarray(p['joint_angles'])
            err = max(float(p['position_error']), float(np.linalg.norm(actual-pos)))
            angle = tilt(rr)
            min_angle = min(min_angle, angle)
            min_error = min(min_error, err)
            if err > .004 or angle > math.radians(limit):
                continue
            low = actual[2]-PAD*math.cos(angle)-jaw_span*math.sin(angle)
            if floor is not None and low < floor+.0002:
                continue
            score = 100*err+2*angle+.035*np.linalg.norm(q-last_q)
            if best is None or score < best[0]:
                best = (score, q, actual, rr, rot)
            if err < .002 and angle < math.radians(3.):
                break
        if best is None:
            api.note('No pose at '+str(pos.tolist())+'; best tilt %.2f deg, position error %.5f' % (math.degrees(min_angle), min_error))
        return best

    def execute(plan, seconds=1., near=False, limit=None):
        nonlocal last_R, last_q
        if plan is None:
            return False
        _, q, planned, rr, requested = plan
        api.move_joints(q.tolist(), seconds=seconds)
        api.wait(.15)
        prop = api.proprioception()
        actual = np.asarray(prop['tool_position'])
        actual_R = np.asarray(prop['tool_rotation'])
        last_R = requested
        last_q = np.asarray(prop['joint_angles'])
        tracking = float(np.linalg.norm(actual-planned))
        allowed = limit if limit is not None else (11. if near else 16.)
        if tracking > (.004 if near else .007) or tilt(actual_R) > math.radians(allowed):
            api.note('Tracking error %.5f, actual tilt %.2f' % (tracking, math.degrees(tilt(actual_R))))
            return False
        return True

    def move(pos, seconds=1., near=False, floor=None):
        return execute(choose(pos, 9. if near else 14., floor), seconds, near)

    clearance = min(.004, thickness*.30)
    grasp_z = table+PAD+clearance
    lift_z = max(grasp_z+.040, tip+PAD+.018)
    pickup_plan = choose(np.r_[hoop_xy, grasp_z], 9., table)
    placement_plan = choose(np.r_[pole_xy, grasp_z+base_top-table+.001], 9., base_top)
    if pickup_plan is None or placement_plan is None:
        return
    api.note('Independent pickup/place tilts %.2f / %.2f deg; pickup clearance %.4f' %
             (math.degrees(tilt(pickup_plan[3])), math.degrees(tilt(placement_plan[3])), clearance))

    def transport_observation(expected_xy, expected_top):
        rgb, w, valid = observe()
        p = red_points(rgb, w, valid)
        select = (np.linalg.norm(p[:, :2]-expected_xy, axis=1) < outer_radius+.014)
        select &= np.abs(p[:, 2]-expected_top) < max(.018, thickness*1.5)
        select &= p[:, 2] > tip+.002
        p = p[select]
        if len(p) < 4:
            return None
        h = float(np.max(p[:, 2]))
        surface = p[p[:, 2] > h-.0025]
        center, radius = circle(surface if len(surface) >= 5 else p)
        if not .5*outer_radius < radius < 1.6*outer_radius:
            return None
        return np.r_[center, h]

    def release_withdraw():
        api.gripper(.8, seconds=.8)
        api.wait(.5)
        released = np.asarray(api.proprioception()['tool_position'])
        # Empty fingers can tilt during withdrawal; the high-pickup workspace
        # does not support the previous strictly upright vertical retreat.
        plan = choose(released+np.array([0., 0., .050]), 35.)
        if plan is not None:
            execute(plan, 1.8, limit=38.)
        else:
            direction = pole_xy-released[:2]
            length = float(np.linalg.norm(direction))
            if length < .005:
                direction = -released[:2]
                length = float(np.linalg.norm(direction))
            direction = direction/max(length, .0001)
            target = released+np.r_[direction*.045, .025]
            execute(choose(target, 40.), 1.8, limit=43.)
        withdrawn = float(np.linalg.norm(np.asarray(api.proprioception()['tool_position'])-released))
        api.note('Open gripper; withdrawal %.4f m' % withdrawn)
        api.wait(3.)

    api.gripper(.8, seconds=.6)
    start = np.asarray(api.proprioception()['tool_position'])
    dest = np.r_[hoop_xy, lift_z]
    for f in [.5, 1.]:
        if not move(start+f*(dest-start), 1.2):
            return
    for height in [grasp_z+.018, grasp_z+.008, grasp_z+.003, grasp_z]:
        if not move(np.r_[hoop_xy, height], .8, True, table):
            return
    prop = api.proprioception()
    actual = np.asarray(prop['tool_position'])
    angle = tilt(np.asarray(prop['tool_rotation']))
    low_pad = actual[2]-PAD*math.cos(angle)-jaw_span*math.sin(angle)
    high_pad = actual[2]-PAD*math.cos(angle)+jaw_span*math.sin(angle)
    if low_pad < table+.0001 or high_pad > top-.0005:
        api.note('Actual fingertip heights do not overlap rim safely: '+str([low_pad, high_pad]))
        return
    api.gripper(-.174, seconds=1.)
    api.wait(.4)
    for height in np.linspace(grasp_z, lift_z, 4)[1:]:
        if not move(np.r_[hoop_xy, height], 1., True):
            release_withdraw()
            return
    expected_top = top+(lift_z-grasp_z)
    measured = transport_observation(hoop_xy, expected_top)
    if measured is None:
        api.wait(.3)
        measured = transport_observation(hoop_xy, expected_top)
    if measured is None:
        api.note('Lifted hoop not visually confirmed')
        release_withdraw()
        return
    prop = api.proprioception()
    tool = np.asarray(prop['tool_position'])
    rr = np.asarray(prop['tool_rotation'])
    local_offset = rr.T @ (measured-tool)
    api.note('Confirmed elevated hoop; local top/tool offset '+str(local_offset.tolist()))
    # The existing lift already clears the tip. Use measured vertical clearance,
    # rather than adding the full three-dimensional offset to another lift.
    travel_z = float(tool[2])
    bottom = measured[2]-thickness
    if bottom < tip+.008:
        travel_z += tip+.008-bottom
        if not move(np.r_[tool[:2], travel_z], 1., True):
            release_withdraw()
            return
    origin = np.asarray(api.proprioception()['tool_position'])
    destination = np.r_[pole_xy, travel_z]
    for f in [.5, 1.]:
        if not move(origin+f*(destination-origin), 1.3, True):
            release_withdraw()
            return
    aligned = False
    for iteration in range(4):
        prop = api.proprioception()
        tool = np.asarray(prop['tool_position'])
        rr = np.asarray(prop['tool_rotation'])
        expected = tool+rr @ local_offset
        measured = transport_observation(expected[:2], expected[2])
        if measured is None:
            api.note('Transport verification unavailable')
            release_withdraw()
            return
        local_offset = rr.T @ (measured-tool)
        error = pole_xy-measured[:2]
        api.note('Alignment residual %.5f m' % np.linalg.norm(error))
        if np.linalg.norm(error) < .001:
            aligned = True
            break
        if not move(np.r_[tool[:2]+error, travel_z], .9, True):
            release_withdraw()
            return
    if not aligned:
        api.note('Alignment did not converge')
        release_withdraw()
        return
    prop = api.proprioception()
    tool = np.asarray(prop['tool_position'])
    rr = np.asarray(prop['tool_rotation'])
    offset = rr @ local_offset
    bottom = tool[2]+offset[2]-thickness
    if bottom < tip+.006:
        if not move(tool+np.array([0., 0., tip+.008-bottom]), 1., True):
            release_withdraw()
            return
    start_z = float(np.asarray(api.proprioception()['tool_position'])[2])
    nominal_seat_z = base_top+thickness-offset[2]+.0005
    steps = max(2, int(math.ceil((start_z-nominal_seat_z)/.004)))
    for height in np.linspace(start_z, nominal_seat_z, steps+1)[1:]:
        prop = api.proprioception()
        rr = np.asarray(prop['tool_rotation'])
        off = rr @ local_offset
        target = np.r_[pole_xy-off[:2], height]
        angle = tilt(rr)
        finger_floor = base_top+PAD*math.cos(angle)+jaw_span*math.sin(angle)+.0004
        seat_height = base_top+thickness-off[2]+.0005
        target[2] = max(height, finger_floor, seat_height)
        plan = choose(target, 9., base_top)
        if plan is None:
            api.note('Descent interrupted; opening and withdrawing')
            release_withdraw()
            return
        # Reuse this plan unless its predicted orientation changes the offset
        # enough to require one corrected preview.
        predicted_off = plan[3] @ local_offset
        predicted_angle = tilt(plan[3])
        corrected = np.r_[pole_xy-predicted_off[:2], max(height,
            base_top+PAD*math.cos(predicted_angle)+jaw_span*math.sin(predicted_angle)+.0004,
            base_top+thickness-predicted_off[2]+.0005)]
        if np.linalg.norm(corrected-target) > .0006:
            plan = choose(corrected, 9., base_top)
        if not execute(plan, .75, True):
            api.note('Descent tracking interrupted; opening and withdrawing')
            release_withdraw()
            return
    # Gravity completes the small remaining seating distance after opening.
    release_withdraw()
    rgb, w, valid = observe()
    p = red_points(rgb, w, valid)
    radial = np.linalg.norm(p[:, :2]-pole_xy, axis=1)
    rim = p[(radial > shaft_radius+.001) & (radial < outer_radius+.005)]
    near_seat = rim[np.abs(rim[:, 2]-(base_top+thickness)) < .002]
    api.note('Visible rim samples near measured seat: %d; visual evidence does not certify success' % len(near_seat))
    api.wait(2.5)
