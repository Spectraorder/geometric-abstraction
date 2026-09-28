"""Sensor-grounded capable R0 for ring-over-peg training.

Grasp search is adapted from the second Full pilot's R3 training proposal.
Transport and release are hand-edited development logic. This is task-specific
prior experience, not a neutral starter or a successful held-out policy.
Only the public sensor and robot API supplies runtime coordinates.
"""
import numpy as np


def unit(x):
    x = np.asarray(x, dtype=float)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def boundaries(vertices, normal, center):
    v = np.asarray(vertices, dtype=float)
    aa = []
    bb = []
    for i in range(len(v)):
        edge = v[(i + 1) % len(v)] - v[i]
        if np.linalg.norm(edge) < 1e-7:
            continue
        a = unit(np.cross(edge, normal))
        if np.dot(a, np.asarray(center) - v[i]) > 0:
            a = -a
        aa.append(a)
        bb.append(float(np.dot(a, v[i])))
    return np.asarray(aa), np.asarray(bb)


def fingers(g):
    v = np.asarray(g['jaw_vertices_m'], dtype=float).reshape(-1, 3)
    f = np.asarray(g['fixed_pad_m'], dtype=float)
    m = np.asarray(g['moving_pad_m'], dtype=float)
    if len(v) == 16:
        groups = [v[:8], v[8:]]
        costs = [np.linalg.norm(np.mean(q, axis=0) - f) - np.linalg.norm(np.mean(q, axis=0) - m) for q in groups]
        if costs[0] > costs[1]:
            groups = [groups[1], groups[0]]
        return groups
    scores = np.sum((v - f) ** 2, axis=1) - np.sum((v - m) ** 2, axis=1)
    order = np.argsort(scores)
    return [v[order[:len(v) // 2]], v[order[len(v) // 2:]]]


def clip_points(v, a, b):
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    if len(v) == 0:
        return v
    s = v @ np.asarray(a) - b
    out = [v[i] for i in range(len(v)) if s[i] <= 1e-9]
    for i in range(len(v)):
        for j in range(i):
            if s[i] * s[j] < -1e-16:
                out.append(v[i] + (v[j] - v[i]) * s[i] / (s[i] - s[j]))
    if not out:
        return np.empty((0, 3))
    return np.unique(np.round(np.asarray(out), 9), axis=0)


def table_z(plane, p):
    return float(plane[0] * p[0] + plane[1] * p[1] + plane[2])


def table_clearance(g, plane):
    v = np.asarray(g['jaw_vertices_m'], dtype=float).reshape(-1, 3)
    return float(np.min(v[:, 2] - plane[0] * v[:, 0] - plane[1] * v[:, 1] - plane[2]))


def ring_geometry(r):
    c = np.asarray(r['hole']['center_m'], dtype=float)
    n = unit(r['hole']['axis'])
    if n[2] < 0:
        n = -n
    h = float(r['height_m'])
    ai, bi = boundaries(r['hole']['inner_boundary_m'], n, c)
    ao, bo = boundaries(r['outer_boundary_m'], n, c)
    return c, n, h, ai, bi, ao, bo


def insertion_clear(r, peg, margin):
    c, n, h, ai, bi, ao, bo = ring_geometry(r)
    p = np.asarray(peg['top_center_m'], dtype=float)
    d = unit(peg['axis'])
    if d[2] < 0:
        d = -d
    nd = float(np.dot(n, d))
    if nd < 0.65:
        return False
    t = float(np.dot(n, c - p) / nd)
    ends = np.asarray([p + t * d, p + (t - h / nd) * d])
    allowance = float(peg['radius_m']) * np.sqrt(1 + ((ai @ d) / nd) ** 2)
    return bool(np.all(ends @ ai.T - bi + allowance < -margin))


def run(api):
    counts = [0, 0]

    def note(s):
        api.note(s)

    def observe_ring():
        for k in range(3):
            try:
                counts[0] += 1
                return api.observe('ring')
            except RuntimeError:
                if k < 2:
                    api.wait(0.15)
        return None

    def preview(p, roll, jaw, direction=None):
        if counts[1] >= 245 or counts[0] >= 480:
            return None
        counts[0] += 1
        counts[1] += 1
        try:
            if direction is None:
                g = api.plan_pad(np.asarray(p).tolist(), float(roll), float(jaw))
            else:
                g = api.plan_grasp(np.asarray(p).tolist(), np.asarray(direction).tolist(), float(jaw))
                if float(g.get('closing_alignment', 1.0)) < 0.965:
                    return None
            f = np.asarray(g['fixed_pad_m'], dtype=float)
            error = max(float(g.get('fixed_pad_error_m', 0.0)), float(np.linalg.norm(f - p)))
            if error > 0.004:
                return None
            if abs(float(g['roll'])) > 2.79:
                return None
            return g
        except (RuntimeError, ValueError):
            return None

    def move(p, roll, duration=1.2):
        counts[0] += 1
        try:
            return api.move_pad(np.asarray(p).tolist(), float(roll), seconds=float(duration))
        except RuntimeError:
            return None

    def grip(jaw):
        counts[0] += 1
        try:
            return api.gripper(float(np.clip(jaw, -0.174, 1.75)), seconds=0.65)
        except RuntimeError:
            return None

    def choose_grasp(r, peg, attempt):
        c, normal, h, ai, bi, ao, bo = ring_geometry(r)
        plane = np.asarray(r['table_plane'], dtype=float)
        resolution = float(r.get('boundary_resolution_m', 0.002))
        gap = max(0.0010, 0.65 * resolution)
        bot = api.robot()
        current = np.asarray(bot['fixed_pad_m'], dtype=float)
        walls = list(r['walls'])
        walls.sort(key=lambda w: float(np.linalg.norm(np.asarray(w['outer_mid_m']) - current)))
        if attempt:
            walls = walls[attempt:] + walls[:attempt]
        candidates = []
        for inside_fixed in (False, True):
            for w in walls:
                if counts[1] >= 205:
                    break
                wall = np.asarray(w['outer_mid_m'], dtype=float)
                outward = unit(w['outward_normal'])
                outward = unit(outward - normal * np.dot(outward, normal))
                s = ai @ outward
                k = ai @ wall - bi
                lower = 0.0
                upper = 1e3
                valid = True
                for j in range(len(s)):
                    if s[j] > 1e-8:
                        lower = max(lower, float(k[j] / s[j]))
                    elif s[j] < -1e-8:
                        upper = min(upper, float(k[j] / s[j]))
                    elif k[j] > 0:
                        valid = False
                if not valid or lower >= upper or lower <= 0:
                    continue
                inner = wall - lower * outward
                direction = outward if inside_fixed else -outward
                target = inner - outward * 0.007 if inside_fixed else wall + outward * 0.007
                g = preview(target, 0.0, 0.60, direction)
                if g is None:
                    continue
                roll = float(g['roll'])
                target = np.asarray(g['fixed_pad_m'], dtype=float)
                target[2] += 0.0012 - table_clearance(g, plane)
                g = preview(target, roll, 0.60)
                if g is None:
                    continue
                fixed = clip_points(fingers(g)[0], normal, float(np.dot(normal, c)) + 0.0005)
                if len(fixed) == 0:
                    continue
                if inside_fixed:
                    shift = float(np.dot(inner, outward) - gap - np.max(fixed @ outward))
                else:
                    shift = float(np.dot(wall, outward) + gap - np.min(fixed @ outward))
                target = np.asarray(g['fixed_pad_m']) + shift * outward
                g = preview(target, roll, 0.60)
                if g is None:
                    continue
                target = np.asarray(g['fixed_pad_m'], dtype=float)
                target[2] += max(0.0, 0.0008 - table_clearance(g, plane))
                options = []
                for jaw in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.40, 1.60):
                    gp = preview(target, roll, jaw)
                    if gp is None or table_clearance(gp, plane) < 0.0003:
                        continue
                    actual_direction = unit(np.asarray(gp['moving_pad_m']) - np.asarray(gp['fixed_pad_m']))
                    if np.dot(actual_direction, direction) < 0.96:
                        continue
                    boxes = fingers(gp)
                    low = [clip_points(v, normal, float(np.dot(normal, c)) + 0.0005) for v in boxes]
                    if any(len(v) == 0 for v in low):
                        continue
                    if any(np.min(v @ normal) > np.dot(c, normal) - min(0.004, 0.28 * h) for v in boxes):
                        continue
                    inner_vertices = low[0] if inside_fixed else low[1]
                    outer_vertices = low[1] if inside_fixed else low[0]
                    room = float(-np.max(inner_vertices @ ai.T - bi))
                    outside_gap = float(np.min(outer_vertices @ outward) - np.dot(wall, outward))
                    if room < gap or outside_gap < 0.0002:
                        continue
                    rim_gap = float(np.dot(inner, outward) - np.max(inner_vertices @ outward))
                    if rim_gap > min(0.014, 0.7 * float(r['hole']['inscribed_radius_m'])):
                        continue
                    score = rim_gap + 0.25 * outside_gap + (0.008 if inside_fixed else 0.0)
                    score += 0.003 * np.linalg.norm(np.asarray(gp['fixed_pad_m']) - current)
                    options.append((score, jaw, gp))
                    if len(options) >= 2:
                        break
                if not options:
                    continue
                options.sort(key=lambda x: x[0])
                score, jaw, gp = options[0]
                fixed = np.asarray(gp['fixed_pad_m'], dtype=float)
                closed = preview(fixed, roll, -0.174)
                if closed is None:
                    continue
                cb = fingers(closed)
                separation = float(np.min(cb[1] @ direction) - np.max(cb[0] @ direction))
                if separation > lower - 0.001:
                    continue
                lift = max(0.055, float(peg['top_center_m'][2]) + 0.020 - float(r['minimum_z_m']))
                high = fixed + np.array([0.0, 0.0, lift])
                if preview(high, roll, jaw) is None:
                    continue
                delta = np.asarray(peg['top_center_m'], dtype=float) - c
                delta[2] = lift
                destination = fixed + delta
                if preview(destination, roll, jaw) is None:
                    continue
                score += 0.001 * abs(roll)
                candidates.append({'score': score, 'pad': fixed, 'roll': roll, 'jaw': jaw, 'high': high, 'inside': inside_fixed})
                if len(candidates) >= 3:
                    break
            if candidates:
                break
        if not candidates:
            return None
        candidates.sort(key=lambda x: x['score'])
        return candidates[min(attempt, len(candidates) - 1)]

    def visual_after_motion():
        r = observe_ring()
        if r is not None:
            return r
        bot = api.robot()
        p = np.asarray(bot['fixed_pad_m'], dtype=float)
        p[2] += 0.008
        if move(p, float(bot['roll']), 0.5) is None:
            return None
        return observe_ring()

    def horizontal_error(r, peg):
        c, n, h, ai, bi, ao, bo = ring_geometry(r)
        mid = c - 0.5 * h * n
        p = np.asarray(peg['top_center_m'], dtype=float)
        d = unit(peg['axis'])
        if abs(d[2]) < 0.5:
            return None
        goal = p + d * ((mid[2] - p[2]) / d[2])
        result = goal - mid
        result[2] = 0.0
        return result

    try:
        api.task()
        r = observe_ring()
        try:
            peg = api.observe('peg')
        except RuntimeError:
            api.wait(0.2)
            peg = api.observe('peg')
        if r is None:
            note('No current ring boundary.')
            return
        held = False
        grasp = None
        for attempt in range(2):
            r = observe_ring()
            if r is None:
                return
            initial_center = np.asarray(r['center_m'], dtype=float)
            initial_bottom = float(r['minimum_z_m'])
            grasp = choose_grasp(r, peg, attempt)
            if grasp is None:
                note('No collision-clear rim approach found.')
                return
            roll = grasp['roll']
            bot = api.robot()
            p = np.asarray(bot['fixed_pad_m'], dtype=float)
            safe_z = max(float(grasp['high'][2]), p[2] + 0.025)
            p[2] = safe_z
            if move(p, float(bot['roll']), 1.0) is None:
                return
            if grip(grasp['jaw']) is None:
                return
            above = grasp['pad'].copy()
            above[2] = safe_z
            if move(above, roll, 1.5) is None:
                continue
            fresh = observe_ring()
            if fresh is None:
                return
            if np.linalg.norm(np.asarray(fresh['center_m']) - initial_center) > 0.004:
                note('Ring moved before approach; replanning.')
                continue
            near = grasp['pad'].copy()
            near[2] += 0.020
            if move(near, roll, 0.8) is None:
                continue
            if move(grasp['pad'], roll, 0.8) is None:
                continue
            if grip(-0.174) is None:
                continue
            api.wait(0.2)
            bot = api.robot()
            lift_target = np.asarray(bot['fixed_pad_m'], dtype=float) + np.array([0.0, 0.0, 0.030])
            if move(lift_target, roll, 1.0) is None:
                return
            lifted = visual_after_motion()
            if lifted is None:
                note('Lift cannot be visually verified.')
                return
            rise = float(lifted['minimum_z_m']) - initial_bottom
            center_rise = float(lifted['center_m'][2]) - initial_center[2]
            if rise > 0.012 and center_rise > 0.015:
                held = True
                r = lifted
                note('Rim grasp verified by observed lift.')
                break
            note('Lift did not carry ring; retrying from current vision.')
            grip(grasp['jaw'])
            bot = api.robot()
            p = np.asarray(bot['fixed_pad_m'], dtype=float)
            p[2] += 0.045
            move(p, roll, 0.8)
            api.wait(0.3)
        if not held:
            return
        roll = float(grasp['roll'])
        top = np.asarray(peg['top_center_m'], dtype=float)
        # Keep the ring and the complete opening sweep above the peg. Reobserve
        # after each motion; rigid grasp attachment is never assumed.
        # Retain the commanded pad target to avoid accumulating gravity/tracking
        # error when issuing lateral moves from the measured, slightly lower pad.
        # This is controller memory, never a propagated object-pose measurement.
        command = np.asarray(api.robot()['fixed_pad_m'], dtype=float)
        for step in range(16):
            r = visual_after_motion()
            if r is None:
                note('Transport stopped: current ring geometry unavailable.')
                return
            bot = api.robot()
            fixed = np.asarray(bot['fixed_pad_m'], dtype=float)
            roll = float(bot['roll'])
            opened = preview(fixed, roll, grasp['jaw'])
            if opened is None:
                note('Transport stopped: release preview unreachable.')
                return
            lowest = min(float(r['minimum_z_m']),
                         float(np.min(np.asarray(bot['jaw_vertices_m'])[:, 2])),
                         float(np.min(np.asarray(opened['jaw_vertices_m'])[:, 2])))
            clearance = lowest - top[2]
            error = horizontal_error(r, peg)
            if error is None:
                note('Transport stopped: peg direction unresolved.')
                return
            if clearance >= 0.010 and np.linalg.norm(error[:2]) <= 0.003:
                if insertion_clear(r, peg, max(0.0015, float(r['boundary_resolution_m']))):
                    break
                note('Transport stopped: observed aperture cannot clear peg.')
                return
            delta = np.zeros(3)
            if clearance < 0.010:
                delta[2] = min(0.045, 0.016-clearance)
            else:
                delta[:2] = error[:2]*min(1., 0.05/max(np.linalg.norm(error[:2]), 1e-9))
            target = command+delta
            if move(target, roll, 1.0) is None:
                note('Transport stopped: target unreachable.')
                return
            command = target
        else:
            note('Transport stopped: alignment budget exhausted.')
            return
        note('Current aperture and jaw clearance verified; releasing above peg.')
        if grip(grasp['jaw']) is None:
            note('Release command failed.')
            return
        api.wait(0.6)
        bot = api.robot()
        retreat = np.asarray(bot['fixed_pad_m'], dtype=float)+np.array([0., 0., 0.075])
        if move(retreat, float(bot['roll']), 1.2) is None:
            note('Retreat failed after release.')
            return
        api.wait(2.0)
        final = observe_ring()
        if final is None:
            note('Release executed; final visual state unavailable.')
            return
        note('Release executed; final visible ring height %.4f m.' % float(final['center_m'][2]))
    except (RuntimeError, ValueError, KeyError, IndexError) as error:
        api.note('Starting policy stopped: '+str(error)[:300])
