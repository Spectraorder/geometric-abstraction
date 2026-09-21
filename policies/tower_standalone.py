"""Three-block tower with verified skills — standalone, lint-compliant version.

Same algorithm as policies/tower_skills.py + aspire_so100/skills.py +
aspire_so100/so100_kinematics.py, folded into one file that passes
agent_loop.validate_policy (math/numpy only, no private attributes) so it
runs under the official bwrap trial runner and can seed the LLM repair loop.

Motion: task-priority IK over the 5 joints — site position (hard), gripper
pointing straight down where the arm allows (minimum tilt otherwise), jaw
closing-axis yaw via the free wrist roll — commanded with
api.move_to_joint_positions. Gripper geometry measured once on the scene.
"""

import math
import numpy as np

ORDER = ["blue_block", "small_red_block", "green_block"]
TABLE_Z = 0.40
JAW_TABLE = np.array([[-0.174, 0.0131], [0.0, 0.0231], [0.1, 0.0290], [0.2, 0.0349], [0.25, 0.0378],
                      [0.3, 0.0408], [0.4, 0.0466], [0.5, 0.0522], [0.6, 0.0576], [0.8, 0.0677],
                      [1.2, 0.0832], [1.75, 0.0910]])
FIXED_PAD_FROM_SITE = 0.0316
PAD_BELOW_SITE = 0.020
KNOWN_HALF = {"blue_block": (0.04, 0.02, 0.025), "small_red_block": (0.025, 0.02, 0.02),
              "green_block": (0.02, 0.02, 0.02)}
LIMITS = np.array([[-1.92, 1.92], [-3.32, 0.174], [-0.174, 3.14], [-1.66, 1.66], [-2.79, 2.79]])
HOME = np.array([0.0, -1.57, 1.57, 1.57, -1.57])


# ----------------------------------------------------------------- math ---

def jaw_for_sep(sep):
    return float(np.clip(np.interp(sep, JAW_TABLE[:, 1], JAW_TABLE[:, 0]), -0.174, 1.75))


def sep_for_jaw(j):
    return float(np.interp(j, JAW_TABLE[:, 0], JAW_TABLE[:, 1]))


def quat_mat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrap_half_pi(a):
    return (a + math.pi / 2) % math.pi - math.pi / 2


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def unit_xy(a):
    return np.array([math.cos(a), math.sin(a), 0.0])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def axis_rot(axis, a):
    axis = np.asarray(axis, dtype=float)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


# --------------------------------------------------------- kinematics ---

CHAIN = (
    (np.array([0.0, -0.0452, 0.0165]), quat_mat((0.707105, 0.707108, 0, 0)), (0, 1, 0)),
    (np.array([0.0, 0.1025, 0.0306]), rot_x(1.57079), (1, 0, 0)),
    (np.array([0.0, 0.11257, 0.028]), rot_x(-1.57079), (1, 0, 0)),
    (np.array([0.0, 0.0052, 0.1349]), rot_x(-1.57079), (1, 0, 0)),
    (np.array([0.0, -0.0601, 0.0]), rot_y(1.57079), (0, 1, 0)),
)
SITE_IN_JAW = np.array([-0.020, -0.085, 0.0])
POINT_IN_JAW = np.array([0.0, -1.0, 0.0])
CLOSE_IN_JAW = np.array([1.0, 0.0, 0.0])


def fk(q):
    R = np.eye(3)
    p = np.array([0.0, 0.0, TABLE_Z])
    for (t, Rf, axis), a in zip(CHAIN, q):
        p = p + R @ t
        R = R @ Rf @ axis_rot(axis, float(a))
    return p + R @ SITE_IN_JAW, R


def closing_yaw(q):
    a = fk(q)[1] @ CLOSE_IN_JAW
    return math.atan2(a[1], a[0])


def tilt_deg(q):
    d = fk(q)[1] @ POINT_IN_JAW
    return math.degrees(math.acos(max(-1.0, min(1.0, -d[2]))))


def ik_errors(q, target, axis_yaw):
    p, R = fk(q)
    d = R @ POINT_IN_JAW
    a = R @ CLOSE_IN_JAW
    e_yaw = (math.atan2(a[1], a[0]) - axis_yaw + math.pi / 2) % math.pi - math.pi / 2
    return p - target, np.array([d[0], d[1], 0.6 * e_yaw])


def ik_jacobians(q, target, axis_yaw, eps=1e-6):
    e1, e2 = ik_errors(q, target, axis_yaw)
    J1, J2 = np.zeros((3, 5)), np.zeros((3, 5))
    for j in range(5):
        dq = np.zeros(5)
        dq[j] = eps
        f1, f2 = ik_errors(q + dq, target, axis_yaw)
        J1[:, j] = (f1 - e1) / eps
        J2[:, j] = (f2 - e2) / eps
    return e1, e2, J1, J2


def ik_solve_from(target, axis_yaw, q0, iters=80):
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    q = np.clip(np.asarray(q0, dtype=float).copy(), lo, hi)
    frozen = False
    hist = []
    for _ in range(iters):
        e1, e2, J1, J2 = ik_jacobians(q, target, axis_yaw)
        dq1 = -J1.T @ np.linalg.solve(J1 @ J1.T + 1e-4 * np.eye(3), e1)
        n1 = float(np.linalg.norm(dq1))
        if n1 > 0.3:
            dq1 *= 0.3 / n1
        dq2 = np.zeros(5)
        e1n = float(np.linalg.norm(e1))
        if not frozen and e1n < 0.003:
            N1 = np.eye(5) - np.linalg.pinv(J1, rcond=1e-6) @ J1
            J2n = J2 @ N1
            dq2 = -N1 @ (J2n.T @ np.linalg.solve(J2n @ J2n.T + 0.05 ** 2 * np.eye(3), e2))
            for j in range(5):
                if (q[j] <= lo[j] + 1e-6 and dq2[j] < 0) or (q[j] >= hi[j] - 1e-6 and dq2[j] > 0):
                    dq2[j] = 0.0
            n2 = float(np.linalg.norm(dq2))
            if n2 > 0.05:
                dq2 *= 0.05 / n2
            hist.append(float(np.linalg.norm(e2)))
            if len(hist) >= 8 and hist[-8] - hist[-1] < 1e-3:
                frozen = True
        q = np.clip(q + dq1 + dq2, lo, hi)
        if e1n < 0.0004 and (frozen or float(np.linalg.norm(dq2)) < 1e-5):
            break
    for _ in range(5):
        e1, _, J1, _ = ik_jacobians(q, target, axis_yaw)
        if float(np.linalg.norm(e1)) < 0.0002:
            break
        q = np.clip(q - J1.T @ np.linalg.solve(J1 @ J1.T + 1e-4 * np.eye(3), e1), lo, hi)
    p = fk(q)[0]
    return q, float(np.linalg.norm(p - target)), tilt_deg(q)


def ik_solve(target, axis_yaw, q0):
    target = np.asarray(target, dtype=float)
    seeds = [np.asarray(q0, dtype=float)]
    for s in (HOME, np.array([0.0, -1.2, 1.9, 0.9, 0.0]), np.array([0.0, -0.8, 1.1, 1.3, 0.0])):
        s = s.copy()
        s[0] = math.atan2(target[0], -target[1])
        s[4] = float(np.clip(wrap_half_pi(axis_yaw - math.pi / 2 - s[0]), -2.79, 2.79))
        seeds.append(s)
    best = None
    for k, seed in enumerate(seeds):
        q, err, tilt = ik_solve_from(target, axis_yaw, seed)
        cost = err * 100 + 0.01 * tilt
        if best is None or cost < best[3]:
            best = (q, err, tilt, cost)
        if err <= 0.0008 and (k > 0 or tilt <= 1.0):
            break
    return best[0], best[1]


# ------------------------------------------------------------- skills ---

class Block:
    def __init__(self, name, state):
        self.name = name
        self.pos = np.asarray(state["position"], dtype=float)
        self.R = quat_mat(state["quaternion_wxyz"])
        self.yaw = math.atan2(self.R[1, 0], self.R[0, 0])
        self.half = np.array(KNOWN_HALF[name], dtype=float)
        ext = state.get("half_extent_world")
        if ext is not None:
            self.half[2] = float(ext[2])
        self.top_z = float(self.pos[2] + np.abs(self.R[2]) @ self.half)
        self.lowest_z = float(self.pos[2] - np.abs(self.R[2]) @ self.half)
        self.tilt = math.degrees(math.acos(min(1.0, max(-1.0, float(np.max(np.abs(self.R[2])))))))


class Obs:
    def __init__(self, raw):
        self.raw = raw
        self.ee = np.asarray(raw["end_effector_position"], dtype=float)
        self.joints = np.asarray(raw["robot_joint_positions"], dtype=float)

    def block(self, name):
        return Block(name, self.raw["objects"][name])

    def names(self):
        return list(self.raw["objects"].keys())


class Skills:
    def __init__(self, api):
        self.api = api
        self.min_transit_z = TABLE_Z + 0.09

    def observe(self):
        return Obs(self.api.get_observation())

    def axis_yaw_now(self, obs):
        return closing_yaw(obs.joints)

    def move(self, pos, axis_yaw, seconds=1.2, precise=True, tol=0.0015, max_iter=2):
        target = np.asarray(pos, dtype=float)
        obs = self.observe()
        q, err = ik_solve(target, axis_yaw, obs.joints)
        if err > 0.004:
            wr = wrap_half_pi(axis_yaw - math.pi / 2 - math.atan2(target[0], -target[1]))
            self.api.move_end_effector(target.tolist(), seconds, wr)
            return self.observe()
        self.api.move_to_joint_positions(q.tolist(), seconds)
        obs = self.observe()
        if not precise:
            return obs
        cmd = target.copy()
        for _ in range(max_iter):
            resid = target - obs.ee
            n = float(np.linalg.norm(resid))
            if n <= tol or n > 0.03:
                break
            cmd = cmd + resid * min(1.0, 0.010 / n)
            q, err = ik_solve(cmd, axis_yaw, obs.joints)
            if err > 0.004:
                break
            self.api.move_to_joint_positions(q.tolist(), 0.5)
            obs = self.observe()
        return obs

    def set_width(self, w, seconds=0.6):
        self.api.set_gripper(jaw_for_sep(w), seconds)

    def grasp_geometry(self, blk, axis):
        hx, hy, hz = blk.half
        if axis == "x" or (axis == "short" and hx < hy):
            width, axis_yaw = 2 * hx, blk.yaw
        else:
            width, axis_yaw = 2 * hy, blk.yaw + math.pi / 2
        rest_z = blk.pos[2] - hz
        clearance = 0.010 if rest_z < TABLE_Z + 0.006 else 0.014
        grasp_z = min(rest_z + PAD_BELOW_SITE + clearance, blk.pos[2] + hz - 0.006)
        return wrap_half_pi(axis_yaw), width, max(grasp_z, blk.pos[2])

    def grasp(self, name, retries=2, lift=0.06, min_lift=0.025):
        axes = ["short", "x", "y"]
        for attempt in range(retries + 1):
            obs = self.observe()
            blk = obs.block(name)
            axis_yaw, width, grasp_z = self.grasp_geometry(blk, axes[min(attempt, 2)])
            open_w = min(width + 0.030, sep_for_jaw(1.75) - 0.004)
            self.set_width(open_w, 0.4)
            open_w = sep_for_jaw(jaw_for_sep(open_w))
            gap_centre = FIXED_PAD_FROM_SITE - open_w / 2
            squeeze = 0.004 + 0.002 * attempt
            above = np.array([blk.pos[0], blk.pos[1], grasp_z + 0.05])
            try:
                obs = self.move(above, axis_yaw, 1.0, precise=False)
                psi = self.axis_yaw_now(obs)
                obs = self.move(above - gap_centre * unit_xy(psi), axis_yaw, 0.6)
                psi = self.axis_yaw_now(obs)
                blk = obs.block(name)
                down = np.array([blk.pos[0], blk.pos[1], grasp_z]) - gap_centre * unit_xy(psi)
                self.move(down + np.array([0.0, 0.0, 0.03]), axis_yaw, 0.7, precise=False)
                obs = self.move(down, axis_yaw, 0.8)
                if float(np.linalg.norm(obs.ee - down)) > 0.012:
                    raise RuntimeError("descent blocked")
            except RuntimeError:
                continue
            self.set_width(max(width - squeeze, 0.012), 0.6)
            self.api.wait(0.2)
            before = self.observe().block(name)
            try:
                obs = self.move(np.array([down[0], down[1], grasp_z + lift]), axis_yaw, 1.0, precise=False)
            except RuntimeError:
                self.set_width(open_w, 0.4)
                continue
            after = obs.block(name)
            rose = float(after.pos[2] - before.pos[2])
            gripper_rose = float(obs.ee[2] - down[2])
            d = after.pos - obs.ee
            if rose >= min_lift and float(np.linalg.norm(d[:2])) < 0.035 and gripper_rose - rose <= 0.012:
                psi = self.axis_yaw_now(obs)
                return {"ok": True, "name": name, "axis_yaw": psi, "width": width, "half": blk.half.copy(),
                        "offset_jaw": rot_z(-psi) @ d, "yaw_rel": wrap_half_pi(after.yaw - psi)}
            self.set_width(open_w, 0.4)
            self.api.wait(0.2)
        return {"ok": False, "name": name}

    def clearance(self, obs, hz, held, start_xy, goal_xy):
        a, b = np.asarray(start_xy, dtype=float), np.asarray(goal_xy, dtype=float)
        ab = b - a
        L2 = float(ab @ ab)
        top = TABLE_Z
        for n in obs.names():
            if n == held:
                continue
            blk = obs.block(n)
            p = blk.pos[:2]
            t = 0.0 if L2 < 1e-9 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
            if float(np.linalg.norm(p - (a + t * ab))) <= float(np.hypot(blk.half[0], blk.half[1])) + 0.09:
                top = max(top, blk.top_z)
        return max(self.min_transit_z, top + 2 * hz + 0.045)

    def place(self, g, support, release_gap=0.002):
        name = g["name"]
        obs = self.observe()
        sup = obs.block(support)
        hz = float(g["half"][2])
        goal = np.array([sup.pos[0], sup.pos[1], sup.top_z + hz])
        transit = self.clearance(obs, hz, name, obs.ee[:2], goal[:2])
        psi_place = sup.yaw - g["yaw_rel"]
        try:
            cur = obs.ee
            obs = self.move([cur[0], cur[1], max(transit, cur[2])], self.axis_yaw_now(obs), 0.8, precise=False)
            above = goal - rot_z(psi_place) @ g["offset_jaw"]
            above[2] = transit
            obs = self.move(above, psi_place, 1.2, precise=False)
            psi = self.axis_yaw_now(obs)
            d_w = rot_z(psi) @ g["offset_jaw"]
            d_now = obs.block(name).pos - obs.ee
            if float(np.linalg.norm(d_now - d_w)) < 0.02:
                d_w = d_now
            ee_goal = goal - d_w
            self.move(ee_goal + np.array([0.0, 0.0, 0.020]), psi_place, 1.0)
            # descend on the held block's lowest point until it is release_gap above the support
            obs = self.move(ee_goal + np.array([0.0, 0.0, 0.008]), psi_place, 0.6)
            cmd = np.array(obs.ee, dtype=float)
            rel0 = float(obs.block(name).pos[2] - obs.ee[2])
            for _ in range(12):
                blk = obs.block(name)
                gap = blk.lowest_z - obs.block(support).top_z
                if gap <= release_gap or (blk.pos[2] - obs.ee[2]) - rel0 > 0.0015:
                    break
                cmd = cmd + np.array([0.0, 0.0, -min(0.002, max(gap - release_gap * 0.5, 0.0005))])
                obs = self.move(cmd, psi_place, 0.25, precise=False)
        except RuntimeError:
            return False
        self.set_width(min(g["width"] + 0.036, 0.088), 0.5)
        self.api.wait(0.3)
        obs = self.observe()
        psi = self.axis_yaw_now(obs)
        try:
            self.move(obs.ee + 0.006 * unit_xy(psi), psi, 0.4, precise=False)
            obs = self.observe()
            self.move([obs.ee[0], obs.ee[1], obs.ee[2] + 0.045], psi, 0.8, precise=False)
        except RuntimeError:
            pass
        self.api.wait(0.4)
        obs = self.observe()
        blk, sup = obs.block(name), obs.block(support)
        e_sup = sup.R[:2, :2].T @ (blk.pos[:2] - sup.pos[:2])
        tol = np.minimum(sup.half[:2] - 0.007, 0.012)
        z_err = abs(float(blk.pos[2]) - (sup.top_z + hz))
        return bool(np.all(np.abs(e_sup) <= tol)) and blk.tilt <= 6.0 and z_err <= 0.006

    def level_ok(self, obs, block, support):
        b, s = obs.block(block), obs.block(support)
        e = s.R[:2, :2].T @ (b.pos[:2] - s.pos[:2])
        if np.any(np.abs(e) > s.half[:2] - 0.003):
            return False
        if abs(b.lowest_z - s.top_z) > 0.004 or b.tilt > 8.0:
            return False
        return True

    def free_spot(self, obs, exclude, preferred):
        others = [obs.block(n).pos[:2] for n in obs.names() if n != exclude]
        if all(float(np.linalg.norm(preferred - o)) >= 0.115 for o in others):
            return preferred
        best = None
        for r in np.linspace(0.18, 0.28, 6):
            for th in np.linspace(-0.95, 0.95, 13):
                p = np.array([r * math.sin(th), -r * math.cos(th)])
                if all(float(np.linalg.norm(p - o)) >= 0.115 for o in others):
                    score = abs(th) + 2 * abs(r - 0.23)
                    if best is None or score < best[0]:
                        best = (score, p)
        return best[1] if best else preferred

    def park(self, name, xy):
        g = self.grasp(name)
        if not g["ok"]:
            return False
        spot = self.free_spot(self.observe(), name, np.asarray(xy, dtype=float))
        # place on the table: reuse place() machinery against a virtual support
        obs = self.observe()
        hz = float(g["half"][2])
        goal = np.array([spot[0], spot[1], TABLE_Z + hz])
        try:
            transit = self.clearance(obs, hz, name, obs.ee[:2], goal[:2])
            obs = self.move([obs.ee[0], obs.ee[1], max(transit, obs.ee[2])], self.axis_yaw_now(obs), 0.8, precise=False)
            above = goal - rot_z(g["axis_yaw"]) @ g["offset_jaw"]
            above[2] = transit
            obs = self.move(above, g["axis_yaw"], 1.2, precise=False)
            d_w = rot_z(self.axis_yaw_now(obs)) @ g["offset_jaw"]
            self.move(goal - d_w + np.array([0.0, 0.0, 0.004]), g["axis_yaw"], 1.0)
        except RuntimeError:
            pass
        self.set_width(min(g["width"] + 0.036, 0.088), 0.5)
        self.api.wait(0.3)
        self.go_clear()
        return True

    def go_clear(self):
        obs = self.observe()
        try:
            self.move([obs.ee[0], obs.ee[1], max(self.min_transit_z, obs.ee[2])], self.axis_yaw_now(obs), 0.8, precise=False)
        except RuntimeError:
            pass

    def pick_and_place(self, name, support, retries=1):
        for _ in range(retries + 1):
            g = self.grasp(name)
            if not g["ok"]:
                return False
            if self.place(g, support):
                return True
        return False


def run(api):
    sk = Skills(api)
    park_xy = {n: sk.observe().block(n).pos[:2].copy() for n in ORDER}
    i, repairs = 1, 0
    while i < len(ORDER):
        ok = sk.pick_and_place(ORDER[i], ORDER[i - 1])
        sk.go_clear()
        check = api.check_structure(ORDER[: i + 1], 2.0)
        if ok and check["stable"]:
            i += 1
            continue
        repairs += 1
        if repairs > 2:
            return
        # level-aware repair: find the lowest broken level, park everything above it, rebuild
        obs = sk.observe()
        bad = i
        for j in range(1, i + 1):
            if not sk.level_ok(obs, ORDER[j], ORDER[j - 1]):
                bad = j
                break
        for j in range(i, bad - 1, -1):
            sk.park(ORDER[j], park_xy[ORDER[j]])
        i = bad
