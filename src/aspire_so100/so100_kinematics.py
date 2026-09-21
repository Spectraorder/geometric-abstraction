"""SO-100 forward kinematics and a 5-DOF inverse kinematics with gripper
pointing control.

The arm has base yaw, three parallel pitch joints and a wrist roll about the
gripper's pointing axis. Position-only IK (the environment's built-in
solver) leaves the gripper pitch free, and the damped-least-squares solution
it lands on can pitch the gripper 25 deg at high transit poses — long blocks
then rotate in the narrow pads and drop. Pointing the gripper straight down
costs one constraint on the pitch sum, position takes base yaw plus two
pitches, and wrist roll remains free for yaw alignment: the system is
exactly determined, so we solve it directly here and command joints.

Frames are taken verbatim from ``assets/trs_so_arm100/so_arm100.xml`` with
the robot root at world ``(0, 0, TABLE_HEIGHT)``; ``validate_against_mujoco``
checks the chain against the compiled scene.
"""

from __future__ import annotations

import math

import numpy as np

TABLE_HEIGHT = 0.40
JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
JOINT_LIMITS = np.array([[-1.92, 1.92], [-3.32, 0.174], [-0.174, 3.14], [-1.66, 1.66], [-2.79, 2.79]])
HOME = np.array([0.0, -1.57, 1.57, 1.57, -1.57])


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _quat_to_mat(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _axis_rot(axis, a):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


# (fixed translation, fixed rotation, joint axis) for each link, in parent frame
_CHAIN = (
    (np.array([0.0, -0.0452, 0.0165]), _quat_to_mat(0.707105, 0.707108, 0, 0), (0, 1, 0)),   # Rotation_Pitch
    (np.array([0.0, 0.1025, 0.0306]), _rot_x(1.57079), (1, 0, 0)),                            # Upper_Arm
    (np.array([0.0, 0.11257, 0.028]), _rot_x(-1.57079), (1, 0, 0)),                           # Lower_Arm
    (np.array([0.0, 0.0052, 0.1349]), _rot_x(-1.57079), (1, 0, 0)),                           # Wrist_Pitch_Roll
    (np.array([0.0, -0.0601, 0.0]), _rot_y(1.57079), (0, 1, 0)),                              # Fixed_Jaw
)
SITE_IN_JAW = np.array([-0.020, -0.085, 0.0])
POINT_IN_JAW = np.array([0.0, -1.0, 0.0])      # gripper pointing direction
CLOSE_IN_JAW = np.array([1.0, 0.0, 0.0])       # jaw closing axis (+ = fixed-pad side)


class SO100Kinematics:
    def __init__(self, base_pos=(0.0, 0.0, TABLE_HEIGHT)):
        self.base_pos = np.asarray(base_pos, dtype=float)

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Return (site position, Fixed_Jaw rotation matrix) in world."""
        R = np.eye(3)
        p = self.base_pos.copy()
        for (t, Rf, axis), a in zip(_CHAIN, q):
            p = p + R @ t
            R = R @ Rf @ _axis_rot(axis, float(a))
        return p + R @ SITE_IN_JAW, R

    def site(self, q) -> np.ndarray:
        return self.fk(q)[0]

    def pointing(self, q) -> np.ndarray:
        return self.fk(q)[1] @ POINT_IN_JAW

    def closing_axis(self, q) -> np.ndarray:
        return self.fk(q)[1] @ CLOSE_IN_JAW

    def closing_yaw(self, q) -> float:
        a = self.closing_axis(q)
        return math.atan2(a[1], a[0])

    def tilt_deg(self, q) -> float:
        d = self.pointing(q)
        return math.degrees(math.acos(max(-1.0, min(1.0, -d[2]))))

    # ------------------------------------------------------------------
    def solve(self, target, axis_yaw: float | None, q0, *, vertical: bool = True,
              pos_tol: float = 0.0008, ang_tol_deg: float = 1.0, iters: int = 80) -> tuple[np.ndarray, dict]:
        """Task-priority damped least squares over the 5 joints.

        Priority 1: site position (hard). Priority 2: gripper pointing
        straight down, solved in the nullspace of position — exact where the
        arm allows it, minimum tilt otherwise. Priority 3: closing-axis yaw
        (mod pi) in the remaining nullspace (the wrist roll).
        Returns the joint vector and diagnostics (pos_err, tilt_deg,
        yaw_err_deg, ok); ``ok`` requires only the *position* tolerance and
        yaw tolerance — tilt is reported, not required. Tries the given seed
        first, then a few canonical seeds, keeping the lowest-tilt solution.
        """
        target = np.asarray(target, dtype=float)
        seeds = [np.asarray(q0, dtype=float)]
        for s in (HOME, np.array([0.0, -1.2, 1.9, 0.9, 0.0]), np.array([0.0, -0.8, 1.1, 1.3, 0.0])):
            s = s.copy()
            s[0] = math.atan2(target[0] - self.base_pos[0], -(target[1] - self.base_pos[1]))
            if axis_yaw is not None:
                s[4] = float(np.clip(((axis_yaw - math.pi / 2 - s[0]) + math.pi / 2) % math.pi - math.pi / 2, -2.79, 2.79))
            seeds.append(s)
        best = None
        for k, seed in enumerate(seeds):
            q, info = self._solve_from(target, axis_yaw, seed, vertical, pos_tol, ang_tol_deg, iters)
            if best is None or info["cost"] < best[1]["cost"]:
                best = (q, info)
            # the current-configuration seed is preferred when it is good enough
            if info["ok"] and (k == 0 and info["tilt_deg"] <= ang_tol_deg or k > 0):
                break
        return best

    def _errors(self, q, target, axis_yaw, vertical):
        """Per-priority error vectors e = actual - desired."""
        p, R = self.fk(q)
        e1 = p - target
        errs = [e1]
        if vertical:
            d = R @ POINT_IN_JAW
            errs.append(np.array([d[0], d[1]]))            # want (0, 0, -1)
        if axis_yaw is not None:
            a = R @ CLOSE_IN_JAW
            errs.append(np.array([(math.atan2(a[1], a[0]) - axis_yaw + math.pi / 2) % math.pi - math.pi / 2]))
        return errs

    def _jacobians(self, q, target, axis_yaw, vertical, eps=1e-6):
        base = self._errors(q, target, axis_yaw, vertical)
        Js = [np.zeros((e.size, 5)) for e in base]
        for j in range(5):
            dq = np.zeros(5)
            dq[j] = eps
            pert = self._errors(q + dq, target, axis_yaw, vertical)
            for J, e0, e1 in zip(Js, base, pert):
                J[:, j] = (e1 - e0) / eps
        return base, Js

    @staticmethod
    def _dpinv(J, lam):
        return J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(J.shape[0]), np.eye(J.shape[0]))

    def _solve_from(self, target, axis_yaw, q0, vertical, pos_tol, ang_tol_deg, iters):
        q = np.clip(np.asarray(q0, dtype=float).copy(), JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        lo, hi = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
        w_sec = (1.0, 0.6)          # pointing, yaw — one least-squares secondary task
        sec_frozen = False
        sec_hist: list[float] = []
        for it in range(iters):
            errs, Js = self._jacobians(q, target, axis_yaw, vertical)
            e1, J1 = errs[0], Js[0]
            # primary: damped Gauss-Newton on position
            dq1 = -J1.T @ np.linalg.solve(J1 @ J1.T + 1e-4 * np.eye(3), e1)
            n1 = float(np.linalg.norm(dq1))
            if n1 > 0.3:
                dq1 *= 0.3 / n1
            dq2 = np.zeros(5)
            e1n = float(np.linalg.norm(e1))
            if len(errs) > 1 and not sec_frozen and e1n < 0.003:
                # Secondary task only once position has (nearly) converged,
                # in the exact nullspace of the position Jacobian, with a
                # small step: the next primary step repairs any leak.
                N1 = np.eye(5) - np.linalg.pinv(J1, rcond=1e-6) @ J1
                e2 = np.concatenate([(w_sec[k] if k < len(w_sec) else 1.0) * e for k, e in enumerate(errs[1:])])
                J2 = np.vstack([(w_sec[k] if k < len(w_sec) else 1.0) * J for k, J in enumerate(Js[1:])])
                J2n = J2 @ N1
                dq2 = -N1 @ (J2n.T @ np.linalg.solve(J2n @ J2n.T + 0.05 ** 2 * np.eye(J2n.shape[0]), e2))
                # never push a saturated joint further into its limit
                for j in range(5):
                    if (q[j] <= lo[j] + 1e-6 and dq2[j] < 0) or (q[j] >= hi[j] - 1e-6 and dq2[j] > 0):
                        dq2[j] = 0.0
                n2 = float(np.linalg.norm(dq2))
                if n2 > 0.05:
                    dq2 *= 0.05 / n2
                # freeze the secondary task when it stops making progress
                # (pointing infeasible at this pose: minimum tilt reached)
                sec_hist.append(float(np.linalg.norm(e2)))
                if len(sec_hist) >= 8 and sec_hist[-8] - sec_hist[-1] < 1e-3:
                    sec_frozen = True
            q = np.clip(q + dq1 + dq2, lo, hi)
            if e1n < pos_tol * 0.5 and (len(errs) == 1 or sec_frozen or float(np.linalg.norm(dq2)) < 1e-5):
                break
        # final polish on position only
        for _ in range(5):
            errs, Js = self._jacobians(q, target, None, False)
            e1, J1 = errs[0], Js[0]
            if float(np.linalg.norm(e1)) < pos_tol * 0.25:
                break
            q = np.clip(q - J1.T @ np.linalg.solve(J1 @ J1.T + 1e-4 * np.eye(3), e1), lo, hi)
        p, R = self.fk(q)
        pos_err = float(np.linalg.norm(target - p))
        tilt = self.tilt_deg(q)
        if axis_yaw is not None:
            a = R @ CLOSE_IN_JAW
            yaw_err = math.degrees((math.atan2(a[1], a[0]) - axis_yaw + math.pi / 2) % math.pi - math.pi / 2)
        else:
            yaw_err = 0.0
        ok = pos_err <= pos_tol and abs(yaw_err) <= max(ang_tol_deg, 2.0)
        cost = pos_err * 100 + (0.01 * tilt if vertical else 0.0) + 0.002 * abs(yaw_err)
        return q, {"ok": ok, "pos_err": pos_err, "tilt_deg": tilt, "yaw_err_deg": yaw_err, "cost": cost}


def validate_against_mujoco(model, data, arm_qpos_addresses, site_id, jaw_body_id, n: int = 200, seed: int = 0):
    """Compare this FK with MuJoCo's on random joint configurations."""
    import mujoco

    kin = SO100Kinematics()
    rng = np.random.default_rng(seed)
    worst_p, worst_ang = 0.0, 0.0
    dd = mujoco.MjData(model)
    for _ in range(n):
        q = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        dd.qpos[:] = data.qpos
        dd.qpos[arm_qpos_addresses] = q
        mujoco.mj_forward(model, dd)
        p, R = kin.fk(q)
        worst_p = max(worst_p, float(np.linalg.norm(p - dd.site_xpos[site_id])))
        Rm = dd.xmat[jaw_body_id].reshape(3, 3)
        cosang = (np.trace(R.T @ Rm) - 1) / 2
        worst_ang = max(worst_ang, math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
    return worst_p, worst_ang
