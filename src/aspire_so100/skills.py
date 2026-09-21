"""Reusable pick-and-stack skills over the SO-100 policy API.

Everything here uses only the ``TowerAPI`` surface — ``get_observation``,
``move_to_joint_positions`` / ``move_end_effector``, ``set_gripper``,
``wait`` — so the same skills run on ``StackingEnv`` (a superset) and under
the tower trial runner. No simulator internals.

Motion goes through :mod:`so100_kinematics`: a task-priority IK that holds
the site position exactly, points the gripper straight down wherever the
arm allows it (minimum tilt otherwise), and uses the free wrist roll to
align the jaw closing axis. The environment's position-only IK is kept as a
fallback. Gripper geometry was measured once with a probe script:

* pad separation vs jaw angle: piecewise-linear table ``_JAW_TABLE``
* fixed-pad inner face is 31.6 mm from ``ee_site`` along +closing axis;
  the moving pad sweeps in from the other side, so a block ends up
  pressed against the fixed pad
* pads extend ~20 mm below the site along the gripper pointing direction
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from aspire_so100.so100_kinematics import SO100Kinematics

TABLE_TOP_Z = 0.40

# (jaw joint angle [rad], pad separation [m]) — measured with FK.
_JAW_TABLE = np.array(
    [
        [-0.174, 0.0131], [0.0, 0.0231], [0.1, 0.0290], [0.2, 0.0349],
        [0.25, 0.0378], [0.3, 0.0408], [0.4, 0.0466], [0.5, 0.0522],
        [0.6, 0.0576], [0.8, 0.0677], [1.2, 0.0832], [1.75, 0.0910],
    ]
)
JAW_MIN, JAW_MAX = -0.174, 1.75
FIXED_PAD_FROM_SITE = 0.0316   # m, +closing axis, inner face of the fixed pad
PAD_BELOW_SITE = 0.020         # m, pads reach this far below ee_site

# Scene knowledge: local half-sizes (x, y, z) of the boxes. Used only when
# the observation does not let us recover them (see ``local_half_extents``).
KNOWN_HALF = {
    "small_red_block": (0.025, 0.020, 0.020),
    "blue_block": (0.040, 0.020, 0.025),
    "red_block": (0.040, 0.020, 0.025),
    "yellow_block": (0.040, 0.020, 0.025),
    "green_block": (0.020, 0.020, 0.020),   # tower scene cube
}


# ---------------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------------

def jaw_for_separation(sep_m: float) -> float:
    return float(np.clip(np.interp(sep_m, _JAW_TABLE[:, 1], _JAW_TABLE[:, 0]), JAW_MIN, JAW_MAX))


def separation_for_jaw(jaw: float) -> float:
    return float(np.interp(jaw, _JAW_TABLE[:, 0], _JAW_TABLE[:, 1]))


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_half_pi(a: float) -> float:
    """Wrap an *axis* angle (period pi) into (-pi/2, pi/2]."""
    return (a + math.pi / 2) % math.pi - math.pi / 2


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def unit_xy(a: float) -> np.ndarray:
    return np.array([math.cos(a), math.sin(a), 0.0])


# Calibrated relation used only for the position-only fallback IK:
# closing-axis yaw = pi/2 + base_rotation + wrist_roll (gripper near vertical).
def jaw_yaw(base_rot: float, wrist_roll: float) -> float:
    return math.pi / 2 + base_rot + wrist_roll


def wrist_roll_for_axis(axis_yaw: float, base_rot: float) -> float:
    return wrap_half_pi(axis_yaw - math.pi / 2 - base_rot)


def base_rot_guess(p) -> float:
    return math.atan2(float(p[0]), -float(p[1]))


def local_half_extents(name: str, ext_world, yaw: float, known: dict) -> np.ndarray:
    """Recover local box half-sizes from world-aligned half-extents.

    ext_world = |Rz(yaw)| @ half  ->  a 2x2 solve for (hx, hy); hz passes
    through. Falls back to scene knowledge when yaw ~ 45 deg (singular) or
    the observation carries no extents.
    """
    if ext_world is None:
        return np.array(known[name], dtype=float)
    ex, ey, ez = (float(v) for v in ext_world)
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    det = c * c - s * s
    if abs(det) < 0.25:  # within ~15 deg of 45: ill-conditioned
        if name in known:
            return np.array(known[name], dtype=float)
        return np.array([ex, ey, ez])
    hx = (c * ex - s * ey) / det
    hy = (c * ey - s * ex) / det
    if hx <= 0.004 or hy <= 0.004:
        return np.array(known.get(name, (ex, ey, ez)), dtype=float)
    return np.array([hx, hy, ez])


# ---------------------------------------------------------------------------
# Observation wrapper
# ---------------------------------------------------------------------------

@dataclass
class BlockState:
    name: str
    pos: np.ndarray
    quat: np.ndarray
    R: np.ndarray
    yaw: float
    half: np.ndarray            # local half-sizes (x, y, z)
    lin_vel: np.ndarray
    ang_vel: np.ndarray

    @property
    def tilt_deg(self) -> float:
        """Angle between the most-vertical local axis and world Z. A box that
        rolled 90 deg onto another face reads 0 (same convention as the
        tower evaluator); only genuine edge/corner tilts read non-zero."""
        return math.degrees(math.acos(float(np.clip(np.max(np.abs(self.R[2])), -1.0, 1.0))))

    @property
    def top_z(self) -> float:
        return float(self.pos[2] + np.abs(self.R[2]) @ self.half)

    @property
    def speed(self) -> float:
        return max(float(np.linalg.norm(self.lin_vel)), float(np.linalg.norm(self.ang_vel)))


class Obs:
    def __init__(self, raw: dict, known_half: dict):
        self.raw = raw
        self.ee = np.asarray(raw["end_effector_position"], dtype=float)
        self.joints = np.asarray(raw["robot_joint_positions"], dtype=float)
        self.base_rot = float(self.joints[0])
        self.wrist_roll = float(self.joints[4])
        self._known = known_half

    @property
    def names(self) -> list[str]:
        return list(self.raw["objects"].keys())

    def block(self, name: str) -> BlockState:
        s = self.raw["objects"][name]
        q = np.asarray(s["quaternion_wxyz"], dtype=float)
        R = quat_to_mat(q)
        yaw = math.atan2(R[1, 0], R[0, 0])
        if s.get("half_extent_local") is not None:
            half = np.asarray(s["half_extent_local"], dtype=float)
        else:
            half = local_half_extents(name, s.get("half_extent_world"), yaw, self._known)
        return BlockState(
            name=name,
            pos=np.asarray(s["position"], dtype=float),
            quat=q,
            R=R,
            yaw=yaw,
            half=half,
            lin_vel=np.asarray(s.get("linear_velocity", (0, 0, 0)), dtype=float),
            ang_vel=np.asarray(s.get("angular_velocity", (0, 0, 0)), dtype=float),
        )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class Grasp:
    ok: bool
    name: str
    axis_yaw: float = 0.0                     # actual closing-axis yaw at grasp (+ = fixed pad)
    width: float = 0.04                       # closing dimension of the block
    half: np.ndarray = field(default_factory=lambda: np.zeros(3))
    offset_jaw: np.ndarray = field(default_factory=lambda: np.zeros(3))  # block - ee, jaw frame
    yaw_rel: float = 0.0                      # block yaw - closing-axis yaw (mod pi)
    attempts: int = 0
    reason: str = ""


@dataclass
class Placement:
    ok: bool
    name: str
    xy_err: float = 1.0
    z_err: float = 1.0
    tilt_deg: float = 90.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

class Skills:
    """Pick-and-stack skills. ``api`` is a TowerAPI-like object."""

    def __init__(
        self,
        api: Any,
        table_z: float = TABLE_TOP_Z,
        known_half: dict | None = None,
        log: Callable[[str], None] | None = None,
        precise: bool = True,
        min_transit_clearance: float = 0.09,
    ):
        self.api = api
        self.table_z = table_z
        self.known = dict(KNOWN_HALF if known_half is None else known_half)
        self._log = log or (lambda s: None)
        self.precise = precise
        self.min_transit_z = table_z + min_transit_clearance
        self.kin = SO100Kinematics(base_pos=(0.0, 0.0, table_z))
        self.events: list[str] = []
        self.keep_out: list = []          # extra xy positions to avoid when parking
        self.ik_fallbacks = 0

    # -- basics -----------------------------------------------------------

    def log(self, msg: str) -> None:
        self.events.append(msg)
        self._log(msg)

    def observe(self) -> Obs:
        return Obs(self.api.get_observation(), self.known)

    def axis_yaw_now(self, obs: Obs) -> float:
        """Actual world yaw of the jaw closing axis (+ = fixed-pad side)."""
        return self.kin.closing_yaw(obs.joints)

    def move(self, pos, axis_yaw: float | None, seconds: float = 1.2, precise: bool | None = None,
             tol: float = 0.0015, max_iter: int = 2) -> Obs:
        """Move the ee_site to ``pos`` with the closing axis at ``axis_yaw``
        (mod pi) and the gripper as vertical as the arm allows. Optionally
        closes the loop on the measured site position to cancel servo sag.
        Raises RuntimeError when the position is unreachable."""
        target = np.asarray(pos, dtype=float)
        obs = self.observe()
        q, info = self.kin.solve(target, axis_yaw, q0=obs.joints)
        if info["pos_err"] > 0.004:
            # unreachable for our solver: try the environment's own IK with
            # the wrist roll pinned so the closing axis keeps its yaw
            self.ik_fallbacks += 1
            self.log(f"move: kin IK residual {info['pos_err']*1000:.1f}mm, falling back to env IK")
            wr = None if axis_yaw is None else wrist_roll_for_axis(axis_yaw, base_rot_guess(target))
            self.api.move_end_effector(target.tolist(), seconds, wr)
            return self.observe()
        self.api.move_to_joint_positions(q.tolist(), seconds)
        obs = self.observe()
        if not (self.precise if precise is None else precise):
            return obs
        cmd = target.copy()
        for _ in range(max_iter):
            resid = target - obs.ee
            n = float(np.linalg.norm(resid))
            if n <= tol:
                break
            if n > 0.03:
                self.log(f"move: residual {n*1000:.0f}mm, not correcting")
                break
            cmd = cmd + resid * min(1.0, 0.010 / n)   # bounded correction step
            q, info = self.kin.solve(cmd, axis_yaw, q0=obs.joints)
            if info["pos_err"] > 0.004:
                break
            self.api.move_to_joint_positions(q.tolist(), 0.5)
            obs = self.observe()
        return obs

    def set_width(self, width_m: float, seconds: float = 0.6) -> None:
        self.api.set_gripper(jaw_for_separation(width_m), seconds)

    # -- grasp ------------------------------------------------------------

    def _grasp_geometry(self, blk: BlockState, axis: str):
        """Return (closing_axis_yaw, width, grasp_z) for a block."""
        hx, hy, hz = blk.half
        if axis == "x":
            width, axis_yaw = 2 * hx, blk.yaw
        elif axis == "y":
            width, axis_yaw = 2 * hy, blk.yaw + math.pi / 2
        else:  # short
            if hy <= hx:
                width, axis_yaw = 2 * hy, blk.yaw + math.pi / 2
            else:
                width, axis_yaw = 2 * hx, blk.yaw
        # Site height: pads reach PAD_BELOW_SITE below the site. On the table
        # keep them ~10 mm off the surface; on top of another block keep
        # them higher (14 mm) so a slightly tilted gripper cannot brush the
        # block underneath. Never above the top face.
        rest_z = blk.pos[2] - hz
        on_table = rest_z < self.table_z + 0.006
        clearance = 0.010 if on_table else 0.014
        grasp_z = min(rest_z + PAD_BELOW_SITE + clearance, blk.pos[2] + hz - 0.006)
        grasp_z = max(grasp_z, blk.pos[2])          # not below the block centre
        return wrap_half_pi(axis_yaw), width, grasp_z

    def grasp(self, name: str, axis: str = "short", retries: int = 2,
              lift: float = 0.06, min_lift: float = 0.025) -> Grasp:
        """Yaw-aligned top-down grasp with a lift postcondition.

        Postcondition: after closing and lifting, the block rose with the
        gripper (no slip) and stayed near the site. A close command
        returning is never taken as evidence of a grasp.
        """
        axes = [axis] + (["x", "y"] if axis == "short" else ["short"])
        last_reason = ""
        for attempt in range(retries + 1):
            obs = self.observe()
            blk = obs.block(name)
            if blk.tilt_deg > 20:
                self.log(f"grasp[{name}] block tilted {blk.tilt_deg:.0f} deg; proceeding")
            ax = axes[min(attempt, len(axes) - 1)]
            axis_yaw, width, grasp_z = self._grasp_geometry(blk, ax)
            open_w = min(width + 0.030, separation_for_jaw(JAW_MAX) - 0.004)
            self.set_width(open_w, 0.4)
            open_w = separation_for_jaw(jaw_for_separation(open_w))   # what the jaw table can realise
            squeeze = 0.004 + 0.002 * attempt

            def centred(p, psi):
                # Centre the block between the *open* pads: the fixed pad face
                # is FIXED_PAD_FROM_SITE along +axis, the moving pad face is
                # (open_w - FIXED_PAD_FROM_SITE) along -axis.
                gap_centre = FIXED_PAD_FROM_SITE - open_w / 2      # from site, along +axis
                return p - gap_centre * unit_xy(psi)

            above = np.array([blk.pos[0], blk.pos[1], grasp_z + 0.05])
            try:
                obs = self.move(above, axis_yaw, 1.0, precise=False)
                psi = self.axis_yaw_now(obs)
                obs = self.move(centred(above, psi), axis_yaw, 0.6)
                psi = self.axis_yaw_now(obs)
                blk = obs.block(name)                     # re-read: may have been nudged
                down = centred(np.array([blk.pos[0], blk.pos[1], grasp_z]), psi)
                # two-stage descent keeps the solution continuous
                self.move(down + np.array([0.0, 0.0, 0.03]), axis_yaw, 0.7, precise=False)
                obs = self.move(down, axis_yaw, 0.8)
                if float(np.linalg.norm(obs.ee - down)) > 0.012:
                    raise RuntimeError(f"descent blocked ({np.linalg.norm(obs.ee - down)*1000:.0f}mm off)")
            except RuntimeError as exc:
                last_reason = f"ik: {exc}"
                self.log(f"grasp[{name}] attempt {attempt}: {last_reason}")
                continue

            self.set_width(max(width - squeeze, 0.012), 0.6)
            self.api.wait(0.2)
            before = self.observe().block(name)
            up = np.array([down[0], down[1], grasp_z + lift])
            try:
                obs = self.move(up, axis_yaw, 1.0, precise=False)
            except RuntimeError as exc:
                last_reason = f"ik lift: {exc}"
                self.log(f"grasp[{name}] attempt {attempt}: {last_reason}")
                self.set_width(open_w, 0.4)
                continue
            after = obs.block(name)
            rose = float(after.pos[2] - before.pos[2])
            gripper_rose = float(obs.ee[2] - down[2])
            d = after.pos - obs.ee
            near = float(np.linalg.norm(d[:2])) < 0.035
            slipped = gripper_rose - rose > 0.012        # block slid down in the jaws
            if rose >= min_lift and near and not slipped:
                psi = self.axis_yaw_now(obs)
                g = Grasp(
                    ok=True, name=name, axis_yaw=psi, width=width, half=blk.half.copy(),
                    offset_jaw=rot_z(-psi) @ d,
                    yaw_rel=wrap_half_pi(after.yaw - psi),
                    attempts=attempt + 1,
                )
                self.log(f"grasp[{name}] ok axis={ax} rose={rose*1000:.0f}mm "
                         f"offset={np.round(d*1000,1).tolist()}mm tilt={self.kin.tilt_deg(obs.joints):.1f}deg "
                         f"attempts={attempt+1}")
                return g
            last_reason = (f"lift check failed rose={rose*1000:.1f}mm of {gripper_rose*1000:.0f}mm "
                           f"near={near} slipped={slipped}")
            self.log(f"grasp[{name}] attempt {attempt}: {last_reason}")
            # release whatever we have and retry from above
            self.set_width(open_w, 0.4)
            self.api.wait(0.2)
        return Grasp(ok=False, name=name, attempts=retries + 1, reason=last_reason)

    # -- place ------------------------------------------------------------

    def place(self, grasp: Grasp, support: str, offset_xy=(0.0, 0.0), yaw_rel: float = 0.0,
              mode: str = "contact", release_gap: float = 0.002, xy_tol: float = 0.012,
              tilt_tol: float = 6.0) -> Placement:
        """Put the held block on ``support`` (live pose), release, retreat.

        ``mode='contact'`` descends in small steps until the held block's
        bottom is within ``release_gap`` of the live support top (or it
        stops following the gripper); ``'geometric'`` releases
        ``release_gap`` above the computed height.
        """
        obs = self.observe()
        sup = obs.block(support)
        off = sup.R[:2, :2] @ np.asarray(offset_xy, dtype=float)
        return self.place_at(grasp, (sup.pos[0] + off[0], sup.pos[1] + off[1]), sup.top_z,
                             sup.yaw + yaw_rel, support=support, offset_xy=offset_xy,
                             mode=mode, release_gap=release_gap, xy_tol=xy_tol, tilt_tol=tilt_tol)

    def place_on_table(self, grasp: Grasp, xy, yaw: float = 0.0, **kw) -> Placement:
        """Put the held block down on the table at world ``xy``."""
        return self.place_at(grasp, xy, self.table_z, yaw, support=None, **kw)

    def clearance_height(self, obs: Obs, hz: float, held: str | None, start_xy, goal_xy,
                         corridor: float = 0.09) -> float:
        """Transit height for a straight move from ``start_xy`` to
        ``goal_xy``: clear every object whose footprint is within
        ``corridor`` of the path, with the held block hanging below the site.
        Lower transits keep the gripper closer to vertical."""
        a, b = np.asarray(start_xy, dtype=float), np.asarray(goal_xy, dtype=float)
        ab = b - a
        L2 = float(ab @ ab)
        top = self.table_z
        for n in obs.names:
            if n == held:
                continue
            blk = obs.block(n)
            p = blk.pos[:2]
            t = 0.0 if L2 < 1e-9 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
            dist = float(np.linalg.norm(p - (a + t * ab)))
            reach = float(np.hypot(*blk.half[:2])) + corridor
            if dist <= reach:
                top = max(top, blk.top_z)
        hang = 2 * hz + 0.020            # block bottom is about this far below the site
        return max(self.min_transit_z, top + hang + 0.025)

    def place_at(self, grasp: Grasp, xy, surface_z: float, desired_yaw: float, support: str | None,
                 offset_xy=(0.0, 0.0), mode: str = "contact", release_gap: float = 0.002,
                 xy_tol: float = 0.012, tilt_tol: float = 6.0) -> Placement:
        name = grasp.name
        obs = self.observe()
        hz = float(grasp.half[2])
        goal = np.array([float(xy[0]), float(xy[1]), surface_z + hz])
        transit = self.clearance_height(obs, hz, name, obs.ee[:2], goal[:2])
        # closing-axis yaw that puts the block at the desired yaw
        psi_place = desired_yaw - grasp.yaw_rel

        try:
            cur = obs.ee
            obs = self.move([cur[0], cur[1], max(transit, cur[2])], self.axis_yaw_now(obs), 0.8, precise=False)
            # move above the goal; the in-hand offset rotates with the jaw, so
            # iterate once with the actual closing-axis yaw after the move
            d_w = rot_z(psi_place) @ grasp.offset_jaw
            above = goal - d_w
            above[2] = transit
            obs = self.move(above, psi_place, 1.2, precise=False)
            psi = self.axis_yaw_now(obs)
            d_w = rot_z(psi) @ grasp.offset_jaw
            held = obs.block(name)
            d_now = held.pos - obs.ee
            if float(np.linalg.norm(d_now - d_w)) < 0.02:
                d_w = d_now                            # trust the live in-hand offset
            ee_goal = goal - d_w
            hover = ee_goal + np.array([0.0, 0.0, 0.020])
            obs = self.move(hover, psi_place, 1.0)
        except RuntimeError as exc:
            return Placement(False, name, reason=f"ik transit: {exc}")

        try:
            if mode == "contact":
                self._descend_on_block(name, support, hz, ee_goal, psi_place, release_gap, surface_z)
            else:
                self.move(ee_goal + np.array([0, 0, release_gap]), psi_place, 0.8)
        except RuntimeError as exc:
            return Placement(False, name, reason=f"ik descend: {exc}")

        # Release: open wide, let the block settle, back the fixed pad away
        # from the block face (it was pressed against it during the grasp,
        # and lifting with it in contact drags the block), then leave
        # straight up.
        self.set_width(min(grasp.width + 0.036, 0.088), 0.5)
        self.api.wait(0.3)
        obs = self.observe()
        psi = self.axis_yaw_now(obs)
        try:
            self.move(obs.ee + 0.006 * unit_xy(psi), psi, 0.4, precise=False)
            obs = self.observe()
            self.move([obs.ee[0], obs.ee[1], obs.ee[2] + 0.045], psi, 0.8, precise=False)
        except RuntimeError as exc:
            self.log(f"place[{name}] retreat ik: {exc}")
        self.api.wait(0.4)

        obs = self.observe()
        blk = obs.block(name)
        if support is not None:
            sup = obs.block(support)
            want_xy = sup.pos[:2] + (sup.R[:2, :2] @ np.asarray(offset_xy, dtype=float))
            want_z = sup.top_z + hz
            e_sup = sup.R[:2, :2].T @ (blk.pos[:2] - want_xy)
            tol_axis = np.minimum(sup.half[:2] - 0.007, xy_tol)
            centred_ok = bool(np.all(np.abs(e_sup) <= tol_axis))
        else:
            want_xy = np.asarray(xy, dtype=float)
            want_z = surface_z + hz
            centred_ok = float(np.linalg.norm(blk.pos[:2] - want_xy)) <= xy_tol
        xy_err = float(np.linalg.norm(blk.pos[:2] - want_xy))
        z_err = float(abs(blk.pos[2] - want_z))
        ok = centred_ok and blk.tilt_deg <= tilt_tol and z_err <= 0.006
        res = Placement(ok, name, xy_err, z_err, blk.tilt_deg,
                        "" if ok else f"xy={xy_err*1000:.1f}mm z={z_err*1000:.1f}mm tilt={blk.tilt_deg:.1f}")
        self.log(f"place[{name} on {support or 'table'}] {'ok' if ok else 'BAD'} "
                 f"xy={xy_err*1000:.1f}mm z={z_err*1000:.1f}mm tilt={blk.tilt_deg:.1f}deg")
        return res

    def _descend_on_block(self, name: str, support: str | None, hz: float, ee_goal: np.ndarray,
                          psi: float, release_gap: float, surface_z: float, step: float = 0.002,
                          max_steps: int = 12) -> None:
        """Close the loop on the *held block's* height: step the gripper down
        until the block's lowest point is within ``release_gap`` of the live
        support top (or the block stops following — it has touched down).

        The lowest point comes from the block's live orientation: a block
        held at the arm's minimum tilt leads with a corner several mm below
        its centre-frame bottom, and releasing on the centre height would
        keep pushing that corner into the support.
        """
        obs = self.move(ee_goal + np.array([0, 0, 0.008]), psi, 0.6)
        cmd = np.array(obs.ee, dtype=float)
        rel0 = float(obs.block(name).pos[2] - obs.ee[2])
        for _ in range(max_steps):
            blk = obs.block(name)
            top = obs.block(support).top_z if support is not None else surface_z
            lowest = float(blk.pos[2] - np.abs(blk.R[2]) @ blk.half)
            gap = lowest - top
            rel = float(blk.pos[2] - obs.ee[2])
            if gap <= release_gap:
                self.log(f"place: block lowest point {gap*1000:.1f}mm above support "
                         f"(tilt {blk.tilt_deg:.1f}deg) -> release")
                return
            if rel - rel0 > 0.0015:
                self.log(f"place: block stopped following gripper (+{(rel-rel0)*1000:.1f}mm) -> touched down")
                return
            cmd = cmd + np.array([0.0, 0.0, -min(step, max(gap - release_gap * 0.5, 0.0005))])
            obs = self.move(cmd, psi, 0.25, precise=False)
        self.log("place: descent step budget exhausted")

    # -- structure inspection & repair -------------------------------------

    def level_ok(self, obs: Obs, block: str, support: str, offset_xy=(0.0, 0.0),
                 margin: float = 0.003) -> tuple[bool, str]:
        """Does ``block`` sit properly on ``support``? Mirrors the evaluator:
        COM inside the support face with ``margin``, correct height, flat."""
        b, s = obs.block(block), obs.block(support)
        want_xy = s.pos[:2] + (s.R[:2, :2] @ np.asarray(offset_xy, dtype=float))
        e_sup = s.R[:2, :2].T @ (b.pos[:2] - want_xy)
        if np.any(np.abs(e_sup) > s.half[:2] - margin):
            return False, f"{block}: off {support} face by {np.round(e_sup*1000,1).tolist()}mm"
        z_gap = float(b.pos[2] - np.abs(b.R[2]) @ b.half - s.top_z)
        if abs(z_gap) > 0.004:
            return False, f"{block}: height gap {z_gap*1000:.1f}mm over {support}"
        if b.tilt_deg > 8.0:
            return False, f"{block}: tilted {b.tilt_deg:.1f}deg"
        return True, ""

    def first_bad_level(self, chain: list[tuple[str, str, tuple]]) -> int | None:
        """``chain`` = [(block, support, offset_xy), ...] bottom-up. Returns the
        index of the lowest level that is not properly placed, or None."""
        obs = self.observe()
        for i, (block, support, off) in enumerate(chain):
            ok, why = self.level_ok(obs, block, support, off)
            if not ok:
                self.log(f"level {i} bad: {why}")
                return i
        return None

    def free_spot(self, obs: Obs, exclude: str, preferred=None, min_dist: float = 0.115,
                  r_range=(0.18, 0.28)) -> np.ndarray:
        """A table position at least ``min_dist`` from every other block
        (and any keep-out points), inside the reach band. Prefers
        ``preferred`` when it is free; otherwise scans a polar grid nearest
        the robot's forward direction first."""
        others = [obs.block(n).pos[:2] for n in obs.names if n != exclude]
        others += [np.asarray(p[:2], dtype=float) for p in self.keep_out]

        def ok(p):
            return all(float(np.linalg.norm(p - q)) >= min_dist for q in others)

        if preferred is not None and ok(np.asarray(preferred, dtype=float)):
            return np.asarray(preferred, dtype=float)
        best = None
        for r in np.linspace(r_range[0], r_range[1], 6):
            for th in np.linspace(-0.95, 0.95, 13):
                p = np.array([r * math.sin(th), -r * math.cos(th)])
                if ok(p):
                    score = abs(th) + 2 * abs(r - 0.23)
                    if best is None or score < best[0]:
                        best = (score, p)
        return best[1] if best else np.asarray(preferred if preferred is not None else (0.0, -0.23), dtype=float)

    def park(self, name: str, xy) -> bool:
        """Move a block out of the way onto a free table spot near ``xy``."""
        g = self.grasp(name)
        if not g.ok:
            self.log(f"park[{name}] grasp failed: {g.reason}")
            return False
        spot = self.free_spot(self.observe(), exclude=name, preferred=xy)
        r = self.place_on_table(g, spot, 0.0, xy_tol=0.03)
        self.go_clear()
        return r.ok

    # -- composite --------------------------------------------------------

    def pick_and_place(self, name: str, support: str, offset_xy=(0.0, 0.0), yaw_rel: float = 0.0,
                       place_retries: int = 1, **place_kw) -> Placement:
        """Grasp ``name`` and place it on ``support``; re-grasp and re-place
        once if the placement postcondition fails."""
        result = Placement(False, name, reason="not attempted")
        for attempt in range(place_retries + 1):
            g = self.grasp(name)
            if not g.ok:
                self.log(f"pick_and_place[{name}] grasp failed: {g.reason}")
                return Placement(False, name, reason=f"grasp: {g.reason}")
            result = self.place(g, support, offset_xy, yaw_rel, **place_kw)
            if result.ok:
                return result
            self.log(f"pick_and_place[{name}] placement off ({result.reason}); retry {attempt+1}")
        return result

    def go_clear(self) -> None:
        """Lift straight up to a clear height (call before stability checks)."""
        obs = self.observe()
        z = max(self.min_transit_z, obs.ee[2])
        try:
            self.move([obs.ee[0], obs.ee[1], z], self.axis_yaw_now(obs), 0.8, precise=False)
        except RuntimeError:
            pass
