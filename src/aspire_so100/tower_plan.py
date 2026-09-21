"""Tower plans: a typed spec compiled into placement steps for the skill layer.

The plan is the only thing that varies with the *task*; block positions vary
with the *scene* and are read at run time by the skills. Compilation does
three things an LLM should not be trusted to get right every time:

1. resolves block *selectors* (name, colour, size, footprint) against the
   perceived inventory — so "a small red block" or "the widest block" bind
   to whatever is actually on the table;
2. checks the support DAG is ordered (every support precedes its load) and
   the static geometry is plausible (centre of mass inside the support face
   with margin, base within reach);
3. emits ordered ``Step``s that ``execute_plan`` runs with the skills,
   re-perceiving every support live before each placement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aspire_so100.skills import Obs, Placement, Skills, TABLE_TOP_Z

COM_MARGIN = 0.003          # evaluator margin on the support face
REACH_MIN, REACH_MAX = 0.14, 0.31   # radial reach band for the tower base [m]


@dataclass
class PlacementSpec:
    block: Any                          # str name, or selector dict
    support: Any = "table"              # "table" or index of an earlier placement
    at: tuple | None = None             # world xy for a table placement (None: leave where it is)
    offset: tuple = (0.0, 0.0)          # xy offset in the support's frame [m]
    yaw_rel: float = 0.0                # yaw relative to the support [rad]


@dataclass
class TowerPlan:
    placements: list[PlacementSpec]
    name: str = "tower"

    @classmethod
    def from_dict(cls, d: dict) -> "TowerPlan":
        ps = [PlacementSpec(p["block"], p.get("support", "table"),
                            tuple(p["at"]) if p.get("at") is not None else None,
                            tuple(p.get("offset", (0.0, 0.0))), float(p.get("yaw_rel", 0.0)))
              for p in d["placements"]]
        return cls(ps, d.get("name", "tower"))


@dataclass
class Step:
    block: str
    support: str | None                 # block name, or None for the table
    at: tuple | None                    # world xy (table placements)
    offset: tuple
    yaw_rel: float
    move_required: bool = True


@dataclass
class CompiledPlan:
    steps: list[Step]
    assignment: dict[int, str]
    warnings: list[str] = field(default_factory=list)


class PlanError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

def _footprint(half) -> tuple[float, float]:
    """(long, short) horizontal side lengths."""
    a, b = 2 * float(half[0]), 2 * float(half[1])
    return (max(a, b), min(a, b))


def matches(selector: Any, name: str, half) -> bool:
    if isinstance(selector, str):
        return selector == name
    sel = dict(selector)
    if "name" in sel and sel["name"] != name:
        return False
    if "color" in sel:
        tokens = name.split("_")
        if sel["color"] not in tokens:
            return False
    if "size" in sel:
        small = name.startswith("small_")
        if (sel["size"] == "small") != small:
            return False
    fp = _footprint(half)
    if "min_footprint" in sel:
        lo = _footprint([v / 2 for v in sel["min_footprint"]] + [0])
        if fp[0] < lo[0] - 1e-6 or fp[1] < lo[1] - 1e-6:
            return False
    if "max_footprint" in sel:
        hi = _footprint([v / 2 for v in sel["max_footprint"]] + [0])
        if fp[0] > hi[0] + 1e-6 or fp[1] > hi[1] + 1e-6:
            return False
    if "extent" in sel:
        want = np.sort(np.asarray(sel["extent"], dtype=float))
        have = np.sort(2 * np.asarray(half, dtype=float))
        if np.any(np.abs(want - have) > 0.004):
            return False
    if "height" in sel and abs(2 * float(half[2]) - float(sel["height"])) > 0.004:
        return False
    return True


def _selector_specificity(selector: Any) -> int:
    return 100 if isinstance(selector, str) else len(selector)


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

def compile_plan(plan: TowerPlan, obs: Obs, table_z: float = TABLE_TOP_Z) -> CompiledPlan:
    inventory = {n: obs.block(n) for n in obs.names}
    assignment: dict[int, str] = {}
    warnings: list[str] = []

    # 1. bind selectors, most specific first, then in plan order
    order = sorted(range(len(plan.placements)),
                   key=lambda i: (-_selector_specificity(plan.placements[i].block), i))
    for i in order:
        sel = plan.placements[i].block
        cands = [n for n, b in inventory.items() if n not in assignment.values() and matches(sel, n, b.half)]
        if not cands:
            raise PlanError(f"placement {i}: no unassigned block matches {sel!r}")
        # deterministic tie-break: prefer the block closest to the robot base
        cands.sort(key=lambda n: float(np.linalg.norm(inventory[n].pos[:2])))
        assignment[i] = cands[0]

    # 2. support DAG + static geometry
    steps: list[Step] = []
    for i, p in enumerate(plan.placements):
        name = assignment[i]
        blk = inventory[name]
        if p.support == "table":
            if p.at is None:
                r = float(np.linalg.norm(blk.pos[:2]))
                if not REACH_MIN <= r <= REACH_MAX:
                    warnings.append(f"{name}: base at radius {r:.2f} m is outside the comfortable reach band")
                steps.append(Step(name, None, None, (0.0, 0.0), p.yaw_rel, move_required=False))
            else:
                r = float(np.linalg.norm(np.asarray(p.at)))
                if not REACH_MIN <= r <= REACH_MAX:
                    raise PlanError(f"{name}: requested base position {p.at} is out of reach (r={r:.2f})")
                far = float(np.linalg.norm(blk.pos[:2] - np.asarray(p.at))) > 0.012
                steps.append(Step(name, None, tuple(p.at), (0.0, 0.0), p.yaw_rel, move_required=far))
            continue
        if not isinstance(p.support, int) or not 0 <= p.support < i:
            raise PlanError(f"placement {i}: support must be 'table' or an earlier placement index, got {p.support!r}")
        sup_name = assignment[p.support]
        sup = inventory[sup_name]
        # COM of this block must land inside the support's top face with margin
        # (rough static check; the upper-substack COM is checked by the evaluator)
        ox, oy = (abs(float(v)) for v in p.offset)
        if ox > sup.half[0] - COM_MARGIN or oy > sup.half[1] - COM_MARGIN:
            raise PlanError(f"placement {i}: offset {p.offset} puts {name}'s COM off {sup_name}'s face "
                            f"(half {sup.half[:2].round(3).tolist()})")
        # footprint sanity: overhang beyond half the block is a tipping risk
        c, s = abs(math.cos(p.yaw_rel)), abs(math.sin(p.yaw_rel))
        fx = c * blk.half[0] + s * blk.half[1]
        fy = s * blk.half[0] + c * blk.half[1]
        if fx - sup.half[0] > blk.half[0] or fy - sup.half[1] > blk.half[1]:
            warnings.append(f"placement {i}: {name} overhangs {sup_name} by more than half its size")
        steps.append(Step(name, sup_name, None, tuple(p.offset), p.yaw_rel))

    return CompiledPlan(steps, assignment, warnings)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_plan(api: Any, plan: TowerPlan | dict, check=None, log=None,
                 place_retries: int = 1, structure_retries: int = 2) -> dict:
    """Compile ``plan`` against the live scene and build it with the skills.

    ``check(names, duration)`` — optional structure check callable (e.g. the
    harness's ``check_structure``); when given, each placement is followed
    by a 2 s check. An unstable result triggers *level-aware repair*: the
    lowest broken level is found from the live scene, everything above it
    is parked back on the table, and the tower is rebuilt from that level.
    Returns a report dict with the compiled steps and per-step outcomes.
    """
    if isinstance(plan, dict):
        plan = TowerPlan.from_dict(plan)
    sk = Skills(api, log=log)
    obs0 = sk.observe()
    compiled = compile_plan(plan, obs0, sk.table_z)
    for w in compiled.warnings:
        sk.log(f"plan warning: {w}")
    park_xy = {n: obs0.block(n).pos[:2].copy() for n in obs0.names}   # free spots once vacated
    report = {"plan": plan.name, "assignment": compiled.assignment,
              "steps": [], "warnings": compiled.warnings, "ok": True}

    # base
    base = compiled.steps[0]
    built: list[str] = [base.block]
    outcome = {"block": base.block, "support": "table", "ok": True}
    if base.move_required:
        g = sk.grasp(base.block)
        if not g.ok:
            outcome.update(ok=False, reason=f"grasp: {g.reason}")
        else:
            r = sk.place_on_table(g, base.at, base.yaw_rel)
            outcome.update(ok=r.ok, reason=r.reason)
        sk.go_clear()
    report["steps"].append(outcome)
    if not outcome["ok"]:
        report.update(ok=False, built=built)
        return report

    # levels: chain[i] = (block, support, offset, yaw_rel) for level i+1
    levels = [(s.block, s.support, s.offset, s.yaw_rel) for s in compiled.steps[1:]]

    def place_level(i: int) -> Placement:
        block, support, off, yaw_rel = levels[i]
        r = sk.pick_and_place(block, support, off, yaw_rel, place_retries=place_retries, mode="contact")
        sk.go_clear()
        return r

    i = 0
    repairs = 0
    while i < len(levels):
        block = levels[i][0]
        outcome = {"block": block, "support": levels[i][1], "ok": True}
        r = place_level(i)
        outcome.update(ok=r.ok, reason=r.reason, xy_err=r.xy_err, z_err=r.z_err)
        built = [base.block] + [lv[0] for lv in levels[: i + 1]]
        stable = True
        if check is not None:
            res = check(built, 2.0)
            outcome["check"] = res
            stable = bool(res["stable"])
        if stable and r.ok:
            report["steps"].append(outcome)
            i += 1
            continue
        # --- repair ---
        repairs += 1
        if repairs > structure_retries:
            outcome["ok"] = False
            report["steps"].append(outcome)
            report.update(ok=False, built=built[:-1])
            return report
        chain = [(lv[0], lv[1], lv[2]) for lv in levels[: i + 1]]
        bad = sk.first_bad_level(chain)
        bad = i if bad is None else bad
        sk.log(f"repair {repairs}: level {bad} of {i} broken; parking {[lv[0] for lv in levels[bad:i+1]]}")
        for j in range(i, bad - 1, -1):
            sk.park(levels[j][0], park_xy[levels[j][0]])
        # rebuild from the broken level; the loop re-places level `bad`
        outcome.update(ok=False, repaired_from=bad)
        report["steps"].append(outcome)
        i = bad

    report["built"] = [base.block] + [lv[0] for lv in levels]
    return report
