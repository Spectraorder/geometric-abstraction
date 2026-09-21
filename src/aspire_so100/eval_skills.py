"""In-process evaluator for skill policies on the stacking and tower tasks.

Mirrors the official protocols without the bwrap subprocess: the stacking
task follows ``pipeline.evaluate_candidate`` (run, wait 0.8 s, evaluate);
the tower task follows ``agent_loop.worker`` (run with TowerAPI, then the
runner's own 5 s whole-structure check, then evaluate).

    uv run python -m aspire_so100.eval_skills --task stack --seeds 51-55 \
        --candidate policies/stack_skills.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings("ignore")

import mujoco  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def load_run(path: Path):
    spec = importlib.util.spec_from_file_location(f"cand_{path.stem}_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.run


def apply_contact_fix(model: mujoco.MjModel) -> None:
    """Restore the contact options the SO-100 MJCF asks for (mjlab's Scene
    overrides them with defaults): elliptic cones, impratio 10."""
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = 10.0


def run_stack(run, seeds, out: Path, contact_fix: bool, verbose: bool) -> dict:
    from aspire_so100.stacking_env import StackingEnv

    env = StackingEnv(out, render=False)
    if contact_fix:
        apply_contact_fix(env.model)
    results = []
    try:
        for seed in seeds:
            env.output_dir = out / f"seed_{seed:04d}"
            env.output_dir.mkdir(parents=True, exist_ok=True)
            env.reset(seed)
            err = None
            t0 = time.time()
            try:
                run(env)
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
            env.wait(0.8)
            m = env.evaluate()
            env.write_trace(m, err)
            r = {"seed": seed, "error": err, "metrics": asdict(m), "wall_s": time.time() - t0}
            results.append(r)
            if verbose:
                x = r["metrics"]
                print(f"  seed {seed:4d}  {'OK ' if x['success'] else 'FAIL'}  score={x['score']:.3f} "
                      f"h={x['horizontal_error_m']*1000:5.1f}mm v={x['vertical_error_m']*1000:5.1f}mm "
                      f"upright={str(x['target_upright'])[0]} contact={str(x['target_on_base_contact'])[0]} "
                      f"{r['wall_s']:.1f}s {('ERR ' + err) if err else ''}", flush=True)
    finally:
        env.close()
    return summarize(results)


def run_tower(run, seeds, out: Path, contact_fix: bool, verbose: bool, randomize: float) -> dict:
    from aspire_so100.tower import TowerEnv, TowerAPI, ORDER

    class Env(TowerEnv):
        def reset(self, seed):
            if randomize <= 0:
                return super().reset(seed)
            # Wider layout variation than the stock task: positions jittered
            # by +/- randomize metres and yaw by +/- 25 deg.
            import numpy as np
            from aspire_so100.stacking_env import TABLE_HEIGHT
            StackingEnv_reset = super(TowerEnv, self).reset
            StackingEnv_reset(seed)
            self.built = [ORDER[0]]; self.checks = []; self.failure = None
            rng = np.random.default_rng(seed + 100_000)
            layout = (("blue_block", (0.09, -0.25), .025),
                      ("small_red_block", (-.12, -.21), .02),
                      ("green_block", (-.10, -.32), .02))
            for name, xy, hh in layout:
                j = rng.uniform(-randomize, randomize, 2)
                yaw = float(rng.uniform(-0.44, 0.44))
                self._set_freejoint_pose(name, (xy[0] + j[0], xy[1] + j[1], TABLE_HEIGHT + hh), yaw)
            mujoco.mj_forward(self.model, self.data)
            self._step_for(.4)
            return self.observation()

    results = []
    for seed in seeds:
        d = out / f"seed_{seed:04d}"
        d.mkdir(parents=True, exist_ok=True)
        env = Env(d)
        if contact_fix:
            apply_contact_fix(env.model)
        err = None
        t0 = time.time()
        try:
            env.reset(seed)
            try:
                run(TowerAPI(env))
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                env.failure = err
            env.check_structure(list(ORDER), duration=5.0)
            m = env.evaluate()
            env.write_trace(m, err)
            r = {"seed": seed, "error": err, "metrics": asdict(m), "wall_s": time.time() - t0,
                 "final_reasons": env.checks[-1]["reasons"] if env.checks else []}
        finally:
            env.close()
        results.append(r)
        if verbose:
            x = r["metrics"]
            print(f"  seed {seed:4d}  {'OK ' if x['success'] else 'FAIL'}  placed={x['placed_blocks']}/2 "
                  f"stable={str(x['structure_stable'])[0]} {r['wall_s']:.1f}s "
                  f"{r['final_reasons'] if not x['success'] else ''} {('ERR ' + err) if err else ''}", flush=True)
    return summarize(results)


# ---------------------------------------------------------------------------
# Plan mode: arbitrary tower plans on the 5-box shape scene, randomized layout
# ---------------------------------------------------------------------------

BOXES = ("red_block", "blue_block", "green_block", "yellow_block", "small_red_block")
# fixed distractors (x, y, keep-out radius)
FIXED = {"blue_cylinder": (-0.18, -0.29, 0.05), "yellow_tetrahedron": (0.0, -0.17, 0.06),
         "red_arch_bridge": (0.12, -0.48, 0.08), "green_hemisphere": (-0.07, -0.48, 0.05)}

PLANS = {
    "p2_small_on_wide": {"placements": [
        {"block": {"min_footprint": [0.07, 0.03]}},
        {"block": {"size": "small"}, "support": 0}]},
    "p2_relocate_base": {"placements": [
        {"block": "green_block", "at": [0.0, -0.24]},
        {"block": {"size": "small"}, "support": 0}]},
    "p3_named": {"placements": [
        {"block": "blue_block"}, {"block": "red_block", "support": 0}, {"block": "green_block", "support": 1}]},
    "p3_selectors": {"placements": [
        {"block": "yellow_block"}, {"block": {"color": "green"}, "support": 0}, {"block": {"size": "small"}, "support": 1}]},
    "p3_staircase": {"placements": [
        {"block": "blue_block"}, {"block": "red_block", "support": 0, "offset": [0.02, 0.0]},
        {"block": "yellow_block", "support": 1, "offset": [0.02, 0.0]}]},
    "p3_crossed": {"placements": [
        {"block": "red_block"}, {"block": "green_block", "support": 0, "yaw_rel": 1.5708},
        {"block": "small_red_block", "support": 1}]},
    "p4_tall": {"placements": [
        {"block": "blue_block"}, {"block": "red_block", "support": 0}, {"block": "green_block", "support": 1},
        {"block": "yellow_block", "support": 2}]},
    "p5_all": {"placements": [
        {"block": "blue_block"}, {"block": "red_block", "support": 0}, {"block": "green_block", "support": 1},
        {"block": "yellow_block", "support": 2}, {"block": "small_red_block", "support": 3}]},
}


def plan_env_classes():
    """Return (PlanEnv, PlanAPI): the 5-box shape scene with a randomized
    layout per seed and a TowerAPI-equivalent surface whose structure check
    accepts any bottom-up block list."""
    import math
    import numpy as np
    from aspire_so100.stacking_env import StackingEnv, TABLE_HEIGHT
    from aspire_so100.tower import TowerEnv

    class PlanEnv(TowerEnv):
        def __init__(self, output_dir):
            StackingEnv.__init__(self, output_dir, render=False, tower=False)
            self.built, self.checks, self.failure = [], [], None

        def reset(self, seed):
            StackingEnv.reset(self, seed)
            self.checks, self.failure = [], None
            rng = np.random.default_rng(seed + 7_000_000)
            placed = []
            for name in BOXES:
                for _ in range(1000):
                    r = rng.uniform(0.17, 0.30)
                    th = rng.uniform(math.radians(-55), math.radians(55))
                    x, y = r * math.sin(th), -r * math.cos(th)
                    if all(math.hypot(x - px, y - py) >= 0.115 for px, py in placed) and \
                       all(math.hypot(x - fx, y - fy) >= fr + 0.065 for fx, fy, fr in FIXED.values()):
                        break
                placed.append((x, y))
                hz = float(self.model.geom(f"{name}/{name}_geom").size[2])
                self._set_freejoint_pose(name, (x, y, TABLE_HEIGHT + hz), float(rng.uniform(-0.6, 0.6)))
            mujoco.mj_forward(self.model, self.data)
            self._step_for(0.4)
            return self.observation()

        def observation(self):
            obs = StackingEnv.observation(self)
            obs["task"] = "tower plan"
            obs["objects"] = {n: self._object_state(n) for n in BOXES}
            return obs

    class PlanAPI:
        """TowerAPI-equivalent surface; check_structure accepts any bottom-up list."""
        def __init__(self, env):
            self.env = env

        def get_observation(self):
            obs = self.env.get_observation()
            for name, state in obs["objects"].items():
                state["half_extent_world"] = self.env.extent(name).copy()
                state["half_extent_local"] = self.env.model.geom(f"{name}/{name}_geom").size.copy()
            return obs

        def move_end_effector(self, position, seconds=1.2, wrist_roll=None):
            return self.env.move_end_effector(position, seconds, wrist_roll)

        def move_to_joint_positions(self, joint_positions, seconds=1.2):
            return self.env.move_to_joint_positions(joint_positions, seconds)

        def set_gripper(self, position, seconds=0.6):
            return self.env.set_gripper(position, seconds)

        def open_gripper(self):
            return self.env.open_gripper()

        def close_gripper(self):
            return self.env.close_gripper()

        def wait(self, seconds=0.5):
            return self.env.wait(seconds)

        def check_structure(self, names, duration=2.0):
            return self.env.check_structure(list(names), duration)

    return PlanEnv, PlanAPI


def run_plans(plan_names, seeds, out: Path, contact_fix: bool, verbose: bool) -> dict:
    from aspire_so100.tower_plan import execute_plan, PlanError

    PlanEnv, PlanAPI = plan_env_classes()
    results = []
    for pname in plan_names:
        plan = dict(PLANS[pname], name=pname)
        n_pl = len(plan["placements"])
        for seed in seeds:
            d = out / pname / f"seed_{seed:04d}"
            d.mkdir(parents=True, exist_ok=True)
            env = PlanEnv(d)
            if contact_fix:
                apply_contact_fix(env.model)
            events: list[str] = []
            err = None
            report = None
            t0 = time.time()
            try:
                env.reset(seed)
                api = PlanAPI(env)
                try:
                    report = execute_plan(api, plan, check=api.check_structure, log=events.append)
                except PlanError as exc:
                    err = f"PlanError: {exc}"
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {exc}"
                built = report["built"] if report else []
                final = env.check_structure(built, 5.0) if len(built) >= 2 else {"stable": False, "reasons": ["nothing built"]}
                success = bool(report and report["ok"] and len(built) == n_pl and final["stable"])
                score = (len(built) - 1) / max(n_pl - 1, 1) if final["stable"] or not built else 0.0
                r = {"plan": pname, "seed": seed, "error": err, "wall_s": time.time() - t0,
                     "metrics": {"success": success, "score": score, "built": len(built), "required": n_pl},
                     "final_reasons": final["reasons"], "report": report, "events": events}
                (d / "plan_result.json").write_text(json.dumps(r, indent=2, default=str) + "\n")
            finally:
                env.close()
            results.append(r)
            if verbose:
                x = r["metrics"]
                print(f"  {pname:18s} seed {seed:4d}  {'OK ' if x['success'] else 'FAIL'}  built={x['built']}/{x['required']} "
                      f"{r['wall_s']:.1f}s  {r['final_reasons'] if not x['success'] else ''} "
                      f"{('ERR ' + err) if err else ''}", flush=True)
    # per-plan summary
    by_plan = {}
    for r in results:
        by_plan.setdefault(r["plan"], []).append(r["metrics"]["success"])
    for p, v in by_plan.items():
        print(f"    {p:18s} {sum(v)}/{len(v)}")
    return summarize(results)


def summarize(results):
    n = len(results)
    s = sum(r["metrics"]["success"] for r in results)
    return {"successes": s, "trials": n, "success_rate": s / n if n else 0.0,
            "mean_score": sum(r["metrics"]["score"] for r in results) / n if n else 0.0,
            "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["stack", "tower", "plan"], default="stack")
    ap.add_argument("--candidate", type=Path, help="policy file with run(api); not used in plan mode")
    ap.add_argument("--plans", default=",".join(PLANS), help="plan mode: comma-separated plan names")
    ap.add_argument("--seeds", type=parse_seeds, default=parse_seeds("51-55"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--contact-fix", action="store_true",
                    help="use elliptic cones / impratio 10 as the robot MJCF requests")
    ap.add_argument("--randomize", type=float, default=0.0,
                    help="tower only: +/- metres of extra position jitter (and +/-25 deg yaw)")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    label = a.candidate.stem if a.candidate else "plans"
    out = a.out or PROJECT_ROOT / "outputs" / "eval_skills" / f"{a.task}_{label}_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.task == "plan":
        summary = run_plans([p for p in a.plans.split(",") if p], a.seeds, out, a.contact_fix, not a.quiet)
    else:
        if a.candidate is None:
            ap.error("--candidate is required for stack/tower")
        run = load_run(a.candidate)
        if a.task == "stack":
            summary = run_stack(run, a.seeds, out, a.contact_fix, not a.quiet)
        else:
            summary = run_tower(run, a.seeds, out, a.contact_fix, not a.quiet, a.randomize)
    summary.update(task=a.task, candidate=str(a.candidate), seeds=a.seeds,
                   contact_fix=a.contact_fix, randomize=a.randomize, wall_s=time.time() - t0)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"{a.task} {label}: {summary['successes']}/{summary['trials']} "
          f"= {summary['success_rate']*100:.0f}%  mean_score={summary['mean_score']:.3f}  "
          f"({summary['wall_s']:.1f}s)  -> {out}")


if __name__ == "__main__":
    main()
