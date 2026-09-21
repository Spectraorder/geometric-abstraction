"""Record an MP4 of a policy rollout: free-camera view with the wrist and
overhead cameras alongside, captioned with sim time and the running
primitive.

    MUJOCO_GL=egl uv run python -m aspire_so100.record_video --task tower \
        --candidate policies/tower_skills.py --seed 107 --randomize 0.03 \
        --out outputs/videos/tower_107.mp4

Rendering is offscreen through MuJoCo's Renderer; on a host without a
working GPU driver EGL falls back to software rendering (slow but fine).
The physics is stepped exactly as in the evaluators; frames are taken every
``stride`` physics steps so that ``--speed`` seconds of simulation play per
video second.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

warnings.filterwarnings("ignore")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from aspire_so100.eval_skills import PLANS, apply_contact_fix, load_run, plan_env_classes  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class VideoMixin:
    """Mixin for StackingEnv subclasses: renders a composite frame every
    ``stride`` physics steps and tracks the running primitive for captions."""

    def __init__(self, output_dir, *args, **kwargs):
        super().__init__(output_dir, *args, **kwargs)
        self.video_frames: list[np.ndarray] = []
        self.stride = 17
        self.main_size = (640, 480)
        self.side_size = (320, 240)
        self.views = ("free", "wrist", "overhead")
        self.current_primitive = ""
        self.caption_extra = ""
        self._vstep = 0
        self._main = None
        self._side = None
        self._cam = None

    # -- rendering -----------------------------------------------------------

    def _renderers(self):
        if self._main is None:
            w, h = self.main_size
            self._main = mujoco.Renderer(self.model, height=h, width=w)
            sw, sh = self.side_size
            self._side = mujoco.Renderer(self.model, height=sh, width=sw)
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._cam.lookat[:] = (0.0, -0.27, 0.40 + 0.08)
            self._cam.distance = 0.78
            self._cam.azimuth = 135
            self._cam.elevation = -30
        return self._main, self._side, self._cam

    def render_composite(self) -> np.ndarray:
        main, side, cam = self._renderers()
        main.update_scene(self.data, camera=cam)
        img = Image.fromarray(main.render())
        draw = ImageDraw.Draw(img)
        text = f"t = {self.data.time:6.2f} s   {self.current_primitive}"
        if self.caption_extra:
            text += f"   {self.caption_extra}"
        draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
        draw.text((6, 4), text, fill=(255, 255, 255))
        tiles = [np.asarray(img)]
        sides = []
        if "wrist" in self.views:
            side.update_scene(self.data, camera="robot/wrist_camera")
            sides.append(self._label(side.render(), "wrist camera"))
        if "overhead" in self.views:
            side.update_scene(self.data, camera="table/overhead_camera")
            sides.append(self._label(side.render(), "overhead camera"))
        if sides:
            col = np.concatenate(sides, axis=0)
            if col.shape[0] < tiles[0].shape[0]:
                pad = np.zeros((tiles[0].shape[0] - col.shape[0], col.shape[1], 3), dtype=np.uint8)
                col = np.concatenate([col, pad], axis=0)
            tiles.append(col[: tiles[0].shape[0]])
        return np.concatenate(tiles, axis=1)

    @staticmethod
    def _label(frame: np.ndarray, text: str) -> np.ndarray:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
        draw.text((4, 2), text, fill=(255, 255, 255))
        return np.asarray(img)

    # -- hooks -----------------------------------------------------------------

    def _step(self) -> None:
        super()._step()
        self._vstep += 1
        if self._vstep % self.stride == 0:
            self.video_frames.append(self.render_composite())

    def _run_primitive(self, name, inputs, operation):
        prev = self.current_primitive
        self.current_primitive = name
        try:
            return super()._run_primitive(name, inputs, operation)
        finally:
            self.current_primitive = prev

    def close(self) -> None:
        for r in (self._main, self._side):
            if r is not None:
                r.close()
        super().close()


def write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    import mediapy

    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), frames, fps=fps, codec="h264", bps=6_000_000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["stack", "tower", "plan"], default="tower")
    ap.add_argument("--candidate", type=Path, help="policy with run(api) (stack/tower)")
    ap.add_argument("--plan", default="p3_crossed", help="plan name (plan task)")
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--randomize", type=float, default=0.0, help="tower: extra layout jitter (m)")
    ap.add_argument("--contact-fix", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=2.0, help="simulated seconds per video second")
    ap.add_argument("--views", default="free,wrist,overhead")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    from aspire_so100.stacking_env import StackingEnv, TABLE_HEIGHT
    from aspire_so100.tower import TowerEnv, TowerAPI, ORDER

    out = a.out or PROJECT_ROOT / "outputs" / "videos" / f"{a.task}_{a.plan if a.task == 'plan' else a.candidate.stem}_seed{a.seed}.mp4"
    work = out.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)

    if a.task == "plan":
        PlanEnv, PlanAPI = plan_env_classes()
        Base = PlanEnv
    elif a.task == "tower":
        Base = TowerEnv
        if a.randomize > 0:
            class Base(TowerEnv):  # noqa: F811
                def reset(self, seed):
                    super(TowerEnv, self).reset(seed)
                    self.built = [ORDER[0]]; self.checks = []; self.failure = None
                    rng = np.random.default_rng(seed + 100_000)
                    layout = (("blue_block", (0.09, -0.25), .025), ("small_red_block", (-.12, -.21), .02),
                              ("green_block", (-.10, -.32), .02))
                    for name, xy, hh in layout:
                        j = rng.uniform(-a.randomize, a.randomize, 2)
                        self._set_freejoint_pose(name, (xy[0] + j[0], xy[1] + j[1], TABLE_HEIGHT + hh),
                                                 float(rng.uniform(-0.44, 0.44)))
                    mujoco.mj_forward(self.model, self.data)
                    self._step_for(.4)
                    return self.observation()
    else:
        Base = StackingEnv

    class Env(VideoMixin, Base):
        pass

    env = Env(work)
    env.views = tuple(v.strip() for v in a.views.split(",") if v.strip())
    env.stride = max(1, int(round(a.speed / (env.model.opt.timestep * a.fps))))
    if a.contact_fix:
        apply_contact_fix(env.model)
    t0 = time.time()
    result = {}
    try:
        env.reset(a.seed)
        env.current_primitive = "start"
        env.video_frames.append(env.render_composite())
        env.wait(0.5)
        if a.task == "plan":
            from aspire_so100.tower_plan import execute_plan
            api = PlanAPI(env)
            plan = dict(PLANS[a.plan], name=a.plan)
            env.caption_extra = f"plan {a.plan}"
            report = execute_plan(api, plan, check=api.check_structure)
            built = report["built"]
            env.current_primitive = "final structure check"
            final = env.check_structure(built, 5.0) if len(built) >= 2 else {"stable": False}
            result = {"success": bool(report["ok"] and final["stable"] and len(built) == len(plan["placements"])),
                      "built": built, "final": final}
        elif a.task == "tower":
            run = load_run(a.candidate)
            run(TowerAPI(env))
            env.current_primitive = "final structure check (5 s)"
            env.check_structure(list(ORDER), duration=5.0)
            m = env.evaluate()
            result = {"success": m.success, "score": m.score}
        else:
            run = load_run(a.candidate)
            run(env)
            env.current_primitive = "settle"
            env.wait(0.8)
            m = env.evaluate()
            result = {"success": m.success, "score": m.score, "h_err_mm": m.horizontal_error_m * 1000}
        env.current_primitive = "done"
        env.wait(1.0)
    finally:
        frames = env.video_frames
        env.close()
    write_video(frames, out, a.fps)
    info = {"video": str(out), "frames": len(frames), "video_seconds": len(frames) / a.fps,
            "sim_seconds": len(frames) * env.stride * env.model.opt.timestep, "wall_seconds": time.time() - t0,
            "result": result}
    (work / "video_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
