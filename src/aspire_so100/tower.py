"""Sequential construction with a time-window test of the entire support chain."""
from dataclasses import asdict, dataclass
import json
import math

import mujoco
import numpy as np

from aspire_so100.stacking_env import StackingEnv, TABLE_HEIGHT, _jsonable

ORDER = ("blue_block", "small_red_block", "green_block")


@dataclass
class TowerMetrics:
    success: bool
    score: float
    placed_blocks: int
    required_placements: int
    structure_stable: bool
    failure: str | None


class TowerEnv(StackingEnv):
    def __init__(self, output_dir, render=False):
        super().__init__(output_dir, render, tower=True)
        self.built = [ORDER[0]]
        self.checks = []
        self.failure = None

    def reset(self, seed, layout="nominal"):
        if layout not in ("nominal", "random_xy"):
            raise ValueError(f"Unknown layout: {layout}")
        super().reset(seed)
        self.built = [ORDER[0]]
        self.checks = []
        self.failure = None
        rng = np.random.default_rng(seed)
        if layout == "random_xy":
            from aspire_so100.layouts import random_block_positions
            random_positions = random_block_positions(seed)
        for name, xy, half_height in (
            ("blue_block", (0.09, -0.25), .025),
            ("small_red_block", (-.12, -.21), .02),
            ("green_block", (-.10, -.32), .02),
        ):
            position = (random_positions[name] if layout == "random_xy"
                        else np.array(xy) + rng.uniform(-.004, .004, 2))
            self._set_freejoint_pose(name, (*position, TABLE_HEIGHT + half_height), 0.)
        mujoco.mj_forward(self.model, self.data)
        self._step_for(.4)
        return self.observation()

    def observation(self):
        obs = super().observation()
        obs["task"] = "build blue/red/green tower"
        obs["objects"] = {name: self._object_state(name) for name in ORDER}
        return obs

    def extent(self, name):
        geom = self.model.geom(f"{name}/{name}_geom")
        return np.abs(self.data.geom_xmat[geom.id].reshape(3, 3)) @ self.model.geom_size[geom.id]

    def structure_status(self, names):
        reasons = []
        states = {name: self._object_state(name) for name in names}
        for name, state in states.items():
            if np.linalg.norm(state["linear_velocity"]) > .015:
                reasons.append(f"{name}: translating")
            if np.linalg.norm(state["angular_velocity"]) > .10:
                reasons.append(f"{name}: rotating")
            rotation = self.data.xmat[self.model.body(f"{name}/{name}").id].reshape(3, 3)
            if np.max(np.abs(rotation[2])) < math.cos(math.radians(8)):
                reasons.append(f"{name}: tilted")
            if self._objects_in_contact(name, "robot"):
                reasons.append(f"{name}: still touching robot")
        if not self._objects_in_contact(names[0], "table"):
            reasons.append("base: missing table support")
        for lower, upper in zip(names, names[1:]):
            p = states[upper]["position"]
            q = states[lower]["position"]
            if not self._objects_in_contact(lower, upper):
                reasons.append(f"{upper}: missing contact with {lower}")
            height_error = abs(p[2] - self.extent(upper)[2] - q[2] - self.extent(lower)[2])
            if height_error > .004:
                reasons.append(f"{upper}: incorrect support height")
            # Approximate upper-substack COM margin on the lower box's top face.
            rotation = self.data.xmat[self.model.body(f"{lower}/{lower}").id].reshape(3,3)
            geom = self.model.geom(f"{lower}/{lower}_geom")
            above = names[names.index(upper):]
            masses = np.array([self.model.body(f"{n}/{n}").mass[0] for n in above])
            com = np.average([self.data.xipos[self.model.body(f"{n}/{n}").id] for n in above], axis=0, weights=masses)
            offset = com - q
            offset[2] = 0
            local = rotation.T @ offset
            vertical_axis = np.argmax(np.abs(rotation[2]))
            axes = [i for i in range(3) if i != vertical_axis]
            if min(self.model.geom_size[geom.id][axes] - np.abs(local[axes])) < .003:
                reasons.append(f"{lower}: insufficient support margin")
        return {"stable": not reasons, "reasons": reasons}

    def check_structure(self, names, duration=2.0):
        def operation():
            self._step_for(.8)
            start = {n: self._object_state(n)["position"].copy() for n in names}
            reasons = set()
            samples = max(1, math.ceil(duration / .05))
            for _ in range(samples):
                self._step_for(.05)
                status = self.structure_status(names)
                reasons.update(status["reasons"])
                for name in names:
                    if np.linalg.norm(self._object_state(name)["position"] - start[name]) > .003:
                        reasons.add(f"{name}: drift exceeds 3 mm")
            result = {"stable": not reasons, "reasons": sorted(reasons), "blocks": list(names), "duration_s": samples * .05, "time": self.data.time}
            self.checks.append(result)
            print(f"Structure check ({len(names)} blocks): {result}", flush=True)
            return result
        return self._run_primitive("check_structure", {"blocks": names, "duration": duration}, operation)

    def evaluate(self):
        count = 0
        for length in (2, 3):
            if self.structure_status(list(ORDER[:length]))["stable"]:
                count = length - 1
            else:
                break
        final_check = self.checks[-1] if self.checks else {}
        intermediate = any(c["stable"] and c["blocks"] == list(ORDER[:2])
                           and c["duration_s"] >= 2 and c["time"] < final_check.get("time", 0)
                           for c in self.checks)
        stable = bool(final_check.get("stable") and final_check.get("blocks") == list(ORDER)
                      and final_check.get("duration_s", 0) >= 5 and count == 2)
        return TowerMetrics(stable and intermediate and self.failure is None, count / 2, count, 2, stable, self.failure)

    def write_trace(self, metrics, error=None):
        path = self.output_dir / "trace.json"
        path.write_text(json.dumps(_jsonable({"task": "tower", "seed": self.seed, "error": error, "metrics": asdict(metrics), "stability_checks": self.checks, "primitives": self.traces}), indent=2) + "\n")
        return path


class TowerAPI:
    """Policy interface. Evaluation and reset are owned by the trial runner."""
    def __init__(self, env):
        self._env = env

    def get_observation(self):
        obs = self._env.get_observation()
        for name, state in obs["objects"].items():
            state["half_extent_world"] = self._env.extent(name).copy()
        return obs

    def get_overhead_rgbd(self):
        return self._env.get_overhead_rgbd()

    def move_end_effector(self, position, seconds=1.2, wrist_roll=None):
        if not np.isfinite(position).all() or len(position) != 3:
            raise ValueError("position must be three finite world coordinates")
        if not 0.1 <= seconds <= 10.:
            raise ValueError("motion duration must be 0.1–10 seconds")
        return self._env.move_end_effector(position, seconds, wrist_roll)

    def move_to_joint_positions(self, joint_positions, seconds=1.2):
        if not 0.1 <= seconds <= 10.:
            raise ValueError("motion duration must be 0.1–10 seconds")
        return self._env.move_to_joint_positions(joint_positions, seconds)

    def open_gripper(self):
        return self._env.open_gripper()

    def close_gripper(self):
        return self._env.close_gripper()

    def set_gripper(self, position, seconds=.6):
        if not 0.1 <= seconds <= 10.:
            raise ValueError("motion duration must be 0.1–10 seconds")
        return self._env.set_gripper(position, seconds)

    def wait(self, seconds=.5):
        if not 0. <= seconds <= 10.:
            raise ValueError("wait duration must be 0–10 seconds")
        return self._env.wait(seconds)

    def check_structure(self, names, duration=2.):
        if list(names) not in [list(ORDER[:i]) for i in range(1, 4)]:
            raise ValueError("names must be a prefix of blue_block, small_red_block, green_block")
        if not 1. <= duration <= 10.:
            raise ValueError("stability duration must be 1–10 seconds")
        return self._env.check_structure(list(names), duration)
