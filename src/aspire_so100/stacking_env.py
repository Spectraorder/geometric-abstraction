from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import mujoco
import numpy as np
from aspire_so100.layouts import BLOCK_HALF_SIZES


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCENE_SPEC = importlib.util.spec_from_file_location(
    "aspire_so100_scene", _PROJECT_ROOT / "view_scene.py"
)
if _SCENE_SPEC is None or _SCENE_SPEC.loader is None:
    raise RuntimeError("Could not load the SO-100 scene definition")
_SCENE_MODULE = importlib.util.module_from_spec(_SCENE_SPEC)
sys.modules[_SCENE_SPEC.name] = _SCENE_MODULE
_SCENE_SPEC.loader.exec_module(_SCENE_MODULE)
TABLE_HEIGHT = _SCENE_MODULE.TABLE_HEIGHT
build_scene = _SCENE_MODULE.build_scene


TARGET_OBJECT = "small_red_block"
BASE_OBJECT = "blue_block"
TARGET_HALF_HEIGHT = BLOCK_HALF_SIZES[TARGET_OBJECT][2]
BASE_HALF_HEIGHT = BLOCK_HALF_SIZES[BASE_OBJECT][2]

_T = TypeVar("_T")


def _qpos_address(joint: Any) -> int:
    return int(joint.qposadr[0])


def _dof_address(joint: Any) -> int:
    return int(joint.dofadr[0])


@dataclass(frozen=True)
class StackMetrics:
    success: bool
    score: float
    horizontal_error_m: float
    vertical_error_m: float
    target_upright: bool
    target_stable: bool
    target_on_base_contact: bool


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class StackingEnv:
    """SO-100 stacking environment with traceable code-as-policy primitives."""

    ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")

    def __init__(self, output_dir: Path, render: bool = False, *, tower: bool = False, prism: bool = False):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.render_enabled = render
        self.scene, self.model, self.data = build_scene(tower=tower, prism=prism, legacy=not (tower or prism))
        self.site_id = self.model.site("robot/ee_site").id
        self.arm_joints = [self.model.joint(f"robot/{name}") for name in self.ARM_JOINTS]
        self.arm_qpos_addresses = np.array(
            [_qpos_address(joint) for joint in self.arm_joints]
        )
        self.arm_dof_addresses = np.array(
            [_dof_address(joint) for joint in self.arm_joints]
        )
        self._renderer: mujoco.Renderer | None = None
        self._overhead_rgbd = None
        self.seed = 0
        self.traces: list[dict[str, Any]] = []
        self._frame_index = 0

    def close(self) -> None:
        if self._overhead_rgbd is not None:
            self._overhead_rgbd.close()
        if self._renderer is not None:
            self._renderer.close()

    def get_overhead_rgbd(self):
        """On-demand pixels and calibration; never embed arrays in text traces."""
        if self._overhead_rgbd is None:
            from aspire_so100.rgbd import OverheadRGBD
            self._overhead_rgbd = OverheadRGBD(self.model)
        return self._overhead_rgbd.capture(self.data)

    def get_wrist_rgb(self):
        if self._overhead_rgbd is None:
            from aspire_so100.rgbd import OverheadRGBD
            self._overhead_rgbd = OverheadRGBD(self.model)
        return self._overhead_rgbd.capture_wrist(self.data)

    def _set_freejoint_pose(
        self, object_name: str, position: tuple[float, float, float], yaw: float
    ) -> None:
        joint = self.model.joint(f"{object_name}/{object_name}_joint")
        qpos_address = _qpos_address(joint)
        self.data.qpos[qpos_address : qpos_address + 3] = position
        self.data.qpos[qpos_address + 3 : qpos_address + 7] = (
            math.cos(yaw / 2),
            0.0,
            0.0,
            math.sin(yaw / 2),
        )
        dof_address = _dof_address(joint)
        self.data.qvel[dof_address : dof_address + 6] = 0.0

    def reset(self, seed: int) -> dict[str, Any]:
        """Reset one deterministic task instance without crossing seed splits."""
        self.seed = seed
        self.traces = []
        self._frame_index = 0
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        rng = np.random.default_rng(seed)

        target_position = (
            float(rng.uniform(-0.16, -0.10)),
            float(rng.uniform(-0.25, -0.18)),
            TABLE_HEIGHT + TARGET_HALF_HEIGHT,
        )
        base_position = (
            float(rng.uniform(0.08, 0.14)),
            float(rng.uniform(-0.31, -0.25)),
            TABLE_HEIGHT + BASE_HALF_HEIGHT,
        )
        self._set_freejoint_pose(TARGET_OBJECT, target_position, rng.uniform(-0.35, 0.35))
        self._set_freejoint_pose(BASE_OBJECT, base_position, rng.uniform(-0.35, 0.35))
        mujoco.mj_forward(self.model, self.data)
        self._step_for(0.25)
        return self.observation()

    def _step_for(self, seconds: float) -> None:
        steps = max(1, math.ceil(seconds / self.model.opt.timestep))
        for _ in range(steps):
            self._step()

    def _step(self) -> None:
        """Advance physics; playback subclasses can record each resulting state."""
        mujoco.mj_step(self.model, self.data)

    def _object_state(self, name: str) -> dict[str, Any]:
        body = self.model.body(f"{name}/{name}")
        joint = self.model.joint(f"{name}/{name}_joint")
        dof_address = _dof_address(joint)
        return {
            "position": self.data.xpos[body.id].copy(),
            "quaternion_wxyz": self.data.xquat[body.id].copy(),
            "linear_velocity": self.data.qvel[dof_address : dof_address + 3].copy(),
            "angular_velocity": self.data.qvel[dof_address + 3 : dof_address + 6].copy(),
        }

    def observation(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "task": f"stack {TARGET_OBJECT} on {BASE_OBJECT}",
            "end_effector_position": self.data.site_xpos[self.site_id].copy(),
            "robot_joint_positions": self.data.qpos[self.arm_qpos_addresses].copy(),
            "objects": {
                TARGET_OBJECT: self._object_state(TARGET_OBJECT),
                BASE_OBJECT: self._object_state(BASE_OBJECT),
            },
        }

    def _render_frame(self, label: str) -> str | None:
        if not self.render_enabled:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.0, -0.27, TABLE_HEIGHT + 0.06)
        camera.distance = 0.72
        camera.azimuth = 90
        camera.elevation = -28
        self._renderer.update_scene(self.data, camera=camera)
        image = self._renderer.render()

        from PIL import Image

        frame_path = self.output_dir / "frames" / f"{self._frame_index:04d}_{label}.png"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(frame_path)
        self._frame_index += 1
        return str(frame_path.relative_to(self.output_dir))

    def _run_primitive(
        self, name: str, inputs: dict[str, Any], operation: Callable[[], _T]
    ) -> _T:
        started = time.monotonic()
        before = self.observation()
        before_frame = self._render_frame(f"{name}_before")
        status = "success"
        error = None
        try:
            result = operation()
            return result
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            after = self.observation()
            after_frame = self._render_frame(f"{name}_after")
            self.traces.append(
                {
                    "primitive": name,
                    "inputs": _jsonable(inputs),
                    "status": status,
                    "error": error,
                    "duration_s": time.monotonic() - started,
                    "before": _jsonable(before),
                    "after": _jsonable(after),
                    "before_frame": before_frame,
                    "after_frame": after_frame,
                }
            )

    def get_observation(self) -> dict[str, Any]:
        return self._run_primitive("get_observation", {}, self.observation)

    def solve_ik(
        self,
        target_position: np.ndarray,
        wrist_roll: float | None = None,
        tolerance: float = 0.004,
        max_iterations: int = 250,
    ) -> np.ndarray:
        """Solve position-only damped-least-squares IK for the five arm joints."""
        target = np.asarray(target_position, dtype=float)
        ik_data = mujoco.MjData(self.model)
        ik_data.qpos[:] = self.data.qpos
        ik_data.qvel[:] = 0.0
        active_dofs = self.arm_dof_addresses
        active_joints = self.arm_joints
        if wrist_roll is not None:
            ik_data.qpos[self.arm_qpos_addresses[-1]] = wrist_roll
            active_dofs = self.arm_dof_addresses[:-1]
            active_joints = self.arm_joints[:-1]
        mujoco.mj_forward(self.model, ik_data)
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))

        for _ in range(max_iterations):
            error = target - ik_data.site_xpos[self.site_id]
            if np.linalg.norm(error) <= tolerance:
                return ik_data.qpos[self.arm_qpos_addresses].copy()

            mujoco.mj_jacSite(
                self.model,
                ik_data,
                jacobian_position,
                jacobian_rotation,
                self.site_id,
            )
            jacobian = jacobian_position[:, active_dofs]
            damping = 0.015
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(3), error
            )
            delta_norm = np.linalg.norm(delta)
            if delta_norm > 0.08:
                delta *= 0.08 / delta_norm

            for local_index, joint in enumerate(active_joints):
                qpos_address = _qpos_address(joint)
                joint_id = joint.id
                lower, upper = self.model.jnt_range[joint_id]
                ik_data.qpos[qpos_address] = np.clip(
                    ik_data.qpos[qpos_address] + delta[local_index], lower, upper
                )
            mujoco.mj_forward(self.model, ik_data)

        residual = np.linalg.norm(target - ik_data.site_xpos[self.site_id])
        raise RuntimeError(f"IK failed for {target.tolist()}; residual={residual:.4f} m")

    def move_to_joint_positions(self, joint_positions: np.ndarray, seconds: float = 1.2) -> None:
        target = np.asarray(joint_positions, dtype=float)
        if target.shape != (5,):
            raise ValueError(f"Expected five arm joints, received shape {target.shape}")

        def operation() -> None:
            start = self.data.ctrl[:5].copy()
            steps = max(1, math.ceil(seconds / self.model.opt.timestep))
            for step in range(steps):
                alpha = (step + 1) / steps
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                self.data.ctrl[:5] = start + alpha * (target - start)
                self._step()
            if not np.isfinite(self.data.qpos).all():
                raise RuntimeError("Simulation became non-finite during joint motion")

        self._run_primitive(
            "move_to_joint_positions",
            {"joint_positions": target, "seconds": seconds},
            operation,
        )

    def move_end_effector(
        self,
        position: list[float],
        seconds: float = 1.2,
        wrist_roll: float | None = None,
    ) -> None:
        target = np.asarray(position, dtype=float)

        def operation() -> None:
            joints = self.solve_ik(target, wrist_roll=wrist_roll)
            start = self.data.ctrl[:5].copy()
            steps = max(1, math.ceil(seconds / self.model.opt.timestep))
            for step in range(steps):
                alpha = (step + 1) / steps
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                self.data.ctrl[:5] = start + alpha * (joints - start)
                self._step()

        self._run_primitive(
            "move_end_effector",
            {"position": target, "seconds": seconds, "wrist_roll": wrist_roll},
            operation,
        )

    def set_gripper(self, position: float, seconds: float = 0.6) -> None:
        target = float(np.clip(position, -0.174, 1.75))

        def operation() -> None:
            start = float(self.data.ctrl[5])
            steps = max(1, math.ceil(seconds / self.model.opt.timestep))
            for step in range(steps):
                alpha = (step + 1) / steps
                self.data.ctrl[5] = start + alpha * (target - start)
                self._step()

        self._run_primitive(
            "set_gripper", {"position": target, "seconds": seconds}, operation
        )

    def open_gripper(self) -> None:
        self.set_gripper(0.80)

    def close_gripper(self) -> None:
        # About a 4 cm pad separation, matching the blocks' short dimension.
        self.set_gripper(0.25)

    def wait(self, seconds: float = 0.5) -> None:
        self._run_primitive("wait", {"seconds": seconds}, lambda: self._step_for(seconds))

    def _objects_in_contact(self, first: str, second: str) -> bool:
        first_prefix = f"{first}/"
        second_prefix = f"{second}/"
        for contact in self.data.contact:
            geom1 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            )
            geom2 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            )
            if geom1 is None or geom2 is None:
                continue
            if (geom1.startswith(first_prefix) and geom2.startswith(second_prefix)) or (
                geom1.startswith(second_prefix) and geom2.startswith(first_prefix)
            ):
                return True
        return False

    def evaluate(self) -> StackMetrics:
        target = self._object_state(TARGET_OBJECT)
        base = self._object_state(BASE_OBJECT)
        target_position = target["position"]
        base_position = base["position"]
        horizontal_error = float(np.linalg.norm(target_position[:2] - base_position[:2]))
        desired_target_z = base_position[2] + BASE_HALF_HEIGHT + TARGET_HALF_HEIGHT
        vertical_error = float(abs(target_position[2] - desired_target_z))

        target_body = self.model.body(f"{TARGET_OBJECT}/{TARGET_OBJECT}")
        target_rotation = self.data.xmat[target_body.id].reshape(3, 3)
        upright = bool(target_rotation[2, 2] > 0.92)
        speed = max(
            float(np.linalg.norm(target["linear_velocity"])),
            float(np.linalg.norm(target["angular_velocity"])),
        )
        stable = speed < 0.15
        contact = self._objects_in_contact(TARGET_OBJECT, BASE_OBJECT)

        horizontal_score = math.exp(-horizontal_error / 0.04)
        vertical_score = math.exp(-vertical_error / 0.03)
        score = (
            0.35 * horizontal_score
            + 0.25 * vertical_score
            + 0.15 * float(contact)
            + 0.15 * float(upright)
            + 0.10 * float(stable)
        )
        success = (
            horizontal_error < 0.018
            and vertical_error < 0.012
            and stable
            and contact
        )
        return StackMetrics(
            success=success,
            score=score,
            horizontal_error_m=horizontal_error,
            vertical_error_m=vertical_error,
            target_upright=upright,
            target_stable=stable,
            target_on_base_contact=contact,
        )

    def write_trace(self, metrics: StackMetrics, error: str | None = None) -> Path:
        trace_path = self.output_dir / "trace.json"
        payload = {
            "task": f"stack {TARGET_OBJECT} on {BASE_OBJECT}",
            "seed": self.seed,
            "error": error,
            "metrics": asdict(metrics),
            "primitives": self.traces,
        }
        trace_path.write_text(json.dumps(payload, indent=2) + "\n")
        return trace_path
