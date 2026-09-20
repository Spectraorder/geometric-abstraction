"""Display a recorded physical rollout with mjviser's playback controls."""

from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from mjviser import Viewer

from aspire_so100.pipeline import DEFAULT_CANDIDATE, _load_candidate
from aspire_so100.stacking_env import StackingEnv
from aspire_so100.camera_view import run_camera_viewer
from aspire_so100.tower import TowerEnv, TowerAPI


class RecordingMixin:
    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.frames: list[np.ndarray] = []
        self.state_type = mujoco.mjtState.mjSTATE_INTEGRATION

    def capture(self) -> None:
        state = np.empty(mujoco.mj_stateSize(self.model, self.state_type))
        mujoco.mj_getState(self.model, self.data, state, self.state_type)
        self.frames.append(state)

    def _step(self) -> None:
        super()._step()
        self.capture()


class RecordedStackingEnv(RecordingMixin, StackingEnv):
    pass


class RecordedTowerEnv(RecordingMixin, TowerEnv):
    pass


class RolloutPlayback:
    def __init__(self, env: RecordedStackingEnv):
        self.frames = env.frames
        self.state_type = env.state_type
        self.index = 0

    def restore(self, model, data) -> None:
        mujoco.mj_setState(model, data, self.frames[self.index], self.state_type)
        mujoco.mj_forward(model, data)

    def step(self, model, data) -> None:
        self.index = (self.index + 1) % len(self.frames)
        self.restore(model, data)

    def reset(self, model, data) -> None:
        self.index = 0
        self.restore(model, data)


def watch_stacking(seed: int = 51, task: str = "single", candidate_path=None) -> None:
    print(f"Simulating stacking policy on seed {seed} for playback...", flush=True)
    with TemporaryDirectory(prefix="aspire-playback-") as directory:
        env = (RecordedTowerEnv if task == "tower" else RecordedStackingEnv)(Path(directory))
        try:
            env.reset(seed)
            env.frames.clear()
            env.capture()
            env.wait(1.0)
            _, policy = _load_candidate(candidate_path or DEFAULT_CANDIDATE)
            error = None
            try:
                policy(TowerAPI(env) if task == "tower" else env)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"Policy failed: {error}", flush=True)
            env.wait(0.8)
            if task == "tower":
                env.check_structure(["blue_block", "small_red_block", "green_block"], duration=5.)
            metrics = env.evaluate()
            print(f"Stacking success: {metrics.success}; score: {metrics.score:.3f}", flush=True)
            env.wait(2.0)
            playback = RolloutPlayback(env)
            playback.reset(env.model, env.data)
            print("Open the viewer URL below. The physical rollout repeats; use Pause/Reset and speed controls.", flush=True)
            run_camera_viewer(env.model, env.data, step_fn=playback.step, reset_fn=playback.reset)
        finally:
            env.close()
