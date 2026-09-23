"""Display a recorded physical rollout with mjviser's playback controls."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import subprocess
import sys

import mujoco
import numpy as np
from mjviser import Viewer

from aspire_so100.pipeline import DEFAULT_CANDIDATE, _load_candidate
from aspire_so100.stacking_env import StackingEnv
from aspire_so100.camera_view import run_camera_viewer
from aspire_so100.tower import TowerEnv, TowerAPI
from aspire_so100.prism import PrismEnv


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


class RecordedPrismEnv(RecordingMixin, PrismEnv):
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


def load_vision_rollout(trial: Path, model: mujoco.MjModel, fps: int = 50):
    """Validate and sample a recorded physical rollout for interactive playback."""
    if fps <= 0:
        raise ValueError("Playback FPS must be positive")
    with np.load(trial / "rollout.npz", allow_pickle=False) as record:
        states = record["states"]
        timestep = float(record["timestep"])
        state_type = mujoco.mjtState(int(record["state_type"]))
    if states.ndim != 2 or len(states) == 0:
        raise ValueError("Vision rollout has no physical states")
    if states.shape[1] != mujoco.mj_stateSize(model, state_type):
        raise ValueError("Vision rollout state size does not match this scene")
    if not np.isfinite(states).all() or not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("Vision rollout contains invalid states or timestep")
    if not np.isclose(timestep, model.opt.timestep):
        raise ValueError("Vision rollout timestep does not match this scene")
    stride = max(1, round(1 / (fps * timestep)))
    indices = np.arange(0, len(states), stride)
    if indices[-1] != len(states) - 1:
        indices = np.r_[indices, len(states) - 1]
    duration = (len(states) - 1) * timestep
    # mjviser schedules step_fn using model.opt.timestep. A sampled frame must
    # therefore represent its share of the ORIGINAL rollout's elapsed time.
    playback_timestep = duration / (len(indices) - 1) if len(indices) > 1 else timestep
    return states[indices], state_type, duration, playback_timestep


def watch_vision_rollout(trial: Path) -> None:
    """Show a saved RGB-D policy execution in Viser, without rerunning it."""
    trial = Path(trial).resolve()
    if not (trial / "rollout.npz").is_file():
        raise FileNotFoundError(f"No rollout.npz in {trial}")
    with TemporaryDirectory(prefix="aspire-vision-viewer-") as directory:
        result_path = trial / 'result.json'
        task = json.loads(result_path.read_text()).get('task', 'tower') if result_path.is_file() else 'tower'
        env = (RecordedPrismEnv if task == 'prism' else RecordedTowerEnv)(Path(directory))
        try:
            frames, state_type, seconds, playback_timestep = load_vision_rollout(trial, env.model)
            env.frames = frames
            env.state_type = state_type
            # This viewer only restores saved states; it never integrates
            # physics. Set its pacing step to the sampled-frame interval so
            # the GUI's 1x speed means 1 simulated second per wall second.
            env.model.opt.timestep = playback_timestep
            playback = RolloutPlayback(env)
            playback.reset(env.model, env.data)
            result_path = trial / "result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                print(f"Vision rollout: seed={result.get('seed')}, success={result.get('success')}, "
                      f"policy_error={result.get('policy_error')}", flush=True)
            print(f"Replaying {seconds:.1f}s of saved physics in Viser at 1x simulated time "
                  f"({len(frames)} frames, {playback_timestep:.5f}s/frame); "
                  "use Pause/Reset and speed controls.", flush=True)
            run_camera_viewer(env.model, env.data, step_fn=playback.step, reset_fn=playback.reset)
        finally:
            env.close()


def watch_vision_policy(candidate: Path, seed: int = 81, output: Path | None = None,
                        layout: str = "nominal", task: str = "tower") -> None:
    """Execute the sensor-only policy once, then replay its actual trajectory."""
    candidate = Path(candidate).resolve()
    runner = Path(__file__).resolve().parents[2] / "scripts" / "run_vision_trial.py"
    if output is None:
        with TemporaryDirectory(prefix="aspire-vision-trial-") as directory:
            _run_and_watch_vision(candidate, seed, Path(directory) / "trial", runner, layout, task)
    else:
        _run_and_watch_vision(candidate, seed, Path(output).resolve(), runner, layout, task)


def _run_and_watch_vision(candidate: Path, seed: int, output: Path, runner: Path,
                         layout: str = "nominal", task: str = "tower") -> None:
    if output.exists():
        raise FileExistsError(f"Vision trial output already exists: {output}")
    print(f"Executing RGB-D policy on seed {seed}; Viser opens after the rollout completes...", flush=True)
    subprocess.run([sys.executable, str(runner), "--candidate", str(candidate),
                    "--seed", str(seed), "--layout", layout, "--task", task,
                    "--output", str(output)], check=True)
    watch_vision_rollout(output)
