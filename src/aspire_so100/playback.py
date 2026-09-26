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
from aspire_so100.inventory_prism import InventoryPrismEnv
from aspire_so100.inventory_tower import InventoryTowerEnv
from aspire_so100.inventory_structure import InventoryStructureEnv


class RecordingMixin:
    def __init__(self, output_dir: Path, **kwargs):
        super().__init__(output_dir, **kwargs)
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


class RecordedInventoryPrismEnv(RecordingMixin, InventoryPrismEnv):
    pass


class RecordedInventoryTowerEnv(RecordingMixin, InventoryTowerEnv):
    pass


class RecordedInventoryStructureEnv(RecordingMixin, InventoryStructureEnv):
    pass


def recorded_env_for_result(result, output_dir):
    """Reconstruct the exact sampled object identities before restoring states."""
    if result.get('task') in ('inventory_prism', 'inventory_tower', 'inventory_structure'):
        metadata = result['scene']
        if result['task'] == 'inventory_structure':
            env = RecordedInventoryStructureEnv(output_dir, scene_seed=result['seed'], object_count=metadata['object_count'],
                                                 recipe=metadata['recipe'], color_map=metadata['color_map'])
        elif result['task'] == 'inventory_tower':
            env = RecordedInventoryTowerEnv(output_dir, scene_seed=result['seed'], object_count=metadata['object_count'],
                                             recipe=metadata['recipe'], recipe_order=metadata['order'])
        else:
            env = RecordedInventoryPrismEnv(output_dir, scene_seed=result['seed'], object_count=metadata['object_count'])
        if env.scene_metadata != metadata:
            env.close()
            raise ValueError('Recorded inventory scene does not match this generator')
        return env
    return (RecordedPrismEnv if result.get('task') == 'prism' else RecordedTowerEnv)(output_dir)


class RolloutPlayback:
    def __init__(self, env: RecordedStackingEnv, times=None):
        self.frames = env.frames
        self.state_type = env.state_type
        self.index = 0
        self.times = (np.asarray(times, float) if times is not None
                      else np.arange(len(self.frames), dtype=float))
        if self.times.shape != (len(self.frames),):
            raise ValueError('Playback times must match saved frames')

    def restore(self, model, data) -> None:
        mujoco.mj_setState(model, data, self.frames[self.index], self.state_type)
        mujoco.mj_forward(model, data)

    def step(self, model, data) -> None:
        self.index = (self.index + 1) % len(self.frames)
        self.restore(model, data)

    def reset(self, model, data) -> None:
        self.index = 0
        self.restore(model, data)

    def seek(self, model, data, seconds: float) -> None:
        """Restore the closest saved frame on the relative rollout clock."""
        index = int(np.searchsorted(self.times, float(seconds), side='left'))
        if index and (index == len(self.times) or
                      seconds-self.times[index-1] <= self.times[index]-seconds):
            index -= 1
        self.index = min(max(index, 0), len(self.frames)-1)
        self.restore(model, data)


def load_trajectory_review(trial: Path, model: mujoco.MjModel, data: mujoco.MjData,
                           states: np.ndarray, state_type: mujoco.mjtState, site_id: int):
    """Reconstruct actual TCP motion and timestamp policy commands offline."""
    positions = np.empty((len(states), 3), float)
    absolute_times = np.empty(len(states), float)
    robot_body_ids = np.asarray([
        body_id for body_id in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or '').startswith('robot/')
        and np.any(model.geom_bodyid == body_id)
    ], dtype=int)
    robot_positions = np.empty((len(states), len(robot_body_ids), 3), float)
    robot_quaternions = np.empty((len(states), len(robot_body_ids), 4), float)
    for index, state in enumerate(states):
        mujoco.mj_setState(model, data, state, state_type)
        mujoco.mj_forward(model, data)
        positions[index] = data.site_xpos[site_id]
        absolute_times[index] = data.time
        robot_positions[index] = data.xpos[robot_body_ids]
        robot_quaternions[index] = data.xquat[robot_body_ids]
    if np.any(np.diff(absolute_times) < 0):
        raise ValueError('Rollout timestamps are not monotonic')
    relative_times = absolute_times-absolute_times[0]
    calls_path = trial/'api_calls.json'
    calls = json.loads(calls_path.read_text()) if calls_path.is_file() else []
    events = []
    plans = []
    motions = []
    visible_methods = {'plan_tool','move_tool','move_joints','gripper','wait','note','report'}
    for call in calls:
        if call.get('method') not in visible_methods:
            continue
        when = float(call.get('time', absolute_times[0]))-absolute_times[0]
        params = call.get('params', {})
        method = call['method']
        label = method
        if method in ('plan_tool','move_tool'):
            target = np.asarray(params.get('position'), float)
            if target.shape == (3,) and np.isfinite(target).all():
                item = {'time':when, 'position':target, 'method':method,
                        'seconds':float(params.get('seconds', 0.))}
                (plans if method=='plan_tool' else motions).append(item)
                label += ' ['+', '.join(f'{x:.3f}' for x in target)+']'
        elif method=='gripper':
            label += f" angle={float(params.get('angle', 0.)):.3f}"
        elif method=='wait':
            label += f" {float(params.get('seconds', 0.)):.2f}s"
        elif method=='report':
            values=params.get('values', {})
            label += ':'+str(values.get('stage', values.get('reason', params.get('stage',''))))
        elif method=='note':
            label += ': '+str(params.get('text',''))[:80]
        if call.get('error'):
            label += ' ERROR '+str(call['error'])[:100]
        events.append({'time':when, 'label':label, 'method':method})
    result_path=trial/'result.json'
    result=json.loads(result_path.read_text()) if result_path.is_file() else {}
    policy_end=float(result.get('policy_sim_seconds', relative_times[-1]))
    return {'times':relative_times, 'positions':positions, 'events':events,
            'plans':plans, 'motions':motions, 'duration':float(relative_times[-1]),
            'policy_end':policy_end, 'robot_body_ids':robot_body_ids,
            'robot_positions':robot_positions, 'robot_quaternions':robot_quaternions}


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
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        env = recorded_env_for_result(result, Path(directory))
        try:
            frames, state_type, seconds, playback_timestep = load_vision_rollout(trial, env.model)
            env.frames = frames
            env.state_type = state_type
            review = load_trajectory_review(trial, env.model, env.data, frames,
                                            state_type, env.site_id)
            # This viewer only restores saved states; it never integrates
            # physics. Set its pacing step to the sampled-frame interval so
            # the GUI's 1x speed means 1 simulated second per wall second.
            env.model.opt.timestep = playback_timestep
            playback = RolloutPlayback(env, review['times'])
            playback.reset(env.model, env.data)
            result_path = trial / "result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                print(f"Vision rollout: seed={result.get('seed')}, success={result.get('success')}, "
                      f"policy_error={result.get('policy_error')}", flush=True)
            print(f"Replaying {seconds:.1f}s of saved physics in Viser at 1x simulated time "
                  f"({len(frames)} frames, {playback_timestep:.5f}s/frame); "
                  "use Pause/Reset and speed controls.", flush=True)
            run_camera_viewer(env.model, env.data, step_fn=playback.step, reset_fn=playback.reset,
                              trajectory_review=review, seek_fn=playback.seek,
                              playback_index_fn=lambda: playback.index)
        finally:
            env.close()


def watch_vision_policy(candidate: Path, seed: int = 81, output: Path | None = None,
                        layout: str = "nominal", task: str = "tower", object_count: int | None = None,
                        tower_recipe: str | None = None, tower_levels: int | None = None,
                        tower_order: list[str] | None = None, structure_recipe: str | None = None,
                        structure_color_map: dict | None = None) -> None:
    """Execute the sensor-only policy once, then replay its actual trajectory."""
    candidate = Path(candidate).resolve()
    runner = Path(__file__).resolve().parents[2] / "scripts" / "run_vision_trial.py"
    if output is None:
        with TemporaryDirectory(prefix="aspire-vision-trial-") as directory:
            _run_and_watch_vision(candidate, seed, Path(directory) / "trial", runner, layout, task, object_count, tower_recipe, tower_levels, tower_order, structure_recipe, structure_color_map)
    else:
        _run_and_watch_vision(candidate, seed, Path(output).resolve(), runner, layout, task, object_count, tower_recipe, tower_levels, tower_order, structure_recipe, structure_color_map)


def _run_and_watch_vision(candidate: Path, seed: int, output: Path, runner: Path,
                         layout: str = "nominal", task: str = "tower", object_count: int | None = None,
                         tower_recipe: str | None = None, tower_levels: int | None = None,
                         tower_order: list[str] | None = None, structure_recipe: str | None = None,
                         structure_color_map: dict | None = None) -> None:
    if output.exists():
        raise FileExistsError(f"Vision trial output already exists: {output}")
    print(f"Executing RGB-D policy on seed {seed}; Viser opens after the rollout completes...", flush=True)
    subprocess.run([sys.executable, str(runner), "--candidate", str(candidate),
                    "--seed", str(seed), "--layout", layout, "--task", task,
                    "--output", str(output)] + ([] if object_count is None else ['--object-count', str(object_count)])
                    + ([] if tower_recipe is None else ['--tower-recipe', tower_recipe])
                    + ([] if tower_levels is None else ['--tower-levels', str(tower_levels)])
                    + ([] if tower_order is None else ['--tower-order', *tower_order])
                    + ([] if structure_recipe is None else ['--structure-recipe', structure_recipe])
                    + ([] if structure_color_map is None else ['--structure-color-map', json.dumps(structure_color_map)]), check=True)
    watch_vision_rollout(output)
