"""Verify a private NVIDIA runtime and real scene pacing without model requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
import time

from run_online_extension_study import ROOT, digest, read, require, write


def environment(manifest):
    runtime = read(manifest)
    version = Path('/proc/driver/nvidia/version').read_text().splitlines()[0]
    require(runtime['version'] in version.split(), 'Loaded NVIDIA module differs from private runtime')
    for filename, sha in runtime['file_sha256'].items():
        require(digest(Path(filename)) == sha, 'Private GPU runtime changed: ' + filename)
    env = dict(os.environ, **runtime['environment'])
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               PATH=runtime['bin'] + os.pathsep + env['PATH'])
    return env


def renderer_info():
    from OpenGL import GL
    result = {key: GL.glGetString(value).decode() for key, value in
              [('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER), ('version', GL.GL_VERSION)]}
    require(result['vendor'] == 'NVIDIA Corporation', 'Rendering is not on NVIDIA GPU')
    return result


def gpu_check(manifest):
    """Check both management API and actual EGL rendering in a fresh process."""
    env = environment(manifest)
    query = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.used',
                            '--format=csv,noheader'], env=env, text=True, capture_output=True,
                           timeout=15, check=True)
    probe = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--phase', 'egl'],
                           env=env, text=True, capture_output=True, timeout=20, check=True)
    import json
    return dict(time=time.time(), nvidia_smi=query.stdout.strip(), renderer=json.loads(probe.stdout))


def egl_check():
    import mujoco
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        renderer.update_scene(data)
        frame = renderer.render()
        require(frame.max() > 0, 'Empty EGL image')
        return renderer_info()


def scene_probe(output, task, seed):
    """Exercise initialization, both cameras, local geometry and paced physics."""
    import numpy as np
    from ring_peg_development import installed
    output.mkdir(parents=True, exist_ok=False)
    env = None
    try:
        with installed() if task == 'ring' else nullcontext():
            if task == 'ring':
                from online_ring_scene import OnlineRingEnv
                from online_ring_loop import OnlineRingSensors, compact_geometry
                env = OnlineRingEnv(output / 'env', scene_seed=seed)
                api = OnlineRingSensors(env, output)
            else:
                from run_online_bound_development import TargetSensorAPI, ContactCompleteEnv
                from online_relocation_scene import RelocationEnv
                env = (RelocationEnv if task == 'relocation' else ContactCompleteEnv)(
                    output / 'env', scene_seed=seed)
                api = TargetSensorAPI(env, output)
            env.reset(seed)
            env.start_episode(12., realtime=True)
            if task != 'ring':
                from online_bound_decisions import ExplicitSpatialSkills, reference_feedback
                from online_relocation_runtime import RelocationSkills
                skills = (RelocationSkills if task == 'relocation' else ExplicitSpatialSkills)(api)
            checks, next_frame = [], time.monotonic()
            while time.monotonic() < env.deadline - .1:
                env.advance_while_waiting()
                if time.monotonic() >= next_frame:
                    for view in ('side', 'wrist'):
                        frame = api.camera(view)
                        require(np.isfinite(frame['depth_m'][frame['valid_depth']]).all(), 'Invalid depth')
                        require(frame['valid_depth'].any() and frame['rgb'].std() > 0, 'Empty sensor frame')
                        checks.append(dict(view=view, **renderer_info()))
                    if task == 'ring':
                        compact_geometry(api)
                    else:
                        reference_feedback(skills)
                    next_frame = time.monotonic() + 1.
            env.advance_while_waiting()
            timing = env.timing()
            require(timing['reset_count'] == 1 and timing['max_simulation_lag_seconds'] <= 2.,
                    'Scene pacing failed')
            require(len(checks) >= 4 and timing['simulated_seconds'] >= 10., 'Incomplete pacing probe')
            result = dict(status='passed', task=task, seed=seed, model_requests=0,
                          timing=timing, renders=checks, scope='Development-only sensing/pacing; no task SR')
            write(output / 'result.json', result)
            return result
    finally:
        if env is not None:
            env.close()


def preflight(output, manifest):
    output.mkdir(parents=True, exist_ok=False)
    gpu = gpu_check(manifest)
    env = environment(manifest)
    probes = [('tower', 2028100101), ('ring', 1), ('relocation', 1), ('ring', 2)]

    def launch(item):
        index, (task, seed) = item
        path = output / f'{index}_{task}'
        with (output / f'{index}_{task}.log').open('x') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--phase', 'scene',
                            '--output', str(path), '--task', task, '--seed', str(seed)],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=60, check=True)
        return read(path / 'result.json')

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(launch, enumerate(probes)))
    report = dict(status='passed', time=time.time(), concurrency=4, gpu=gpu,
                  runtime_manifest_sha256=digest(manifest), results=results,
                  model_requests=0, reserved_layouts_used=False)
    write(output / 'report.json', report)
    return report


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('egl', 'scene', 'preflight'), required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--task', choices=('tower', 'ring', 'relocation'))
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.phase == 'egl':
        print(json.dumps(egl_check()))
    elif args.phase == 'scene':
        scene_probe(args.output, args.task, args.seed)
    else:
        preflight(args.output, args.manifest)
