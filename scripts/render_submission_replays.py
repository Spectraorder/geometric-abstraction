#!/usr/bin/env python3
"""Encode six sparse observation replays from frozen primary-study archives."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/evaluations/videos'
TITLES = {'bridge': 'Bridge + cylinder', 'stacking': 'Three-cube stacking',
          'ring': 'Ring-on-peg', 'tower': 'Bridge-tower', 'relocation': 'Tower relocation'}
FPS = 12
MAX_OBSERVATIONS = 32
FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
BOLD = FONT.with_name('DejaVuSans-Bold.ttf')
SOURCES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register(path):
    relative = str(path.relative_to(ROOT))
    SOURCES[relative] = digest(path)
    return relative


def rows(name):
    path = ROOT / name
    register(path)
    with path.open() as stream:
        return list(csv.DictReader(stream))


def candidates():
    trials = rows('docs/evaluations/single_policy/test_outcomes.csv')
    curves = rows('docs/evaluations/single_policy/chain_curves.csv')
    selections = []
    for task in ('bridge', 'stacking', 'ring'):
        incumbents = sorted([r for r in curves if r['task'] == task and r['arm'] == 'Full'
                             and r['subset'] == 'all_100' and r['revision'] == '3'],
                            key=lambda r: int(r['replica']))
        for incumbent in incumbents:
            matches = [r for r in trials if r['task'] == task and r['arm'] == 'Full'
                       and r['replica'] == incumbent['replica']
                       and r['revision'] == incumbent['retained_revision']
                       and r['success'] == '1']
            if matches:
                row = min(matches, key=lambda r: int(r['seed']))
                selections.append(dict(row, mode='single_policy'))
                break
        else:
            raise ValueError(f'No successful retained Full example: {task}')
    episodes = rows('docs/evaluations/online/episodes.csv')
    for task in ('tower', 'ring', 'relocation'):
        matches = [r for r in episodes if r['task'] == task and r['arm'] == 'Full' and r['success'] == '1']
        # Coverage determines illustrative selection, not experimental outcomes.
        row = min(matches, key=lambda r: (-len(list((ROOT/r['source']).parent.glob('*_side.npz'))), int(r['seed'])))
        selections.append(dict(row, mode='online'))
    return selections


def observations(directory):
    paths = sorted(directory.glob('*_side.npz'))
    if not paths:
        paths = sorted(directory.glob('camera_*.npz'))
    records = []
    external_pose = None
    seen_times = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            pose = archive['camera_to_world']
            if external_pose is None:
                external_pose = pose.copy()
            # Unnamed legacy archives interleave external and wrist camera calls.
            if not np.allclose(pose, external_pose, atol=1e-7):
                continue
            time = float(archive['time'])
            key = round(time, 8)
            if key in seen_times:
                continue
            seen_times.add(key)
            records.append(dict(time=time, rgb=archive['rgb'].copy(), path=path,
                                kind='Recorded side-camera observation'))
    records.sort(key=lambda r: r['time'])
    assert records, directory
    indices = np.linspace(0, len(records)-1, min(len(records), MAX_OBSERVATIONS)).round().astype(int)
    selected = [records[i] for i in indices]
    for r in selected:
        register(r['path'])
    return selected, len(records)


def terminal_snapshot(directory):
    """Render an archived final state; never step physics or execute a policy."""
    import mujoco
    model_path = directory/'private_model.mjb'
    state_path = directory/'private_terminal_state.npz'
    register(model_path)
    register(state_path)
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    with np.load(state_path, allow_pickle=False) as state:
        for key in ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart'):
            getattr(data, key)[:] = state[key]
        data.time = float(state['time'])
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 480
    with mujoco.Renderer(model, width=640, height=480) as renderer:
        renderer.update_scene(data, camera='table/overhead_camera')
        rgb = renderer.render().copy()
    return dict(time=data.time, rgb=rgb, path=state_path,
                kind='Terminal state re-render (archived)', mujoco_version=mujoco.__version__)


def text(draw, pos, value, size=22, color='#273641', bold=False):
    draw.text(pos, value, fill=color, font=ImageFont.truetype(str(BOLD if bold else FONT), size))


def panel(row, record, index, total, times, outcome=False):
    canvas = Image.new('RGB', (1280, 720), '#FAFBFC')
    draw = ImageDraw.Draw(canvas)
    title = ('Single policy' if row['mode'] == 'single_policy' else 'Online') + ' | ' + TITLES[row['task']]
    text(draw, (26, 20), title, 30, bold=True)
    identity = f"Full  |  seed {row['seed']}"
    if row['mode'] == 'single_policy':
        identity += f"  |  chain {row['replica']}, retained R{row['revision']}"
    text(draw, (26, 65), identity, 20, color='#64717C')
    draw.line((26, 102, 1254, 102), fill='#D7DFE4', width=2)
    picture = Image.fromarray(record['rgb'])
    native = picture.size
    picture = picture.resize((720, 540), Image.Resampling.NEAREST if native[0] < 640 else Image.Resampling.LANCZOS)
    canvas.paste(picture, (26, 116))
    label = 'ARCHIVED RESULT' if outcome else 'SPARSE OBSERVATION REPLAY'
    text(draw, (777, 122), label, 22, color='#286F73', bold=True)
    text(draw, (777, 171), f"Simulation t = {record['time']:.2f} s", 24, bold=True)
    kind = record['kind']
    if 're-render' in kind:
        lines = ['Terminal state re-render', 'Saved model + final state', 'No physics rollout or new actions']
    else:
        lines = ['Recorded side-camera image', f'Original resolution: {native[0]} x {native[1]}', f'Snapshot {index + 1} of {total}']
    for i, value in enumerate(lines):
        text(draw, (777, 221 + 33*i), value, 20)
    if outcome:
        text(draw, (777, 355), 'Recorded outcome: SUCCESS', 23, color='#286F73', bold=True)
        if row['mode'] == 'online':
            text(draw, (777, 398), f"Episode wall time: {float(row['success_seconds']):.2f} s", 20)
        else:
            text(draw, (777, 398), 'Held-out evaluation example', 20)
    else:
        text(draw, (777, 355), 'Snapshots use uniform holds.', 20)
        text(draw, (777, 390), 'Gaps and actions may be missing.', 20)
        text(draw, (777, 425), 'No motion interpolation.', 20)
    text(draw, (777, 490), 'Playback is not execution speed.', 19, color='#64717C')
    text(draw, (777, 523), 'Times label the original recording.', 19, color='#64717C')
    start, end = times[0], times[-1]
    draw.line((784, 599, 1230, 599), fill='#D7DFE4', width=4)
    for t in times:
        x = 784 + 446*(t-start)/max(end-start, 1e-6)
        draw.ellipse((x-2, 597, x+2, 601), fill='#8C9DA8')
    x = 784 + 446*(record['time']-start)/max(end-start, 1e-6)
    draw.ellipse((x-6, 593, x+6, 605), fill='#BD6833')
    text(draw, (784, 617), f'{start:.1f} s', 17, color='#64717C')
    text(draw, (1150, 617), f'{end:.1f} s', 17, color='#64717C')
    text(draw, (26, 681), 'Selected successful Full example. Aggregate results include every evaluation, including failures.', 18, color='#64717C')
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    register(Path(__file__))
    results = []
    previews = []
    for row in candidates():
        result_path = ROOT/row['source']
        assert digest(result_path) == row['source_sha256'], result_path
        register(result_path)
        result = json.loads(result_path.read_text())
        assert result['success'] is True
        snapshots, available = observations(result_path.parent)
        if row['mode'] == 'online':
            terminal = terminal_snapshot(result_path.parent)
            assert terminal['time'] >= snapshots[-1]['time']
            snapshots.append(terminal)
        times = [r['time'] for r in snapshots]
        key = row['mode'] + '_' + row['task']
        destination = OUT/f'{key}.mp4'
        visual_frames = []
        with imageio.get_writer(destination, fps=FPS, codec='libx264', quality=8,
                                pixelformat='yuv420p', macro_block_size=1,
                                ffmpeg_params=['-movflags', '+faststart']) as writer:
            for i, record in enumerate(snapshots):
                frame = panel(row, record, i, len(snapshots), times)
                visual_frames.append(frame)
                seconds = 4 if len(snapshots) <= 4 else (2 if i in (0, len(snapshots)-1) else 1)
                for _ in range(FPS*seconds):
                    writer.append_data(np.asarray(frame))
            frame = panel(row, snapshots[-1], len(snapshots)-1, len(snapshots), times, outcome=True)
            for _ in range(FPS*3):
                writer.append_data(np.asarray(frame))
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
            'format=duration:stream=codec_name,width,height,nb_frames', '-of', 'json', str(destination)]))
        info = dict(key=key, file=destination.name, task=TITLES[row['task']], mode=row['mode'],
                    arm='Full', seed=int(row['seed']), replica=row.get('replica'), revision=row.get('revision'),
                    success=True, result_source=register(result_path),
                    original_distinct_side_observations=available,
                    shown_side_observations=len(snapshots)-(row['mode']=='online'),
                    terminal_state_rerender=row['mode']=='online',
                    first_simulation_time=times[0], last_simulation_time=times[-1],
                    video_seconds=float(probe['format']['duration']),
                    fps=FPS, video_sha256=digest(destination),
                    frames=[dict(source=str(r['path'].relative_to(ROOT)), simulation_time=r['time'],
                                 kind=r['kind'], native_resolution=list(reversed(r['rgb'].shape[:2]))) for r in snapshots])
        results.append(info)
        previews.append([visual_frames[0], visual_frames[len(visual_frames)//2], frame])
        print(f"{key}: seed={row['seed']}, {info['shown_side_observations']}/{available} observations, {info['video_seconds']:.1f}s", flush=True)
    sheet = Image.new('RGB', (1280, 240*len(previews)), 'white')
    for row, images in enumerate(previews):
        for col, im in enumerate(images):
            sheet.paste(im.resize((426, 240), Image.Resampling.LANCZOS), (col*426, row*240))
    sheet.save(OUT/'preview.png')
    manifest = dict(kind='archived_sparse_observation_replays',
        selection='Full successful examples only: first successful seed of the first chain with a successful final training-retained policy; online most archived side observations, tie lowest seed.',
        timing='Uniform snapshot holds, not wall-time playback. Timestamps are original simulation time.',
        terminal_rendering='Online final frames render recorded models and states with mj_forward only; no mj_step, policy, model request, or new evaluator run.',
        source_sha256=SOURCES, videos=results)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    columns=['file','task','mode','arm','seed','replica','revision','shown_side_observations','original_distinct_side_observations','terminal_state_rerender','video_seconds','result_source','video_sha256']
    with (OUT/'index.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(results)
    lines=['# Sparse observation replays','',
           'Six selected successful Full examples from the primary studies. These videos',
           'show archived observations, not continuous recordings. They illustrate task',
           'behavior; the complete result tables include all successes and failures.','',
           '| Video | Seed | Policy | Archived side frames shown | Duration |',
           '| --- | ---: | --- | ---: | ---: |']
    for r in results:
        policy=f"Chain {r['replica']}, retained R{r['revision']}" if r['mode']=='single_policy' else 'Online Full'
        lines.append(f"| [{r['mode']}: {r['task']}]({r['file']}) | {r['seed']} | {policy} | {r['shown_side_observations']}/{r['original_distinct_side_observations']} | {r['video_seconds']:.0f} s |")
    lines += ['', '## Reading the replays','',
        '- Timestamps are original simulation time. Equal display holds do not preserve real-time duration; playback must not be used to compare speed.',
        '- Repeated same-time observations are deduplicated. Up to 32 distinct external-camera frames are sampled evenly in frame order; no intermediate motion is generated.',
        '- Each online clip ends with a labeled re-render of its archived terminal state. This is not an image delivered to the agent. Only forward kinematics/rendering is run, without advancing physics.',
        '- Bridge-tower and Relocation have only two archived external observations in the selected episodes. Their intermediate actions are not reconstructable from these images.',
        '- Single-policy clips use the final training-retained policy, which can be an earlier revision. Online Ring includes its shared grasp/lift observations.',
        '- Success labels come from original result records, not a new assessment of these videos. The final camera observation may precede task verification.',
        '- Video canvases are 1280 x 720; original camera images are 320 x 240 or 640 x 480. Enlarging them does not add sensor detail.',
        '', '## Selection and provenance','',manifest['selection'], '',
        '[index.csv](index.csv) identifies each example; [manifest.json](manifest.json)',
        'records source and video hashes, frame timestamps, and selection rules.',
        'Source paths identify archives in the research repository, not files included',
        'in this submission bundle. No experiment was rerun to produce these clips.',
        '', '[Preview contact sheet](preview.png)', '']
    (OUT/'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
