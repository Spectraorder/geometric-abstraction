"""Prepare RGB-only same-color trials and score frozen proposals offline.

Simulator state and segmentation belong to the host evaluator only. Never pass
this script, host_manifest.json, or RGB-D archives to a pointing subagent.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from aspire_so100.block_catalog import BLOCK_COLORS, SHAPE_DIMENSIONS
from aspire_so100.stacking_env import StackingEnv
from score_pointing_replays import point_outcome

DESCRIPTIONS = {
    'cube': 'cube',
    'thin_prism': 'slender rectangular prism (thin beam)',
    'prism': 'thick rectangular prism (long block)',
    'cylinder': 'cylinder',
    'triangular_prism': 'triangular prism (wedge, a cube cut diagonally in half)',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene(entry, output):
    required = tuple(entry['color'] + '_' + shape for shape in SHAPE_DIMENSIONS)
    env = StackingEnv(output, scene_task='support_graph', scene_seed=entry['seed'],
                      object_count=8, required_objects=required)
    env.pixel_aligned_rgbd = True
    if entry['long_blocks_flat']:
        for shape in ('thin_prism', 'prism'):
            name = entry['color'] + '_' + shape
            address = int(env.model.joint(f'{name}/{name}_joint').qposadr[0])
            geom = env.model.geom(f'{name}/{name}_geom')
            env.model.key_qpos[0, address+2] += float(geom.size[0]-geom.size[2])
            env.model.key_qpos[0, address+3:address+7] = [np.sqrt(.5), 0, np.sqrt(.5), 0]
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    mujoco.mj_forward(env.model, env.data)
    env._step_for(.65)
    return env


def prepare(root):
    if (root / 'host_manifest.json').exists():
        raise ValueError('Refusing to overwrite a frozen scene manifest')
    root.mkdir(parents=True, exist_ok=True)
    entries, tasks = [], []
    pairs = [(color, shape) for color in BLOCK_COLORS for shape in SHAPE_DIMENSIONS]
    np.random.default_rng(926).shuffle(pairs)
    for i, (color, shape) in enumerate(pairs):
        key = f'S{i+1:02d}'
        entry = dict(task_id=key, seed=2501+i, color=color, shape=shape,
                     long_blocks_flat=(i % 2 == 0), evaluation_target_object=color+'_'+shape,
                     frame_path=str(root / key / 'frame.npz'))
        env = scene(entry, root / key)
        try:
            frame = env.get_overhead_rgbd()
            np.savez_compressed(entry['frame_path'], **frame)
            image_path = root / key / 'rgb.png'
            Image.fromarray(frame['rgb']).save(image_path)
            task = dict(task_id=key, image_path=str(image_path), image_width=320,
                        image_height=240, target=f"{color} {DESCRIPTIONS[shape]}")
            entry['image_sha256'] = digest(image_path)
            entry['frame_sha256'] = digest(Path(entry['frame_path']))
            entries.append(entry)
            tasks.append(task)
            print(key, task['target'], flush=True)
        finally:
            env.close()
    (root / 'host_manifest.json').write_text(json.dumps(entries, indent=2)+'\n')
    (root / 'tasks.json').write_text(json.dumps(tasks, indent=2)+'\n')
    for i, batch in enumerate((tasks[:7], tasks[7:14], tasks[14:])):
        (root / f'batch_{i+1}.json').write_text(json.dumps(batch, indent=2)+'\n')


def score(root):
    entries = json.loads((root / 'host_manifest.json').read_text())
    tasks = {t['task_id']: t for t in json.loads((root / 'tasks.json').read_text())}
    prediction_path = root / 'predictions.json'
    raw = json.loads(prediction_path.read_text())
    predictions = {p['task_id']: p for p in raw}
    if len(predictions) != len(raw) or set(predictions) != set(tasks):
        raise ValueError('Require exactly one frozen prediction per task')
    rows, masks, panels = [], {}, []
    for entry in entries:
        key = entry['task_id']
        task = tasks[key]
        if digest(Path(task['image_path'])) != entry['image_sha256'] or digest(Path(entry['frame_path'])) != entry['frame_sha256']:
            raise ValueError('Frozen input changed')
        with np.load(entry['frame_path'], allow_pickle=False) as archive:
            frame = dict(archive)
        env = scene(entry, root / key)
        try:
            replay = env.get_overhead_rgbd()
            alignment = {k: bool(np.array_equal(frame[k], replay[k])) for k in
                         ('time', 'rgb', 'depth_m', 'valid_depth', 'intrinsics', 'camera_to_world')}
            if not all(alignment.values()):
                raise ValueError('Scene reconstruction differs: '+key+str(alignment))
            sensor = env._overhead_rgbd
            sensor.renderer.enable_segmentation_rendering()
            sensor.renderer.update_scene(env.data, camera=sensor.camera_id)
            segmentation = sensor.renderer.render().copy()
            geom_ids = segmentation[..., 0]
            is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            mask = np.zeros(is_geom.shape, dtype=bool)
            same_color_visibility = {}
            for shape in SHAPE_DIMENSIONS:
                name = entry['color']+'_'+shape
                ids = [i for i in range(env.model.ngeom) if env.model.geom(i).name.startswith(name+'/')]
                current = is_geom & np.isin(geom_ids, ids)
                same_color_visibility[name] = int(current.sum())
                if name == entry['evaluation_target_object']:
                    mask = current
            prediction = predictions[key]
            outcome = point_outcome(mask, prediction)
            hit_geom = None
            point = prediction.get('point_px')
            if point is not None and outcome not in ('invalid_prediction', 'out_of_bounds'):
                u, v = map(int, point)
                if is_geom[v, u]:
                    hit_geom = env.model.geom(int(geom_ids[v, u])).name
            row = dict(**entry, target=task['target'], prediction=prediction, outcome=outcome,
                       visible_target_pixels=int(mask.sum()), geom_at_point=hit_geom,
                       same_color_visible_pixels=same_color_visibility, replay_alignment=alignment)
            rows.append(row)
            masks[key] = mask
            panel = Image.new('RGB', (640, 530), 'white')
            panel.paste(Image.fromarray(frame['rgb']).resize((640,480)), (0,50))
            draw = ImageDraw.Draw(panel)
            draw.text((8,5), key+' '+task['target'], fill='black')
            draw.text((8,25), outcome+' | '+str(int(mask.sum()))+' visible target pixels', fill='black')
            if point is not None and outcome not in ('invalid_prediction', 'out_of_bounds'):
                x, y = point[0]*2, point[1]*2+50
                draw.ellipse((x-7,y-7,x+7,y+7), outline='magenta', width=2)
                draw.line((x-11,y,x+11,y), fill='magenta', width=2)
                draw.line((x,y-11,x,y+11), fill='magenta', width=2)
            panels.append(panel)
            panel.save(root / key / 'scored.png')
            print(key, outcome, hit_geom, flush=True)
        finally:
            env.close()
    summary = {name: sum(r['outcome'] == name for r in rows) for name in
               ('target_hit','correct_abstention','false_abstention','wrong_pixel',
                'false_positive_absent','invalid_prediction','out_of_bounds')}
    summary['total'] = len(rows)
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), summary=summary,
                  scope='Twenty new same-color RGB pointing queries; no robot actions or policy success measurement.',
                  protocol='All five shapes of target color plus three other-color objects; fixed seeds 2501-2520, alternating flat/upright long blocks. No post-prediction corrections.',
                  information_boundary='Subagents receive RGB PNG and public target description only. Segmentation is rendered exclusively after predictions freeze, for offline identity scoring.',
                  predictions_sha256=digest(prediction_path), tasks_sha256=digest(root/'tasks.json'),
                  script_sha256=digest(Path(__file__)), cases=rows)
    (root / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(root / 'offline_masks.npz', **masks)
    gallery = Image.new('RGB', (1280, 530*((len(panels)+1)//2)), '#dddddd')
    for i, panel in enumerate(panels):
        gallery.paste(panel, ((i%2)*640, (i//2)*530))
    gallery.save(root / 'overview.jpg', quality=90)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare','score'))
    parser.add_argument('--output', type=Path, default=Path('outputs/same_color_pointing_01'))
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else score)(args.output.resolve())
