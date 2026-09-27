"""Offline camera coverage diagnostic; never supply simulator geometry to actors."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aspire_so100.inventory_structure import InventoryStructureEnv
from view_scene import SIDE_CAMERA_DISTANCE_SCALE


def project(frame, points):
    pose = frame['camera_to_world']
    camera = (np.asarray(points)-pose[:3, 3]) @ pose[:3, :3]
    pixels = camera @ frame['intrinsics'].T
    return pixels[:, :2]/pixels[:, 2:], camera[:, 2]


def check(output, seeds):
    output.mkdir(parents=True, exist_ok=True)
    # Conservative envelopes include object extents and low-rise building height.
    envelopes = {'task_workspace': ((-.18, .18), (-.40, -.16), (.40, .60)),
                 'full_inventory': ((-.236, .236), (-.476, -.164), (.40, .60))}
    report = {'scope': 'Static framing and reset images only; no policy success or occlusion guarantee.',
              'distance_scale': SIDE_CAMERA_DISTANCE_SCALE, 'image_size': [320, 240],
              'spawn_distribution_changed': False, 'seeds': []}
    for seed in seeds:
        env = InventoryStructureEnv(output/f'seed_{seed}', scene_seed=seed, object_count=4,
                                    recipe='thin_beam_bridge')
        try:
            env.reset(seed)
            cid = env.model.camera('table/overhead_camera').id
            current = env.model.cam_pos[cid].copy()
            mast = env.model.geom('table/camera_mast').id
            mount = env.model.geom('table/camera_mount').id
            rig = {g: (env.model.geom_pos[g].copy(), env.model.geom_size[g].copy()) for g in (mast, mount)}
            row = {'seed': seed, 'views': {}}
            for name, position in [('before', np.array([.48, -.30, .725])), ('closer', current)]:
                env.model.cam_pos[cid] = position
                for g, (pos, size) in rig.items():
                    env.model.geom_pos[g], env.model.geom_size[g] = pos, size
                if name == 'before':
                    env.model.geom_pos[mast] = [.51, -.30, .35]
                    env.model.geom_size[mast, :2] = [.012, .375]
                    env.model.geom_pos[mount] = [.51, -.30, .725]
                frame = env.get_overhead_rgbd()
                path = output/f'{seed}_{name}.png'
                Image.fromarray(frame['rgb']).save(path)
                bounds = {}
                for label, (xs, ys, zs) in envelopes.items():
                    points = [[x,y,z] for x in xs for y in ys for z in zs]
                    uv, depths = project(frame, points)
                    margin = float(min(uv[:,0].min(), uv[:,1].min(),
                                       319-uv[:,0].max(), 239-uv[:,1].max()))
                    bounds[label] = {'minimum_margin_pixels': margin,
                                     'all_in_front': bool((depths > 0).all())}
                row['views'][name] = {'camera_world_m': frame['camera_to_world'][:3,3].tolist(),
                                      'envelopes': bounds, 'image': str(path)}
            report['seeds'].append(row)
        finally:
            env.close()
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if any(v['minimum_margin_pixels'] < 12 or not v['all_in_front']
           for r in report['seeds'] for v in r['views']['closer']['envelopes'].values()):
        raise RuntimeError('Closer camera fails the 12-pixel envelope margin')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(2026092601, 2026092606)))
    args = parser.parse_args()
    report = check(args.output, args.seeds)
    print(json.dumps({'status': 'passed', 'seeds': len(report['seeds']),
                      'closer': report['seeds'][0]['views']['closer']}, indent=2))
