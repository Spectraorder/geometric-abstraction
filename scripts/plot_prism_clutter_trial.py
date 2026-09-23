"""Plot recorded sensor frames and the policy's visually planned relocations."""
import argparse
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy import ndimage

from aspire_so100.cube_geometry import rgbd_points
from aspire_so100.cuboid_geometry import color_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.trial / 'result.json').read_text())
    if result.get('task') != 'prism':
        parser.error('A recorded prism trial is required')
    calls = json.loads((args.trial / 'api_calls.json').read_text())
    camera_index, side = -1, None
    stages, destinations, worksite = [], {}, None
    for call in calls:
        if call['method'] == 'camera':
            camera_index += 1
            if call['params'].get('view', 'side') == 'side':
                side = camera_index
        if call['method'] != 'note':
            continue
        note = call['params']['text']
        if note.startswith('CLUTTER PLAN worksite='):
            worksite = np.array(ast.literal_eval(note.split('worksite=')[1].split(' blockers=')[0]))
            stages.append(('Observe clutter', side))
        elif note.startswith('CLEAR blocker='):
            name, xy = note.split('blocker=')[1].split(' destination=')
            destinations[name] = np.array(ast.literal_eval(xy))
        elif note.startswith('CLEAR confirmed '):
            stages.append(('Clear ' + note.removeprefix('CLEAR confirmed '), side))
    if worksite is None or not stages or side is None:
        parser.error('The recording has no visual clutter plan')
    stages = stages[:3] + [('Final observation', side)]
    frames = []
    for _, index in stages:
        with np.load(args.trial / f'camera_{index:03d}.npz') as data:
            frames.append(dict(data))

    fig = plt.figure(figsize=(12, 7.2), facecolor='#f6f7f9', layout='constrained')
    grid = fig.add_gridspec(2, 3)
    image_cells = [(0, 0), (0, 1), (0, 2), (1, 0)]
    for number, ((title, _), frame, cell) in enumerate(zip(stages, frames, image_cells), 1):
        ax = fig.add_subplot(grid[cell])
        ax.imshow(frame['rgb'], interpolation='nearest')
        ax.set_title(f'{number}. {title}', loc='left', fontsize=12)
        ax.axis('off')

    ax = fig.add_subplot(grid[1, 1:])
    frame = frames[0]
    points, valid = rgbd_points(frame)
    r, g, b = np.asarray(frame['rgb'], float).transpose(2, 0, 1)
    table = valid & (r > 95) & (g > 65) & (r > 1.15*g) & (r < 1.65*g) & (g > 1.35*b)
    cloud = points[table]
    design = np.column_stack((cloud[:, :2], np.ones(len(cloud))))
    keep = abs(cloud[:, 2]-np.median(cloud[:, 2])) < .025
    for _ in range(4):
        plane = np.linalg.lstsq(design[keep], cloud[keep, 2], rcond=None)[0]
        keep = abs(cloud[:, 2]-design @ plane) < .003
    height = points[:, :, 2]-points[:, :, :2] @ plane[:2]-plane[2]
    palette = {'blue': '#225ccc', 'red': '#de383a', 'green': '#1b9c47',
               'orange': '#e88716', 'purple': '#a23aca', 'cyan': '#109ea9'}
    observed = {}
    for name, color in palette.items():
        labels, count = ndimage.label(color_mask(frame['rgb'], name) & valid & (height > .006) & (height < .10))
        components = [points[labels == label] for label in range(1, count+1)]
        components = [q for q in components if len(q) >= 6]
        if not components:
            continue
        # Do not span unrelated colored islands with a single bounding box.
        # The figure labels the dominant component; the policy checks all islands.
        q = max(components, key=len)
        lo, hi = np.percentile(q[:, :2], [1, 99], axis=0)
        center = (lo+hi)/2
        observed[name] = center.tolist()
        ax.scatter(q[:, 0]*100, q[:, 1]*100, s=2, color=color, alpha=.5)
        ax.add_patch(Rectangle(lo*100, *(100*(hi-lo)), fill=False, edgecolor=color))
        ax.annotate(name, center*100, xytext=(4, 5), textcoords='offset points', fontsize=9, color=color)
        if name in destinations:
            destination = destinations[name]
            ax.annotate('', destination*100, center*100,
                        arrowprops={'arrowstyle': '->', 'color': color, 'lw': 2})
            ax.scatter(*destination*100, marker='x', color=color, s=55)
    ax.scatter(*worksite*100, marker='*', s=150, color='#222222', label='chosen stacking site')
    ax.set(title='Dominant RGB-D components and planned moves', xlabel='World X (cm)', ylabel='World Y (cm)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=.2)
    ax.legend(loc='best', fontsize=9)
    fig.suptitle(f'Recorded simulation · seed {result["seed"]} · {"PASS" if result["success"] else "FAIL"}', fontsize=17)
    fig.supxlabel('Images: recorded side RGB. Map: depth-derived footprints and policy commands; no object-state overlay.', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps({
        'trial': str(args.trial), 'seed': result['seed'], 'success': result['success'],
        'candidate_sha256': result['candidate_sha256'],
        'source_frames': [f'camera_{i:03d}.npz' for _, i in stages],
        'visual_footprint_centers': observed, 'planned_worksite': worksite.tolist(),
        'planned_destinations': {n: xy.tolist() for n, xy in destinations.items()},
        'source': 'recorded RGB-D and policy API notes, no simulator object states',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
