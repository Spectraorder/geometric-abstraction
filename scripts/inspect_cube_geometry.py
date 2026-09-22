"""Export fitted cube centers/corners and a calibrated RGB overlay from sensor NPZs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from aspire_so100.cube_geometry import color_mask, estimate_cube_geometry, rgbd_points


def measured_sides(frame):
    """Initial-view prior: upright cubes resting directly on the visible table."""
    points, valid = rgbd_points(frame)
    r, g, b = np.asarray(frame['rgb'], float).transpose(2, 0, 1)
    mask = valid & (r > 95) & (g > 65) & (r > 1.15*g) & (r < 1.65*g) & (g > 1.35*b)
    p = points[mask]
    p = p[::max(1, len(p)//3000)]
    if len(p) < 50:
        raise ValueError('Initial size view needs visible tabletop')
    design = np.column_stack((p[:, :2], np.ones(len(p))))
    keep = abs(p[:, 2] - np.median(p[:, 2])) < .025
    for _ in range(4):
        plane = np.linalg.lstsq(design[keep], p[keep, 2], rcond=None)[0]
        keep = abs(p[:, 2] - design @ plane) < .003
    sizes = {}
    for color in ('red', 'green', 'blue'):
        p = points[color_mask(frame['rgb'], color) & valid]
        heights = p[:, 2] - p[:, :2] @ plane[:2] - plane[2]
        heights = heights[(heights > .004) & (heights < .10)]
        if len(heights) < 12:
            raise ValueError('Initial cube size unavailable for ' + color)
        level = np.percentile(heights, 90)
        sizes[color] = float(np.median(heights[abs(heights-level) < .002]))
    return sizes


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def project(points, frame):
    pose = frame['camera_to_world']
    camera = (points-pose[:3, 3]) @ pose[:3, :3]
    pixels = camera @ frame['intrinsics'].T
    return pixels[:, :2] / pixels[:, 2:3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--size-frame', type=Path, required=True,
                        help='Unobstructed initial RGB-D view, upright cubes resting on table')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.frame, allow_pickle=False) as record:
        frame = dict(record)
    with np.load(args.size_frame, allow_pickle=False) as record:
        sizes = measured_sides(dict(record))
    results = {}
    for color, side in sizes.items():
        try:
            results[color] = estimate_cube_geometry(frame, color, side)
        except ValueError as exc:
            results[color] = {'error': str(exc)}
    payload = {'frame': str(args.frame), 'size_frame': str(args.size_frame),
               'coordinates': 'world XYZ in meters', 'source': 'RGB-D and calibration only',
               'size_assumption': 'Upright cube on table in size_frame; same rigid cube thereafter',
               'corners': 'Inferred model vertices, including hidden corners; not observed keypoints',
               'cubes': jsonable(results)}
    (args.output/'geometry.json').write_text(json.dumps(payload, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax in axes:
        ax.imshow(frame['rgb'])
        ax.set_axis_off()
    axes[0].set_title('RGB sensor image')
    axes[1].set_title('Fitted centers and inferred cube edges')
    all_pixels = []
    for color, geometry in results.items():
        if 'error' in geometry:
            continue
        vertices = project(geometry['vertices'], frame)
        center = project(geometry['centroid'][None], frame)[0]
        all_pixels.extend(vertices)
        line_color = {'red': '#ff7f7f', 'green': '#70ff70', 'blue': '#65c9ff'}[color]
        for i in range(8):
            for j in range(i+1, 8):
                if (i ^ j) in (1, 2, 4):
                    axes[1].plot(vertices[[i,j],0], vertices[[i,j],1], '--', color=line_color, lw=1.2)
        axes[1].scatter(vertices[:,0], vertices[:,1], s=9, color=line_color)
        axes[1].scatter(*center, marker='+', s=65, color='yellow')
        quality = '3 faces' if geometry['reliable'] else 'uncertain'
        axes[1].annotate(f'{color}: {quality}', center, xytext=(7, -12),
                         textcoords='offset points', color=line_color, fontsize=9,
                         bbox={'facecolor':'black','alpha':.6,'pad':1})
    if all_pixels:
        p = np.asarray(all_pixels)
        axes[1].set_xlim(max(0, p[:,0].min()-18), min(frame['rgb'].shape[1], p[:,0].max()+30))
        axes[1].set_ylim(min(frame['rgb'].shape[0], p[:,1].max()+18), max(0, p[:,1].min()-18))
    fig.text(.5, .02, 'Dashed edges are inferred from a cube model. Yellow + marks the fitted solid centroid.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(args.output/'geometry_overlay.png', dpi=170, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)
    print(args.output/'geometry.json')
    print(args.output/'geometry_overlay.png')


if __name__ == '__main__':
    main()
