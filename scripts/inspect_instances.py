"""Visualize CPU instances from a real saved sensor frame, without state labels."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from aspire_so100.instance_geometry import InstanceTracker, inventory_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.frame, allow_pickle=False) as archive:
        frame = dict(archive)
    tracker = InstanceTracker()
    instances = tracker.update(frame)
    points, _ = inventory_points(frame)
    palette = dict(red='#e34747', yellow='#b58b00', blue='#4388eb', green='#258a56')
    fig, (image, geometry) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1.35, 1]})
    image.imshow(frame['rgb'])
    all_pixels = np.concatenate([np.argwhere(t['mask']) for t in instances])
    lo, hi = all_pixels.min(axis=0), all_pixels.max(axis=0)
    image.set_xlim(lo[1]-25, hi[1]+25)
    image.set_ylim(hi[0]+30, lo[0]-30)
    records = []
    for index, track in enumerate(instances):
        color = palette[track['color']]
        pixels = np.argwhere(track['mask'])
        low, high = pixels.min(axis=0), pixels.max(axis=0)
        image.add_patch(Rectangle(low[::-1]-.5, *(high-low)[::-1]+1, fill=False, edgecolor=color, lw=1.5))
        image.text(high[1]+2, low[0], str(index+1), color='white', fontsize=10,
                   bbox=dict(facecolor=color, edgecolor='white', pad=2))
        cloud = points[track['mask']]
        geometry.scatter(cloud[:, 0]*100, cloud[:, 1]*100, s=4, color=color, alpha=.4)
        center, half = track['center'][:2]*100, track['half']*100
        geometry.add_patch(Rectangle(center-half, *half*2, fill=False, edgecolor=color, ls='--', lw=1))
        geometry.scatter(*center, marker='+', s=60, color=color)
        role = ' TARGET' if track['color']=='red' and track['shape']=='thin_prism' else (' BASE' if track['color']=='blue' and track['shape']=='cube' else '')
        geometry.annotate(str(index+1)+role, center, xytext=(6, 5), textcoords='offset points', fontsize=8)
        records.append(dict(id=track['id'], color=track['color'], shape=track['shape'],
                            measured_dimensions_m=track['dimensions'].tolist(),
                            inferred_center_m=track['center'].tolist(),
                            features=track['features'], scores=track['shape_scores']))
    image.set_title('Recorded side RGB + depth-separated instances', fontsize=12)
    image.set_axis_off()
    geometry.set_title('CPU geometry in tabletop coordinates', fontsize=12)
    geometry.set_xlabel('World X (cm)')
    geometry.set_ylabel('World Y (cm)')
    geometry.set_aspect('equal')
    geometry.grid(alpha=.2)
    labels = ['%d  %s: %s' % (i+1, t['id'], t['shape']) for i, t in enumerate(instances)]
    fig.text(.05, .13, '\n'.join(labels[:3]), fontsize=9, family='monospace', va='top')
    fig.text(.5, .13, '\n'.join(labels[3:]), fontsize=9, family='monospace', va='top')
    fig.text(.05, .015, 'Dots: observed depth. Dashed boxes: visible XY extents. Centers/hidden volume are inferred; no simulator object state.', fontsize=9)
    fig.tight_layout(rect=(0, .17, 1, 1))
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'instances.png', dpi=180)
    plt.close(fig)
    (args.output/'instances.json').write_text(json.dumps(dict(frame=str(args.frame), instances=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
