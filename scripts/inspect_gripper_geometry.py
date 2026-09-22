"""Overlay visible gripper geometry using a saved camera frame and robot tool pose."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from aspire_so100.gripper_geometry import estimate_gripper_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--tool-position', type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--support-z', type=float)
    parser.add_argument('--wrist', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.frame, allow_pickle=False) as record:
        frame = dict(record)
    geometry = estimate_gripper_geometry(frame, np.asarray(args.tool_position), args.support_z)
    pose = frame['camera_to_world']

    def project(points):
        local = (np.asarray(points)-pose[:3, 3]) @ pose[:3, :3]
        pixels = local @ frame['intrinsics'].T
        return pixels[:, :2]/pixels[:, 2:3]

    figure, axes = plt.subplots(1, 2 if args.wrist else 1, figsize=(12 if args.wrist else 7, 5), squeeze=False)
    ax = axes[0, 0]
    ax.imshow(frame['rgb'])
    for key, color, label in [('visible_points', 'cyan', 'Visible gripper'),
                              ('tip_points', 'magenta', 'Lowest visible surfaces')]:
        uv = project(geometry[key])
        ax.scatter(uv[:, 0], uv[:, 1], s=4, c=color, alpha=.7, label=label)
    uv = project([args.tool_position])[0]
    ax.scatter(*uv, s=70, c='yellow', marker='+', label='Robot tool reference')
    ax.set_title('Depth-derived gripper geometry\nHidden surfaces remain unobserved')
    ax.legend(loc='lower left', fontsize=8)
    ax.axis('off')
    if args.wrist:
        with np.load(args.wrist, allow_pickle=False) as record:
            wrist = dict(record)
        axes[0, 1].imshow(wrist['rgb'])
        axes[0, 1].set_title('Wrist RGB — no depth channel')
        axes[0, 1].axis('off')
    figure.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output/'gripper_overlay.png', dpi=160)
    plt.close(figure)
    serial = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in geometry.items()}
    serial.update(frame=str(args.frame), tool_position=args.tool_position)
    (args.output/'gripper_geometry.json').write_text(json.dumps(serial, indent=2)+'\n')


if __name__ == '__main__':
    main()
