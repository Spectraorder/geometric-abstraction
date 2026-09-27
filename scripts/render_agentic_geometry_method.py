"""Draw the paper methodology figure from a recorded RGB-D frame and schematics.

Only panel (a) is a recorded observation. Other panels illustrate the interface;
their numeric state is synthetic and is not a fit to the displayed camera frame.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME = ROOT / 'outputs/geometry_cube_stack_preflight_02/basic/geometry_preflight/trials/seed_2027100302/camera_000.npz'
INK = '#24384c'
MUTED = '#6c7b88'
BLUE = '#e1edf6'
SAGE = '#e7f0e7'
CREAM = '#fff2dc'
LAVENDER = '#f1eef7'
GREEN = '#268276'
ORANGE = '#c58c3c'
PURPLE = '#8a7ba2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(frame_path, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [output.with_suffix(ext) for ext in ('.png', '.pdf', '.svg', '.json')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Choose a new output stem to preserve existing figures')
    with np.load(frame_path) as archive:
        rgb = archive['rgb'].copy()
        stamp = float(archive['time'])
        for name in ('depth_m', 'valid_depth', 'intrinsics', 'camera_to_world'):
            if name not in archive:
                raise ValueError('Expected a calibrated RGB-D observation')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 7.5,
        'text.color': INK, 'axes.edgecolor': INK,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    width, height = 7.6, 3.2
    fig = plt.figure(figsize=(width, height), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height), aspect='equal')
    ax.axis('off')
    texts = []

    def text(x, y, label, size=7.5, color=INK, weight='normal', **kwargs):
        artist = ax.text(x, y, label, fontsize=size, color=color,
                         fontweight=weight, va='top', **kwargs)
        texts.append(artist)
        return artist

    def box(x, y, w, h, fill, dashed=False):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=PURPLE if dashed else INK,
                               linewidth=.7, linestyle=(0, (3, 2)) if dashed else '-'))

    def arrow(start, end, dashed=False, color=INK, style='-|>', lw=.85):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style,
                     mutation_scale=8, linewidth=lw, color=color,
                     linestyle=(0, (3, 2)) if dashed else '-', shrinkA=0, shrinkB=0))

    columns = [(0.14, 1.45), (1.87, 1.48), (3.65, 1.79), (5.74, 1.72)]
    for (x, _), heading in zip(columns, (
            '(a) RGB-D observation', '(b) Metric geometry',
            '(c) Text + symbols', '(d) Agentic policy')):
        text(x, 3.03, heading, 8.4, weight='bold')

    # The source sensor pixels are displayed without a fitted overlay.
    image_ax = fig.add_axes([.14/width, 1.12/height, 1.45/width, 1.64/height])
    image_ax.imshow(rgb, interpolation='nearest', aspect='auto')
    image_ax.set_xlim(111, 252)
    image_ax.set_ylim(178, 18)
    image_ax.axis('off')
    ax.add_patch(Rectangle((.14, 1.12), 1.45, 1.64, fill=False,
                           edgecolor=INK, linewidth=.55))
    text(.14, .99, 'Recorded side RGB', 7.0)
    text(.14, .83, '+ calibrated depth', 7.0, MUTED)

    # Planar projection of a cuboid: solid visible edges, dashed hidden edges.
    text(1.88, 2.76, 'Points, edges, and faces', 7.4)
    text(1.88, 2.59, 'Schematic geometry', 6.1, MUTED)
    vertices = np.array([
        [2.07, 1.47], [2.83, 1.47], [2.83, 2.21], [2.07, 2.21],
        [2.34, 1.73], [3.10, 1.73], [3.10, 2.47], [2.34, 2.47],
    ])
    ax.add_patch(Polygon(vertices[[0, 1, 2, 3]], facecolor=CREAM,
                         edgecolor='none'))
    ax.add_patch(Polygon(vertices[[3, 2, 6, 7]], facecolor='#f6f7f7',
                         edgecolor='none'))
    solid = [(0, 1), (1, 2), (2, 3), (3, 0), (3, 7), (7, 6), (6, 2), (6, 5), (5, 1)]
    hidden = [(0, 4), (4, 5), (4, 7)]
    for edges, style in ((solid, '-'), (hidden, (0, (2, 2)))):
        for i, j in edges:
            ax.plot(vertices[[i, j], 0], vertices[[i, j], 1],
                    color=MUTED, linewidth=.85, linestyle=style)
    ax.scatter(vertices[:, 0], vertices[:, 1], s=6, color=ORANGE, zorder=4)
    center = vertices.mean(axis=0)
    ax.scatter(*center, s=12, color=GREEN, zorder=5)
    ax.plot([center[0], 3.17], [center[1], 2.14], color=GREEN, lw=.7)
    text(2.80, 2.18, 'center', 6.8, GREEN)
    ax.plot([2.64, 2.89], [1.53, 1.31], color=ORANGE, lw=.65)
    text(2.22, 1.29, 'bottom_face', 7.0, ORANGE, family='DejaVu Sans Mono')
    box(1.87, .68, 1.48, .41, SAGE)
    text(1.95, 1.00, 'Geometry + evidence', 7.0, weight='bold')
    text(1.95, .83, 'Hidden surfaces are inferred', 6.2, MUTED)

    # Synthetic state values agree with the face-alignment example in the text.
    box(3.65, 1.30, 1.79, 1.46, BLUE)
    text(3.76, 2.65, 'object: green', 7.6, weight='bold', family='DejaVu Sans Mono')
    text(3.76, 2.42, 'center [m]', 7.1)
    text(3.76, 2.26, '  (0.12, -0.01, 0.10)', 6.8, family='DejaVu Sans Mono')
    text(3.76, 2.02, 'goal: place on blue', 7.1, weight='bold')
    text(3.76, 1.80, 'to_goal [m]', 7.1)
    text(3.76, 1.63, '  (-0.02, 0.03, -0.04)', 6.8, family='DejaVu Sans Mono')
    text(3.76, 1.43, 'Illustrative values; world frame', 5.9, MUTED)
    box(3.65, .68, 1.79, .48, SAGE)
    text(3.76, 1.07, 'green.bottom_face', 7.1, family='DejaVu Sans Mono')
    text(3.76, .87, 'blue.top_face', 7.1, family='DejaVu Sans Mono')

    box(5.74, 2.10, 1.72, .66, SAGE)
    text(5.86, 2.65, 'LLM reasoning', 8.2, weight='bold')
    text(5.86, 2.42, 'Generate / revise a policy', 6.8)
    text(5.86, 2.26, 'Between training rollouts', 6.1, MUTED)
    arrow((6.60, 2.07), (6.60, 1.87))
    box(5.74, 1.27, 1.72, .57, CREAM)
    text(5.86, 1.73, 'Policy execution', 8.0, weight='bold')
    text(5.86, 1.50, 'Local sensing + pick / place', 6.5)
    box(5.74, .68, 1.72, .42, LAVENDER, dashed=True)
    text(5.86, 1.01, 'Prior policy experience', 7.0, weight='bold')
    text(5.86, .83, 'Planned online reuse', 6.2, PURPLE)

    ax.plot([5.74, 5.60, 5.60], [.89, .89, 2.19],
            color=PURPLE, lw=.75, linestyle=(0, (3, 2)))
    arrow((5.60, 2.19), (5.74, 2.19), dashed=True, color=PURPLE)

    arrow((1.64, 2.03), (1.81, 2.03))
    arrow((3.40, 2.03), (3.59, 2.03))
    arrow((5.49, 2.43), (5.68, 2.43))

    # The continuous loop is explicitly future work, distinct from training.
    ax.plot([7.49, 7.53, 7.53, .82, .82],
            [1.53, 1.53, .39, .39, .64],
            color=PURPLE, lw=.8, linestyle=(0, (3, 2)))
    arrow((.82, .64), (.82, .70), dashed=True, color=PURPLE)
    text(1.39, .32, 'Planned online loop: reason, execute, observe — no reset until success or timeout',
         6.4, PURPLE)
    text(3.80, .10, 'Gray dashed: hidden geometry     Purple dashed: planned online use', 6.0, MUTED,
         ha='center')

    fig.canvas.draw()
    canvas = fig.bbox
    for artist in texts:
        extent = artist.get_window_extent(fig.canvas.get_renderer())
        if not (canvas.x0 <= extent.x0 and extent.x1 <= canvas.x1 and
                canvas.y0 <= extent.y0 and extent.y1 <= canvas.y1):
            raise ValueError('Text lies outside the figure: '+artist.get_text())
    for suffix in ('.pdf', '.svg'):
        fig.savefig(output.with_suffix(suffix), facecolor='white')
    fig.savefig(output.with_suffix('.png'), dpi=450, facecolor='white')
    plt.close(fig)
    evidence = {
        'purpose': 'Paper methodology illustration, not an experimental result',
        'source_sensor': str(frame_path.relative_to(ROOT)),
        'source_sha256': digest(frame_path), 'observation_time': stamp,
        'panel_a': 'Recorded preflight RGB; crop x=111..252, y=18..178; calibrated depth is available but not pictured',
        'panels_b_c_d': 'Schematic geometry, illustrative metric values, and conceptual interfaces; not fitted to panel (a)',
        'illustrative_object': 'green', 'illustrative_support': 'blue',
        'center_m': [.12, -.01, .10], 'desired_center_m': [.10, .02, .06],
        'remaining_translation_m': [-.02, .03, -.04],
        'solid_paths': 'Implemented geometry interface and policy training',
        'dashed_paths': 'Planned continuous online decisions and reuse of policy experience',
        'source_script_sha256': digest(Path(__file__)),
        'outputs': {path.name: digest(path) for path in paths[:3]},
    }
    output.with_suffix('.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return {'outputs': [str(path.relative_to(ROOT)) for path in paths],
            'png_size_px': [round(width*450), round(height*450)]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=Path, default=DEFAULT_FRAME)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/media/agentic_geometry_method')
    args = parser.parse_args()
    print(json.dumps(render(args.frame.resolve(), args.output.resolve()), indent=2))
