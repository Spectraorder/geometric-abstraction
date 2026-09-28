#!/usr/bin/env python3
"""Compose experiment flow diagrams with existing simulator task schematics."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / 'docs/evaluations/task_illustrations'
INK = '#23394E'
EDGE = '#63758A'
BLUE = '#E2EBF4'
GREEN = '#E6EFE5'
PURPLE = '#ECE7F2'
GEO = '#F5E7D5'


def box(ax, x, y, w, h, title, body='', color=BLUE, size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1',
                               facecolor=color, edgecolor=EDGE, linewidth=.9))
    if body:
        ax.text(x+w/2, y+h-5.5, title, ha='center', va='center', fontsize=size,
                weight='bold', color=INK)
        ax.text(x+w/2, y+5.8, body, ha='center', va='center', fontsize=9.1,
                color=INK, linespacing=1.4)
    else:
        ax.text(x+w/2, y+h/2, title, ha='center', va='center', fontsize=size, color=INK)


def arrow(ax, points, label=None, label_at=None, dashed=False):
    style = '--' if dashed else '-'
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=EDGE, lw=1.15, linestyle=style)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                               mutation_scale=12, linewidth=1.15, color=EDGE, linestyle=style))
    if label:
        ax.text(*label_at, label, fontsize=8.6, ha='center', va='center', color=INK,
                bbox=dict(facecolor='white', edgecolor='none', pad=1.5))


def single_flow(ax):
    box(ax, 3, 60, 38, 19, 'LLM revision')
    box(ax, 60, 60, 37, 19, 'Python policy', color=GREEN)
    box(ax, 60, 86, 37, 13, 'Geometric abstraction\n(Full only)', color=GEO, size=9)
    arrow(ax, [(78.5, 86), (78.5, 79)], dashed=True)
    box(ax, 60, 31, 37, 22, 'Rollout', 'Reset · no LLM calls', GREEN)
    box(ax, 3, 31, 38, 22, 'Feedback', color=PURPLE)
    arrow(ax, [(41, 69.5), (60, 69.5)])
    arrow(ax, [(78.5, 60), (78.5, 53)])
    arrow(ax, [(60, 42), (41, 42)])
    arrow(ax, [(22, 53), (22, 60)])
    box(ax, 3, 2, 94, 16, 'Freeze → held-out evaluation', color='#F5F7F9')
    arrow(ax, [(78.5, 31), (78.5, 18)])
    ax.text(36, 24, 'After training', ha='center', va='center', fontsize=9, color=EDGE)


def online_flow(ax):
    box(ax, 60, 82, 37, 12, 'Goal + prior skills', color=PURPLE, size=10)
    box(ax, 3, 54, 38, 19, 'Observe')
    box(ax, 3, 82, 38, 16, 'Geometric abstraction\n(Full only)', color=GEO, size=9)
    arrow(ax, [(22, 73), (22, 82)], dashed=True)
    arrow(ax, [(41, 90), (50, 90), (50, 70), (60, 70)], dashed=True)
    box(ax, 60, 54, 37, 19, 'LLM decision')
    arrow(ax, [(78.5, 82), (78.5, 73)])
    arrow(ax, [(41, 63.5), (60, 63.5)])
    box(ax, 60, 23, 37, 19, 'Execute action', color=GREEN)
    box(ax, 3, 23, 38, 19, 'Same scene', 'No reset', GREEN)
    arrow(ax, [(78.5, 54), (78.5, 42)])
    arrow(ax, [(60, 32.5), (41, 32.5)])
    arrow(ax, [(22, 42), (22, 54)])
    ax.text(50, 6, 'Success or 200 s timeout', ha='center', fontsize=10, color=INK)


def compose(records, mode):
    fig = plt.figure(figsize=(12, 3.8), facecolor='white')
    title = '(a) Single-policy acquisition' if mode == 'single_policy' else '(b) Online execution'
    fig.text(.025, .94, title, fontsize=14, weight='bold', color=INK)
    ax = fig.add_axes([.015, .065, .475, .815])
    ax.set(xlim=(0, 100), ylim=(0, 100)); ax.axis('off')
    (single_flow if mode == 'single_policy' else online_flow)(ax)
    fig.add_artist(plt.Line2D([.505, .505], [.08, .92], transform=fig.transFigure,
                             lw=.8, color='#D6DFE6'))
    # Three task columns, each with a start frame above its goal frame.
    fig.text(.527, .709, 'Start', fontsize=10, rotation=90, ha='center', va='center', color=INK)
    fig.text(.527, .259, 'Goal', fontsize=10, rotation=90, ha='center', va='center', color=INK)
    selected = [r for r in records if r['mode'] == mode]
    labels = {
        'single_bridge': 'Bridge + cylinder', 'single_stacking': 'Stacking',
        'single_ring': 'Ring-on-peg', 'online_tower': 'Bridge-tower',
        'online_ring': 'Ring-on-peg', 'online_relocation': 'Relocation',
    }
    for col, record in enumerate(selected):
        left = .548 + .15*col
        fig.text(left+.07, .94, labels[record['key']], fontsize=11,
                 ha='center', color=INK, weight='bold')
        for row, index in enumerate((0, 2)):
            bottom = .545 - row*.45
            panel = fig.add_axes([left, bottom, .14, .328])
            with Image.open(OUT / record['frames'][index]) as im:
                panel.imshow(im)
            panel.axis('off')
        fig.add_artist(FancyArrowPatch((left+.07, .53), (left+.07, .437),
                       transform=fig.transFigure, arrowstyle='-|>',
                       mutation_scale=11, linewidth=1.05, color=EDGE))
    fig.text(.768, .025, 'Task schematics', fontsize=8, ha='center', color=EDGE)
    name = f'{mode}_workflow_tasks'
    fig.savefig(OUT/f'{name}.pdf', metadata={'CreationDate': None}, dpi=300)
    fig.savefig(OUT/f'{name}.png', dpi=300)
    fig.savefig(OUT/f'{name}.svg', dpi=300)
    svg = OUT / f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    fig.savefig(Path('/tmp')/f'{name}_preview.png', dpi=110)
    plt.close(fig)
    return [OUT/f'{name}.{suffix}' for suffix in ('pdf', 'png', 'svg')]


def main():
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none'})
    manifest = OUT/'manifest.json'
    records = json.loads(manifest.read_text())['records']
    inputs = [manifest, Path(__file__), ROOT/'docs/ONLINE_EXPERIMENTS.md',
              ROOT/'docs/SINGLE_POLICY_EXPERIMENTS.md']
    for r in records:
        inputs.extend(OUT/r['frames'][i] for i in (0, 2))
    for study in ('online_compact_tower_200_01', 'online_ring_200_01', 'online_relocation_200_01'):
        protocol = ROOT/'outputs'/study/'protocol.json'
        data = json.loads(protocol.read_text())
        if 'model' in data:
            assert data['model'] == 'gpt-6-astra' and data['reasoning'] == 'medium'
        else:
            assert 'Astra medium' in data['routing']
        inputs.append(protocol)
    outputs = []
    for mode in ('single_policy', 'online'):
        outputs.extend(compose(records, mode))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = dict(kind='workflow_and_constructed_task_schematics',
                   source_sha256={str(p.relative_to(ROOT)): digest(p) for p in inputs},
                   output_sha256={p.name: digest(p) for p in outputs},
                   online_model='gpt-6-astra', online_reasoning='medium',
                   single_policy_model='gpt-6-astra', single_policy_reasoning='high',
                   frame_indices=[0, 2],
                   interpretation='Task configurations are illustrative, not measured trajectories. No experiments were run.')
    (OUT/'workflow_manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('Created two workflow/task figures: PDF, PNG, and editable SVG.')


if __name__ == '__main__':
    main()
