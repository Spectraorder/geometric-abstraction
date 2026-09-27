"""Render actual sensor observations, inferred geometry and constructed targets."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from aspire_so100.ablation_geometry import observe_cube
from aspire_so100.mixed_rigid_state import observe_mixed

COLORS = {'blue': '#3989ff', 'red': '#ff5555', 'green': '#39bd75', 'yellow': '#d6a500'}
EDGES = [(i, j) for i in range(8) for j in range(i+1, 8) if i ^ j in (1, 2, 4)]


def geometry_edges(obj):
    if obj.get('shape') == 'cylinder':
        n = len(obj['vertices'])//2
        return [(offset+i, offset+(i+1)%n) for offset in (0,n) for i in range(n)] + [(i, i+n) for i in range(0,n,8)]
    return EDGES


def project(vertices, frame):
    pose = frame['camera_to_world']
    local = (np.asarray(vertices) - pose[:3, 3]) @ pose[:3, :3]
    uv = local @ frame['intrinsics'].T
    return uv[:, :2] / uv[:, 2:], local[:, 2] > 0


def overlay(ax, obj, frame, color, style='-'):
    uv, visible = project(obj['vertices'], frame)
    for i, j in geometry_edges(obj):
        if visible[i] and visible[j]:
            ax.plot(uv[[i, j], 0], uv[[i, j], 1], style, color=color, lw=1.2, alpha=.9)


def state_plot(ax, record):
    all_vertices = []
    for name, obj in record['objects'].items():
        vertices = np.asarray(obj['vertices'])
        ax.add_collection3d(Line3DCollection(vertices[geometry_edges(obj)], colors=COLORS[name], linewidths=1.3))
        all_vertices.extend(vertices)
    for name, target in record['targets'].items():
        vertices = np.asarray(target['vertices'])
        ax.add_collection3d(Line3DCollection(vertices[geometry_edges(target)], colors=COLORS[name], linestyles='dashed', linewidths=1))
        all_vertices.extend(vertices)
    if all_vertices:
        points = np.asarray(all_vertices)
        middle = (points.min(0)+points.max(0))/2
        radius = max(float(np.ptp(points, axis=0).max())*.58, .055)
        ax.set_xlim(middle[0]-radius, middle[0]+radius)
        ax.set_ylim(middle[1]-radius, middle[1]+radius)
        ax.set_zlim(middle[2]-radius, middle[2]+radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(25, -60)
    ax.set_xlabel('World X (m)', fontsize=8)
    ax.set_ylabel('World Y (m)', fontsize=8)
    ax.set_zlabel('World Z (m)', fontsize=8)
    ax.tick_params(labelsize=7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', type=Path, required=True)
    args = parser.parse_args()
    trial = args.demo/'trial'
    records = json.loads((trial/'geometry_records.json').read_text())
    calls = json.loads((trial/'api_calls.json').read_text())
    result = json.loads((trial/'result.json').read_text())
    counter, latest, frames = 0, {}, []
    for call in calls:
        if call['method'] == 'camera':
            latest[call['params'].get('view', 'side')] = f'camera_{counter:03d}.npz'
            counter += 1
        elif call['method'] == 'record_geometry':
            frames.append(dict(latest))
    assert len(records) == len(frames)
    for record, names in zip(records, frames):
        record['sensor_files'] = names
    summary = {'purpose': 'single_example_not_a_success_rate_estimate', 'seed': result['seed'],
               'private_evaluator_success': result['success'], 'policy_error': result['policy_error'],
               'task': records[0].get('task', 'cube_triple'),
               'wrist_depth': result['wrist_depth'], 'record_count': len(records),
               'new_wrist_geometry_controls_actions': False, 'views': {},
               'final_errors': records[-1]['errors'], 'final_relations': records[-1]['relations']}
    timings = []
    for view in ('side', 'wrist'):
        values = [obj[view] for record in records for obj in record['per_view'].values() if view in obj]
        summary['views'][view] = {'attempts': len(values),
            'available': sum('unavailable' not in value for value in values),
            'full_reliable_pose': sum(value.get('geometry_reliable', False) for value in values),
            'selected': sum(obj['selected_view'] == view for record in records for obj in record['objects'].values())}
    # Replay only policy-visible RGB-D with retained visual dimensions. This is
    # geometry compute latency, not online sensor/RPC/render/agent latency.
    for record in records:
        for view, filename in record['sensor_files'].items():
            with np.load(trial/filename) as data:
                frame = dict(data)
            if 'depth_m' not in frame:
                continue
            for color, observation in record['per_view'].items():
                measured = observation.get(view, {})
                if 'model' not in measured:
                    continue
                started = time.perf_counter()
                if record.get('task') == 'thin_beam_bridge':
                    node = next(node for node in record['goal'] if node['color']==color)
                    observe_mixed(frame, color, node['shape'], measured['model'], record['table_plane'], node['orientation'])
                else:
                    observe_cube(frame, color, measured['model'], record['table_plane'])
                timings.append((time.perf_counter()-started)*1000)
    summary['geometry_replay_latency_ms'] = {'samples': len(timings), 'median': float(np.median(timings)) if timings else None,
        'p95': float(np.percentile(timings, 95)) if timings else None, 'scope': 'one successful object fit; excludes acquisition, RPC, rendering, agent and actions'}
    final = records[-1]
    (args.demo/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.demo/'geometry_states.json').write_text(json.dumps(records, indent=2)+'\n')
    packet = {key: final[key] for key in ('label', 'time', 'coordinate_frame', 'units', 'target_anchor_label', 'errors', 'relations')}
    packet['objects'] = {name: {key: obj[key] for key in ('shape', 'center', 'rotation', 'height', 'dims', 'axis', 'radius', 'end_centers', 'vertices_kind', 'selected_view', 'observed_face_count', 'position_fully_observed', 'inferred_orientation') if key in obj}
                         for name, obj in final['objects'].items()}
    packet['targets'] = {name: {key: target[key] for key in ('shape', 'center', 'rotation', 'height', 'dims', 'axis', 'radius', 'end_centers', 'support', 'source') if key in target} for name, target in final['targets'].items()}
    (args.demo/'agent_packet.json').write_text(json.dumps(packet, indent=2)+'\n')
    top_color = 'red' if records[0].get('task') == 'thin_beam_bridge' else 'green'
    releases = [i for i, record in enumerate(records) if record['label'].startswith(top_color+'_') and 'release offset=' in record['label']]
    selected = [0, releases[-1] if releases else max(0, len(records)//2), len(records)-1]
    fig = plt.figure(figsize=(15, 11), facecolor='white')
    for row, index in enumerate(selected):
        record = records[index]
        with np.load(trial/frames[index]['side']) as data:
            side = dict(data)
        with np.load(trial/frames[index]['wrist']) as data:
            wrist = dict(data)
        ax = fig.add_subplot(3, 3, row*3+1)
        ax.imshow(side['rgb'])
        for name, obj in record['objects'].items():
            overlay(ax, obj, side, COLORS[name])
        for name, target in record['targets'].items():
            overlay(ax, target, side, COLORS[name], '--')
        ax.set_xlim(0, 320); ax.set_ylim(240, 0); ax.axis('off')
        stage = ('Initial', 'Before top release' if releases else 'Intermediate', 'Final')[row]
        ax.set_title(f'{stage}: side RGB + geometry | t={record["time"]:.1f}s', fontsize=10)
        ax.text(.02, .02, 'Goal: initial base (provisional)' if record['target_anchor_label'].startswith('initial')
                else 'Goal: recorded visual plan', transform=ax.transAxes, fontsize=8, color='white',
                bbox={'facecolor': 'black', 'alpha': .55, 'pad': 2})
        ax = fig.add_subplot(3, 3, row*3+2)
        if 'depth_m' in wrist:
            depth = np.where(wrist['valid_depth'], wrist['depth_m'], np.nan)
            ax.imshow(depth, cmap='inferno', vmin=0, vmax=.5)
            for name, view in record['per_view'].items():
                if 'vertices' in view.get('wrist', {}):
                    overlay(ax, view['wrist'], wrist, COLORS[name])
            ax.set_title('Actual wrist depth (0–0.5 m) + wrist fits', fontsize=10)
        else:
            ax.imshow(wrist['rgb']); ax.set_title('Wrist RGB (depth disabled)', fontsize=10)
        ax.set_xlim(0, 320); ax.set_ylim(240, 0); ax.axis('off')
        ax = fig.add_subplot(3, 3, row*3+3, projection='3d')
        state_plot(ax, record)
        ax.set_title('Inferred current state / dashed desired state', fontsize=10)
    outcome = 'PASS' if result['success'] else 'FAIL'
    fig.suptitle(f'RGB-D → rigid geometry → desired state → observed error | seed {result["seed"]} | {outcome}', fontsize=15)
    fig.text(.5, .025, 'Solid = inferred geometry; cylinder rims are model samples. Dashed = constructed goal.\n'
             'Wrist estimates are diagnostic; existing CPU controller executes actions. No object truth in geometry.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .96))
    fig.savefig(args.demo/'geometry_pipeline.png', dpi=160)
    plt.close(fig)
    index = selected[1]
    record = records[index]
    with np.load(trial/frames[index]['wrist']) as data:
        wrist = dict(data)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(wrist['rgb'])
    for pad in record['gripper']['pads']:
        overlay(ax, pad, wrist, '#00ffff')
    for name, obj in record['objects'].items():
        overlay(ax, obj, wrist, COLORS[name])
    ax.set_xlim(0, 320); ax.set_ylim(240, 0); ax.axis('off')
    ax.set_title('Actual wrist RGB + model pad boxes (cyan)')
    ax = fig.add_subplot(2, 2, 2)
    if 'depth_m' in wrist:
        ax.imshow(np.where(wrist['valid_depth'], wrist['depth_m'], np.nan), cmap='inferno', vmin=0, vmax=.5)
    else:
        ax.imshow(wrist['rgb'])
    ax.axis('off'); ax.set_title('Wrist observation at the same instant')
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    state_plot(ax, record)
    for pad in record['gripper']['pads']:
        ax.add_collection3d(Line3DCollection(np.asarray(pad['vertices'])[EDGES], colors='#007f99', linewidths=.7))
    ax.set_title('Robot-model pads + current and desired objects')
    ax = fig.add_subplot(2, 2, 4); ax.axis('off')
    lines = ['STRUCTURED GEOMETRY (CPU)', f"time: {record['time']:.2f} s", 'frame: world; distances below in mm', '',
             'Gripper: 8 pad boxes from robot FK', 'Objects: RGB-D boxes / cylinder axis and rims', 'Dashed objects: constructed task goals', '']
    for color in record['per_view']:
        obj = record['objects'].get(color)
        if obj is None:
            lines.append(color + ': unavailable'); continue
        error = record['errors'].get(color)
        center = ', '.join(f'{x*1000:.1f}' for x in obj['center'])
        lines.extend([f'{color}: center [{center}]',
                      f"  view={obj['selected_view']}; full={obj['geometry_reliable']}",
                      f"  estimated XY error={error['xy_error_m']*1000:.2f}" if error else "  target unavailable"])
    lines.extend(['', 'Partial observations remain uncertain.', 'No contact / stability claim from these values.'])
    ax.text(0, .98, '\n'.join(lines), va='top', family='monospace', fontsize=9)
    fig.suptitle(f'Gripper geometry and object geometry | actual seed {result["seed"]} observation')
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(args.demo/'gripper_and_state.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
