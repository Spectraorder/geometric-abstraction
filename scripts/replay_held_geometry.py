"""Training-only replay of conditional held-center predictions from robot poses.

Read only policy-visible checkpoints and robot proprioception. Never read rollout
states, evaluator labels or object truth. Predictions are not current observations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def robot_pose(observation):
    position = np.asarray(observation['tool_position'], dtype=float)
    rotation = np.asarray(observation['tool_rotation'], dtype=float)
    if (position.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(position).all() or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or np.linalg.det(rotation) < .999999):
        raise ValueError('Require finite robot position and proper rotation')
    return position, rotation


def predict_center(center, earlier_robot, later_robot):
    """Assume a fixed object center in the tool frame; slip is not ruled out."""
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Require a finite estimated center')
    before, first = robot_pose(earlier_robot)
    after, second = robot_pose(later_robot)
    return after + second @ (first.T @ (center - before))


def replay_records(records, calls):
    poses = {round(float(c['time']), 6): c['observation'] for c in calls
             if c['method'] == 'proprioception' and 'observation' in c}
    gripper_times = [float(c['time']) for c in calls if c['method'] == 'gripper']
    anchor = None
    results = []
    for entry in records:
        if entry.get('kind') != 'geometry_step':
            continue
        row = entry['record']; time = float(row['time']); key = round(time, 6)
        action = row.get('action_status')
        if anchor is not None and (row['object'] != anchor['object']
                or any(anchor['time']+1e-6 < t <= time+1e-6 for t in gripper_times)):
            anchor = None
        if action not in (None, 'lift_visually_confirmed', 'placement_unconfirmed'):
            anchor = None
        if row['status'] != 'current_estimate' or key not in poses:
            anchor = None
            continue
        if row.get('frame') != 'world' or row.get('units') != 'meters':
            raise ValueError('Require current world-meter checkpoint estimates')
        if action not in ('lift_visually_confirmed', 'placement_unconfirmed'):
            continue
        geometry = row['geometry']
        center = np.asarray(geometry.get('center_m', geometry.get('center')), dtype=float)
        current = {'object': row['object'], 'time': time, 'center': center,
                   'robot': poses[key], 'observed': geometry['position_fully_observed']}
        if anchor is not None and action == 'placement_unconfirmed' and time > anchor['time']:
            predicted = predict_center(anchor['center'], anchor['robot'], current['robot'])
            results.append({'object': row['object'], 'from_time': anchor['time'], 'to_time': time,
                            'prediction_kind': 'conditional_rigid_attachment',
                            'condition': 'Object remains attached to tool without slip; not verified by contact sensing.',
                            'predicted_center_m': predicted.tolist(), 'later_visual_center_m': center.tolist(),
                            'visual_residual_m': (center-predicted).tolist(),
                            'visual_residual_norm_m': float(np.linalg.norm(center-predicted)),
                            'visual_center_change_m': float(np.linalg.norm(center-anchor['center'])),
                            'both_visual_positions_fully_observed': bool(anchor['observed'] and current['observed']),
                            'fresh_robot_poses': True, 'intervening_gripper_commands': False})
        anchor = current
    return results


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, output):
    sources = {}; rows = []; trial_count = 0
    for path in sorted(root.glob('*/**/trials/seed_*/geometry_records.json')):
        if 'heldout' in path.parts or 'reserved_revisions' in path.parts:
            raise ValueError('Reserved data is excluded')
        calls_path = path.parent/'api_calls.json'; trial_count += 1
        for source in (path, calls_path): sources[str(source.relative_to(root))] = digest(source)
        pairs = replay_records(json.loads(path.read_text()), json.loads(calls_path.read_text()))
        rows.extend(dict(row, trial=str(path.parent.relative_to(root))) for row in pairs)
    primary = [r for r in rows if r['both_visual_positions_fully_observed']]
    residuals = [r['visual_residual_norm_m'] for r in primary]
    first_pairs = {}
    for row in primary:
        first_pairs.setdefault(row['trial'], row)
    result = {'scope':'Offline training only, policy-visible estimated centers and robot proprioception. No object truth, success labels or physical prediction certification.',
              'script_sha256':digest(Path(__file__)), 'input_sha256':sources,
              'training_trials_read':trial_count, 'eligible_pairs':len(rows),
              'fully_observed_pairs':len(primary), 'fully_observed_trial_count':len({r['trial'] for r in primary}),
              'fully_observed_distinct_seed_count':len({Path(r['trial']).name for r in primary}),
              'first_pair_per_trial_residuals_m':[r['visual_residual_norm_m'] for r in first_pairs.values()],
              'first_pair_per_trial_visual_changes_m':[r['visual_center_change_m'] for r in first_pairs.values()],
              'median_visual_residual_m':float(np.median(residuals)) if residuals else None,
              'max_visual_residual_m':max(residuals, default=None),
              'limits':['Condition not verified; visual estimates may share bias.',
                        'Only explicit held checkpoints can be scored; no accuracy claim during unobserved intervals.',
                        'Repeated retries and related policy versions are not independent physical replications.',
                        'No live state fusion, controller change or SR benefit tested.'], 'pairs':rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','input_sha256')},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args();main(args.root.resolve(),args.output)
