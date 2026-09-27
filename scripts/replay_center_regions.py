"""Compare conditional transported center regions with current visual constraints.

Offline development only. Uses visual-fit logs and public robot proprioception;
never reads evaluator labels, contacts or simulator object state. No pose is fused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from replay_held_geometry import robot_pose


def region(basis, bounds):
    basis, bounds = np.asarray(basis, float), np.asarray(bounds, float)
    if (basis.shape != (3, 3) or bounds.shape != (3, 2)
            or not np.isfinite(basis).all() or not np.isfinite(bounds).all()
            or not np.allclose(basis @ basis.T, np.eye(3), atol=3e-6)
            or np.any(bounds[:, 0] > bounds[:, 1])):
        raise ValueError('Require a finite orthonormal basis and ordered bounds')
    return basis, bounds


def transport(basis, bounds, earlier_robot, later_robot):
    """Transform the whole region under a conditional rigid attachment."""
    basis, bounds = region(basis, bounds)
    p0, r0 = robot_pose(earlier_robot)
    p1, r1 = robot_pose(later_robot)
    rotation = r1 @ r0.T
    translation = p1 - rotation @ p0
    transformed_basis = basis @ rotation.T
    return transformed_basis, bounds + (transformed_basis @ translation)[:, None]


def compare_regions(first_basis, first_bounds, second_basis, second_bounds):
    """Minimum uniform outward expansion of both regions' half-space bounds.

    Zero means the closed sets overlap. Positive values refute joint compatibility
    only under the visual-size/orientation and no-slip assumptions. This is neither
    Euclidean set distance nor a calibrated uncertainty/confidence interval.
    """
    b0, q0 = region(first_basis, first_bounds)
    b1, q1 = region(second_basis, second_bounds)
    matrix = np.vstack((b0, -b0, b1, -b1))
    rhs = np.concatenate((q0[:, 1], -q0[:, 0], q1[:, 1], -q1[:, 0]))
    solution = linprog([0., 0., 0., 1.],
        A_ub=np.column_stack((matrix, -np.ones(12))), b_ub=rhs,
        bounds=[(None, None)] * 3 + [(0, None)], method='highs')
    if not solution.success:
        raise ValueError('Region comparison failed: ' + solution.message)
    slack = max(0., float(solution.x[3]))
    return {'minimum_uniform_halfspace_expansion_m': slack,
            'overlap_at_solver_tolerance': slack <= 1e-7,
            'solver_tolerance_m': 1e-7,
            'position_recovered': False, 'attachment_certified': False}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(trial):
    record_path, call_path = trial/'geometry_records.json', trial/'api_calls.json'
    records = json.loads(record_path.read_text())
    calls = json.loads(call_path.read_text())
    # Keep every same-time public pose instead of silently selecting one when
    # repeated queries differ slightly. Report the resulting diagnostic range.
    poses = {}
    for call in calls:
        if call['method'] == 'proprioception' and 'observation' in call:
            poses.setdefault(round(float(call['time']), 6), []).append(call['observation'])
    gripper_times = [float(c['time']) for c in calls if c['method'] == 'gripper']
    fits = {}; anchor = None; results = []; invalidations = []
    for item in records:
        if item.get('kind') == 'observer_fit':
            fit = item['fit']
            fits[(item['object'], round(float(fit['observation_time']), 6))] = fit
            continue
        if item.get('kind') != 'geometry_step':
            continue
        row = item['record']; stamp = float(row['time']); key = round(stamp, 6)
        if anchor is not None:
            changed = any(anchor['time']+1e-6 < t <= stamp+1e-6 for t in gripper_times)
            if row['object'] != anchor['object'] or changed:
                invalidations.append({'time':stamp, 'reason':'focus_change' if row['object'] != anchor['object'] else 'gripper_command'})
                anchor = None
        if row.get('action_status') == 'lift_visually_confirmed':
            anchor = None
            fit = fits.get((row['object'], key), {})
            if ('finite_center_bounds_m' in fit and key in poses
                    and fit.get('coordinate_frame') == 'world' and fit.get('units') == 'meters'):
                basis, bounds = region(np.asarray(fit['rotation']).T, fit['finite_center_bounds_m'])
                anchor = {'object':row['object'], 'time':stamp, 'basis':basis, 'bounds':bounds,
                          'poses':poses[key], 'fit':fit}
            continue
        current = row.get('constraints')
        if (anchor is None or current is None or not row['step'].startswith('place')
                or key not in poses or stamp <= anchor['time']):
            continue
        if current['object'] != anchor['object'] or abs(float(current['time'])-stamp) > 1e-6:
            raise ValueError('Stale or different-object visual constraints')
        comparisons = []
        for before in anchor['poses']:
            for after in poses[key]:
                basis, bounds = transport(anchor['basis'], anchor['bounds'], before, after)
                comparisons.append(compare_regions(basis, bounds, current['basis_world'], current['bounds_m']))
        slacks = [r['minimum_uniform_halfspace_expansion_m'] for r in comparisons]
        fit = anchor['fit']
        results.append({'object':row['object'], 'from_time':anchor['time'], 'time':stamp,
            'step':row['step'], 'kind':'conditional_center_region_comparison',
            'condition':'Fitted visual dimensions and orientations, rigid tool attachment without slip.',
            'anchor_position_fully_observed':fit['position_fully_observed'],
            'anchor_widths_m':(anchor['bounds'][:,1]-anchor['bounds'][:,0]).tolist(),
            'anchor_axis_dimensions_m':fit['axis_dimensions'],
            'current_visual_dimensions_m':current['visual_dimensions_m'],
            'same_time_pose_pair_count':len(comparisons),
            'minimum_uniform_expansion_range_m':[min(slacks), max(slacks)],
            'overlap_for_any_same_time_pose_pair':any(r['overlap_at_solver_tolerance'] for r in comparisons),
            'current_constraint_display_precision_m':1e-6,
            'position_recovered':False, 'attachment_certified':False})
        # These are independent comparisons to the original anchor, not a tracker
        # that keeps accepting a refuted attachment assumption or fuses estimates.
    return {'trial':str(trial), 'comparisons':results, 'invalidations':invalidations,
            'input_sha256':{str(p):digest(p) for p in (record_path, call_path)}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite diagnostic evidence')
    receipt_path = args.trial.parent/'equivalence.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'passed':
        raise ValueError('Require a behavior-equivalent logging replay')
    report = replay(args.trial)
    report.update(scope=__doc__, model_calls=0, heldout_seeds_used=[], actor_feedback=False,
        equivalence_receipt_sha256=digest(receipt_path), script_sha256=digest(Path(__file__)),
        robot_pose_helper_sha256=digest(Path(__file__).with_name('replay_held_geometry.py')),
        limitations=['No calibrated dimension, orientation or proprioception uncertainty.',
                     'A disjoint result cannot distinguish slip from visual-model error.',
                     'Overlapping regions would not establish a grasp or recover a pose.',
                     'A development case, not an intervention or success-rate evaluation.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['comparisons'], indent=2))
