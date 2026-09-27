"""Reproduce a logged visual fit failure and isolate model/mask effects offline."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def plain(value):
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def run(root, output, attachment_replay=None):
    certificate = read(root/'equivalence.json')
    if certificate['status'] != 'passed': raise ValueError('Require equivalent logging replay')
    trial = root/f"seed_{certificate['seed']}"
    library = root/'logging_library.py'
    ns = {}; exec(compile(library.read_text(), str(library), 'exec'), ns)
    records = read(trial/'geometry_records.json')
    inputs = [r for r in records if r.get('kind') == 'observer_input']
    current = inputs[-1]; color = current['object']; stamp = current['time']
    initial = next(r for r in inputs if r['object'] == color and r['visual_model'] is not None)
    frames = []; frame_paths = []
    for path in sorted(trial.glob('camera_*.npz')):
        with np.load(path) as data:
            if abs(float(data['time']) - stamp) > 1e-6: continue
            frame = {k: data[k] for k in data.files}
        frames.append(frame); frame_paths.append(path)
    if len(frames) != 2: raise ValueError('Require exactly the failing side/wrist pair')
    # The original callback always requests the side view before wrist fallback.
    frames = list(zip(('side', 'wrist'), frames))
    masked = [(view, ns['filter_robot_visual_geometry'](frame, current['robot_geometry'])[0])
              for view, frame in frames]
    object_spec = dict(color=color, **current['descriptor'])
    results = []
    for model_name, packet in (('current', current), ('initial', initial)):
        for mask_name, views in (('robot_mask', masked), ('no_robot_mask', frames)):
            observer = ns['MixedPnPObservation']([object_spec], correspondence=True)
            model = copy.deepcopy(packet['visual_model'])
            result = {'model': model_name, 'mask': mask_name, 'input_model': model}
            try:
                measured = observer.observe_views(views, color, copy.deepcopy(model), current['table_plane'])
                result.update(status='accepted', center_m=measured['center'],
                    position_fully_observed=measured['position_fully_observed'])
            except (ValueError, RuntimeError) as error:
                result.update(status='rejected', error=str(error))
            # Inspect finite bounds only; this does not change the live fitter's
            # 1.5 mm gate or certify an accepted/recovered pose.
            try:
                fit = ns['observe_cuboid_correspondence'](views, color, object_spec['shape'],
                    copy.deepcopy(model), current['table_plane'], max_hidden_width_m=1.)
                result['diagnostic_bounds'] = {key: fit[key] for key in (
                    'rotation', 'center', 'constrained_axes', 'finite_center_bounds_m',
                    'finite_center_width_m', 'axis_dimensions', 'temporal_pose_prior_used')}
            except (ValueError, RuntimeError) as error:
                result['diagnostic_bounds_error'] = str(error)
            results.append(plain(result))
    if results[0]['status'] != 'rejected': raise ValueError('Exact input did not reproduce failure')
    sources = [Path(__file__), library, root/'equivalence.json', trial/'geometry_records.json', *frame_paths]
    comparison = None
    if attachment_replay is not None:
        sources.append(attachment_replay)
        replay = read(attachment_replay)
        predictions = [r for t in replay['trials'] if Path(t['trial']).name == trial.name
                       for r in t['records'] if r['object'] == color]
        if len(predictions) != 1: raise ValueError('Require one matching conditional prediction')
        prediction = predictions[0]
        # Verify the replay's current camera time, not just the object label.
        if abs(prediction['time'] - stamp) > 1e-6:
            raise ValueError('Conditional prediction has a different camera time')
        fit = results[0]['diagnostic_bounds']
        local = np.asarray(prediction['predicted_center_m']) @ np.asarray(fit['rotation'])
        bounds = np.asarray(fit['finite_center_bounds_m'])
        gap = np.maximum(np.maximum(bounds[:, 0] - local, local - bounds[:, 1]), 0)
        comparison = plain({'predicted_center_m': prediction['predicted_center_m'],
            'prediction_in_fitted_basis_m': local, 'outside_visual_bounds_m': gap,
            'inside_all_visual_bounds': bool(np.all(gap == 0)),
            'pose_fused': False,
            'interpretation': 'A non-contradicted enclosing sphere does not establish compatibility with tighter visual constraints.'})
    report = {'scope': __doc__, 'seed': certificate['seed'], 'object': color,
        'time': stamp, 'initial_model_time': initial['time'], 'variants': results,
        'pose_recovered': False, 'live_observer_changed': False,
        'conditional_prediction_comparison': comparison,
        'limitations': ['Single development failure, not success-rate evidence.',
            'Relaxed bounds are diagnostic only; they are not calibrated confidence intervals.',
            'Public robot geometry and RGB-D-derived dimensions only; no object truth or success labels.'],
        'input_sha256': {str(p): digest(p) for p in sources}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attachment-replay', type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.attachment_replay)
