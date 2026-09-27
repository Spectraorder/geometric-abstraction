"""OFFLINE EVALUATOR ONLY: score frozen points against replay segmentation.

Reads privileged saved simulator states exclusively for scoring. Never import
this module into a runtime policy or provide its outputs to a pointing agent.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np


def point_outcome(mask, prediction):
    visible = bool(np.any(mask))
    if prediction.get('status') == 'abstain' and prediction.get('point_px') is None:
        return 'false_abstention' if visible else 'correct_abstention'
    point = prediction.get('point_px')
    if (prediction.get('status') != 'point' or not isinstance(point, list) or len(point) != 2
            or any(type(x) not in (int, float) for x in point)
            or not np.all(np.isfinite(point))):
        return 'invalid_prediction'
    u, v = point
    if not (0 <= u < mask.shape[1] and 0 <= v < mask.shape[0]):
        return 'out_of_bounds'
    if mask[int(v), int(u)]:
        return 'target_hit'
    return 'wrong_pixel' if visible else 'false_positive_absent'


def main():
    import mujoco
    from aspire_so100.playback import recorded_env_for_result

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frames = json.loads(args.frames.read_text())
    raw = json.loads(args.predictions.read_text())
    predictions = {p['task_id']: p for p in raw}
    ids = [f['task_id'] for f in frames]
    if len(predictions) != len(raw) or len(set(ids)) != len(ids) or set(ids) != predictions.keys():
        raise ValueError('Exactly one frozen prediction and frame per unique task required')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in frames:
        trial = Path(entry['trial'])
        result = json.loads((trial / 'result.json').read_text())
        with np.load(entry['frame_path'], allow_pickle=False) as data:
            frame = dict(data)
        with np.load(trial / 'rollout.npz', allow_pickle=False) as archive:
            states = archive['states']
            state_index = int(np.argmin(abs(states[:, 0] - float(frame['time']))))
            state = states[state_index].copy()
            state_type = int(archive['state_type'])
        with tempfile.TemporaryDirectory() as temp:
            env = recorded_env_for_result(result, Path(temp))
            env.pixel_aligned_rgbd = result.get('sensor', {}).get('depth_sampling') == 'pixel_center'
            try:
                mujoco.mj_setState(env.model, env.data, state, state_type)
                mujoco.mj_forward(env.model, env.data)
                replay = env.get_overhead_rgbd()
                time_error = abs(float(replay['time']) - float(frame['time']))
                depth_error = float(np.max(abs(replay['depth_m'] - frame['depth_m'])))
                calibration_equal = all(np.allclose(replay[k], frame[k], rtol=0, atol=1e-12)
                                        for k in ('intrinsics', 'camera_to_world'))
                if (time_error > 1e-8 or depth_error > 1e-6 or not calibration_equal
                        or not np.array_equal(replay['valid_depth'], frame['valid_depth'])):
                    raise ValueError('Replay does not align to archived sensor: ' + entry['task_id'])
                sensor = env._overhead_rgbd
                sensor.renderer.enable_segmentation_rendering()
                sensor.renderer.update_scene(env.data, camera=sensor.camera_id)
                segmentation = sensor.renderer.render().copy()
                is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
                target_name = entry['evaluation_target_object']
                target_geoms = [i for i in range(env.model.ngeom)
                                if env.model.geom(i).name.startswith(target_name + '/')]
                mask = is_geom & np.isin(segmentation[..., 0], target_geoms)
                prediction = predictions[entry['task_id']]
                outcome = point_outcome(mask, prediction)
                hit_name = None
                point = prediction.get('point_px')
                if outcome not in ('invalid_prediction', 'out_of_bounds') and point is not None:
                    u, v = map(int, point)
                    if is_geom[v, u]:
                        hit_name = env.model.geom(int(segmentation[v, u, 0])).name
                np.savez_compressed(args.output / (entry['task_id'] + '_offline_mask.npz'), target_mask=mask)
                rows.append({'task_id': entry['task_id'], 'target_object': target_name,
                             'target_in_scene': bool(target_geoms), 'visible_target_pixels': int(mask.sum()),
                             'stratum': entry['stratum'], 'distractors': entry['distractors'],
                             'outcome': outcome, 'geom_at_proposed_pixel': hit_name,
                             'replay_alignment': {'state_index': state_index, 'time_error_seconds': time_error,
                                                  'max_depth_error_m': depth_error, 'calibration_equal': calibration_equal,
                                                  'rgb_equal': bool(np.array_equal(replay['rgb'], frame['rgb']))}})
                print(entry['task_id'], outcome, 'visible_pixels', int(mask.sum()), flush=True)
            finally:
                env.close()
    summary = {name: sum(row['outcome'] == name for row in rows) for name in
               ('target_hit', 'correct_abstention', 'false_abstention', 'wrong_pixel',
                'false_positive_absent', 'invalid_prediction', 'out_of_bounds')}
    summary['total'] = len(rows)
    summary['visible_target_cases'] = sum(row['visible_target_pixels'] > 0 for row in rows)
    summary['absent_or_invisible_cases'] = len(rows) - summary['visible_target_cases']
    summary['correct'] = summary['target_hit'] + summary['correct_abstention']
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'boundary': 'Privileged simulator replay segmentation used ONLY after frozen RGB-only predictions, for offline scoring. Not runtime input.',
              'predictions_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
              'score_definition': 'Target hit iff floor(u),floor(v) is in the visible target geom mask; correct abstention iff no target pixels. No snapping or tolerance radius.',
              'alignment_note': 'Require matching timestamp, calibration, validity and depth (<=1e-6 m). Replayed RGB is logged separately and is never substituted for actor RGB.',
              'summary': summary, 'cases': rows}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
